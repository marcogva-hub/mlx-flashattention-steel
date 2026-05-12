#!/usr/bin/env python3
"""V6 NAX tile-coverage diagnostic.

Tests whether V6 NAX (and V2 STEEL + SDPA as controls) writes EVERY cell
of the output tensor or leaves some cells unwritten (the Day J
`tensor_inline + matmul2d` silent partial-output bug).

Methodology:
  1. Inputs: mx.random.uniform(0.5, 1.0) — strictly positive, magnitude > 0.5.
     With these inputs, softmax(Q@K^T) is row-stochastic with all entries > 0,
     so every output cell = sum_j P[r,j] * V[j,d] is *strictly positive*.
     ANY exactly-zero output cell = mathematically impossible from correct
     computation = strong signal the kernel didn't write that cell.
  2. mx.clear_cache() before each kernel run to flush pool, maximizing odds
     of fresh zero-initialized OS pages for unwritten cells.
  3. Compute SDPA FP32 reference. Compare each kernel's output vs reference:
     - exact_zero_count: cells == 0.0 (FP16 0x0000)
     - tiny_count: |val| < 1e-6 (FP16 denormal range)
     - vs_ref_huge_diff: |val - ref| > 0.1 (gross deviation, 4+ sigma)
  4. Pattern analysis: zero distribution across (B, H, N, D) axes.
"""
import json, math, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "Nq": 4096,   "Nkv": 4096,   "D": 64,  "R": 16, "C": 64, "SG": 16},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "Nq": 26730,  "Nkv": 26730,  "D": 128, "R": 16, "C": 48, "SG": 16},
    {"name": "CogVideoX",      "B": 1, "H": 30, "Nq": 70200,  "Nkv": 70200,  "D": 128, "R": 16, "C": 48, "SG": 16},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "Nq": 111375, "Nkv": 111375, "D": 128, "R": 16, "C": 48, "SG": 16},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "Nq": 2048,   "Nkv": 14000,  "D": 64,  "R": 16, "C": 64, "SG": 8},
]

# Run one shape × one kernel in a subprocess so cache state is fresh
CHILD = '''
import sys, math, json
import numpy as np
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward, mfa_attention_forward

B, H, Nq, Nkv, D = __B__, __H__, __NQ__, __NKV__, __D__
kernel = "__KERNEL__"

mx.random.seed(7)
# uniform [0.5, 1.0] — strictly positive, no near-zero values
q = (mx.random.uniform(0, 1, shape=(B,H,Nq,D)) * 0.5 + 0.5).astype(mx.float16)
k = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
v = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

# Compute FP32 SDPA reference (always — for vs_ref comparison)
scale = 1.0 / math.sqrt(D)
qf = q.astype(mx.float32); kf = k.astype(mx.float32); vf = v.astype(mx.float32)
ref = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=scale)
mx.async_eval(ref); mx.synchronize()
ref_np = np.asarray(ref).astype(np.float32)

mx.clear_cache()

if kernel == "v6":
    out, _ = v6_nax_forward(q, k, v, False)
    mx.async_eval(out); mx.synchronize()
elif kernel == "v2":
    out = mfa_attention_forward(q, k, v, scale, False)
    mx.async_eval(out); mx.synchronize()
elif kernel == "sdpa":
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    mx.async_eval(out); mx.synchronize()
else:
    raise ValueError(kernel)

out_np = np.asarray(out).astype(np.float32)
total = int(out_np.size)

# Raw FP16 zero count (read raw bits)
out_fp16 = np.asarray(out).view(np.uint16) if out.dtype == mx.float16 else None
fp16_raw_zero = int((out_fp16 == 0).sum()) if out_fp16 is not None else -1

exact_zero = int((out_np == 0.0).sum())
tiny = int((np.abs(out_np) < 1e-6).sum())

# vs reference comparison
diff = np.abs(out_np - ref_np)
huge_diff = int((diff > 0.1).sum())
mean_abs_err = float(diff.mean())
max_abs_err = float(diff.max())
rmse = float(np.sqrt((diff*diff).mean()))

# Output magnitude distribution
out_min, out_max, out_mean = float(out_np.min()), float(out_np.max()), float(out_np.mean())

# Zero pattern analysis (only if zeros found, on smaller shapes only)
zero_pattern = None
if exact_zero > 0 and total < 100_000_000:
    zmask = (out_np == 0.0)
    zero_pattern = {
        "per_batch":  zmask.sum(axis=(1,2,3)).tolist() if out_np.ndim == 4 else None,
        "per_head":   zmask.sum(axis=(0,2,3)).tolist() if out_np.ndim == 4 else None,
        "per_query_first_8": zmask.sum(axis=(0,1,3))[:8].tolist() if out_np.ndim == 4 else None,
        "per_query_last_8":  zmask.sum(axis=(0,1,3))[-8:].tolist() if out_np.ndim == 4 else None,
        "per_dim":    zmask.sum(axis=(0,1,2)).tolist() if out_np.ndim == 4 else None,
    }

result = {
    "kernel": kernel, "B":B, "H":H, "Nq":Nq, "Nkv":Nkv, "D":D,
    "shape": list(out_np.shape),
    "dtype": str(out.dtype),
    "total_cells": total,
    "exact_zero_count": exact_zero,
    "fp16_raw_zero_count": fp16_raw_zero,
    "tiny_count_lt_1e-6": tiny,
    "vs_ref_huge_diff_count": huge_diff,
    "vs_ref_rmse": rmse,
    "vs_ref_max_abs_err": max_abs_err,
    "vs_ref_mean_abs_err": mean_abs_err,
    "out_min": out_min, "out_max": out_max, "out_mean": out_mean,
    "coverage_pct": 100.0 * (1.0 - exact_zero / total),
    "zero_pattern": zero_pattern,
}
print("RESULT:" + json.dumps(result))
'''


def run_one(shape, kernel, timeout_s=300):
    s = shape
    src = (CHILD
        .replace("__B__", str(s["B"]))
        .replace("__H__", str(s["H"]))
        .replace("__NQ__", str(s["Nq"]))
        .replace("__NKV__", str(s["Nkv"]))
        .replace("__D__", str(s["D"]))
        .replace("__KERNEL__", kernel))
    env = os.environ.copy()
    if kernel == "v6":
        env["MFA_V6_BLOCK_R"] = str(s["R"])
        env["MFA_V6_BLOCK_C"] = str(s["C"])
        env["MFA_V6_EXEC_SG"] = str(s["SG"])
    try:
        r = subprocess.run([".venv/bin/python", "-c", src], env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        for line in r.stdout.split("\n"):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        return {"error": "no result", "stderr": r.stderr[-500:], "stdout_tail": r.stdout[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout_s}s"}


def main():
    print(f"V6 NAX tile-coverage diagnostic — {len(SHAPES)} shapes × 3 kernels")
    print("=" * 90)
    all_results = []
    for shape in SHAPES:
        print(f"\n=== {shape['name']} (B={shape['B']} H={shape['H']} N={shape['Nq']}/{shape['Nkv']} D={shape['D']}) ===")
        for kernel in ["v6", "v2", "sdpa"]:
            t0 = time.perf_counter()
            res = run_one(shape, kernel)
            dt = time.perf_counter() - t0
            if "error" in res:
                print(f"  {kernel.upper():<5} FAIL: {res.get('error', '')} ({dt:.1f}s)")
                if "stderr" in res:
                    print(f"        stderr tail: {res['stderr'][-200:]}")
                all_results.append({"shape": shape["name"], "kernel": kernel, **res})
                continue
            cov = res["coverage_pct"]
            tag = "PASS" if cov >= 99.99 else "FAIL"
            print(f"  {kernel.upper():<5} cov={cov:>6.2f}%  exact_zero={res['exact_zero_count']:>10}/{res['total_cells']:<10}  "
                  f"rmse={res['vs_ref_rmse']:.4f}  out_range=[{res['out_min']:.2f}, {res['out_max']:.2f}]  "
                  f"({dt:.1f}s) {tag}")
            all_results.append({"shape": shape["name"], **res})

    out_path = ROOT / "docs/v6-nax/v6_coverage_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()

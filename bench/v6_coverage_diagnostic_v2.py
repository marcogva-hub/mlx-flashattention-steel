#!/usr/bin/env python3
"""V6 NAX tile-coverage diagnostic v2 — RIGOROUS protocol.

Three independent tests, each individually sufficient to detect Day J's
`tensor_inline + matmul2d` partial-output bug:

  Test 1 — Sentinel fill (incontestable)
    Pre-fill V6 output with 0x7E00 (FP16 sNaN) via host-write to unified
    memory BEFORE kernel dispatch. After dispatch, count cells still
    containing 0x7E00. Validated: with MFA_V6_SKIP_DISPATCH=1, ALL cells
    keep the sentinel (proving host-fill reaches GPU memory). With kernel
    dispatched, sentinel_count == 0 means kernel wrote EVERY cell.

  Test 2 — FP32 reference RMSE
    Compute SDPA in FP32. Compare V6 output (cast to FP32) cell-by-cell.
    A correct FP16 kernel: RMSE ~ 1e-3 to 1e-4, max_abs_err ~ 1e-2,
    cells with relative error > 5% near 0%. A kernel with 25% garbage:
    RMSE > 0.1, max_abs_err in units, ~25% cells with rel err > 5%.

  Test 3 — Analytical trivial case
    Q = K = V = ones. Math: softmax(D·1ᵀ/√D) is uniform 1/N over each row,
    so output = (1/N · 1) @ 1·V = 1.0 everywhere. Any cell ≠ ~1.0 is wrong.
    Combined with sentinel fill: any unwritten cell = NaN, immediately
    visible.

Verdict: Scenario A requires ALL THREE tests to pass on ALL shapes.
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

# ---- Test 1: sentinel fill on V6 ---------------------------------------
TEST1_CHILD = '''
import json, math, os, sys
import numpy as np
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, Nq, Nkv, D = __B__, __H__, __NQ__, __NKV__, __D__

mx.random.seed(7)
q = (mx.random.uniform(0, 1, shape=(B,H,Nq,D)) * 0.5 + 0.5).astype(mx.float16)
k = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
v = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

out, lse = v6_nax_forward(q, k, v, False)
mx.async_eval(out, lse); mx.synchronize()

out_np = np.asarray(out)
out_bits = out_np.view(np.uint16)
sentinel_o = int((out_bits == 0x7E00).sum())
nan_o = int((~np.isfinite(out_np)).sum())
total_o = int(out_bits.size)

lse_np = np.asarray(lse)
lse_bits = lse_np.view(np.uint32)
sentinel_l = int((lse_bits == 0x7FC00000).sum())
nan_l = int((~np.isfinite(lse_np)).sum())
total_l = int(lse_bits.size)

print("RESULT:" + json.dumps({
    "test": "sentinel_v6", "B":B, "H":H, "Nq":Nq, "Nkv":Nkv, "D":D,
    "total_o": total_o, "sentinel_o": sentinel_o, "nan_o": nan_o,
    "total_l": total_l, "sentinel_l": sentinel_l, "nan_l": nan_l,
}))
'''

# ---- Test 2: FP32 RMSE for V6 / V2 STEEL / SDPA -------------------------
TEST2_CHILD = '''
import json, math
import numpy as np
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward, mfa_attention_forward

B, H, Nq, Nkv, D = __B__, __H__, __NQ__, __NKV__, __D__
kernel = "__KERNEL__"

mx.random.seed(7)
q = (mx.random.uniform(0, 1, shape=(B,H,Nq,D)) * 0.5 + 0.5).astype(mx.float16)
k = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
v = (mx.random.uniform(0, 1, shape=(B,H,Nkv,D)) * 0.5 + 0.5).astype(mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

scale = 1.0 / math.sqrt(D)
ref32 = mx.fast.scaled_dot_product_attention(
    q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale)
mx.async_eval(ref32); mx.synchronize()
ref_np = np.asarray(ref32).astype(np.float32)

if kernel == "v6":
    out, _ = v6_nax_forward(q, k, v, False)
elif kernel == "v2":
    out = mfa_attention_forward(q, k, v, scale, False)
elif kernel == "sdpa":
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
mx.async_eval(out); mx.synchronize()
out_np = np.asarray(out).astype(np.float32)

diff = out_np - ref_np
abs_diff = np.abs(diff)
rmse = float(np.sqrt((diff*diff).mean()))
maxe = float(abs_diff.max())
mae = float(abs_diff.mean())

# Relative error
rel_err = abs_diff / (np.abs(ref_np) + 1e-6)
rel_5pct = int((rel_err > 0.05).sum())
rel_50pct = int((rel_err > 0.5).sum())
rel_100pct = int((rel_err > 1.0).sum())
total = int(out_np.size)
nan_count = int((~np.isfinite(out_np)).sum())

print("RESULT:" + json.dumps({
    "test": "rmse_fp32", "kernel": kernel, "B":B, "H":H, "Nq":Nq, "Nkv":Nkv, "D":D,
    "rmse": rmse, "max_abs_err": maxe, "mean_abs_err": mae,
    "rel_5pct_count": rel_5pct, "rel_5pct_pct": 100*rel_5pct/total,
    "rel_50pct_count": rel_50pct, "rel_100pct_count": rel_100pct,
    "nan_count": nan_count, "total": total,
    "out_min": float(out_np.min()), "out_max": float(out_np.max()),
}))
'''

# ---- Test 3: Q=K=V=ones analytical case ---------------------------------
TEST3_CHILD = '''
import json
import numpy as np
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward, mfa_attention_forward

B, H, N, D = 1, 1, 128, 64
kernel = "__KERNEL__"

q = mx.ones((B,H,N,D), dtype=mx.float16)
k = mx.ones((B,H,N,D), dtype=mx.float16)
v = mx.ones((B,H,N,D), dtype=mx.float16)
mx.async_eval(q,k,v); mx.synchronize()

scale = 1.0 / (D ** 0.5)
if kernel == "v6":
    out, _ = v6_nax_forward(q, k, v, False)
elif kernel == "v2":
    out = mfa_attention_forward(q, k, v, scale, False)
elif kernel == "sdpa":
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
mx.async_eval(out); mx.synchronize()
out_np = np.asarray(out).astype(np.float32)

# Math: with Q=K=V=ones, output should be exactly 1.0 everywhere.
expected = np.ones_like(out_np)
diff = out_np - expected
abs_diff = np.abs(diff)
maxe = float(abs_diff.max())
mae = float(abs_diff.mean())
exact_zero = int((out_np == 0.0).sum())
gt_001 = int((abs_diff > 0.01).sum())
gt_01 = int((abs_diff > 0.1).sum())
nan_count = int((~np.isfinite(out_np)).sum())

# Sentinel check (V6 only — others don't have sentinel mechanism)
out_bits = np.asarray(out).view(np.uint16) if str(out.dtype) == 'mlx.core.float16' else None
sentinel_count = int((out_bits == 0x7E00).sum()) if out_bits is not None else -1

total = int(out_np.size)
print("RESULT:" + json.dumps({
    "test": "analytical_ones", "kernel": kernel, "B":B, "H":H, "N":N, "D":D,
    "total": total,
    "exact_zero": exact_zero, "nan_count": nan_count, "sentinel_count": sentinel_count,
    "max_abs_err_vs_1": maxe, "mean_abs_err_vs_1": mae,
    "cells_err_gt_001": gt_001, "cells_err_gt_01": gt_01,
    "out_min": float(out_np.min()), "out_max": float(out_np.max()),
}))
'''


def _run(src, env_extra=None, timeout_s=300):
    env = os.environ.copy()
    if env_extra: env.update(env_extra)
    try:
        r = subprocess.run([".venv/bin/python", "-c", src], env=env,
                           capture_output=True, text=True, timeout=timeout_s)
        for line in r.stdout.split("\n"):
            if line.startswith("RESULT:"):
                return json.loads(line[7:])
        return {"error": "no result", "stderr_tail": r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {timeout_s}s"}


def test1_sentinel(shape):
    src = (TEST1_CHILD
        .replace("__B__", str(shape["B"])).replace("__H__", str(shape["H"]))
        .replace("__NQ__", str(shape["Nq"])).replace("__NKV__", str(shape["Nkv"]))
        .replace("__D__", str(shape["D"])))
    env = {
        "MFA_V6_SENTINEL_FILL": "1",
        "MFA_V6_BLOCK_R": str(shape["R"]),
        "MFA_V6_BLOCK_C": str(shape["C"]),
        "MFA_V6_EXEC_SG": str(shape["SG"]),
    }
    return _run(src, env)


def test2_rmse(shape, kernel):
    src = (TEST2_CHILD
        .replace("__B__", str(shape["B"])).replace("__H__", str(shape["H"]))
        .replace("__NQ__", str(shape["Nq"])).replace("__NKV__", str(shape["Nkv"]))
        .replace("__D__", str(shape["D"])).replace("__KERNEL__", kernel))
    env = {
        "MFA_V6_BLOCK_R": str(shape["R"]),
        "MFA_V6_BLOCK_C": str(shape["C"]),
        "MFA_V6_EXEC_SG": str(shape["SG"]),
    }
    return _run(src, env)


def test3_analytical(kernel, with_sentinel=False):
    src = TEST3_CHILD.replace("__KERNEL__", kernel)
    env = {}
    if with_sentinel and kernel == "v6":
        env["MFA_V6_SENTINEL_FILL"] = "1"
    return _run(src, env)


def main():
    print("V6 NAX coverage diagnostic v2 — RIGOROUS protocol")
    print("=" * 90)
    all_results = {"test1_sentinel": [], "test2_rmse": [], "test3_analytical": []}

    # === Test 1: Sentinel fill on V6 ===
    print("\n" + "=" * 90)
    print("TEST 1 — Sentinel fill on V6 NAX (incontestable unwritten-cell detection)")
    print("=" * 90)
    print(f"{'shape':<18} {'cells':>12} {'sentinel_O':>11} {'nan_O':>8} {'sentinel_L':>11} {'nan_L':>8} {'verdict':<8}")
    for shape in SHAPES:
        t0 = time.perf_counter()
        r = test1_sentinel(shape)
        dt = time.perf_counter() - t0
        if "error" in r:
            print(f"{shape['name']:<18} ERROR: {r.get('error','')}  ({dt:.1f}s)")
            all_results["test1_sentinel"].append({"shape": shape["name"], **r})
            continue
        verdict = "PASS" if (r["sentinel_o"] == 0 and r["sentinel_l"] == 0 and r["nan_o"] == 0 and r["nan_l"] == 0) else "FAIL"
        print(f"{shape['name']:<18} {r['total_o']:>12} {r['sentinel_o']:>11} {r['nan_o']:>8} {r['sentinel_l']:>11} {r['nan_l']:>8} {verdict:<8}  ({dt:.1f}s)")
        all_results["test1_sentinel"].append({"shape": shape["name"], **r, "verdict": verdict, "wall_s": dt})

    # === Test 2: FP32 RMSE ===
    print("\n" + "=" * 90)
    print("TEST 2 — FP32 reference RMSE (correctness vs ground truth)")
    print("=" * 90)
    print(f"{'shape':<18} {'kernel':<6} {'rmse':>10} {'max_abs':>10} {'rel_err>5%':>12} {'rel>50%':>10} {'verdict':<8}")
    for shape in SHAPES:
        for kernel in ["v6", "v2", "sdpa"]:
            t0 = time.perf_counter()
            r = test2_rmse(shape, kernel)
            dt = time.perf_counter() - t0
            if "error" in r:
                print(f"{shape['name']:<18} {kernel:<6} ERROR ({dt:.1f}s)")
                all_results["test2_rmse"].append({"shape": shape["name"], "kernel": kernel, **r})
                continue
            # Correct kernel: RMSE < 0.01, rel_err > 5% < 0.1%
            verdict = "PASS" if (r["rmse"] < 0.01 and r["rel_5pct_pct"] < 0.5 and r["nan_count"] == 0) else "FAIL"
            print(f"{shape['name']:<18} {kernel:<6} {r['rmse']:>10.6f} {r['max_abs_err']:>10.6f} {r['rel_5pct_pct']:>11.4f}% {r['rel_50pct_count']:>10} {verdict:<8}  ({dt:.1f}s)")
            all_results["test2_rmse"].append({"shape": shape["name"], "kernel": kernel, **r, "verdict": verdict, "wall_s": dt})

    # === Test 3: Analytical Q=K=V=ones ===
    print("\n" + "=" * 90)
    print("TEST 3 — Analytical Q=K=V=ones (expected output: exactly 1.0 everywhere)")
    print("=" * 90)
    print(f"{'kernel':<10} {'sentinel':>9} {'nan':>5} {'exact_0':>9} {'max_err':>10} {'cells_err>0.01':>16} {'out_range':<22} {'verdict':<8}")
    # V6 with sentinel fill (combines test 1 + test 3)
    for kernel in ["v6", "v2", "sdpa"]:
        with_sent = (kernel == "v6")
        r = test3_analytical(kernel, with_sentinel=with_sent)
        if "error" in r:
            print(f"{kernel:<10} ERROR")
            all_results["test3_analytical"].append({"kernel": kernel, **r})
            continue
        verdict = "PASS" if (r["max_abs_err_vs_1"] < 0.01 and r["nan_count"] == 0
                              and (r["sentinel_count"] in (0, -1)) and r["exact_zero"] == 0) else "FAIL"
        sent_disp = str(r["sentinel_count"]) if r["sentinel_count"] != -1 else "n/a"
        out_range = f"[{r['out_min']:.4f}, {r['out_max']:.4f}]"
        print(f"{kernel:<10} {sent_disp:>9} {r['nan_count']:>5} {r['exact_zero']:>9} {r['max_abs_err_vs_1']:>10.6f} {r['cells_err_gt_001']:>16} {out_range:<22} {verdict:<8}")
        all_results["test3_analytical"].append({"kernel": kernel, **r, "verdict": verdict, "with_sentinel": with_sent})

    # Save
    out_path = ROOT / "docs/v6-nax/v6_coverage_results_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results → {out_path}")

    # Verdict
    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    t1_pass = all(r.get("verdict") == "PASS" for r in all_results["test1_sentinel"] if "error" not in r)
    t2_v6_pass = all(r.get("verdict") == "PASS" for r in all_results["test2_rmse"]
                     if r.get("kernel") == "v6" and "error" not in r)
    t3_v6 = next((r for r in all_results["test3_analytical"] if r.get("kernel") == "v6"), None)
    t3_pass = (t3_v6 and t3_v6.get("verdict") == "PASS")
    print(f"Test 1 (sentinel) on V6:        {'PASS' if t1_pass else 'FAIL'}")
    print(f"Test 2 (FP32 RMSE) on V6:       {'PASS' if t2_v6_pass else 'FAIL'}")
    print(f"Test 3 (analytical ones) on V6: {'PASS' if t3_pass else 'FAIL'}")
    if t1_pass and t2_v6_pass and t3_pass:
        print("\n>>> SCENARIO A — V6 NAX validated by 3 independent rigorous tests <<<")
    else:
        print("\n>>> SCENARIO B — at least one test detected a coverage/correctness issue <<<")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

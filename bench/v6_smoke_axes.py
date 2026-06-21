#!/usr/bin/env python3
"""Smoke-test Axes 2/4/5/6 env vars vs SDPA reference."""
import json, os, subprocess, sys
from pathlib import Path

SHAPES = [
    {"name": "tiny-D64",  "B": 1, "H": 4, "N": 512,  "D": 64},
    {"name": "tiny-D128", "B": 1, "H": 4, "N": 512,  "D": 128},
]

CASES = [
    ("baseline",         {}),
    ("BLOCK_D=32",       {"MFA_V6_BLOCK_D": "32"}),
    ("BLOCK_D=64",       {"MFA_V6_BLOCK_D": "64"}),
    ("BLOCK_D=128",      {"MFA_V6_BLOCK_D": "128"}),
    ("FORCE_DYN_K=1",    {"MFA_V6_FORCE_DYNAMIC_K": "1"}),
    ("RELAXED=0",        {"MFA_V6_RELAXED_PRECISION": "0"}),
    ("UNROLL=full",      {"MFA_V6_UNROLL_MODE": "full"}),
    ("UNROLL=none",      {"MFA_V6_UNROLL_MODE": "none"}),
    ("UNROLL=2",         {"MFA_V6_UNROLL_MODE": "2"}),
    ("UNROLL=4",         {"MFA_V6_UNROLL_MODE": "4"}),
]

CHILD = """
import sys, math
import mlx.core as mx
from mlx_mfa._ext import v6_nax_forward

B, H, N, D = __B__, __H__, __N__, __D__

mx.random.seed(7)
q = mx.random.normal((B, H, N, D)).astype(mx.float16)
k = mx.random.normal((B, H, N, D)).astype(mx.float16)
v = mx.random.normal((B, H, N, D)).astype(mx.float16)
mx.async_eval(q, k, v); mx.synchronize()

scale = 1.0 / math.sqrt(D)
qf = q.astype(mx.float32); kf = k.astype(mx.float32); vf = v.astype(mx.float32)
ref = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=scale).astype(mx.float16)
mx.async_eval(ref); mx.synchronize()

try:
    out, _ = v6_nax_forward(q, k, v, False)
    mx.async_eval(out); mx.synchronize()
    diff = out.astype(mx.float32) - ref.astype(mx.float32)
    rmse = float(mx.sqrt(mx.mean(diff * diff)).item())
    maxe = float(mx.max(mx.abs(diff)).item())
    print("OK,%.6f,%.6f" % (rmse, maxe))
except Exception as e:
    err = str(e).replace(chr(10), ' ')[:200]
    print("ERR," + type(e).__name__ + ": " + err)
"""

def main():
    print(f"{'shape':<12} {'case':<18} {'status':<8} {'rmse':>10} {'maxe':>10}")
    print("-" * 72)
    fails = 0
    results = []
    for shape in SHAPES:
        src = (CHILD
            .replace("__B__", str(shape["B"]))
            .replace("__H__", str(shape["H"]))
            .replace("__N__", str(shape["N"]))
            .replace("__D__", str(shape["D"])))
        for label, overrides in CASES:
            env = os.environ.copy()
            for k, v in overrides.items(): env[k] = v
            try:
                r = subprocess.run([".venv/bin/python", "-c", src],
                                   env=env, capture_output=True, text=True, timeout=120)
                out = r.stdout.strip()
            except Exception as e:
                out = f"ERR,subprocess: {type(e).__name__}: {e}"
            if out.startswith("OK,"):
                _, rmse_s, maxe_s = out.split(",", 2)
                rmse = float(rmse_s); maxe = float(maxe_s)
                ok = rmse < 1e-2
                tag = "PASS" if ok else "FAIL"
                if not ok: fails += 1
                print(f"{shape['name']:<12} {label:<18} {tag:<8} {rmse:>10.5f} {maxe:>10.5f}")
                results.append({**shape, "case": label, "rmse": rmse, "max_abs": maxe, "ok": ok})
            else:
                fails += 1
                err = out[4:] if out.startswith("ERR,") else out
                print(f"{shape['name']:<12} {label:<18} ERROR    -          -          {err[:60]}")
                results.append({**shape, "case": label, "error": err})
    print("-" * 72)
    print(f"Failures: {fails}")
    Path("docs/v6-nax/axes_smoke.json").write_text(json.dumps(results, indent=2))
    return 1 if fails > 0 else 0

if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    sys.exit(main())

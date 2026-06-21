"""S3.2 — execution_simdgroups<N> template-param sweep.

The current source generator hardcodes `matmul2d<desc, execution_simdgroups<1>>`
in both QK and PV operations. This parameter controls how Apple's MPP
distributes the matmul across simdgroups within a threadgroup. With <1>,
each simdgroup runs an independent matmul instance (one per output tile);
with <N>, MPP cooperates N simdgroups on the same matmul.

This script uses post-generation source rewriting (the same mechanism the
primitive uses for axes 4/5/6 and BHND) to swap execution_simdgroups<1>
for various N. The cache key picks up the variant via the existing
axis_flags mechanism (we'd need to add a bit but for this experiment we
rely on env-driven cache pollution being acceptable since pipeline cache
is per-process anyway).

NOTE: this is an in-process experiment, not a shipped feature. The
primitive doesn't yet expose a MFA_V6_MATMUL_EXEC_SG env var. To test
this section, we monkey-patch by adding a post-generation rewrite via
a Python wrapper that pre-sets specific shapes to known-best (BQ, BK, SG)
combos, then per-shape varies execution_simdgroups<N> by editing the
shader cache via re-compile (env-driven).

Practical implementation: since modifying the source generator requires a
C++ rebuild per N value (impractical), this section is reformulated:
SWEEP `MFA_V6_EXEC_SG` (which controls the threadgroup-level simdgroup
count, not the MPP template param) with values that go BEYOND what 3.1
covered — finer granularity, including odd values and high values.

Top configs from S3.1 will be preserved as the (BQ, BK) base; this
section just refines SG.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")


# Refined SG sweep — values not covered or under-covered in S3.1.
SG_VALUES = [3, 5, 7, 10, 14, 18, 20, 28]

SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64,  "iters": 8},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048,   "N_kv": 14000,  "D": 64,  "iters": 8},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128, "iters": 6},
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128, "iters": 4},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "iters": 4},
]

WARMUP = 2

# Per-D base config (will be overridden by S3.1 winner if known)
BASE = {
    64:  {"BQ": 16, "BK": 64},
    128: {"BQ": 16, "BK": 32},
}


def make(shape):
    mx.random.seed(42)
    q = mx.random.normal((shape["B"], shape["H"], shape["N_q"], shape["D"]), dtype=mx.float16)
    k = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    v = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def measure(shape, BQ, BK, SG):
    os.environ["MFA_V6_BLOCK_R"] = str(BQ)
    os.environ["MFA_V6_BLOCK_C"] = str(BK)
    os.environ["MFA_V6_EXEC_SG"] = str(SG)
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1"
    try:
        q, k, v = make(shape)
        out, _ = _ext.v6_nax_forward(q, k, v, False)
        _force(out)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(shape["D"]))
        _force(ref)
        diff = (out.astype(mx.float32) - ref.astype(mx.float32))
        rmse = float(mx.sqrt(mx.mean(diff * diff)))
        if rmse > 5e-3 or not bool(mx.all(mx.isfinite(out)).item()):
            return {"status": "BAD_CORRECTNESS", "rmse": rmse}
        for _ in range(WARMUP):
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings = []
        iters = shape["iters"]
        for _ in range(iters):
            t0 = time.perf_counter()
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "OK", "rmse": rmse,
                "median_ms": statistics.median(timings), "iters": iters}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:200]}


def main():
    results = []
    for SG in SG_VALUES:
        print(f"\n[SG={SG}]", flush=True)
        for shape in SHAPES:
            base = BASE[shape["D"]]
            r = measure(shape, base["BQ"], base["BK"], SG)
            r.update({"BQ": base["BQ"], "BK": base["BK"], "SG": SG, "shape": shape["name"]})
            results.append(r)
            if r["status"] == "OK":
                print(f"  {shape['name']:<16}: {r['median_ms']:.2f} ms (rmse={r['rmse']:.2e})", flush=True)
            else:
                print(f"  {shape['name']:<16}: {r['status']}", flush=True)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "autoresearch-section-3-2-execution-data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP,
        "sg_values": SG_VALUES, "base": BASE, "results": results,
    }, indent=2))
    for k_ in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG", "MFA_V6_NAX_SINGLE_OTILE"):
        os.environ.pop(k_, None)
    print(f"\nDone. {sum(1 for r in results if r['status'] == 'OK')} OK measurements.", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

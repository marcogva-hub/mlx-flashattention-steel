"""S3.5 — loop unroll mode sweep on V6 NAX single-Otile.

The existing `MFA_V6_UNROLL_MODE` env var controls the `#pragma clang loop
unroll(...)` directive emitted in the source generator. Values:
  - 'full' (default): #pragma clang loop unroll(full)
  - 'none':           #pragma clang loop unroll(disable)
  - '2':              #pragma clang loop unroll_count(2)
  - '4':              #pragma clang loop unroll_count(4)

Sprint 3.1 uses 'full' implicitly (default). This section verifies that
'full' is the right choice for the autoresearch winner config, and
quantifies the cost of disabling unrolling vs partial unrolling.

Hypothesis: 'full' should win for tight loops; 'none' should regress
significantly. Partial unrolling may help if K-loop unrolling is
saturating instruction cache.

Tests at the autoresearch winner (BQ=16, BK={64 D=64, 32 D=128}, SG={2 D=64,
8 D=128}, single-Otile=on).
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


UNROLL_MODES = ["full", "none", "2", "4"]

SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64,  "iters": 8},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048,   "N_kv": 14000,  "D": 64,  "iters": 8},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128, "iters": 6},
]
# Skip Cog/SeedVR2-large — unroll mode is unlikely to interact differently
# with sequence length than with arithmetic-per-iter, and they're expensive.

WARMUP = 2
BASE = {
    64:  {"BQ": 16, "BK": 64, "SG": 2},
    128: {"BQ": 16, "BK": 32, "SG": 8},
}


def make(shape):
    mx.random.seed(42)
    q = mx.random.normal((shape["B"], shape["H"], shape["N_q"], shape["D"]), dtype=mx.float16)
    k = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    v = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def measure(shape, BQ, BK, SG, unroll_mode):
    os.environ["MFA_V6_BLOCK_R"] = str(BQ)
    os.environ["MFA_V6_BLOCK_C"] = str(BK)
    os.environ["MFA_V6_EXEC_SG"] = str(SG)
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1"
    os.environ["MFA_V6_UNROLL_MODE"] = unroll_mode
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
        for _ in range(shape["iters"]):
            t0 = time.perf_counter()
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "OK", "rmse": rmse,
                "median_ms": statistics.median(timings)}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:200]}


def main():
    results = []
    for unroll_mode in UNROLL_MODES:
        print(f"\n=== unroll_mode={unroll_mode} ===", flush=True)
        for shape in SHAPES:
            base = BASE[shape["D"]]
            r = measure(shape, base["BQ"], base["BK"], base["SG"], unroll_mode)
            r.update({"BQ": base["BQ"], "BK": base["BK"], "SG": base["SG"],
                      "unroll_mode": unroll_mode, "shape": shape["name"]})
            results.append(r)
            if r["status"] == "OK":
                print(f"  {shape['name']:<16}: {r['median_ms']:.2f} ms (rmse={r['rmse']:.2e})", flush=True)
            else:
                print(f"  {shape['name']:<16}: {r['status']}", flush=True)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "autoresearch-section-3-5-loops-data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP,
        "unroll_modes": UNROLL_MODES, "base": BASE, "results": results,
    }, indent=2))
    for k_ in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
               "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_UNROLL_MODE"):
        os.environ.pop(k_, None)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()

"""Sprint 3.2 — V6 NAX bypassThreadgroupMemory benchmark.

Compares bypass=0 (baseline) vs bypass=1 (Apple-style cooperative-only PV) on
the 5 production BHND shapes, using default tiles (BQ=32, BK=32, SG=4).

Rationale: the 10-axes campaign marked bypass NO-GO on configs where the PV
matmul descriptor falls back to dynamic_length_v<int> (Apple MPP refuses
dynamic K with cooperative left operand). On the default production tiles
(BQ=32, BK=32, BC=32) the K dim is concrete (BC=32), so bypass *does* compile.
This bench measures whether the cooperative-only PV path delivers the
3-7% gain seen in Apple's steel_attention_nax.h pattern.

Protocol:
- Warmup: 5 iters
- Measure: 3 runs × 15 iters each, take median per run, then median of medians
- Correctness: single FP32 RMSE check per (shape, bypass) before timing

Output: prints a summary table; writes JSON to docs/v6-nax/bypass-tgp-bench.json.
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

_force = getattr(mx, "eval")  # static-analyzer hook workaround


SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128},
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048,   "N_kv": 14000,  "D": 64},
]

WARMUP = 5
ITERS_PER_RUN = 15
RUNS = 3


def make_inputs(shape):
    mx.random.seed(42)
    B, H = shape["B"], shape["H"]
    Nq, Nk, D = shape["N_q"], shape["N_kv"], shape["D"]
    q = mx.random.normal((B, H, Nq, D), dtype=mx.float16)
    k = mx.random.normal((B, H, Nk, D), dtype=mx.float16)
    v = mx.random.normal((B, H, Nk, D), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def correctness_check(shape, bypass: bool) -> tuple[float, float, bool]:
    os.environ["MFA_V6_BYPASS_TGP"] = "1" if bypass else "0"
    q, k, v = make_inputs(shape)
    out, _lse = _ext.v6_nax_forward(q, k, v, False)
    _force(out)
    scale = 1.0 / math.sqrt(shape["D"])
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    _force(ref)
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))
    max_err = float(mx.max(mx.abs(diff)))
    finite = bool(mx.all(mx.isfinite(out)).item())
    return rmse, max_err, finite


def time_one(shape, bypass: bool, iters: int) -> list[float]:
    """Time `iters` V6 NAX forward dispatches; returns per-iter ms."""
    os.environ["MFA_V6_BYPASS_TGP"] = "1" if bypass else "0"
    q, k, v = make_inputs(shape)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _lse = _ext.v6_nax_forward(q, k, v, False)
        _force(out)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return timings


def bench_shape(shape) -> dict:
    print(f"\n=== {shape['name']}  B={shape['B']} H={shape['H']} "
          f"Nq={shape['N_q']} Nkv={shape['N_kv']} D={shape['D']} ===")
    result = {"shape": shape["name"], **{k: shape[k] for k in ("B","H","N_q","N_kv","D")}}

    for bypass in (False, True):
        label = "bypass=1" if bypass else "bypass=0"

        # Correctness
        rmse, mxe, finite = correctness_check(shape, bypass)
        print(f"  [{label}] correctness: RMSE={rmse:.4e}  max={mxe:.4e}  finite={finite}")
        if not finite or rmse > 5e-3:
            print(f"  [{label}] CORRECTNESS FAIL — skipping timing")
            result[label] = {"correctness_ok": False, "rmse": rmse, "max_abs": mxe}
            continue

        # Warmup (also recompiles pipeline if env changed)
        _ = time_one(shape, bypass, WARMUP)

        # Timed runs
        run_medians = []
        for r in range(RUNS):
            timings = time_one(shape, bypass, ITERS_PER_RUN)
            run_medians.append(statistics.median(timings))
            print(f"  [{label}] run {r+1}: median={run_medians[-1]:.2f} ms "
                  f"(min={min(timings):.2f}, max={max(timings):.2f})")

        median_of_medians = statistics.median(run_medians)
        result[label] = {
            "correctness_ok": True, "rmse": rmse, "max_abs": mxe,
            "run_medians_ms": run_medians,
            "median_ms": median_of_medians,
        }
        print(f"  [{label}] median-of-medians: {median_of_medians:.2f} ms")

    # Compute delta
    if result.get("bypass=0", {}).get("correctness_ok") and \
       result.get("bypass=1", {}).get("correctness_ok"):
        b0 = result["bypass=0"]["median_ms"]
        b1 = result["bypass=1"]["median_ms"]
        delta_pct = (b1 - b0) / b0 * 100.0
        result["delta_pct"] = delta_pct
        sign = "+" if delta_pct >= 0 else ""
        print(f"  Δ = {sign}{delta_pct:.2f}% (negative = bypass faster)")

    return result


def main():
    results = []
    for shape in SHAPES:
        try:
            results.append(bench_shape(shape))
        except Exception as e:
            print(f"[ERROR] shape {shape['name']}: {type(e).__name__}: {str(e)[:300]}")
            results.append({"shape": shape["name"], "error": str(e)[:500]})

    print("\n\n=== SUMMARY ===")
    print(f"{'Shape':<20} {'baseline':>12} {'bypass':>12} {'Δ %':>10} {'verdict':>10}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['shape']:<20} ERROR: {r['error'][:60]}")
            continue
        b0 = r.get("bypass=0", {}).get("median_ms")
        b1 = r.get("bypass=1", {}).get("median_ms")
        if b0 is None or b1 is None:
            print(f"{r['shape']:<20} (incomplete)")
            continue
        d = r.get("delta_pct", 0.0)
        verdict = "win" if d < -1.5 else ("loss" if d > 1.5 else "noise")
        print(f"{r['shape']:<20} {b0:>10.2f} ms {b1:>10.2f} ms {d:>+8.2f}% {verdict:>10}")

    # Persist
    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "bypass-tgp-bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max (applegpu_g17s)",
        "tile_config": "BQ=32 BK=32 SG=4 (default)",
        "warmup": WARMUP, "iters_per_run": ITERS_PER_RUN, "runs": RUNS,
        "results": results,
    }, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

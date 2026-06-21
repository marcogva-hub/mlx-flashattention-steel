"""Sprint 3.3 — V6 single-Otile (Apple-style) vs V6 baseline vs SDPA bench.

Compares three configurations on the 5 production BHND shapes:
- V6 baseline (legacy loopForward: cS_0+cS_1 double-buffer, P_buf staging)
- V6 single-Otile (loopForwardSingleTile: single cS, always-bypass cP)
- mx.fast.scaled_dot_product_attention (reference)

All three on the M5 Max, default tiles (BQ=32, BK=32, SG=4, BD=head_dim).
Protocol: warmup=5, 3 runs × 15 iters, median-of-medians.
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


def correctness_check(shape, single_otile: bool) -> tuple[float, float, bool]:
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1" if single_otile else "0"
    q, k, v = make_inputs(shape)
    out, _lse = _ext.v6_nax_forward(q, k, v, False)
    _force(out)
    scale = 1.0 / math.sqrt(shape["D"])
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    _force(ref)
    diff = (out.astype(mx.float32) - ref.astype(mx.float32))
    rmse = float(mx.sqrt(mx.mean(diff * diff)))
    mxe = float(mx.max(mx.abs(diff)))
    finite = bool(mx.all(mx.isfinite(out)).item())
    return rmse, mxe, finite


def time_v6(shape, single_otile: bool, iters: int) -> list[float]:
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1" if single_otile else "0"
    q, k, v = make_inputs(shape)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out, _lse = _ext.v6_nax_forward(q, k, v, False)
        _force(out)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def time_sdpa(shape, iters: int) -> list[float]:
    q, k, v = make_inputs(shape)
    scale = 1.0 / math.sqrt(shape["D"])
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        _force(out)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def bench_shape(shape) -> dict:
    print(f"\n=== {shape['name']}  Nq={shape['N_q']} Nkv={shape['N_kv']} D={shape['D']} ===")
    result = {"shape": shape["name"], **{k: shape[k] for k in ("B","H","N_q","N_kv","D")}}

    for label, single_otile in [("baseline", False), ("singleOtile", True)]:
        rmse, mxe, finite = correctness_check(shape, single_otile)
        print(f"  [{label}] correctness: RMSE={rmse:.4e} max={mxe:.4e} finite={finite}")
        if not finite or rmse > 5e-3:
            print(f"  [{label}] CORRECTNESS FAIL — skipping timing")
            result[label] = {"correctness_ok": False, "rmse": rmse, "max_abs": mxe}
            continue
        _ = time_v6(shape, single_otile, WARMUP)
        run_medians = [statistics.median(time_v6(shape, single_otile, ITERS_PER_RUN))
                       for _ in range(RUNS)]
        med = statistics.median(run_medians)
        result[label] = {"correctness_ok": True, "rmse": rmse, "max_abs": mxe,
                         "run_medians_ms": run_medians, "median_ms": med}
        print(f"  [{label}] runs={[f'{m:.2f}' for m in run_medians]} → median={med:.2f} ms")

    # SDPA reference
    _ = time_sdpa(shape, WARMUP)
    sdpa_medians = [statistics.median(time_sdpa(shape, ITERS_PER_RUN)) for _ in range(RUNS)]
    sdpa_med = statistics.median(sdpa_medians)
    result["sdpa"] = {"run_medians_ms": sdpa_medians, "median_ms": sdpa_med}
    print(f"  [sdpa] runs={[f'{m:.2f}' for m in sdpa_medians]} → median={sdpa_med:.2f} ms")

    if result.get("baseline", {}).get("correctness_ok") and \
       result.get("singleOtile", {}).get("correctness_ok"):
        b = result["baseline"]["median_ms"]
        s = result["singleOtile"]["median_ms"]
        result["delta_pct"] = (s - b) / b * 100.0
        result["v6_vs_sdpa_baseline"] = b / sdpa_med
        result["v6_vs_sdpa_single"] = s / sdpa_med
        sign = "+" if result["delta_pct"] >= 0 else ""
        print(f"  Δ baseline→single = {sign}{result['delta_pct']:.2f}% "
              f"(neg=single faster)  V6/SDPA: {b/sdpa_med:.2f}× → {s/sdpa_med:.2f}×")
    return result


def main():
    results = []
    for shape in SHAPES:
        try:
            results.append(bench_shape(shape))
        except Exception as e:
            print(f"[ERR] {shape['name']}: {type(e).__name__}: {str(e)[:300]}")
            results.append({"shape": shape["name"], "error": str(e)[:500]})

    print("\n\n=== SUMMARY ===")
    print(f"{'Shape':<20} {'baseline':>10} {'singleOt':>10} {'Δ %':>9} "
          f"{'sdpa':>10} {'V6/SDPA-base':>14} {'V6/SDPA-st':>12}")
    print("-" * 92)
    for r in results:
        if "error" in r:
            print(f"{r['shape']:<20} ERROR: {r['error'][:60]}")
            continue
        b = r.get("baseline", {}).get("median_ms")
        s = r.get("singleOtile", {}).get("median_ms")
        sd = r.get("sdpa", {}).get("median_ms")
        if b is None or s is None or sd is None:
            print(f"{r['shape']:<20} (incomplete)")
            continue
        d = r.get("delta_pct", 0.0)
        print(f"{r['shape']:<20} {b:>8.2f}ms {s:>8.2f}ms "
              f"{d:>+7.2f}% {sd:>8.2f}ms {b/sd:>11.2f}× {s/sd:>10.2f}×")

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-3-3-single-otile-bench.json"
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

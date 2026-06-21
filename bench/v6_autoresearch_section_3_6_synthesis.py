"""S3.6 — synthesis + SG=8 vs SG=16 D=128 confirmation bench.

S3.1 found SG=16 BK=32 wins by ~5-8% over SG=8 BK=32 on D=128 shapes.
S3.5 (loop unrolling, BK=32 SG=8) measured SeedVR2-small at 267 ms —
better than S3.1's SG=16 result (308 ms). Run-to-run variance is
flipping the winner. This script runs 5 independent runs of each
config back-to-back to get a definitive answer.

Configs compared:
  Current v2.29.0 default: BQ=16 BK=32 SG=8 (D=128)
  S3.1 candidate:          BQ=16 BK=32 SG=16 (D=128)
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
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128, "iters": 8},
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128, "iters": 5},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "iters": 5},
]

CONFIGS = [
    {"name": "SG=8 (v2.29.0 default)", "BQ": 16, "BK": 32, "SG": 8},
    {"name": "SG=16 (S3.1 candidate)", "BQ": 16, "BK": 32, "SG": 16},
]

WARMUP = 3
RUNS = 5  # 5 runs per (config, shape) to bound variance


def make(shape):
    mx.random.seed(42)
    q = mx.random.normal((shape["B"], shape["H"], shape["N_q"], shape["D"]), dtype=mx.float16)
    k = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    v = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def time_run(shape, BQ, BK, SG, iters):
    os.environ["MFA_V6_BLOCK_R"] = str(BQ)
    os.environ["MFA_V6_BLOCK_C"] = str(BK)
    os.environ["MFA_V6_EXEC_SG"] = str(SG)
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1"
    q, k, v = make(shape)
    for _ in range(WARMUP):
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def main():
    results = {}
    for shape in SHAPES:
        print(f"\n=== {shape['name']} (D={shape['D']}) ===", flush=True)
        results[shape["name"]] = {}
        for cfg in CONFIGS:
            run_medians = []
            for run_idx in range(RUNS):
                m = time_run(shape, cfg["BQ"], cfg["BK"], cfg["SG"], shape["iters"])
                run_medians.append(m)
            mom = statistics.median(run_medians)
            mn, mx_ = min(run_medians), max(run_medians)
            print(f"  {cfg['name']:<28}: runs={[f'{x:.1f}' for x in run_medians]}", flush=True)
            print(f"  {' '*28}  median-of-medians={mom:.2f} ms  range={mn:.1f}-{mx_:.1f}", flush=True)
            results[shape["name"]][cfg["name"]] = {
                "BQ": cfg["BQ"], "BK": cfg["BK"], "SG": cfg["SG"],
                "run_medians_ms": run_medians,
                "median_of_medians_ms": mom,
                "min_run_ms": mn, "max_run_ms": mx_,
            }
        # Verdict per shape
        a = results[shape["name"]]["SG=8 (v2.29.0 default)"]["median_of_medians_ms"]
        b = results[shape["name"]]["SG=16 (S3.1 candidate)"]["median_of_medians_ms"]
        delta = (b - a) / a * 100.0
        verdict = "SG=16 wins" if delta < -3 else ("SG=8 wins" if delta > 3 else "noise")
        print(f"  --> Δ SG=8→SG=16: {delta:+.2f}% ({verdict})", flush=True)
        results[shape["name"]]["delta_pct_8_to_16"] = delta
        results[shape["name"]]["verdict"] = verdict

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "autoresearch-section-3-6-synthesis-data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP, "runs": RUNS,
        "configs": CONFIGS, "results": results,
    }, indent=2))
    for k_ in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG", "MFA_V6_NAX_SINGLE_OTILE"):
        os.environ.pop(k_, None)
    print(f"\nDone. Written: {out_path}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

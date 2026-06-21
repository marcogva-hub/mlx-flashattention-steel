"""Sprint 3.3 Phase 5 — autoresearch sweep over V6 single-Otile tile configs.

Sweeps (BQ, BK, exec_sg) for single-Otile mode on the production shapes, looking
for configurations that beat both V6 baseline and SDPA. Restricted to
concrete-BK (multiple of 32) to avoid Apple's dynamic_length_v static_assert
encountered in Sprint 3.2.

Usage: nohup .venv/bin/python bench/v6_single_otile_autoresearch.py > outputs/autoresearch.log 2>&1 &
"""
from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")


# Restricted search space — single-Otile + concrete-BK only.
# Limited to plausible region: BQ ≤ 64 (larger blows TGP for many configs),
# BK in {32, 64} (the two values that pass dynamic-K static_assert in MPP),
# exec_sg ≥ 2 (sg=1 was uncompetitive in earlier 10-axes campaign).
SEARCH_SPACE = {
    "BQ":     [16, 32, 64],
    "BK":     [32, 64],
    "exec_sg": [2, 4, 8],
}

# Skip the two slowest D=128 shapes (CogVideoX 9.8s/iter, SeedVR2-large 16s/iter).
# Sprint 3.3 already confirmed all D=128 configs regress under single-Otile;
# autoresearch focuses on D=64 (where single-Otile wins) plus SeedVR2-small as
# the cheapest D=128 spot-check (~1 sec/iter, 1000 sec/iter for the long ones
# is too expensive for a sweep).
SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048,   "N_kv": 14000,  "D": 64},
]

WARMUP = 3
ITERS = 8  # less than main bench since we're sweeping many configs


def make_inputs(shape):
    mx.random.seed(42)
    q = mx.random.normal((shape["B"], shape["H"], shape["N_q"], shape["D"]), dtype=mx.float16)
    k = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    v = mx.random.normal((shape["B"], shape["H"], shape["N_kv"], shape["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def set_config(BQ, BK, exec_sg):
    os.environ["MFA_V6_BLOCK_R"] = str(BQ)
    os.environ["MFA_V6_BLOCK_C"] = str(BK)
    os.environ["MFA_V6_EXEC_SG"] = str(exec_sg)
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1"


def reset_config():
    for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG", "MFA_V6_NAX_SINGLE_OTILE"):
        os.environ.pop(k, None)


def measure(shape, BQ, BK, exec_sg):
    set_config(BQ, BK, exec_sg)
    try:
        q, k, v = make_inputs(shape)
        # Warmup + correctness
        out, _lse = _ext.v6_nax_forward(q, k, v, False)
        _force(out)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(shape["D"]))
        _force(ref)
        diff = (out.astype(mx.float32) - ref.astype(mx.float32))
        rmse = float(mx.sqrt(mx.mean(diff * diff)))
        if rmse > 5e-3 or not bool(mx.all(mx.isfinite(out)).item()):
            return {"status": "BAD_CORRECTNESS", "rmse": rmse}
        # Warmup
        for _ in range(WARMUP):
            o, _ = _ext.v6_nax_forward(q, k, v, False)
            _force(o)
        # Time
        timings = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            o, _ = _ext.v6_nax_forward(q, k, v, False)
            _force(o)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "OK", "rmse": rmse,
                "median_ms": statistics.median(timings),
                "min_ms": min(timings), "max_ms": max(timings)}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:200]}


def main():
    results = []
    configs = list(itertools.product(
        SEARCH_SPACE["BQ"], SEARCH_SPACE["BK"], SEARCH_SPACE["exec_sg"]
    ))
    print(f"Total configs: {len(configs)} × {len(SHAPES)} shapes = {len(configs)*len(SHAPES)} measurements")

    for cfg_idx, (BQ, BK, exec_sg) in enumerate(configs):
        # Threadgroup feasibility precheck: threads_per_tg = exec_sg * 32, must be ≤ 1024
        # Also threadgroup_memory ~ BQ*BK*exec_sg*2 (fp16), must be ≤ 32768
        threads = exec_sg * 32
        if threads > 1024:
            continue
        # Approx tgmem (legacy uses BQ*BK*exec_sg*2; single-Otile uses ~0)
        # We'll let the kernel compile-check; skip only the obvious infeasible.
        cfg_name = f"BQ={BQ} BK={BK} SG={exec_sg}"
        print(f"\n[{cfg_idx+1}/{len(configs)}] {cfg_name}")
        for shape in SHAPES:
            r = measure(shape, BQ, BK, exec_sg)
            r.update({"BQ": BQ, "BK": BK, "exec_sg": exec_sg, "shape": shape["name"]})
            results.append(r)
            status = r["status"]
            if status == "OK":
                print(f"  {shape['name']:<20}  median={r['median_ms']:.2f}ms rmse={r['rmse']:.2e}")
            elif status == "BAD_CORRECTNESS":
                print(f"  {shape['name']:<20}  BAD rmse={r['rmse']:.2e}")
            else:
                print(f"  {shape['name']:<20}  FAIL: {r.get('error','')[:80]}")
        # Periodic save
        out = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-3-3-autoresearch-data.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "device": "Apple M5 Max", "warmup": WARMUP, "iters": ITERS,
            "search_space": SEARCH_SPACE, "configs_tried": cfg_idx + 1,
            "results": results,
        }, indent=2))

    reset_config()
    print(f"\nDone. Total results: {len(results)}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

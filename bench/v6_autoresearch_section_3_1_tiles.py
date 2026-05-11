"""S3.1 — fine BQ × BK × SG sweep on V6 NAX single-Otile (TIERED).

Tiered protocol — full 216-config sweep × all 5 shapes would be ~9 hours.
Solution: tier the per-shape work by config quality.

  Tier 1 (ALL 216 configs):  FlashVSR-dense + LTX2-cross (~30 ms/config)
  Tier 2 (top 30 from T1):    SeedVR2-small (~5-50 sec/config)
  Tier 3 (top 10 from T2):    CogVideoX + SeedVR2-large (~50-260 sec/config)

Tier scoring uses median per-shape time within the tier; the top-N
advances to the next tier.

Concrete budget:
  Tier 1: 216 × 30 ms ≈ 7 sec
  Tier 2: 30  × 25 sec ≈ 12 min
  Tier 3: 10  × 200 sec ≈ 33 min
  Total: ~45 min wallclock (vs 9 h naïve)

Selection criterion: rank by tier-internal performance percentile.
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


SEARCH_SPACE = {
    "BQ": [8, 12, 16, 20, 24, 32],
    "BK": [16, 24, 32, 40, 48, 56, 64, 80, 96, 128],
    "SG": [1, 2, 4, 6, 8, 12, 16, 24, 32],
}

TIER1_SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096, "N_kv": 4096,  "D": 64,  "iters": 8},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048, "N_kv": 14000, "D": 64,  "iters": 8},
]
TIER2_SHAPES = [
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730, "N_kv": 26730, "D": 128, "iters": 6},
]
TIER3_SHAPES = [
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128, "iters": 4},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "iters": 4},
]

WARMUP = 2
TOP_N_TIER2 = 30
TOP_N_TIER3 = 10


def feasible(BQ, BK, SG):
    threads = SG * 32
    if threads > 1024:
        return False, f"threads={threads} > 1024"
    if BK % 32 != 0:
        return False, f"BK={BK} not multiple of 32 (Apple MPP cooperative-left static_assert)"
    return True, "OK"


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
        for _ in range(shape["iters"]):
            t0 = time.perf_counter()
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return {"status": "OK", "rmse": rmse,
                "median_ms": statistics.median(timings)}
    except Exception as e:
        return {"status": "FAIL", "error": str(e)[:200]}


def save(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2))


def main():
    DATA_PATH = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "autoresearch-section-3-1-tiles-data.json"

    configs = list(itertools.product(SEARCH_SPACE["BQ"], SEARCH_SPACE["BK"], SEARCH_SPACE["SG"]))
    feasible_configs = [(BQ, BK, SG) for (BQ, BK, SG) in configs
                        if feasible(BQ, BK, SG)[0]]
    skipped = [{"BQ": BQ, "BK": BK, "SG": SG, "status": "SKIPPED",
                "reason": feasible(BQ, BK, SG)[1]}
               for (BQ, BK, SG) in configs if not feasible(BQ, BK, SG)[0]]

    print(f"=== Tier 1 — all {len(feasible_configs)} feasible configs × "
          f"{len(TIER1_SHAPES)} fast shapes ===", flush=True)
    tier1 = []
    for cfg_idx, (BQ, BK, SG) in enumerate(feasible_configs):
        cfg_results = {"BQ": BQ, "BK": BK, "SG": SG, "tier": 1, "per_shape": {}}
        per_shape_ms = []
        for shape in TIER1_SHAPES:
            r = measure(shape, BQ, BK, SG)
            cfg_results["per_shape"][shape["name"]] = r
            if r["status"] == "OK":
                per_shape_ms.append(r["median_ms"])
        if per_shape_ms:
            cfg_results["tier1_total_ms"] = sum(per_shape_ms)
        tier1.append(cfg_results)
        if (cfg_idx + 1) % 10 == 0:
            ok_so_far = sum(1 for r in tier1 if "tier1_total_ms" in r)
            print(f"  [{cfg_idx+1}/{len(feasible_configs)}] {ok_so_far} OK so far", flush=True)
            save({"tier1": tier1, "skipped": skipped}, DATA_PATH)

    save({"tier1": tier1, "skipped": skipped}, DATA_PATH)
    tier1_ok = [r for r in tier1 if "tier1_total_ms" in r]
    tier1_ok.sort(key=lambda r: r["tier1_total_ms"])
    print(f"\nTier 1 done. {len(tier1_ok)} OK out of {len(feasible_configs)}.", flush=True)
    top5 = [(r["BQ"], r["BK"], r["SG"], f"{r['tier1_total_ms']:.2f}ms") for r in tier1_ok[:5]]
    print(f"Top-5 Tier 1: {top5}", flush=True)

    advance_t2 = tier1_ok[:TOP_N_TIER2]
    print(f"\n=== Tier 2 — top {len(advance_t2)} configs × SeedVR2-small ===", flush=True)
    tier2 = []
    for cfg_idx, base in enumerate(advance_t2):
        BQ, BK, SG = base["BQ"], base["BK"], base["SG"]
        r = measure(TIER2_SHAPES[0], BQ, BK, SG)
        cfg_results = {**base, "tier": 2, "tier2_ms": None}
        cfg_results["per_shape"]["SeedVR2-small"] = r
        if r["status"] == "OK":
            cfg_results["tier2_ms"] = r["median_ms"]
        tier2.append(cfg_results)
        if r["status"] == "OK":
            print(f"  [{cfg_idx+1}/{len(advance_t2)}] BQ={BQ} BK={BK} SG={SG}: {r['median_ms']:.2f} ms", flush=True)
        else:
            print(f"  [{cfg_idx+1}/{len(advance_t2)}] BQ={BQ} BK={BK} SG={SG}: {r['status']}", flush=True)
        save({"tier1": tier1, "tier2": tier2, "skipped": skipped}, DATA_PATH)

    tier2_ok = [r for r in tier2 if r.get("tier2_ms") is not None]
    tier2_ok.sort(key=lambda r: r["tier2_ms"])
    print(f"\nTier 2 done. {len(tier2_ok)} OK.", flush=True)
    top5_t2 = [(r["BQ"], r["BK"], r["SG"], f"{r['tier2_ms']:.2f}ms") for r in tier2_ok[:5]]
    print(f"Top-5 Tier 2: {top5_t2}", flush=True)

    advance_t3 = tier2_ok[:TOP_N_TIER3]
    print(f"\n=== Tier 3 — top {len(advance_t3)} configs × CogVideoX + SeedVR2-large ===", flush=True)
    tier3 = []
    for cfg_idx, base in enumerate(advance_t3):
        BQ, BK, SG = base["BQ"], base["BK"], base["SG"]
        cfg_results = {**base, "tier": 3, "tier3_ms": {}}
        for shape in TIER3_SHAPES:
            r = measure(shape, BQ, BK, SG)
            cfg_results["per_shape"][shape["name"]] = r
            if r["status"] == "OK":
                cfg_results["tier3_ms"][shape["name"]] = r["median_ms"]
                print(f"  [{cfg_idx+1}/{len(advance_t3)}] BQ={BQ} BK={BK} SG={SG} {shape['name']:<16}: "
                      f"{r['median_ms']:.0f} ms", flush=True)
            else:
                print(f"  [{cfg_idx+1}/{len(advance_t3)}] BQ={BQ} BK={BK} SG={SG} {shape['name']:<16}: "
                      f"{r['status']}", flush=True)
        tier3.append(cfg_results)
        save({"tier1": tier1, "tier2": tier2, "tier3": tier3, "skipped": skipped}, DATA_PATH)

    # Strip env at end
    for k_ in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG", "MFA_V6_NAX_SINGLE_OTILE"):
        os.environ.pop(k_, None)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

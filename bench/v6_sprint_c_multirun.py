"""Sprint C — multi-run baseline sweep (rigorous version of S3.1).

Per the v2.30 mandate, ALL deltas <15% require multi-run methodology.
This sweep redoes the BQ × BK × SG exploration with 3 runs per config
× 6-8 iters median per run, median-of-medians final.

Tiered to fit budget:
  Tier 1 (all feasible) × FlashVSR-dense + LTX2-cross  : ~5 min
  Tier 2 (top 30 by T1) × SeedVR2-small × multi-run    : ~30 min
  Tier 3 (top 10 by T2) × CogVideoX + SeedVR2-large × multi-run : ~60-90 min

Total budget: 90-120 min wallclock.
"""
from __future__ import annotations

import itertools, json, math, os, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")


SEARCH_SPACE = {
    "BQ": [8, 12, 16, 20, 24, 32],
    "BK": [32, 64, 96, 128],  # only multiples of 32 (Apple MPP req)
    "SG": [1, 2, 4, 6, 8, 12, 16, 24],
}

TIER1_SHAPES = [
    {"name": "FlashVSR-dense", "B":1, "H":10, "N_q":4096, "N_kv":4096,  "D":64,  "iters":6},
    {"name": "LTX2-cross",     "B":1, "H":8,  "N_q":2048, "N_kv":14000, "D":64,  "iters":6},
]
TIER2_SHAPES = [
    {"name": "SeedVR2-small",  "B":1, "H":20, "N_q":26730,  "N_kv":26730, "D":128, "iters":6},
]
TIER3_SHAPES = [
    {"name": "CogVideoX",      "B":1, "H":30, "N_q":70200,   "N_kv":70200,   "D":128, "iters":4},
    {"name": "SeedVR2-large",  "B":1, "H":20, "N_q":111375,  "N_kv":111375,  "D":128, "iters":4},
]

WARMUP = 2
RUNS_T1 = 1   # tier 1: huge config count, single-run for triage (top 30 advance)
RUNS_T2 = 3   # tier 2: 30 configs, 3 runs each
RUNS_T3 = 3   # tier 3: 10 configs, 3 runs each
TOP_N_TIER2 = 30
TOP_N_TIER3 = 10


def feasible(BQ, BK, SG):
    threads = SG * 32
    if threads > 1024:
        return False, f"threads={threads} > 1024"
    if BK % 32 != 0:
        return False, f"BK={BK} not multiple of 32"
    return True, "OK"


def make(s):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["H"], s["N_q"], s["D"]), dtype=mx.float16)
    k = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    v = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def measure_once(s, BQ, BK, SG, iters, warmup):
    os.environ["MFA_V6_BLOCK_R"] = str(BQ)
    os.environ["MFA_V6_BLOCK_C"] = str(BK)
    os.environ["MFA_V6_EXEC_SG"] = str(SG)
    os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "1"
    try:
        q, k, v = make(s)
        out, _ = _ext.v6_nax_forward(q, k, v, False)
        _force(out)
        ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0/math.sqrt(s["D"]))
        _force(ref)
        diff = (out.astype(mx.float32) - ref.astype(mx.float32))
        rmse = float(mx.sqrt(mx.mean(diff * diff)))
        if rmse > 5e-3 or not bool(mx.all(mx.isfinite(out)).item()):
            return {"status":"BAD","rmse":rmse}
        for _ in range(warmup):
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings = []
        for _ in range(iters):
            t0 = time.perf_counter()
            o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return {"status":"OK","rmse":rmse,"median_ms":statistics.median(timings)}
    except Exception as e:
        return {"status":"FAIL","error":str(e)[:200]}


def measure_multirun(s, BQ, BK, SG, runs):
    """Returns median-of-medians + run list."""
    iters = s["iters"]
    run_medians = []
    for _ in range(runs):
        r = measure_once(s, BQ, BK, SG, iters, WARMUP)
        if r["status"] != "OK":
            return r  # propagate fail
        run_medians.append(r["median_ms"])
    return {"status":"OK","run_medians":run_medians,
            "median_of_medians":statistics.median(run_medians)}


def save(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2))


def main():
    DATA_PATH = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-C-multirun-data.json"

    configs = list(itertools.product(SEARCH_SPACE["BQ"], SEARCH_SPACE["BK"], SEARCH_SPACE["SG"]))
    feasible_configs = [(BQ,BK,SG) for (BQ,BK,SG) in configs if feasible(BQ,BK,SG)[0]]
    print(f"Tier 1: {len(feasible_configs)} feasible × 2 fast shapes (single-run)", flush=True)
    tier1 = []
    for cfg_idx, (BQ,BK,SG) in enumerate(feasible_configs):
        cfg = {"BQ":BQ,"BK":BK,"SG":SG,"per_shape":{}}
        per = []
        for s in TIER1_SHAPES:
            r = measure_once(s, BQ, BK, SG, s["iters"], WARMUP)
            cfg["per_shape"][s["name"]] = r
            if r["status"] == "OK":
                per.append(r["median_ms"])
        if per:
            cfg["t1_total"] = sum(per)
        tier1.append(cfg)
        if (cfg_idx+1) % 20 == 0:
            ok = sum(1 for c in tier1 if "t1_total" in c)
            print(f"  [{cfg_idx+1}/{len(feasible_configs)}] {ok} OK so far", flush=True)
            save({"tier1":tier1}, DATA_PATH)
    save({"tier1":tier1}, DATA_PATH)

    t1_ok = sorted([c for c in tier1 if "t1_total" in c], key=lambda c: c["t1_total"])
    print(f"\nTier 1 done. {len(t1_ok)} OK out of {len(feasible_configs)}.", flush=True)
    top1 = [(c["BQ"], c["BK"], c["SG"], f"{c['t1_total']:.2f}ms") for c in t1_ok[:5]]
    print(f"Top-5 T1: {top1}", flush=True)

    advance_t2 = t1_ok[:TOP_N_TIER2]
    print(f"\nTier 2: {len(advance_t2)} configs × SeedVR2-small × {RUNS_T2} runs", flush=True)
    tier2 = []
    for idx, base in enumerate(advance_t2):
        BQ, BK, SG = base["BQ"], base["BK"], base["SG"]
        r = measure_multirun(TIER2_SHAPES[0], BQ, BK, SG, RUNS_T2)
        cfg = {**base, "tier2_seedvr2_small": r}
        tier2.append(cfg)
        if r["status"] == "OK":
            print(f"  [{idx+1}/{len(advance_t2)}] BQ={BQ} BK={BK} SG={SG}: "
                  f"{r['median_of_medians']:.1f}ms (runs: {[f'{x:.1f}' for x in r['run_medians']]})",
                  flush=True)
        else:
            print(f"  [{idx+1}/{len(advance_t2)}] BQ={BQ} BK={BK} SG={SG}: {r['status']}", flush=True)
        save({"tier1":tier1,"tier2":tier2}, DATA_PATH)

    t2_ok = sorted([c for c in tier2 if c["tier2_seedvr2_small"].get("status") == "OK"],
                   key=lambda c: c["tier2_seedvr2_small"]["median_of_medians"])
    print(f"\nTier 2 done. {len(t2_ok)} OK.", flush=True)
    top2 = [(c["BQ"], c["BK"], c["SG"], f"{c['tier2_seedvr2_small']['median_of_medians']:.1f}ms") for c in t2_ok[:5]]
    print(f"Top-5 T2: {top2}", flush=True)

    advance_t3 = t2_ok[:TOP_N_TIER3]
    print(f"\nTier 3: {len(advance_t3)} configs × CogVideoX + SeedVR2-large × {RUNS_T3} runs", flush=True)
    tier3 = []
    for idx, base in enumerate(advance_t3):
        BQ, BK, SG = base["BQ"], base["BK"], base["SG"]
        cfg = {**base, "tier3":{}}
        for s in TIER3_SHAPES:
            r = measure_multirun(s, BQ, BK, SG, RUNS_T3)
            cfg["tier3"][s["name"]] = r
            if r["status"] == "OK":
                print(f"  [{idx+1}/{len(advance_t3)}] BQ={BQ} BK={BK} SG={SG} {s['name']:<16}: "
                      f"{r['median_of_medians']:.0f}ms (runs: {[f'{x:.0f}' for x in r['run_medians']]})",
                      flush=True)
            else:
                print(f"  [{idx+1}/{len(advance_t3)}] BQ={BQ} BK={BK} SG={SG} {s['name']:<16}: {r['status']}",
                      flush=True)
        tier3.append(cfg)
        save({"tier1":tier1,"tier2":tier2,"tier3":tier3}, DATA_PATH)

    for k_ in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG", "MFA_V6_NAX_SINGLE_OTILE"):
        os.environ.pop(k_, None)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

"""Sprint G — final dispatch v6 vs dispatch v5 bench.

After Sprint C identifies the best (BQ, BK, SG) per shape, this script
benches the proposed dispatch v6 vs the current v5 default on all 5
production shapes with rigorous multi-run methodology (5 runs).

Results determine whether dispatch v6 ships in v2.30.0.
"""
import json, math, os, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")

SHAPES = [
    {"name": "FlashVSR-dense", "B":1, "H":10, "N_q":4096,   "N_kv":4096,   "D":64,  "iters":8},
    {"name": "LTX2-cross",     "B":1, "H":8,  "N_q":2048,   "N_kv":14000,  "D":64,  "iters":8},
    {"name": "SeedVR2-small",  "B":1, "H":20, "N_q":26730,  "N_kv":26730,  "D":128, "iters":8},
    {"name": "CogVideoX",      "B":1, "H":30, "N_q":70200,  "N_kv":70200,  "D":128, "iters":5},
    {"name": "SeedVR2-large",  "B":1, "H":20, "N_q":111375, "N_kv":111375, "D":128, "iters":5},
]
WARMUP = 3
RUNS = 5

# Dispatch v5 (current v2.29.0 + S3.6 default, no env override)
# This is what the auto-defaults produce.

# Dispatch v6 (proposed): per-shape best from Sprint C.
# THIS GETS FILLED IN AFTER SPRINT C COMPLETES.
DISPATCH_V6_PER_SHAPE = {
    # Filled programmatically below from sprint-C-multirun-data.json
}


def make(s):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["H"], s["N_q"], s["D"]), dtype=mx.float16)
    k = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    v = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def time_run(s, env_overrides):
    for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
              "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP"):
        os.environ.pop(k, None)
    for k_, v_ in env_overrides.items():
        os.environ[k_] = v_
    q, k, v = make(s)
    for _ in range(WARMUP):
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
    timings = []
    for _ in range(s["iters"]):
        t0 = time.perf_counter()
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def time_sdpa(s):
    q, k, v = make(s)
    scale = 1.0 / math.sqrt(s["D"])
    for _ in range(WARMUP):
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(o)
    timings = []
    for _ in range(s["iters"]):
        t0 = time.perf_counter()
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def load_v6_per_shape():
    """Pull per-shape best from sprint-C-multirun-data.json."""
    data_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-C-multirun-data.json"
    if not data_path.exists():
        print(f"WARNING: {data_path} not found; using v5 (no v6 candidate)", flush=True)
        return {}
    data = json.loads(data_path.read_text())
    v6_per_shape = {}
    # FlashVSR + LTX2 from Tier 1's t1_total ranking
    t1 = data.get("tier1", [])
    t1_ok = sorted([c for c in t1 if "t1_total" in c], key=lambda c: c["t1_total"])
    if t1_ok:
        # The Tier 1 winner per shape can be different — but the t1_total sums
        # both fast shapes. Per-shape:
        flashvsr_best = sorted(
            [c for c in t1 if c.get("per_shape", {}).get("FlashVSR-dense", {}).get("status") == "OK"],
            key=lambda c: c["per_shape"]["FlashVSR-dense"]["median_ms"])
        ltx2_best = sorted(
            [c for c in t1 if c.get("per_shape", {}).get("LTX2-cross", {}).get("status") == "OK"],
            key=lambda c: c["per_shape"]["LTX2-cross"]["median_ms"])
        if flashvsr_best:
            c = flashvsr_best[0]
            v6_per_shape["FlashVSR-dense"] = {"BQ": c["BQ"], "BK": c["BK"], "SG": c["SG"]}
        if ltx2_best:
            c = ltx2_best[0]
            v6_per_shape["LTX2-cross"] = {"BQ": c["BQ"], "BK": c["BK"], "SG": c["SG"]}
    # SeedVR2-small from Tier 2 multi-run
    t2 = data.get("tier2", [])
    t2_ok = sorted(
        [c for c in t2 if c.get("tier2_seedvr2_small", {}).get("status") == "OK"],
        key=lambda c: c["tier2_seedvr2_small"]["median_of_medians"])
    if t2_ok:
        c = t2_ok[0]
        v6_per_shape["SeedVR2-small"] = {"BQ": c["BQ"], "BK": c["BK"], "SG": c["SG"]}
    # Cog + SeedVR2-large from Tier 3
    t3 = data.get("tier3", [])
    cog_best = sorted(
        [c for c in t3 if c.get("tier3", {}).get("CogVideoX", {}).get("status") == "OK"],
        key=lambda c: c["tier3"]["CogVideoX"]["median_of_medians"])
    seedlarge_best = sorted(
        [c for c in t3 if c.get("tier3", {}).get("SeedVR2-large", {}).get("status") == "OK"],
        key=lambda c: c["tier3"]["SeedVR2-large"]["median_of_medians"])
    if cog_best:
        c = cog_best[0]
        v6_per_shape["CogVideoX"] = {"BQ": c["BQ"], "BK": c["BK"], "SG": c["SG"]}
    if seedlarge_best:
        c = seedlarge_best[0]
        v6_per_shape["SeedVR2-large"] = {"BQ": c["BQ"], "BK": c["BK"], "SG": c["SG"]}
    return v6_per_shape


def main():
    v6_per_shape = load_v6_per_shape()
    print(f"Sprint C-derived dispatch v6 per shape:", flush=True)
    for k, v in v6_per_shape.items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    results = {}
    print(f"{'Shape':<20} {'v5 (default)':>14} {'v6 (per-shape)':>16} {'Δ':>9} {'sdpa':>10} {'v6/sdpa':>10}", flush=True)
    print("-" * 85, flush=True)
    for s in SHAPES:
        # v5: no env override, uses auto-default
        v5_runs = [time_run(s, {}) for _ in range(RUNS)]
        v5_med = statistics.median(v5_runs)
        # v6: per-shape override
        v6 = v6_per_shape.get(s["name"])
        if v6:
            v6_env = {
                "MFA_V6_BLOCK_R": str(v6["BQ"]),
                "MFA_V6_BLOCK_C": str(v6["BK"]),
                "MFA_V6_EXEC_SG": str(v6["SG"]),
                "MFA_V6_NAX_SINGLE_OTILE": "1",
            }
            v6_runs = [time_run(s, v6_env) for _ in range(RUNS)]
            v6_med = statistics.median(v6_runs)
        else:
            v6_runs = []
            v6_med = v5_med  # fallback
        # SDPA
        sdpa_runs = [time_sdpa(s) for _ in range(RUNS)]
        sdpa_med = statistics.median(sdpa_runs)
        delta = (v6_med - v5_med) / v5_med * 100.0
        v6_over_sdpa = v6_med / sdpa_med
        results[s["name"]] = {
            "v5_runs_ms": v5_runs, "v5_median": v5_med,
            "v6_config": v6,
            "v6_runs_ms": v6_runs, "v6_median": v6_med,
            "sdpa_runs_ms": sdpa_runs, "sdpa_median": sdpa_med,
            "delta_pct": delta, "v6_over_sdpa": v6_over_sdpa,
        }
        v6_disp = f"{v6_med:.2f}ms" if v6 else "n/a"
        print(f"{s['name']:<20} {v5_med:>12.2f}ms {v6_disp:>16} {delta:>+7.2f}% {sdpa_med:>8.2f}ms {v6_over_sdpa:>8.2f}×", flush=True)

    # Reset env
    for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
              "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP"):
        os.environ.pop(k, None)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-G-dispatch-v6-bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP, "runs": RUNS,
        "v6_per_shape": v6_per_shape, "results": results,
    }, indent=2))
    print(f"\nDone. {out_path}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

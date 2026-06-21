"""Sprint A.1 — measure the gain from skipping tgmem allocation when
single-Otile + bypass are both on (P_buf is never used).

Multi-run methodology: 5 runs × {6-8} iters median, median-of-medians.
Compares against S3.6 baseline numbers for the same 5 production shapes.

Note: this is post-fix only. The pre-fix numbers come from S3.6's data
which used the same multi-run protocol on the same hardware.
"""
import json, math, os, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")

SHAPES = [
    {"name": "FlashVSR-dense", "B": 1, "H": 10, "N_q": 4096,   "N_kv": 4096,   "D": 64,  "iters": 8},
    {"name": "LTX2-cross",     "B": 1, "H": 8,  "N_q": 2048,   "N_kv": 14000,  "D": 64,  "iters": 8},
    {"name": "SeedVR2-small",  "B": 1, "H": 20, "N_q": 26730,  "N_kv": 26730,  "D": 128, "iters": 8},
    {"name": "CogVideoX",      "B": 1, "H": 30, "N_q": 70200,  "N_kv": 70200,  "D": 128, "iters": 5},
    {"name": "SeedVR2-large",  "B": 1, "H": 20, "N_q": 111375, "N_kv": 111375, "D": 128, "iters": 5},
]
WARMUP = 3
RUNS = 5

# Pre-fix baselines from S3.6 / autoresearch (multi-run validated)
PRE_FIX_S3_6 = {
    "FlashVSR-dense": 1.11,
    "LTX2-cross": 1.59,
    "SeedVR2-small": 290.01,
    "CogVideoX": 3349.30,
    "SeedVR2-large": 7244.44,
}


def make(s):
    mx.random.seed(42)
    q = mx.random.normal((s["B"], s["H"], s["N_q"], s["D"]), dtype=mx.float16)
    k = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    v = mx.random.normal((s["B"], s["H"], s["N_kv"], s["D"]), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def time_run(s, iters):
    q, k, v = make(s)
    for _ in range(WARMUP):
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def main():
    # Use the auto-defaults (no env override)
    for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
              "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP"):
        os.environ.pop(k, None)

    results = {}
    print(f"{'Shape':<20} {'pre-fix':>10} {'post-fix':>10} {'Δ':>9}", flush=True)
    print("-" * 55, flush=True)
    for s in SHAPES:
        run_medians = [time_run(s, s["iters"]) for _ in range(RUNS)]
        med = statistics.median(run_medians)
        results[s["name"]] = {
            "run_medians": run_medians, "median_of_medians": med,
        }
        pre = PRE_FIX_S3_6[s["name"]]
        delta = (med - pre) / pre * 100.0
        print(f"{s['name']:<20} {pre:>8.2f}ms {med:>8.2f}ms {delta:>+7.2f}%", flush=True)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-A-tgmem-fix-bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP, "runs": RUNS,
        "pre_fix_baseline": PRE_FIX_S3_6, "post_fix": results,
    }, indent=2))
    print(f"\nDone. Written: {out_path}", flush=True)


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

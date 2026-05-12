"""Sprint B GQA gain bench — single-Otile + BHND vs legacy double-buffer + BNHD.

Comparison protocol:
  - Single-Otile + BHND (default for GQA in v2.30): no env override.
  - Legacy double-buffer + BNHD (v2.29.0 fallback): MFA_V6_NAX_SINGLE_OTILE=0
    + MFA_V6_BNHD_LEGACY=1.

Multi-run methodology: 3 runs × 6 iters median, median-of-medians.
"""
import json, math, os, statistics, time
from pathlib import Path
import mlx.core as mx
from mlx_mfa import _ext

_force = getattr(mx, "eval")

# Production-relevant GQA shapes (Llama-style proportions)
GQA_SHAPES = [
    ("GQA-Hq32-Hk8 D=128 N=4096",  1, 32, 8, 4096, 4096, 128),
    ("GQA-Hq16-Hk4 D=64 N=8192",   1, 16, 4, 8192, 8192, 64),
    ("GQA-Hq40-Hk8 D=128 N=2048",  1, 40, 8, 2048, 2048, 128),
    ("GQA-Hq8-Hk2  D=64 N=4096",   1,  8, 2, 4096, 4096, 64),
]

WARMUP = 3
RUNS = 3
ITERS = 6


def make(B, Hq, Hk, Nq, Nkv, D):
    mx.random.seed(42)
    q = mx.random.normal((B, Hq, Nq, D), dtype=mx.float16)
    k = mx.random.normal((B, Hk, Nkv, D), dtype=mx.float16)
    v = mx.random.normal((B, Hk, Nkv, D), dtype=mx.float16)
    _force(q, k, v)
    return q, k, v


def time_run(B, Hq, Hk, Nq, Nkv, D):
    q, k, v = make(B, Hq, Hk, Nq, Nkv, D)
    for _ in range(WARMUP):
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
    timings = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        o, _ = _ext.v6_nax_forward(q, k, v, False); _force(o)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(timings)


def reset_env():
    for k in ("MFA_V6_BLOCK_R", "MFA_V6_BLOCK_C", "MFA_V6_EXEC_SG",
              "MFA_V6_NAX_SINGLE_OTILE", "MFA_V6_BYPASS_TGP",
              "MFA_V6_BNHD_LEGACY"):
        os.environ.pop(k, None)


def main():
    results = {}
    print(f"{'Shape':<32} {'legacy (BNHD)':>15} {'single-Otile (BHND)':>22} {'Δ':>9}", flush=True)
    print("-" * 85, flush=True)
    for name, B, Hq, Hk, Nq, Nkv, D in GQA_SHAPES:
        # Legacy: force BNHD + force single_otile=0
        reset_env()
        os.environ["MFA_V6_BNHD_LEGACY"] = "1"
        os.environ["MFA_V6_NAX_SINGLE_OTILE"] = "0"
        legacy_runs = [time_run(B, Hq, Hk, Nq, Nkv, D) for _ in range(RUNS)]
        legacy_med = statistics.median(legacy_runs)

        # Single-Otile: defaults (no override → uses v2.30 GQA single-Otile path)
        reset_env()
        new_runs = [time_run(B, Hq, Hk, Nq, Nkv, D) for _ in range(RUNS)]
        new_med = statistics.median(new_runs)

        delta = (new_med - legacy_med) / legacy_med * 100.0
        results[name] = {"Hq": Hq, "Hk": Hk, "Nq": Nq, "D": D,
                         "legacy_runs_ms": legacy_runs, "legacy_median_ms": legacy_med,
                         "new_runs_ms": new_runs, "new_median_ms": new_med,
                         "delta_pct": delta}
        print(f"{name:<32} {legacy_med:>13.2f}ms {new_med:>20.2f}ms {delta:>+7.2f}%", flush=True)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "v6-nax" / "sprint-B-gqa-bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "device": "Apple M5 Max", "warmup": WARMUP, "runs": RUNS, "iters": ITERS,
        "results": results,
    }, indent=2))
    print(f"\nDone. {out_path}", flush=True)
    reset_env()


if __name__ == "__main__":
    main()

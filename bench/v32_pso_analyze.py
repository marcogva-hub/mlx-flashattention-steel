"""Analyze Phase A.1 PSO cache A/B results.

Compares cold-cache vs warm-cache legacy bench timings against
v2.31.0 baseline (from docs/v6-nax/v34-aba.json) and Phase 0
re-bench (docs/v6-nax/v32-aba.json). Prints a discriminant verdict.
"""
import json
import statistics
from pathlib import Path

OUT = Path("outputs/diagnostic")

# Reference values from prior measurements (legacy mode, median of 3-run subprocess).
V231_LEGACY_MS = {
    "SeedVR2-small": 275.6,   # avg of (265.13, 286.03) from v2.31.0 v34-aba.json
    "CogVideoX":     3669.0,  # avg of (3610.79, 3727.58)
    "SeedVR2-large": 6780.0,  # avg of (6776.12, 6784.13)
}
PHASE0_LEGACY_MS = {
    "SeedVR2-small": 167.75,  # avg of (162.13, 173.37) from v32-aba.json
    "CogVideoX":     2344.0,  # avg of (2335.85, 2352.43)
    "SeedVR2-large": 3982.0,  # avg of (4073.61, 3891.03)
}


def median_of_runs(json_path):
    if not json_path.exists():
        return None, []
    d = json.loads(json_path.read_text())
    if not d.get("records"):
        return None, []
    rec = d["records"][-1]  # last record = our run
    runs = rec.get("v6_runs_ms", [])
    if not runs:
        return None, []
    return statistics.median(runs), runs


def fmt_pct(num):
    return f"{num*100:+.1f}%"


def closest(val, refs):
    """Return (label, ref_val, pct_diff) of closest reference."""
    if val is None:
        return ("?", None, None)
    best = None
    for label, ref in refs.items():
        d = abs(val - ref) / ref
        if best is None or d < best[2]:
            best = (label, ref, d)
    return best


def main():
    shapes = ["SeedVR2-small", "CogVideoX", "SeedVR2-large"]
    cold_files = {
        "SeedVR2-small": OUT / "a1-cold-seedvr2small.json",
        "CogVideoX":     OUT / "a1-cold-cogvideox.json",
        "SeedVR2-large": OUT / "a1-cold-seedvr2large.json",
    }
    warm_files = {
        "SeedVR2-small": OUT / "a1-warm-seedvr2small.json",
        "CogVideoX":     OUT / "a1-warm-cogvideox.json",
        "SeedVR2-large": OUT / "a1-warm-seedvr2large.json",
    }

    print(f"{'Shape':<18} {'cold':>10} {'warm':>10} {'v2.31.0':>10} {'phase0':>10} "
          f"{'cold/v231':>12} {'cold/p0':>10} {'warm/p0':>10} {'cold/warm':>12}")

    cold_close_v231 = 0
    warm_close_p0 = 0
    cold_warm_drift = []

    for s in shapes:
        cold_med, cold_runs = median_of_runs(cold_files[s])
        warm_med, warm_runs = median_of_runs(warm_files[s])
        v231 = V231_LEGACY_MS[s]
        p0 = PHASE0_LEGACY_MS[s]

        if cold_med is None or warm_med is None:
            print(f"{s:<18} MISSING DATA")
            continue

        cold_v231 = (cold_med - v231) / v231
        cold_p0 = (cold_med - p0) / p0
        warm_p0 = (warm_med - p0) / p0
        cold_warm = (cold_med - warm_med) / warm_med
        cold_warm_drift.append(cold_warm)

        print(f"{s:<18} {cold_med:>10.2f} {warm_med:>10.2f} {v231:>10.2f} {p0:>10.2f} "
              f"{fmt_pct(cold_v231):>12} {fmt_pct(cold_p0):>10} {fmt_pct(warm_p0):>10} "
              f"{fmt_pct(cold_warm):>12}")

        if abs(cold_v231) < 0.10:
            cold_close_v231 += 1
        if abs(warm_p0) < 0.10:
            warm_close_p0 += 1

    # Verdict
    print()
    print("=== Verdict ===")
    avg_cold_warm = (statistics.mean(cold_warm_drift) if cold_warm_drift
                     else None)
    print(f"Cold rounds matching v2.31.0 (within ±10%): {cold_close_v231}/{len(shapes)}")
    print(f"Warm rounds matching Phase 0 (within ±10%): {warm_close_p0}/{len(shapes)}")
    print(f"Mean cold→warm speedup: {fmt_pct(avg_cold_warm) if avg_cold_warm else 'n/a'}")
    print()
    if cold_close_v231 >= 2 and warm_close_p0 >= 2:
        print("→ PSO CACHE HYPOTHESIS CONFIRMED.")
        print("  v2.31.0's measurements were effectively cold-cache.")
        print("  Phase 0's measurements were warm-cache.")
        print("  The cross-session drift is fully explained by cache state.")
    elif warm_close_p0 >= 2 and cold_close_v231 == 0:
        print("→ PSO PARTIALLY EXPLAINS PHASE 0 (warm matches), but cold")
        print("  doesn't reach v2.31.0 slowness. v2.31.0 had additional factor")
        print("  (e.g., system state, background load) on top of cold cache.")
    elif cold_close_v231 == 0 and warm_close_p0 == 0:
        print("→ PSO CACHE HYPOTHESIS REJECTED (or non-isolating).")
        print("  Cold and warm both differ from both reference points.")
        print("  Different hypothesis needed (Phase A.2 thermal, A.3 ramp-up).")
    else:
        print("→ MIXED / INCONCLUSIVE — needs Phase A.2/A.3 follow-up.")


if __name__ == "__main__":
    main()

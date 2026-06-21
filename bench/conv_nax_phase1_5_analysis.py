#!/usr/bin/env python3
"""Phase 1.5 ratio analysis + decision tree application.

Reads docs/conv-nax/conv-nax-phase1_5-perfsweep.json (3-session bench)
and produces:
  1. Per-shape cross-session medians + range
  2. Variance flag per shape per Sprint A §B.7
  3. Decision tree application (ship-default / opt-in / shelve)
  4. Summary table

Decision tree (prompt §F.1):
  - ≥ 1.2× across dominant shapes → ship-default
  - 0.9-1.2× → opt-in
  - < 0.9× → shelve

Variance fallback (§B.7):
  - cross-session range < 10% per shape → confident
  - 10-20% range → boundary; default to opt-in regardless of ratio
  - > 20% range on 3+ of 6 → "data inconclusive" → shelve
"""
import json, statistics, sys
from pathlib import Path


def load_data(path):
    return json.load(open(path))


def analyze(sessions):
    """Compute per-shape cross-session medians + variance."""
    shape_ratios = {}  # label -> [ratio_S1, ratio_S2, ratio_S3]
    shape_nax_ms = {}
    shape_mlx_ms = {}
    shape_nax_tf = {}
    shape_mlx_tf = {}
    shape_aba_drift = {}

    for sess in sessions:
        for r in sess["results"]:
            if "error" in r:
                continue
            label = r["shape"]
            shape_ratios.setdefault(label, []).append(r["ratio_mlx_over_nax"])
            shape_nax_ms.setdefault(label, []).append(r["nax_median_ms"])
            shape_mlx_ms.setdefault(label, []).append(r["mlx_median_ms"])
            shape_nax_tf.setdefault(label, []).append(r["nax_TFLOPS"])
            shape_mlx_tf.setdefault(label, []).append(r["mlx_TFLOPS"])
            shape_aba_drift.setdefault(label, []).append(r["aba_drift_pct"])

    summary = {}
    for label, ratios in shape_ratios.items():
        med = statistics.median(ratios)
        rng_pct = (max(ratios) - min(ratios)) / med * 100 if med > 0 else 0
        nax_med = statistics.median(shape_nax_ms[label])
        mlx_med = statistics.median(shape_mlx_ms[label])
        nax_tf = statistics.median(shape_nax_tf[label])
        mlx_tf = statistics.median(shape_mlx_tf[label])
        aba_max = max(shape_aba_drift[label])
        if rng_pct > 20:
            variance_flag = "HIGH (>20% — §B.7 fallback applies)"
        elif rng_pct > 10:
            variance_flag = "boundary (10-20%)"
        else:
            variance_flag = "confident (<10%)"
        summary[label] = {
            "ratios_per_session": ratios,
            "ratio_median": med,
            "range_pct": rng_pct,
            "nax_median_ms": nax_med,
            "mlx_median_ms": mlx_med,
            "nax_TFLOPS": nax_tf,
            "mlx_TFLOPS": mlx_tf,
            "aba_drift_max_pct": aba_max,
            "variance_flag": variance_flag,
        }
    return summary


def decide(summary, dominant_shapes):
    """Apply ship/opt-in/shelve decision tree."""
    dominant_ratios = [s["ratio_median"] for label, s in summary.items()
                       if label in dominant_shapes]
    if not dominant_ratios:
        return {"verdict": "INDETERMINATE", "reason": "no dominant shapes"}

    median_dom = statistics.median(dominant_ratios)
    min_dom = min(dominant_ratios)
    max_dom = max(dominant_ratios)

    # Variance fallback check
    high_var_count = sum(1 for label, s in summary.items()
                         if "HIGH" in s["variance_flag"])
    if high_var_count >= 3:
        return {
            "verdict": "SHELVE",
            "reason": f"{high_var_count} of {len(summary)} shapes have "
                     f">20% cross-session variance — data inconclusive",
            "median_dominant_ratio": median_dom,
            "min_dominant_ratio": min_dom,
            "max_dominant_ratio": max_dom,
            "high_var_shapes": [l for l, s in summary.items()
                                if "HIGH" in s["variance_flag"]],
        }

    # Standard ratio mapping
    if median_dom >= 1.2 and min_dom >= 0.9:
        verdict = "SHIP_DEFAULT"
        reason = f"median dominant ratio {median_dom:.2f}× ≥ 1.2× threshold"
    elif median_dom >= 0.9:
        verdict = "OPT_IN"
        reason = f"median dominant ratio {median_dom:.2f}× in 0.9-1.2× boundary"
    else:
        verdict = "SHELVE"
        reason = f"median dominant ratio {median_dom:.2f}× < 0.9× shelve threshold"

    return {
        "verdict": verdict, "reason": reason,
        "median_dominant_ratio": median_dom,
        "min_dominant_ratio": min_dom,
        "max_dominant_ratio": max_dom,
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "docs/conv-nax/conv-nax-phase1_5-perfsweep.json"
    sessions = load_data(path)
    print(f"sessions: {len(sessions)}")
    summary = analyze(sessions)

    DOMINANT = set(s for s in summary.keys() if "probe" not in s.lower())
    # All 6 shapes are dominant per Phase 1.5 prompt §F.1.

    print(f"\n{'shape':<28} {'S1':>5} {'S2':>5} {'S3':>5} {'median':>7} {'range%':>7} {'variance':>30}")
    print('-' * 100)
    for label, s in summary.items():
        ratios = s["ratios_per_session"]
        rs = ' '.join(f"{r:>5.2f}" for r in ratios)
        print(f"{label:<28} {rs}  {s['ratio_median']:>5.2f} "
              f"{s['range_pct']:>6.1f}%  {s['variance_flag']:>30}")
    print('-' * 100)

    decision = decide(summary, DOMINANT)
    print(f"\nDECISION TREE:")
    print(f"  verdict:                  {decision['verdict']}")
    print(f"  reason:                   {decision['reason']}")
    print(f"  median dominant ratio:    {decision.get('median_dominant_ratio', 'N/A')}")
    print(f"  min dominant ratio:       {decision.get('min_dominant_ratio', 'N/A')}")
    print(f"  max dominant ratio:       {decision.get('max_dominant_ratio', 'N/A')}")

    out = Path("docs/conv-nax/conv-nax-phase1_5-analysis.json")
    out.write_text(json.dumps({"summary": summary, "decision": decision}, indent=2))
    print(f"\nanalysis -> {out}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

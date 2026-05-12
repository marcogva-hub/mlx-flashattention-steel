"""Cross-session aggregator for canonical-protocol bench data.

Reads docs/methodology/canonical-bench-data.json (list of per-session
records produced by canonical_warmup_continuous_harness.py) and
computes:
  - Per-shape cross-session p50 range (informational)
  - Per-shape cross-session ratio range (decision criterion)
  - Verdict flag per docs/methodology/canonical-protocol.md
  - V2 wall-clock per shape (input for decide_auto_version threshold)
  - Comparison vs prior section-4-strict / matched-workload results

Output: docs/methodology/canonical-bench-results.md +
docs/methodology/canonical-bench-analysis.json.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

VARIANCE_CONFIDENT = 10.0
VARIANCE_BOUNDARY = 20.0

# v2.36.0 (section-4-strict) reference data, for comparison column.
# Source: docs/lcsa-nax/lcsa-nax-v2-only-rebench-analysis.json
S4_STRICT_REF = {
    "lcsa_small_seq4k":           {"range_pct": 26.0, "flag": "HIGH"},
    "lcsa_small_seq4k_sparse":    {"range_pct":  4.7, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k":             {"range_pct":  8.6, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k_sparse":      {"range_pct": 37.3, "flag": "HIGH"},
    "lcsa_large_seq16k":          {"range_pct":  3.3, "flag": "CONFIDENT"},
    "lcsa_large_seq16k_sparse":   {"range_pct":  5.8, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k_very_sparse": {"range_pct": 46.0, "flag": "HIGH"},
}


def variance_flag(range_pct: float) -> str:
    if range_pct < VARIANCE_CONFIDENT:
        return "CONFIDENT"
    if range_pct < VARIANCE_BOUNDARY:
        return "BOUNDARY"
    return "HIGH_VARIANCE"


def aggregate(sessions):
    """Per-shape across sessions: v2 p50 range, ratio range, verdict flag."""
    by_shape = {}
    for sess in sessions:
        sid = sess.get("session_id", "?")
        for r in sess.get("production_results", []):
            if "error" in r or r.get("smoke_failed"):
                continue
            by_shape.setdefault(r["shape"], []).append({
                "sid": sid,
                "v2_p50": r["v2"]["p50_ms"],
                "v2_p95": r["v2"]["p95_ms"],
                "v2_p99": r["v2"]["p99_ms"],
                "sdpa_p50": r["sdpa"]["p50_ms"],
                "ratio": r["ratio_sdpa_over_v2"],
                "density_actual": r["density_actual"],
                "B": r["B"], "Hq": r["Hq"], "Hk": r["Hk"],
                "qL": r["qL"], "kL": r["kL"], "D": r["D"], "BT": r["BT"],
            })

    rows = []
    for name, samples in by_shape.items():
        if not samples:
            continue
        v2_p50s = [s["v2_p50"] for s in samples]
        sdpa_p50s = [s["sdpa_p50"] for s in samples]
        ratios = [s["ratio"] for s in samples]
        v2_med = statistics.median(v2_p50s)
        sdpa_med = statistics.median(sdpa_p50s)
        ratio_med = statistics.median(ratios)
        v2_range_pct = ((max(v2_p50s) - min(v2_p50s)) / v2_med * 100
                        if v2_med > 0 else 0)
        ratio_range_pct = ((max(ratios) - min(ratios)) / ratio_med * 100
                           if ratio_med > 0 else 0)
        ref = S4_STRICT_REF.get(name, {})
        rows.append({
            "shape": name,
            "n_sessions": len(samples),
            "density_actual": samples[0]["density_actual"],
            "qL": samples[0]["qL"],
            "kL": samples[0]["kL"],
            "D": samples[0]["D"],
            "work_product": samples[0]["qL"] * samples[0]["kL"] * samples[0]["D"],
            "v2_p50_median_ms": v2_med,
            "sdpa_p50_median_ms": sdpa_med,
            "ratio_median": ratio_med,
            "v2_range_pct": v2_range_pct,
            "ratio_range_pct": ratio_range_pct,
            "variance_flag": variance_flag(ratio_range_pct),
            "v2_default_eligible": (variance_flag(ratio_range_pct)
                                    in ("CONFIDENT", "BOUNDARY")),
            "s4_strict_range_pct": ref.get("range_pct"),
            "s4_strict_flag": ref.get("flag"),
            "samples": samples,
        })
    return rows


def calibrate_threshold(rows):
    """Find the inflection between CONFIDENT/BOUNDARY and HIGH_VARIANCE."""
    eligible_work = [r["work_product"] for r in rows if r["v2_default_eligible"]]
    ineligible_work = [r["work_product"] for r in rows if not r["v2_default_eligible"]]
    if not eligible_work:
        return {"threshold": None,
                "rationale": "No CONFIDENT/BOUNDARY shapes found - V2 stays SHIP_OPT_IN"}
    if not ineligible_work:
        return {"threshold": 0,
                "rationale": "All shapes CONFIDENT/BOUNDARY - V2 ships unconditionally"}
    # Threshold = smallest work product among eligible shapes
    threshold = min(eligible_work)
    return {
        "threshold": threshold,
        "rationale": (
            f"Smallest eligible work product = {threshold:.2e}. "
            f"Largest ineligible = {max(ineligible_work):.2e}. "
            f"Inflection: " +
            ("CLEAN (eligible all > ineligible all)"
             if min(eligible_work) > max(ineligible_work)
             else "OVERLAPPING - reconsider per-shape rule")
        ),
        "eligible_work_min": min(eligible_work),
        "eligible_work_max": max(eligible_work),
        "ineligible_work_min": min(ineligible_work),
        "ineligible_work_max": max(ineligible_work),
        "clean_inflection": min(eligible_work) > max(ineligible_work),
    }


def render_md(rows, calibration, conditions, n_sessions):
    L = []
    L.append("# Canonical re-bench results (Sprint Option beta, v2.36.1)")
    L.append("")
    L.append("**Methodology**: docs/methodology/canonical-protocol.md - "
             "10 warmup + 100 continuous timed iters, V2 and SDPA "
             "back-to-back per shape, 3 subprocess-isolated sessions.")
    L.append(f"**Hardware**: M5 Max 128GB, macOS 26 (fan profile iStat performance).")
    L.append(f"**Sessions**: {n_sessions}.")
    L.append("")
    L.append("## Per-shape verdict")
    L.append("")
    L.append("| Shape | density | qL*kL*D | V2 p50 ms | SDPA p50 ms | "
             "Ratio | V2 range % | Ratio range % | Verdict | V2 default | "
             "s4-strict ratio range % | s4-strict flag |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|:--:|")
    for r in rows:
        s4r = (f"{r['s4_strict_range_pct']:.1f}%"
               if r["s4_strict_range_pct"] is not None else "-")
        s4f = r["s4_strict_flag"] or "-"
        L.append(f"| {r['shape']} | {r['density_actual']:.3f} | "
                 f"{r['work_product']:.2e} | "
                 f"{r['v2_p50_median_ms']:.3f} | "
                 f"{r['sdpa_p50_median_ms']:.3f} | "
                 f"{r['ratio_median']:.2f}x | "
                 f"{r['v2_range_pct']:.1f}% | "
                 f"{r['ratio_range_pct']:.1f}% | "
                 f"{r['variance_flag']} | "
                 f"{'YES' if r['v2_default_eligible'] else 'NO'} | "
                 f"{s4r} | {s4f} |")
    L.append("")
    L.append("## Threshold calibration for decide_auto_version()")
    L.append("")
    L.append(f"- **Threshold**: `qL * kL * D >= {calibration['threshold']:.2e}`"
             if calibration.get("threshold") else "- **No threshold**: see rationale.")
    L.append(f"- **Rationale**: {calibration['rationale']}")
    if calibration.get("threshold") is not None:
        L.append(f"- Eligible work range: "
                 f"[{calibration['eligible_work_min']:.2e}, "
                 f"{calibration['eligible_work_max']:.2e}]")
        L.append(f"- Ineligible work range: "
                 f"[{calibration['ineligible_work_min']:.2e}, "
                 f"{calibration['ineligible_work_max']:.2e}]")
        L.append(f"- Clean inflection: {calibration['clean_inflection']}")
    L.append("")
    L.append("## Per-session samples")
    L.append("")
    for r in rows:
        L.append(f"### {r['shape']}")
        L.append("")
        L.append("| Session | V2 p50 ms | V2 p95 ms | V2 p99 ms | "
                 "SDPA p50 ms | Ratio |")
        L.append("|---|---:|---:|---:|---:|---:|")
        for s in r["samples"]:
            L.append(f"| {s['sid']} | {s['v2_p50']:.3f} | "
                     f"{s['v2_p95']:.3f} | {s['v2_p99']:.3f} | "
                     f"{s['sdpa_p50']:.3f} | {s['ratio']:.2f}x |")
        L.append("")
    L.append("## Comparison vs section-4-strict (v2.36.0 baseline)")
    L.append("")
    L.append("| Shape | section-4-strict range | canonical range | Direction |")
    L.append("|---|---:|---:|:--:|")
    for r in rows:
        s4r = r["s4_strict_range_pct"]
        if s4r is None:
            continue
        canon = r["ratio_range_pct"]
        direction = ("IMPROVED" if canon < s4r else
                     "WORSE" if canon > s4r else "EQUAL")
        L.append(f"| {r['shape']} | {s4r:.1f}% | {canon:.1f}% | {direction} |")
    L.append("")
    L.append("## Session conditions")
    L.append("")
    for sid, c in conditions:
        L.append(f"### {sid}")
        for k, v in c.items():
            if isinstance(v, dict):
                L.append(f"- **{k}**:")
                for kk, vv in v.items():
                    L.append(f"  - {kk}: `{vv}`")
            else:
                L.append(f"- **{k}**: `{v}`")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="docs/methodology/canonical-bench-data.json")
    ap.add_argument("--out-md",
                    default="docs/methodology/canonical-bench-results.md")
    ap.add_argument("--out-json",
                    default="docs/methodology/canonical-bench-analysis.json")
    args = ap.parse_args()

    sessions = json.loads(Path(args.data).read_text())
    rows = aggregate(sessions)
    calibration = calibrate_threshold(rows)
    conditions = [(s.get("session_id", "?"), s.get("conditions", {}))
                  for s in sessions]

    out = {
        "n_sessions": len(sessions),
        "rows": rows,
        "calibration": calibration,
        "session_conditions": [{"session_id": sid, "conditions": c}
                               for sid, c in conditions],
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    Path(args.out_md).write_text(
        render_md(rows, calibration, conditions, len(sessions)))

    print(f"=== Calibration ===")
    if calibration.get("threshold") is not None:
        print(f"  threshold: qL*kL*D >= {calibration['threshold']:.2e}")
    print(f"  rationale: {calibration['rationale']}")
    n_eligible = sum(1 for r in rows if r["v2_default_eligible"])
    print(f"  V2-default eligible: {n_eligible}/{len(rows)} shapes")
    print(f"\nWrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

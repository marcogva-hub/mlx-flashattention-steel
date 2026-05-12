"""Methodology sprint - matched-workload cross-session analysis.

Reads docs/methodology/matched-workload-data.json and computes:
- Per-shape cross-session range (V2 median, ratio_sdpa_over_v2)
- Variance flag per section B.7
- Decision-tree outcome per prompt section E.1 (same as v2.36.0)
- Delta vs v2.36.0 V2-only re-bench (range %, ratio % per shape)
- Three-axis self-validation: warmup-counter check (path entered)

Output: docs/methodology/matched-workload-results.md + analysis.json.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

VARIANCE_CONFIDENT = 10.0
VARIANCE_BOUNDARY = 20.0

# v2.36.0 V2-only re-bench reference ranges (from
# docs/lcsa-nax/lcsa-nax-v2-only-rebench-analysis.json).
V2_36_REF = {
    "lcsa_small_seq4k":           {"range_pct": 26.0, "flag": "HIGH"},
    "lcsa_small_seq4k_sparse":    {"range_pct":  4.7, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k":             {"range_pct":  8.6, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k_sparse":      {"range_pct": 37.3, "flag": "HIGH"},
    "lcsa_large_seq16k":          {"range_pct":  3.3, "flag": "CONFIDENT"},
    "lcsa_large_seq16k_sparse":   {"range_pct":  5.8, "flag": "CONFIDENT"},
    "lcsa_mid_seq8k_very_sparse": {"range_pct": 46.0, "flag": "HIGH"},
}
V2_36_HIGH_SHAPES = {k for k, v in V2_36_REF.items() if v["flag"] == "HIGH"}
V2_36_CONFIDENT_SHAPES = {k for k, v in V2_36_REF.items()
                          if v["flag"] == "CONFIDENT"}


def variance_flag(range_pct):
    if range_pct < VARIANCE_CONFIDENT:
        return "CONFIDENT"
    if range_pct < VARIANCE_BOUNDARY:
        return "BOUNDARY"
    return "HIGH"


def aggregate(sessions):
    shape_data = {}
    for sess in sessions:
        sid = sess.get("session_id", "?")
        for r in sess.get("production_results", []):
            if "error" in r:
                continue
            shape_data.setdefault(r["shape"], []).append({
                "sid": sid,
                "v2_med": r["v2_median_ms"],
                "sdpa_med": r["sdpa_median_ms"],
                "ratio": r["ratio_sdpa_over_v2"],
                "drift": r["aba_drift_pct"],
                "density_actual": r.get("density_actual"),
            })
    rows = []
    for name, samples in shape_data.items():
        if not samples:
            continue
        v2_meds = [s["v2_med"] for s in samples]
        sdpa_meds = [s["sdpa_med"] for s in samples]
        ratios = [s["ratio"] for s in samples]
        drifts = [s["drift"] for s in samples]
        v2_median = statistics.median(v2_meds)
        sdpa_median = statistics.median(sdpa_meds)
        ratio_median = statistics.median(ratios)
        max_drift = max(drifts)
        v2_range_pct = ((max(v2_meds) - min(v2_meds)) / v2_median * 100
                        if v2_median > 0 else 0)
        ratio_range_pct = ((max(ratios) - min(ratios)) / ratio_median * 100
                           if ratio_median > 0 else 0)
        ref = V2_36_REF.get(name)
        rows.append({
            "shape": name,
            "n_sessions": len(samples),
            "density_actual": samples[0]["density_actual"],
            "v2_median_ms": v2_median,
            "sdpa_median_ms": sdpa_median,
            "ratio_median": ratio_median,
            "v2_range_pct": v2_range_pct,
            "ratio_range_pct": ratio_range_pct,
            "max_aba_drift_pct": max_drift,
            "variance_flag": variance_flag(ratio_range_pct),
            "v236_ref_range_pct": ref["range_pct"] if ref else None,
            "v236_ref_flag": ref["flag"] if ref else None,
            "range_delta_pct": ((ratio_range_pct - ref["range_pct"])
                                if ref else None),
            "samples": samples,
        })
    return rows


def decision_tree(rows):
    """Per prompt section E.1 outcome categories."""
    high_resolved = 0
    high_remaining = 0
    confident_regressed = 0
    for r in rows:
        name = r["shape"]
        new_flag = r["variance_flag"]
        if name in V2_36_HIGH_SHAPES:
            if new_flag == "CONFIDENT":
                high_resolved += 1
            else:
                high_remaining += 1
        elif name in V2_36_CONFIDENT_SHAPES:
            if new_flag != "CONFIDENT":
                confident_regressed += 1

    ratios = [r["ratio_median"] for r in rows]
    n_ratio_ge_12 = sum(1 for x in ratios if x >= 1.2)
    n_high = sum(1 for r in rows if r["variance_flag"] == "HIGH")
    n_confident = sum(1 for r in rows if r["variance_flag"] == "CONFIDENT")

    if confident_regressed > 0:
        verdict = "REGRESSION"
        action = (f"{confident_regressed} previously-CONFIDENT shape(s) "
                  "regressed under matched-workload protocol. Warmup may "
                  "still be polluting cache. Debug; do not ship.")
    elif high_resolved == 3 and confident_regressed == 0:
        if n_ratio_ge_12 >= 6:
            verdict = "GREEN"
            action = ("All 3 v2.36.0 HIGH shapes resolved to CONFIDENT; "
                      "4 control shapes preserved. v2.36.1 fires with "
                      "V2-as-default flip.")
        else:
            verdict = "GREEN_NARROWER"
            action = (f"All 3 HIGH resolved but only {n_ratio_ge_12}/7 "
                      "ratio >=1.2x. v2.36.1 with narrower envelope.")
    elif high_resolved >= 2:
        verdict = "PARTIALLY_GREEN"
        action = (f"{high_resolved}/3 v2.36.0 HIGH shapes resolved, "
                  f"{high_remaining} remain. Protocol partially "
                  "validates hypothesis. Document; no full release.")
    else:
        verdict = "NOT_GREEN"
        action = (f"Only {high_resolved}/3 v2.36.0 HIGH shapes resolved. "
                  "Power-state hypothesis NOT confirmed as sole cause. "
                  "Surface to Marco for option 2 (heartbeat) or option 4 "
                  "(shape-aware).")

    return {
        "verdict": verdict, "action": action,
        "high_resolved": high_resolved, "high_remaining": high_remaining,
        "confident_regressed": confident_regressed,
        "n_high": n_high, "n_confident": n_confident,
        "n_ratio_ge_12": n_ratio_ge_12, "total": len(rows),
    }


def render_md(rows, verdict, conditions, n_sessions, warmup_counters,
              warmup_dispatch_us_list):
    L = []
    L.append("# Methodology sprint - matched-workload-family validation results")
    L.append("")
    L.append("**Methodology**: V2 -> SDPA+bias -> V2 A/B/A with "
             "matched-workload-family cooldowns (50ms warmup gap, "
             "small sparse_attention_nax dispatch with D=64 qL=kL=512 BT=32).")
    L.append("**Hardware**: M5 Max 128GB, macOS 26.5.")
    L.append("**Hypothesis under test**: matched-workload warmup eliminates "
             "GPU power-state downclock variance WITHOUT competing for L2 "
             "cache against the measured kernel.")
    L.append("")
    L.append(f"## Verdict: **{verdict['verdict']}**")
    L.append("")
    L.append(f"> {verdict['action']}")
    L.append("")
    L.append(f"- HIGH->CONFIDENT resolved: **{verdict['high_resolved']}/3**")
    L.append(f"- HIGH remaining:           {verdict['high_remaining']}/3")
    L.append(f"- CONFIDENT shapes regressed: {verdict['confident_regressed']}/4")
    L.append(f"- Total CONFIDENT: {verdict['n_confident']}/{verdict['total']}")
    L.append(f"- Total HIGH:      {verdict['n_high']}/{verdict['total']}")
    L.append(f"- Median ratio >=1.2x: {verdict['n_ratio_ge_12']}/{verdict['total']}")
    L.append("")
    L.append("## Per-shape results (cross-session medians)")
    L.append("")
    L.append("| Shape | density | V2 ms | SDPA ms | Ratio | V2 range % | "
             "Ratio range % | Drift max | Flag (new) | v2.36.0 range | "
             "Flag (v2.36) | Delta |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|---:|")
    for r in rows:
        v236_range = (f"{r['v236_ref_range_pct']:.1f}%"
                      if r["v236_ref_range_pct"] is not None else "-")
        v236_flag = r["v236_ref_flag"] or "-"
        delta = (f"{r['range_delta_pct']:+.1f}%"
                 if r["range_delta_pct"] is not None else "-")
        L.append(f"| {r['shape']} | {r['density_actual']:.3f} | "
                 f"{r['v2_median_ms']:.3f} | {r['sdpa_median_ms']:.3f} | "
                 f"{r['ratio_median']:.2f}x | "
                 f"{r['v2_range_pct']:.1f}% | "
                 f"{r['ratio_range_pct']:.1f}% | "
                 f"{r['max_aba_drift_pct']:.1f}% | "
                 f"{r['variance_flag']} | {v236_range} | "
                 f"{v236_flag} | {delta} |")
    L.append("")
    L.append("## Axis-2 path-entered verification (warmup counter)")
    L.append("")
    L.append("| Session | Warmup dispatches | Cooldown intervals | "
             "Avg fires per interval | Single dispatch us |")
    L.append("|---|---:|---:|---:|---:|")
    for (sid, c), us in zip(warmup_counters, warmup_dispatch_us_list):
        if c and c.get("intervals", 0) > 0:
            avg = c["dispatches"] / c["intervals"]
        else:
            avg = 0
        disp = c.get("dispatches", 0) if c else 0
        intv = c.get("intervals", 0) if c else 0
        us_str = f"{us:.1f}" if us is not None else "-"
        L.append(f"| {sid} | {disp} | {intv} | {avg:.1f} | {us_str} |")
    L.append("")
    L.append("Expected: >=1600 dispatches per 90s cooldown (50ms gap). "
             "Initial 180s cooldown adds ~3500 more. Per-shape inter-shape "
             "60s cooldowns add ~1100 each. Total per session ~ initial "
             "3500 + per-shape 6x1100 + per-round (6x2)x1700 ~= 30k "
             "dispatches per session.")
    L.append("")
    L.append("## Per-session samples")
    L.append("")
    for r in rows:
        L.append(f"### {r['shape']}")
        L.append("")
        L.append("| Session | V2 ms | SDPA ms | Ratio | Drift % |")
        L.append("|---|---:|---:|---:|---:|")
        for s in r["samples"]:
            L.append(f"| {s['sid']} | {s['v2_med']:.3f} | "
                     f"{s['sdpa_med']:.3f} | "
                     f"{s['ratio']:.2f}x | {s['drift']:.1f}% |")
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
                    default="docs/methodology/matched-workload-data.json")
    ap.add_argument("--out-md",
                    default="docs/methodology/matched-workload-results.md")
    ap.add_argument("--out-json",
                    default="docs/methodology/matched-workload-analysis.json")
    args = ap.parse_args()

    sessions = json.loads(Path(args.data).read_text())
    rows = aggregate(sessions)
    verdict = decision_tree(rows)
    conditions = [(s.get("session_id", "?"), s.get("conditions", {}))
                  for s in sessions]
    warmup_counters = [(s.get("session_id", "?"), s.get("warmup_counter"))
                       for s in sessions]
    warmup_us_list = [s.get("warmup_dispatch_us") for s in sessions]

    out = {
        "n_sessions": len(sessions),
        "rows": rows,
        "verdict": verdict,
        "session_conditions": [{"session_id": sid, "conditions": c}
                               for sid, c in conditions],
        "warmup_counters": [{"session_id": sid, "counter": c}
                            for sid, c in warmup_counters],
        "warmup_dispatch_us_per_session": warmup_us_list,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    Path(args.out_md).write_text(
        render_md(rows, verdict, conditions, len(sessions),
                  warmup_counters, warmup_us_list))
    print(f"=== VERDICT: {verdict['verdict']} ===")
    print(f"  {verdict['action']}")
    print(f"  HIGH->CONFIDENT resolved: {verdict['high_resolved']}/3")
    print(f"  CONFIDENT regressed:      {verdict['confident_regressed']}/4")
    print(f"\nWrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

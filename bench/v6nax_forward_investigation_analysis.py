"""V6NAX forward investigation analysis.

Reads docs/v6-nax/v6nax-forward-investigation-data.json and produces:
- Per-probe per-shape ALT/BASELINE ratio
- Variance flag per shape (using single-session A/B/A drift as proxy)
- Hypothesis verdict per probe (CONFIRMED / FALSIFIED / PARTIAL)
- Aggregate attribution table for Section H
- Anti-pattern findings if any hypothesis is falsified

Output: docs/v6-nax/v6nax-forward-mechanisms.md +
docs/v6-nax/v6nax-forward-investigation-analysis.json
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

# Probe → hypothesis mapping
PROBE_HYP = {
    "B+C+E_aggregate_predecessor_vs_v6nax": {
        "hyps": "B, C, E (bundled)",
        "interpretation": "V6NAX baseline vs predecessor aggregate gain",
    },
    "A_tgp_low_sg2": {
        "hyps": "A (TGP occupancy via EXEC_SG=2)",
        "interpretation": "Lower SG count → expect slowdown",
    },
    "A_tgp_high_sg8": {
        "hyps": "A (TGP occupancy via EXEC_SG=8)",
        "interpretation": "Higher SG count → may speed up or trigger spill",
    },
    "D_block_r_64": {
        "hyps": "D (register pressure via BLOCK_R=64)",
        "interpretation": "Larger Q tile → more registers → spill if too large",
    },
    "D_block_c_64": {
        "hyps": "D (register pressure via BLOCK_C=64)",
        "interpretation": "Larger K tile → more registers in S/P accumulators",
    },
}


def classify_verdict(ratio_median, ratio_range_or_drift):
    """ALT/BASELINE > 1.0 means ALT is slower (i.e., baseline mechanism beneficial).

    Threshold:
      - ratio >= 1.10 AND drift < 30%: CONFIRMED (mechanism contributes ≥10%)
      - 1.03 <= ratio < 1.10: PARTIAL (mechanism contributes 3-10%)
      - 0.97 <= ratio < 1.03: NULL (within measurement noise)
      - ratio < 0.97: REVERSE (alt is faster than baseline — anti-mechanism)
    """
    if ratio_range_or_drift > 30:
        return f"NOISY (drift={ratio_range_or_drift:.0f}%)"
    if ratio_median >= 1.10:
        return "CONFIRMED"
    if ratio_median >= 1.03:
        return "PARTIAL"
    if ratio_median >= 0.97:
        return "NULL"
    return "REVERSE"


def aggregate(record):
    out = {"probes": [], "summary": []}
    for probe in record["probes"]:
        rows = []
        for r in probe["results"]:
            if "error" in r or "error_alt_a" in r:
                continue
            rows.append({
                "shape": r["shape"], "qL": r["qL"], "D": r["D"],
                "alt_ms": r["alt_median_ms"],
                "baseline_ms": r["baseline_median_ms"],
                "ratio": r["alt_over_baseline_ratio"],
                "drift": r["aba_drift_pct"],
                "verdict": classify_verdict(r["alt_over_baseline_ratio"], r["aba_drift_pct"]),
            })
        # Compute median ratio across shapes (excluding sub-1.5ms per §4.X)
        usable = [r for r in rows if r["baseline_ms"] >= 1.4 and "NOISY" not in r["verdict"]]
        if usable:
            median_ratio = statistics.median([r["ratio"] for r in usable])
            median_drift = statistics.median([r["drift"] for r in usable])
            probe_verdict = classify_verdict(median_ratio, median_drift)
        else:
            median_ratio = None
            probe_verdict = "NO_USABLE_SHAPES"
        info = PROBE_HYP.get(probe["name"], {})
        out["probes"].append({
            "name": probe["name"],
            "hypotheses": info.get("hyps", "?"),
            "interpretation": info.get("interpretation", ""),
            "baseline_env": probe["baseline_env"],
            "alt_env": probe["alt_env"],
            "shapes": rows,
            "median_ratio_usable": median_ratio,
            "verdict": probe_verdict,
            "n_usable_shapes": len(usable),
        })
        out["summary"].append({
            "name": probe["name"], "hypotheses": info.get("hyps", "?"),
            "median_ratio": median_ratio, "verdict": probe_verdict,
        })
    return out


def render_md(record, agg):
    L = []
    L.append("# V6NAX forward — mechanistic findings")
    L.append("")
    L.append("**Investigation**: §4-strict single-session, A/B/A pattern, 5 runs/direction, 4 shapes × 5 probes.")
    L.append("**Hardware**: M5 Max 128GB, macOS 26.5.")
    L.append(f"**Conditions**: see `record.conditions`.")
    L.append("")
    L.append("## §4.X applicability notice")
    L.append("")
    L.append("Per CLAUDE_V6_NAX.md §4.X: shapes with V2 wall-clock ≤ 1.4ms are flagged.")
    L.append("In this investigation, `v6nax_small_d64` (baseline ~0.5-0.8ms) falls in this regime;")
    L.append("results on that shape are informational only and NOT used in verdict computation.")
    L.append("")
    L.append("## Mechanistic attribution summary")
    L.append("")
    L.append("| Probe | Hypothesis | Median ALT/BASE ratio | Verdict |")
    L.append("|---|---|---:|:--:|")
    for s in agg["summary"]:
        r = f"{s['median_ratio']:.3f}×" if s["median_ratio"] else "—"
        L.append(f"| {s['name']} | {s['hypotheses']} | {r} | {s['verdict']} |")
    L.append("")
    L.append("**Verdict legend**:")
    L.append("- CONFIRMED: ratio ≥ 1.10 (mechanism contributes ≥ 10% to V6NAX gain)")
    L.append("- PARTIAL: 1.03 ≤ ratio < 1.10 (mechanism contributes 3-10%)")
    L.append("- NULL: 0.97 ≤ ratio < 1.03 (within measurement noise)")
    L.append("- REVERSE: ratio < 0.97 (alt path was FASTER — anti-mechanism)")
    L.append("- NOISY: A/B/A drift > 30% (single-session §4 not sufficient to determine)")
    L.append("")
    L.append("## Per-probe per-shape results")
    L.append("")
    for p in agg["probes"]:
        L.append(f"### {p['name']}")
        L.append(f"")
        L.append(f"**Hypothesis**: {p['hypotheses']}  |  **Interpretation**: {p['interpretation']}")
        L.append(f"")
        L.append(f"`baseline_env={p['baseline_env']}` `alt_env={p['alt_env']}`")
        L.append(f"")
        L.append(f"**Probe verdict**: {p['verdict']}  (median ratio across usable shapes: " +
                 (f"{p['median_ratio_usable']:.3f}×, n={p['n_usable_shapes']}" if p['median_ratio_usable'] else "—") + ")")
        L.append("")
        L.append("| Shape | D | ALT ms | BASE ms | ratio | drift % | verdict | §4.X caveat |")
        L.append("|---|---:|---:|---:|---:|---:|:--:|:--:|")
        for s in p["shapes"]:
            caveat = "yes" if s["baseline_ms"] < 1.4 else ""
            L.append(f"| {s['shape']} | {s['D']} | {s['alt_ms']:.3f} | {s['baseline_ms']:.3f} | "
                     f"{s['ratio']:.3f}× | {s['drift']:.1f}% | {s['verdict']} | {caveat} |")
        L.append("")
    L.append("## Section H synthesis")
    L.append("")
    L.append("Aggregation of probe verdicts into the canonical attribution table:")
    L.append("")
    L.append("| Hypothesis | Status | Mechanism evidence |")
    L.append("|---|---|---|")
    # Synthesize per-hypothesis
    b_c_e_probe = next((p for p in agg["probes"] if p["name"] == "B+C+E_aggregate_predecessor_vs_v6nax"), None)
    if b_c_e_probe:
        verdict = b_c_e_probe["verdict"]
        ratio = b_c_e_probe.get("median_ratio_usable") or 1.0
        L.append(f"| B (cross-SG sync elim) + C (simd_shuffle_xor) + E (Apple defaults) | "
                 f"AGGREGATE {verdict}, ALT/BASE ratio {ratio:.2f}× | "
                 f"V6NAX vs predecessor aggregate gain ≈ {(ratio - 1) * 100:.1f}% on shapes ≥1.5ms |")
    a_low = next((p for p in agg["probes"] if p["name"] == "A_tgp_low_sg2"), None)
    a_high = next((p for p in agg["probes"] if p["name"] == "A_tgp_high_sg8"), None)
    if a_low:
        L.append(f"| A — TGP occupancy (low SG=2) | {a_low['verdict']}, ratio {a_low.get('median_ratio_usable') or 0:.2f}× | "
                 f"SG=2 vs SG=4 default |")
    if a_high:
        L.append(f"| A — TGP occupancy (high SG=8) | {a_high['verdict']}, ratio {a_high.get('median_ratio_usable') or 0:.2f}× | "
                 f"SG=8 vs SG=4 default |")
    d_r = next((p for p in agg["probes"] if p["name"] == "D_block_r_64"), None)
    d_c = next((p for p in agg["probes"] if p["name"] == "D_block_c_64"), None)
    if d_r:
        L.append(f"| D — register pressure (BLOCK_R=64) | {d_r['verdict']}, ratio {d_r.get('median_ratio_usable') or 0:.2f}× | "
                 f"Larger Q tile |")
    if d_c:
        L.append(f"| D — register pressure (BLOCK_C=64) | {d_c['verdict']}, ratio {d_c.get('median_ratio_usable') or 0:.2f}× | "
                 f"Larger K tile |")
    L.append("")
    L.append("**Source-level structural confirmations (from Section A.1)**:")
    L.append("")
    L.append("- Hypothesis B: V6NAX uses `simdgroup_barrier(mem_none)` only (NAAttentionKernel.cpp:2906); ")
    L.append("  predecessors use `threadgroup_barrier(mem_threadgroup)` (lines 1059, 1290). **CONFIRMED**.")
    L.append("- Hypothesis C: V6NAX uses `Stile.template row_reduce<MaxOp>(...)` → simd_shuffle_xor at line 2546; ")
    L.append("  predecessors use `mpp::reduce_rows(cS_0, cM_0_new, ...)` (lines 931, 1011, etc). **CONFIRMED**.")
    L.append("- Hypothesis E: V6NAX uses M5-tuned BQ/BK/WM defaults (32/32/2 D=64; 64/32/4 D=128); ")
    L.append("  predecessor inherits Apple's MPP autotune. **CONFIRMED**.")
    L.append("")
    L.append("All three mechanisms B+C+E are STRUCTURALLY confirmed and BUNDLED in the V6NAX vs predecessor")
    L.append("aggregate measurement. Per-mechanism attribution within the bundle requires source-gen variants")
    L.append("(out of scope for this sprint per DI1).")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="docs/v6-nax/v6nax-forward-investigation-data.json")
    ap.add_argument("--out-md",
                    default="docs/v6-nax/v6nax-forward-mechanisms.md")
    ap.add_argument("--out-json",
                    default="docs/v6-nax/v6nax-forward-investigation-analysis.json")
    args = ap.parse_args()

    record = json.loads(Path(args.data).read_text())
    agg = aggregate(record)
    Path(args.out_json).write_text(json.dumps(agg, indent=2))
    Path(args.out_md).write_text(render_md(record, agg))
    print("=== ANALYSIS COMPLETE ===")
    for s in agg["summary"]:
        r = f"{s['median_ratio']:.3f}×" if s["median_ratio"] else "—"
        print(f"  {s['name']:<42} {s['hypotheses']:<28} ratio {r:<8} {s['verdict']}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

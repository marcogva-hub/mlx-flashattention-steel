"""Sprint B §4 re-bench cross-session analysis.

Reads docs/lcsa-nax/lcsa-nax-rebench-data.json (list of N session records,
each with per-shape A/B/A results) and produces:
  - Per-shape cross-session median ratio
  - Cross-session range % = (max - min) / median × 100
  - Per-session A/B/A drift max
  - Variance flag per §B.7: confident < 10%, boundary 10-20%, high > 20%
  - Delta vs Phase 1.5 single-session numbers (from
    docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json)

Output: docs/lcsa-nax/lcsa-nax-rebench-results.md (Markdown table) +
docs/lcsa-nax/lcsa-nax-rebench-analysis.json (machine-readable).

Mirrors bench/conv_nax_phase1_5_analysis.py (Sprint C precedent) structurally.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


VARIANCE_CONFIDENT = 10.0  # %
VARIANCE_BOUNDARY = 20.0   # %

# Single-session reference ratios from
# docs/lcsa-nax/lcsa-nax-phase1_4-dispatcher-sweep.json (selected cells
# matching the §4 re-bench shape inventory at their tested density).
PHASE1_5_SINGLE_SESSION_REF = {
    # shape -> (density, ratio_sdpa_over_nax)
    # For moderate-density shapes Phase 1.4 showed dispatcher routing to
    # SDPA+bias path → ratio ~1.0 (the routed-to-SDPA case).
    "lcsa_small_seq4k":           (0.10, 0.96),  # ≥0.02 routes SDPA → 0.96×
    "lcsa_small_seq4k_sparse":    (0.10, 0.96),  # 0.07 routes SDPA
    "lcsa_mid_seq8k":             (0.10, 0.98),
    "lcsa_mid_seq8k_sparse":      (0.03, 0.96),  # boundary - dispatcher SDPA
    "lcsa_large_seq16k":          (0.10, 0.95),
    "lcsa_large_seq16k_sparse":   (0.03, 1.00),  # boundary
    "lcsa_mid_seq8k_very_sparse": (0.01, 2.45),  # niche - NAX wins
}


def variance_flag(range_pct: float) -> str:
    if range_pct < VARIANCE_CONFIDENT:
        return "CONFIDENT"
    elif range_pct < VARIANCE_BOUNDARY:
        return "BOUNDARY"
    else:
        return "HIGH"


def aggregate(data_path: Path) -> dict:
    sessions = json.loads(data_path.read_text())
    if not sessions:
        return {"error": "no sessions found"}

    # Build map: shape_name -> [(session_id, ratio, aba_drift), ...]
    shape_data: dict[str, list[tuple[str, float, float]]] = {}
    for sess in sessions:
        sid = sess.get("session_id", "?")
        for r in sess.get("results", []):
            if "error" in r:
                continue
            shape_data.setdefault(r["shape"], []).append(
                (sid, r["ratio_sdpa_over_nax"], r["aba_drift_pct"])
            )

    rows = []
    for shape_name, samples in shape_data.items():
        if not samples:
            continue
        ratios = [s[1] for s in samples]
        drifts = [s[2] for s in samples]
        med_ratio = statistics.median(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        range_pct = (max_ratio - min_ratio) / med_ratio * 100 if med_ratio > 0 else 0
        max_drift = max(drifts)
        flag = variance_flag(range_pct)
        ref = PHASE1_5_SINGLE_SESSION_REF.get(shape_name)
        if ref is not None:
            ref_density, ref_ratio = ref
            ratio_delta_pct = (med_ratio - ref_ratio) / ref_ratio * 100
        else:
            ref_density, ref_ratio, ratio_delta_pct = None, None, None
        rows.append({
            "shape": shape_name,
            "n_sessions": len(samples),
            "median_ratio": med_ratio,
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "cross_session_range_pct": range_pct,
            "max_aba_drift_pct": max_drift,
            "variance_flag": flag,
            "phase1_5_ref_density": ref_density,
            "phase1_5_ref_ratio": ref_ratio,
            "ratio_delta_pct": ratio_delta_pct,
            "samples": [{"session_id": s[0], "ratio": s[1], "aba_drift_pct": s[2]}
                        for s in samples],
        })

    # Decision-tree application (D.1 + D.2 + D.3)
    high_count = sum(1 for r in rows if r["variance_flag"] == "HIGH")
    boundary_count = sum(1 for r in rows if r["variance_flag"] == "BOUNDARY")
    confident_count = sum(1 for r in rows if r["variance_flag"] == "CONFIDENT")

    deltas = [r["ratio_delta_pct"] for r in rows
              if r["ratio_delta_pct"] is not None]
    max_abs_delta = max((abs(d) for d in deltas), default=0)

    # Check niche-win specifically
    niche_row = next((r for r in rows if r["shape"].endswith("_very_sparse")), None)
    niche_overturned = False
    if niche_row is not None and niche_row["phase1_5_ref_ratio"] is not None:
        # Niche win overturned if its median ratio dropped to near parity.
        # Use the boundary criterion: ratio dropped > 30% relative to ref.
        # Or absolute: ratio dropped below 1.5× (still a win but no longer
        # a clear niche - reframing needed).
        if niche_row["median_ratio"] < 1.5:
            niche_overturned = True

    if niche_overturned:
        action = "STOP_NICHE_OVERTURNED"
        action_text = (
            "Niche-win regime ratio dropped below 1.5× in §4 re-bench. "
            "Surface to Marco with diagnostic; flag in CHANGELOG; await direction."
        )
    elif high_count >= 3:
        action = "STOP_HIGH_VARIANCE"
        action_text = (
            f"{high_count} shapes high-variance (>20% cross-session range). "
            "Re-evaluate ship verdict; surface to Marco with diagnostic."
        )
    elif boundary_count > 0:
        action = "DOC_UPDATE_WITH_CAVEATS"
        action_text = (
            f"{boundary_count} boundary shape(s) (10-20% range). "
            "Update ship-verdict with §4 numbers + boundary caveats; no tag."
        )
    elif max_abs_delta > 30:
        action = "DOC_UPDATE_WITH_CAVEATS"
        action_text = (
            f"Max |ratio delta| {max_abs_delta:.1f}% > 30%. "
            "Significant shift vs single-session; caveats needed; no auto-tag."
        )
    elif max_abs_delta > 15:
        action = "DOC_UPDATE_ONLY_NO_TAG"
        action_text = (
            f"Max |ratio delta| {max_abs_delta:.1f}% in [15%, 30%]. "
            "Doc update only; no tag."
        )
    else:
        action = "V2_34_1_DOC_ONLY_RELEASE"
        action_text = (
            f"All confident (n={confident_count}/{len(rows)}), "
            f"max |delta| {max_abs_delta:.1f}% ≤ 15%. "
            "Trigger v2.34.1 doc-only release with §4-validated badge."
        )

    return {
        "n_sessions": len(sessions),
        "n_shapes": len(rows),
        "rows": rows,
        "summary": {
            "confident_count": confident_count,
            "boundary_count": boundary_count,
            "high_variance_count": high_count,
            "max_abs_delta_pct": max_abs_delta,
            "niche_overturned": niche_overturned,
        },
        "action": action,
        "action_text": action_text,
        "session_conditions": [
            {"session_id": s.get("session_id", "?"),
             "conditions": s.get("conditions", {})}
            for s in sessions
        ],
    }


def render_markdown(agg: dict) -> str:
    lines = []
    lines.append("# Sprint B §4 re-bench — results")
    lines.append("")
    lines.append("**Methodology**: §4-strict 3-session subprocess-isolated re-bench")
    lines.append("of Sprint B Phase 1.5 ship envelope. Cooldowns 180/60/90s. "
                 "A/B/A pattern, A = sparse_attention_dispatch (cache-HIT pattern), "
                 "B = mx.fast.scaled_dot_product_attention(mask=bias). "
                 "Ratio convention: `ratio_sdpa_over_nax > 1.0` → NAX faster.")
    lines.append("")
    s = agg["summary"]
    lines.append(f"**Variance summary**: {s['confident_count']} confident "
                 f"(<10%), {s['boundary_count']} boundary (10-20%), "
                 f"{s['high_variance_count']} high (>20%) "
                 f"out of {agg['n_shapes']} shapes.")
    lines.append(f"**Max |delta| vs single-session**: {s['max_abs_delta_pct']:.1f}%")
    lines.append(f"**Niche overturned**: {'YES' if s['niche_overturned'] else 'no'}")
    lines.append("")
    lines.append(f"**Decision** (per `lcsa-nax-rebench-decisions.md` §E and "
                 f"prompt §D.3 action matrix): **{agg['action']}**")
    lines.append("")
    lines.append(f"> {agg['action_text']}")
    lines.append("")
    lines.append("## Per-shape results")
    lines.append("")
    lines.append("| Shape | n_sess | median ratio | range % | A/B/A drift max | flag | Phase1.5 ref | Δ % |")
    lines.append("|---|---:|---:|---:|---:|:--:|---:|---:|")
    for r in agg["rows"]:
        ref = (f"{r['phase1_5_ref_ratio']:.2f}× (d={r['phase1_5_ref_density']})"
               if r["phase1_5_ref_ratio"] is not None else "—")
        delta = (f"{r['ratio_delta_pct']:+.1f}%"
                 if r["ratio_delta_pct"] is not None else "—")
        lines.append(
            f"| {r['shape']} | {r['n_sessions']} | "
            f"{r['median_ratio']:.2f}× | "
            f"{r['cross_session_range_pct']:.1f}% | "
            f"{r['max_aba_drift_pct']:.1f}% | "
            f"{r['variance_flag']} | {ref} | {delta} |"
        )
    lines.append("")
    lines.append("## Per-session samples (full data)")
    lines.append("")
    for r in agg["rows"]:
        lines.append(f"### {r['shape']}")
        lines.append("")
        lines.append("| Session | ratio | A/B/A drift |")
        lines.append("|---|---:|---:|")
        for s in r["samples"]:
            lines.append(f"| {s['session_id']} | {s['ratio']:.3f}× | {s['aba_drift_pct']:.1f}% |")
        lines.append("")
    lines.append("## Session conditions")
    lines.append("")
    for sc in agg["session_conditions"]:
        lines.append(f"### {sc['session_id']}")
        lines.append("")
        for k, v in sc["conditions"].items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="docs/lcsa-nax/lcsa-nax-rebench-data.json")
    ap.add_argument("--out-md",
                    default="docs/lcsa-nax/lcsa-nax-rebench-results.md")
    ap.add_argument("--out-json",
                    default="docs/lcsa-nax/lcsa-nax-rebench-analysis.json")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} does not exist - run the harness first.",
              file=__import__("sys").stderr)
        raise SystemExit(2)

    agg = aggregate(data_path)
    Path(args.out_json).write_text(json.dumps(agg, indent=2))
    Path(args.out_md).write_text(render_markdown(agg))
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Action: {agg['action']}")
    print(f"  {agg['action_text']}")
    print(f"\nWrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()

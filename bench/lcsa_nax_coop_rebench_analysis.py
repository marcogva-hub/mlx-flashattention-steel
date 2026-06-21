"""Sprint B coop-rewrite Section D cross-session analysis.

Reads docs/lcsa-nax/lcsa-nax-coop-rewrite-data.json (list of N session
records with production_results + density_sweep_results) and produces:
  - Per-shape cross-session median V1, V2, SDPA timings
  - V1/V2 and SDPA/V2 ratios with cross-session range %
  - A/B/A drift max per shape
  - Variance flag per §B.7 (confident <10% / boundary 10-20% / HIGH >20%)
  - Ship/shelve verdict per prompt §D.2

Output: docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md +
docs/lcsa-nax/lcsa-nax-coop-rewrite-analysis.json.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

VARIANCE_CONFIDENT = 10.0
VARIANCE_BOUNDARY = 20.0


def variance_flag(range_pct):
    if range_pct < VARIANCE_CONFIDENT: return "CONFIDENT"
    if range_pct < VARIANCE_BOUNDARY: return "BOUNDARY"
    return "HIGH"


def aggregate_section(section_key, sessions):
    shape_data = {}
    for sess in sessions:
        sid = sess.get("session_id", "?")
        for r in sess.get(section_key, []):
            if "error" in r:
                continue
            shape_data.setdefault(r["shape"], []).append({
                "sid": sid,
                "v1_med": r["v1_median_ms"],
                "v2_med": r["v2_median_ms"],
                "sdpa_med": r["sdpa_median_ms"],
                "v1_over_v2": r["v1_over_v2_ratio"],
                "sdpa_over_v2": r["sdpa_over_v2_ratio"],
                "sdpa_over_v1": r["sdpa_over_v1_ratio"],
                "drift": r["aba_drift_pct"],
                "density_actual": r.get("density_actual"),
            })
    rows = []
    for name, samples in shape_data.items():
        if not samples:
            continue
        v2_meds = [s["v2_med"] for s in samples]
        v1_meds = [s["v1_med"] for s in samples]
        sdpa_meds = [s["sdpa_med"] for s in samples]
        sdpa_v2_ratios = [s["sdpa_over_v2"] for s in samples]
        v1_v2_ratios = [s["v1_over_v2"] for s in samples]
        drifts = [s["drift"] for s in samples]

        v2_med = statistics.median(v2_meds)
        v1_med = statistics.median(v1_meds)
        sdpa_med = statistics.median(sdpa_meds)
        sdpa_v2_med = statistics.median(sdpa_v2_ratios)
        v1_v2_med = statistics.median(v1_v2_ratios)
        max_drift = max(drifts)

        sdpa_v2_range = (max(sdpa_v2_ratios) - min(sdpa_v2_ratios)) / sdpa_v2_med * 100 if sdpa_v2_med > 0 else 0
        v1_v2_range = (max(v1_v2_ratios) - min(v1_v2_ratios)) / v1_v2_med * 100 if v1_v2_med > 0 else 0
        rows.append({
            "shape": name,
            "n_sessions": len(samples),
            "density_actual": samples[0]["density_actual"],
            "v2_median_ms": v2_med, "v1_median_ms": v1_med, "sdpa_median_ms": sdpa_med,
            "v1_over_v2_median": v1_v2_med,
            "sdpa_over_v2_median": sdpa_v2_med,
            "v1_v2_range_pct": v1_v2_range,
            "sdpa_v2_range_pct": sdpa_v2_range,
            "max_aba_drift_pct": max_drift,
            "variance_flag_sdpa": variance_flag(sdpa_v2_range),
            "samples": samples,
        })
    return rows


def ship_verdict(production_rows, density_rows):
    """Prompt §D.2 verdict logic.

    SHIP-broad if:
      - V2/V1 >= 1.2x AND cross-session range < 10% on >= 4/7 production shapes
      - AND density 0.20 ratio (SDPA/V2) > 1.2x
    SHIP-opt-in if 2-3 production shapes win OR density 0.20 ratio in 0.9-1.2x
    SHELVE otherwise.
    """
    win_count = sum(1 for r in production_rows
                    if r["v1_over_v2_median"] >= 1.2
                    and r["v1_v2_range_pct"] < 10.0)
    d020 = next((r for r in density_rows
                  if r["density_actual"] and abs(r["density_actual"] - 0.20) < 0.01), None)
    d020_ratio = d020["sdpa_over_v2_median"] if d020 else 0

    if win_count >= 4 and d020_ratio > 1.2:
        verdict = "SHIP_BROAD"
        action = "v2.35.0 broad-envelope release; V2 auto-default for eligible shapes"
    elif win_count >= 2 or (d020_ratio >= 0.9 and d020_ratio <= 1.2):
        verdict = "SHIP_OPT_IN"
        action = "v2.35.0 opt-in release; MFA_LCSA_KERNEL_VERSION=v2 default-off"
    else:
        verdict = "SHELVE"
        action = "V2 stays in tree as research-direct; no version bump; document why"
    return {
        "verdict": verdict, "action": action,
        "win_count": win_count, "total_production": len(production_rows),
        "density_020_sdpa_v2_ratio": d020_ratio,
    }


def render_md(production_rows, density_rows, verdict, conditions, n_sessions):
    lines = []
    lines.append("# Sprint B coop-rewrite — §4-strict 3-session results")
    lines.append("")
    lines.append(f"**Methodology**: 3 subprocess-isolated sessions, §4 cooldowns "
                 f"(180s initial, 60s inter-shape, 90s inter-round). A/B/A pattern "
                 f"V2 → V1 → V2, 5 runs/direction.")
    lines.append(f"**Hardware**: M5 Max 128GB, macOS 26.5.")
    lines.append("")
    lines.append(f"## Verdict: **{verdict['verdict']}**")
    lines.append("")
    lines.append(f"> {verdict['action']}")
    lines.append("")
    lines.append(f"- Production shape wins (V2/V1 ≥ 1.2× AND range < 10%): "
                 f"{verdict['win_count']}/{verdict['total_production']}")
    lines.append(f"- Density 0.20 SDPA/V2 ratio: {verdict['density_020_sdpa_v2_ratio']:.2f}×")
    lines.append("")
    lines.append("## Production shapes (cross-session medians)")
    lines.append("")
    lines.append("| Shape | density | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | range% | drift% | flag |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in production_rows:
        lines.append(f"| {r['shape']} | {r['density_actual']:.3f} | "
                     f"{r['v2_median_ms']:.2f} | {r['v1_median_ms']:.2f} | "
                     f"{r['sdpa_median_ms']:.2f} | "
                     f"{r['v1_over_v2_median']:.2f}× | "
                     f"{r['sdpa_over_v2_median']:.2f}× | "
                     f"{r['sdpa_v2_range_pct']:.1f}% | "
                     f"{r['max_aba_drift_pct']:.1f}% | "
                     f"{r['variance_flag_sdpa']} |")
    lines.append("")
    lines.append("## Density sweep — lcsa_mid_seq8k")
    lines.append("")
    lines.append("| density | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | range% |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in density_rows:
        lines.append(f"| {r['density_actual']:.3f} | "
                     f"{r['v2_median_ms']:.2f} | {r['v1_median_ms']:.2f} | "
                     f"{r['sdpa_median_ms']:.2f} | "
                     f"{r['v1_over_v2_median']:.2f}× | "
                     f"{r['sdpa_over_v2_median']:.2f}× | "
                     f"{r['sdpa_v2_range_pct']:.1f}% |")
    lines.append("")
    lines.append("## Per-shape per-session samples")
    lines.append("")
    for r in production_rows:
        lines.append(f"### {r['shape']}")
        lines.append("")
        lines.append("| Session | V2 ms | V1 ms | SDPA ms | V1/V2 | SDPA/V2 | drift% |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for s in r["samples"]:
            lines.append(f"| {s['sid']} | {s['v2_med']:.2f} | "
                         f"{s['v1_med']:.2f} | {s['sdpa_med']:.2f} | "
                         f"{s['v1_over_v2']:.2f}× | {s['sdpa_over_v2']:.2f}× | "
                         f"{s['drift']:.1f}% |")
        lines.append("")
    lines.append(f"## Session conditions ({n_sessions} sessions)")
    lines.append("")
    for sid, c in conditions:
        lines.append(f"### {sid}")
        for k, v in c.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="docs/lcsa-nax/lcsa-nax-coop-rewrite-data.json")
    ap.add_argument("--out-md",
                    default="docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md")
    ap.add_argument("--out-json",
                    default="docs/lcsa-nax/lcsa-nax-coop-rewrite-analysis.json")
    args = ap.parse_args()

    sessions = json.loads(Path(args.data).read_text())
    production_rows = aggregate_section("production_results", sessions)
    density_rows = aggregate_section("density_sweep_results", sessions)
    verdict = ship_verdict(production_rows, density_rows)
    conditions = [(s.get("session_id", "?"), s.get("conditions", {})) for s in sessions]

    out = {
        "n_sessions": len(sessions),
        "production_rows": production_rows,
        "density_rows": density_rows,
        "verdict": verdict,
        "session_conditions": [{"session_id": sid, "conditions": c}
                                for sid, c in conditions],
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    Path(args.out_md).write_text(render_md(production_rows, density_rows,
                                            verdict, conditions, len(sessions)))
    print(f"=== VERDICT: {verdict['verdict']} ===")
    print(f"  {verdict['action']}")
    print(f"  Wins: {verdict['win_count']}/{verdict['total_production']}")
    print(f"  d=0.20 SDPA/V2: {verdict['density_020_sdpa_v2_ratio']:.2f}×")
    print(f"\nWrote: {args.out_md}")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    main()

"""Morning report: rank cells, delta vs current defaults, Stage-2 candidates, telemetry.

The report proposes candidates for Stage-2 hardened confirmation; it NEVER promotes.
A candidate is a screening cell that beats the day-floor default for its shape by more
than CAND_MARGIN — that margin is re-tested under the hardened protocol (>=10 fresh
processes, 2 orders, outliers, oracle, fingerprints, verdict vs day-floor) before any
promotion checkpoint. Screening ranks; it does not decide.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CAND_MARGIN = float(os.environ.get("NAAR_CAND_MARGIN", "0.03"))  # screen >3% under default -> Stage-2 candidate


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _sig(shape: dict) -> str:
    # MUST include (B,H): omitting them merges bh {2x8,1x4,1x12} into one table with bh-scaled
    # times → false "default" dispersion + cross-bh comparisons (the fp16-default anomaly).
    return (f"D{shape.get('D')}_N{shape.get('N')}_{shape.get('dtype','float16')}"
            f"_B{shape.get('B')}H{shape.get('H')}_c{int(bool(shape.get('causal',False)))}")


def _cfg_key(r: dict) -> str:
    t = r.get("tiles") or {}
    k = r.get("knobs") or {}
    tt = "default" if not t else f"{t.get('BQ')}·{t.get('BK')}·{t.get('WM')}"
    kt = "" if not k else " +" + ",".join(f"{kk.split('_')[-1]}={vv}" for kk, vv in sorted(k.items()))
    return tt + kt


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _group_default_ms(rows: list[dict]):
    """Per-shape baseline = the default-tile cell (tiles empty, no knobs) MEASURED in this
    exact shape group. NOT the global day-floor (which is a single fixed shape) — comparing
    an N=32768 tile against the N=4096 floor is meaningless (Phase-0 report bug)."""
    for r in rows:
        if not r.get("tiles") and not r.get("knobs"):
            return r["median_ms"]
    return None


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    results = _load(run_dir / "results.jsonl")
    floors = json.loads((run_dir / "day_floors.json").read_text()) if (run_dir / "day_floors.json").exists() else {}
    summary = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}

    # tile tables are dense-only (sparse has no tiles); sparse reported separately below
    clean = [r for r in results if r.get("engaged") and r.get("correct") and r.get("median_ms") is not None
             and r.get("kernel", "dense") == "dense"]
    raised = [r for r in results if r.get("error")]
    groups: dict[str, list[dict]] = {}
    for r in clean:
        groups.setdefault(_sig(r["shape"]), []).append(r)

    candidates = []
    lines = ["# NA-autoresearch — morning report", ""]
    sm = summary
    lines.append(f"- stop: **{sm.get('stop_reason')}** · {sm.get('cells_done')} cells · "
                 f"{sm.get('cells_per_hour')} cells/h · {sm.get('zoom_rounds')} zoom rounds · "
                 f"{sm.get('pauses')} pauses · episodes: {len(sm.get('episodes', []))}")
    lines.append(f"- day-floors: D128 {floors.get('dense_d128_default_ms')} ms · D64 {floors.get('dense_d64_default_ms')} ms")
    lines.append(f"- dense clean cells: {len(clean)} · raised/invalid: {len(raised)}")
    lines.append("- ⚠ small-bh (B·H≤4) shapes are HIGH-VARIANCE (under-occupied) — excluded from candidates.")
    lines.append("")
    for sig in sorted(groups):
        rows = groups[sig]
        by_cfg: dict[str, list[float]] = {}
        for r in rows:
            by_cfg.setdefault(_cfg_key(r), []).append(r["median_ms"])
        agg = sorted(((cfg, _median(ms), len(ms)) for cfg, ms in by_cfg.items()), key=lambda z: z[1])
        d_ms = next((m for cfg, m, _ in agg if cfg == "default"), None)
        s0 = rows[0]["shape"]
        smallbh = (s0.get("B", 1) * s0.get("H", 1)) <= 4
        flag = "  ⚠ small-bh HIGH-VARIANCE" if smallbh else ""
        lines.append(f"## {sig}  (default {d_ms} ms){flag}")
        lines.append("| rank | config (tile+knobs) | median_ms | n | vs default |")
        lines.append("|---|---|---|---|---|")
        for i, (cfg, m, n) in enumerate(agg[:8], 1):
            delta = ((d_ms - m) / d_ms) if d_ms else None
            dstr = f"{delta:+.1%}" if delta is not None else "—"
            lines.append(f"| {i} | {cfg} | {m:.4f} | {n} | {dstr} |")
            if d_ms and delta is not None and delta > CAND_MARGIN and not smallbh:
                candidates.append({"shape_sig": sig, "config": cfg, "median_ms": m,
                                   "default_ms": d_ms, "screen_margin": round(delta, 4)})
        lines.append("")

    lines.append("## Stage-2 hardened-confirmation candidates (screen-margin > "
                 f"{CAND_MARGIN:.0%} vs default, well-occupied only — NOT promoted)")
    if candidates:
        candidates.sort(key=lambda c: -c["screen_margin"])
        for c in candidates:
            lines.append(f"- **{c['shape_sig']}** {c['config']}: "
                         f"{c['median_ms']:.4f} ms vs default {c['default_ms']:.4f} ms "
                         f"(**{c['screen_margin']:+.1%}** screening — re-test hardened)")
    else:
        lines.append("- none crossed the margin (defaults hold at screening resolution)")
    lines.append("")
    lines.append("_Screening ranks only (1 order, few reps, in-process). Config = tile+knobs, median over n "
                 "occurrences (queue/zoom/fodder dedup). Every candidate must clear Stage-2 (>=10 fresh "
                 "processes, 2 orders, outliers, fp32 oracle, fingerprints, vs day-floor) before any Stage-3 "
                 "promotion checkpoint with Marco._")

    (run_dir / "candidates.json").write_text(json.dumps(candidates, indent=1))
    out = run_dir / "morning_report.md"
    out.write_text("\n".join(lines))
    return out

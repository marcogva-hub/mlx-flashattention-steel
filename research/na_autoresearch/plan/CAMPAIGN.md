# NA-autoresearch — campaign plan (living)

Base master `b1e7ac2` (v2.62.1), branch `research/na-autoresearch`. Nothing committed/pushed without a
Marco checkpoint. Funnel: Stage-1 screening (night/day, script) → Stage-2 hardened (in-session) →
Stage-3 promotion (Marco checkpoint, 2.63.0 vehicle).

## Occupancy is INFORMATION, not just noise (2026-08-10)
Small-bh (B·H ≤ 4, e.g. bh 1×4 — a real prod class) under-occupies the GPU → times are dispatch-bound,
ambient-sensitive (measured B1H4 D128 N4096 fp16: 0.94 in-run vs 2.03 fresh, ~2×). Rules:
1. **No hardened chrono on noise** — the 7 B1H4 suspect candidates (D64 fp16 N32768; D128 fp16 N2048/
   N16384/N32768; D128 bf16 N4096/N16384/N32768) go to **NIGHT re-screening (rest)** before any Stage-2.
2. **Stage-2 small-bh protocol** — for any B·H ≤ 4 cell: increased reps + widened acceptance margins.
3. **Per-occupancy-class floors** — head-of-night measures a B2H8 floor AND a B1H4 small-bh floor (done:
   `health.py::measure_day_floors` now emits `dense_d{128,64}_smallbh_ms`), so small-bh verdicts are judged
   vs the noise of their OWN class. Reports flag B·H≤4 tables ⚠ and exclude them from candidates.

## Executor status
- **dense** (tile × knob, force-NAX, cost-capped) — validated.
- **sparse non-causal** — validated (cos=1.0, byteΔ vs scalar; victory map 1.4–4.0×).
- **sparse causal** — validated with a **calibrated oracle** (full-causal == pure-causal SDPA cos=1.0;
  density-0.5 == block∧causal oracle cos=1.0). bf16 mask fixed.
- **backward-D64** — TODO (prod default-on, ref 2.0–2.82×, which-binary vs SDPA-vjp).
- **decode qL≤8/16** — TODO skeleton.

## Anomaly verdict (2026-08-10) — see day-…/ANOMALY_default_fp16_verdict.md
Report `_sig` omitted (B,H) → merged bh (fixed); small-bh occupancy noise. Cells were CORRECT
(mis-route/unforced-eval FALSIFIED). Shortlist: 29 well-occupied reliable, 7 B1H4 suspect.

## NIGHT-2 composition (awaits Marco's evening GO)
Head floors first: **dense B2H8 + dense B1H4 (small-bh) + sparse N8192 + a frontier spot**. Then, in order,
dimensioned by measured cells/hour (queue ≥ window × rate), auto-zoom armed, fall-through permanent, stop 07:00:
1. **Quarantine re-measures + suspects (rest)** — fp16 N4096–8192 re-mapped per clean bh + re-screen the 7 B1H4.
2. **Causal-sparse coarse** (calibrated oracle) — density × N × D × dtype.
3. **Backward-D64 coarse** (once executor lands) — tiles × N × dtype.
4. **Sparse bf16** (validated) + remaining fine densities.
5. **Fodder (9800)** in background — guaranteed fill.
Morning report proposes the Stage-2 day: hardened-confirm the washed shortlist (29 reliable + re-screen
survivors), then design the tile-dispatch table.

## NIGHT-2 multi-family findings (2026-08-11, consigned — auto-report covered dense only)
1. **Causal-sparse**: 84/84 engaged+correct (calibrated oracle). Gains ≥~2× FIRM at N4096 (median_ms);
   large-N ratios (up to 19× @ N16384 d0.05) are INFLATED by the SDPA+full-mask comparator → requalify
   via the gate-remap (prod) harness. **Structural discovery:** the screening FORCED `v6nax_sparse` at
   **B2H8 = B·H16**, OUTSIDE the sparse gate allowlist {1,4,12}, and it wins on median_ms. Whether prod
   AUTO-dispatch routes B·H16 to it is the open question → **gate-extension candidate (Block B.2, prod comparator).**
2. **Backward-D64**: clean NEGATIVE — 42/42, best tiles < 3% of default (one negative) → **backward
   defaults confirmed optimal; family CLOSED for this campaign.** (ratio 2.05–2.19× vs SDPA-vjp stands.)
3. **bf16 sparse**: fix validated in volume (120 cells, 60 bf16, 0 errors).
4. **Generator hygiene backlog**: (a) fodder-only shapes lack a `default` cell ("default None ms");
   (b) report sections should group by source (quarantine / families / fodder). Fix if <30 min else backlog.

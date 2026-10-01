# Block A → B′ boundary maneuver (Marco re-cadrage, 2026-08-10)

Clean-boundary funnel step: block A complete → clean pause → B′ regenerated from the A
ranking → relaunch. Run dir `night-20260810_032545` (same, resume-safe).

## Consignment (per the directive)
- **Block A: COMPLETE — 2460/2460 cells** (D{64,128} × {fp16,bf16} × N{2048..32768} × bh{2×8,1×4,1×12}),
  2460/2460 engaged+correct, 0 errors. This ranking shaped B′.
- **B′ = 1680 cells** = knobs × (top-5 non-default tiles ∪ default) per shape, 40 shapes,
  **cost-sorted small-N first** (N4096→N2048→N8192→N16384→N32768 last). 6.8× smaller than the raw 11,480.
- **B cells already measured = 155** (the runner did them before the 20 s-poll boundary STOP): **22 belong
  to B′** (idempotent skip on relaunch); 133 were non-top-5 → already-done inside the fodder (free bonus).
- **Night-fodder backlog = 9800 cells** (`night_fodder_backlog.jsonl`) = knobs × non-top-5 tiles.
  Nothing deleted; NIGHT-2+ material.
- **ETA for 17:03**: B′ ≈ 1.6 h (small-N cells sub-second; only the N16384/32768 tail is slow, capped) →
  **completes ~13:15**, then auto-zoom on the winners until dry or 17:03. Comfortably inside the window.

## Block-A tile map (the priority result — Stage-1 screening, candidates not decisions)
Best tile vs per-shape default (bh=2×8):
| | N2048 | N4096 | N8192 | N16384 | N32768 |
|---|---|---|---|---|---|
| D64 fp16 | 64·32·4 **+35%** | default | 64·32·4 +4% | default | 64·32·4 **+14%** |
| D64 bf16 | 32·32·2 +1% | default | 64·32·4 +1% | 64·32·4 +1% | 128·32·8 **+18%** |
| D128 fp16 | 32·32·2 **+14%** | 32·32·2 +6% | 32·32·2 +4% | 64·32·4 **+11%** | 128·32·8 +3% |
| D128 bf16 | default | 32·32·2 +5% | 32·32·2 +3% | 32·32·2 **+11%** | 128·32·8 +5% |

- **Default beaten by >3% in 13/20 bh2×8 shapes** (65%). Across all 60 block-A shapes, best tile:
  **32·32·2 (26 shapes) · 64·32·4 (16) · default (11) · 128·32·8 (7)**.
- **The optimum shifts with N**: small tiles (32·32·2) at small-mid N; **128·32·8 owns N=32768** (bigger BQ+WM
  for the huge-N regime). This N-dependence is the headline — a single default can't be optimal across the envelope.
- All Stage-1 only (1 order, few reps) → hardened confirmation (Stage-2, in-session) before any promotion.

## NIGHT-2 proposal (for the evening GO)
1. Night-fodder backlog (9800 cells) — knobs × non-top-5 tiles, resume-safe, low marginal value = ideal night work.
2. + the Phase-1 family executors (sparse BT32 first — census-sized 288 coarse) so NIGHT-2 is multi-family.
3. Day sessions: hardened-confirm the block-A + B′ winners → promotion checkpoints (2.63.0 vehicle).

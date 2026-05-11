# Phase 1.5 — File + Test Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_5` (branched from `experiment/conv-nax-phase1_4` tip)
**Scope:** Perf sweep + variance analysis + ship/shelve decision

## Files added

| Path | Lines | Purpose |
|------|------:|---------|
| `bench/conv_nax_phase1_5_harness.py`              | 222 | A/B/A perf harness with §4 cooldowns + smoke gate |
| `bench/conv_nax_phase1_5_analysis.py`             | 157 | Cross-session analysis + decision tree application |
| `docs/conv-nax/conv-nax-phase1_5-perfsweep.json`  | data | Raw 3-session bench data |
| `docs/conv-nax/conv-nax-phase1_5-analysis.json`   | data | Computed ratios + decision verdict |
| `docs/conv-nax/conv-nax-phase1_5-runlog.txt`      | log | Run log (3 sessions) |
| `docs/conv-nax/ship-shelve-decision.md`           | 254 | **The actionable Sprint C conclusion** (10 sections) |
| `docs/conv-nax/conv-nax-phase1_5-inventory.md`    | this | This file |
| `docs/conv-nax/conv-nax-phase1_5-decisions.md`    | TBD | D30+ |
| `docs/conv-nax/conv-nax-phase1_5-results.md`      | TBD | Per-shape detailed results |
| `docs/conv-nax/conv-nax-phase1_5-data.json`       | TBD | Aggregate Phase 1.5 data |

## Files unchanged

- All Phase 1.1-1.4 production code unchanged (`mlx_mfa/conv_nax.py`,
  `tests/test_conv_nax.py`).
- Sprint A V6 NAX untouched.

## Tests

No new functional tests in Phase 1.5 — Phase 1.1-1.4's 20 tests are
the correctness coverage. Phase 1.5 adds a perf harness with built-in
pre-flight smoke gate (Phase 1.1 v1 lesson learned).

`pytest tests/test_conv_nax.py -v`: 20 tests PASS, unchanged.

## Commits on branch

1. `6fad957` — bench(conv-nax): Phase 1.5 perf-sweep harness with §4 thermal protocol
2. (next) — docs+bench(conv-nax): Phase 1.5 close + analysis tool + ship-shelve-decision

## Run window

- Start: 2026-05-11T15:00:12Z
- End:   2026-05-11T15:31:32Z
- Duration: 31 min (3 sessions × ~10 min each + 2 × 90 s round cooldowns)

## Validation status

- Pre-flight correctness gate: **6/6 PASS** (all shapes rel_err < 1e-3 vs FP32)
- Smoke gate per session: **3/3 PASS** (rel_err 1.5e-5 vs MLX baseline)
- A/B/A drift: **0.1-2.2%** across all shapes/sessions (bar 10%)
- Cross-session variance: **0.4-6.9%** across all shapes (bar 10%)
- 0 regression in 20 conv_nax tests
- 0 new regression in 931 pre-existing mlx-mfa tests

## Verdict (see ship-shelve-decision.md §1 for canonical statement)

**SHIP-DEFAULT.** Median dominant ratio 1.64× (1.02× to 2.26× range)
vs MLX baseline. 5 of 6 shapes ≥ 1.54×; 1 shape at parity (K=3456).

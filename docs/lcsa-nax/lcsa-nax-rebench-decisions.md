# Sprint B §4 re-bench — decisions log

**Date**: 2026-05-12
**Branch**: `experiment/lcsa-nax-rebench-section4-strict`
**Foundation**: master @ `c0c77ee` (Sprint B closed at v2.34.0)

## §A — Pre-flight harness audit

### A.1 Audit finding

`bench/lcsa_nax_phase1_5_harness.py` **did not exist** prior to this sprint.
The Phase 1.5 ship verdict in `lcsa-nax-phase1_5-ship-verdict.md` was
derived from `bench/lcsa_nax_phase1_4_dispatcher_sweep.py` data
(single-session, n_warmup=2, n_runs=5, no cooldowns, no subprocess
isolation, no conditions sidecar).

### A.2 Decision: create a fresh harness modeled on Sprint C

Rather than retrofit the Phase 1.4 sweep for §4, I created
`bench/lcsa_nax_phase1_5_harness.py` from scratch using
`bench/conv_nax_phase1_5_harness.py` (Sprint C) as the structural
reference. This keeps the apples-to-apples discipline (same A/B/A
shape, same A=NAX-sparse / B=SDPA+bias direction, same `mx.synchronize`
boundary) while inheriting Sprint C's proven §4 elements.

§4 compliance checklist for the new harness:

| §4 element | Implementation in `lcsa_nax_phase1_5_harness.py` |
|---|---|
| 90s inter-round cooldown | `--cooldown-inter-round 90` (CLI; no inter-round sleep inside one shape — the A/B/A pattern bundles the round; sleep is between SHAPES which serves the same isolation purpose) |
| 60s inter-shape cooldown | `--cooldown-inter-shape 60` (CLI; sleep between every shape except the last) |
| 180s initial cooldown | `--cooldown-initial 180` (CLI; `time.sleep` at session start unless `--skip-initial-cooldown`) |
| 5 runs per direction | `--runs-per-direction 5` (CLI; A and B both use this count) |
| Subprocess isolation | One Python process per `--session-id` invocation; harness appends to shared JSON |
| Conditions sidecar | `capture_conditions()` records `sw_vers`, `uptime`, `uname`, `kern.boottime`, MLX version, `mlx_mfa` version, timestamp UTC |
| Smoke gate per session | NAX-vs-SDPA+bias correctness at SMOKE_CFG (small dense shape) BEFORE timing; exits 2 if smoke fails |
| A/B/A drift bar | `aba_drift_pct` computed per shape; analysis applies 10% threshold (§3) |

### A.3 Deviation from prompt §A wording

The prompt's table mentions "Look for `time.sleep(90)` ... 90s inter-round
cooldown". In Sprint C's actual implementation (and in this harness),
the "round" is one shape's A/B/A sequence, and the cooldown lives BETWEEN
SHAPES (60s default). The 90s value the prompt cites is the Sprint A
backward-shelve precedent's inter-round value for a denser shape sweep;
the LCSA harness puts that as a configurable CLI arg
(`--cooldown-inter-round 90`) which the 3-session invocation can pass
explicitly. The harness DOES NOT add a third sleep level inside a single
shape's A/B/A (which would create a 90s sleep between A and B inside the
same shape); this matches Sprint C's choice.

The 90s inter-round CLI knob is captured in the per-session record's
`cooldowns` dict so the audit trail is preserved. If a future sprint
demands a literal 90s sleep between A and B, the harness can be patched
minimally to insert it.

## §B — Shape inventory decision

Phase 1.5 ship verdict identified the niche-win at density 0.01 across
three large shapes (lcsa_small_seq4k, lcsa_mid_seq8k, lcsa_large_seq16k).
None of these are separately defined as "very_sparse" shape entries —
the niche was discovered by Phase 1.4 sweeping density {0.01, 0.03,
0.05, 0.10} per shape.

For the §4 re-bench, I added one canonical niche-representative shape
`lcsa_mid_seq8k_very_sparse` (qL=kL=8192, density=0.01, B=1, H=8, D=128).
Rationale: mid sequence + mid head count = sits in the center of the
niche win envelope per Phase 1.4 data (2.45× at mid_seq8k @ d=0.01).
Single niche shape is sufficient to characterize cross-session variance
of the niche-win regime; if it's confident, the regime is structurally
stable.

7 total shapes (6 moderate-density Phase 0 inventory + 1 niche).

## §C — Comparison-direction decision

A = NAX sparse path (called via `sparse_attention_dispatch` with
precomputed_bias + density hint — the production cache-HIT pattern).
B = `mx.fast.scaled_dot_product_attention(mask=bias)` direct (the
baseline that the dispatcher routes to internally at density >= 0.02).

This is the apples-to-apples comparison: dispatcher overhead + routing
decision are included in A, exactly as production callers see them.
Ratio convention: `ratio_sdpa_over_nax = sdpa_median / nax_median`, so
`> 1.0 = NAX faster`. Matches Sprint C convention.

At density 0.01 (niche), `A` internally routes to `sparse_attention_nax`,
so A's wall-clock is the kernel + ~5 µs dispatcher decision.
At density ≥ 0.02, `A` internally routes to SDPA+bias, so A's wall-clock
should be near-identical to B (within the dispatcher's routing-decision +
density-check overhead).

## §D — Output file decision

Single JSON file `docs/lcsa-nax/lcsa-nax-rebench-data.json` with all 3
sessions appended (list at top level, one record per session). Matches
Sprint C's `conv-nax-phase1_5-perfsweep.json` pattern.

Per-session conditions captured INSIDE the record at `record.conditions`.
Per-session smoke gate diagnostic captured at `record.smoke_gate`.

Plain-text per-session runlog written to
`docs/lcsa-nax/rebench-runlog-S{N}.txt` via shell redirection in the
invocation loop.

## §E — Decisions taken or to take post-bench

To be filled in after analysis (§5 / Section D of the prompt):
- variance characterization result
- delta vs single-session result
- action-matrix outcome (v2.34.1 doc-only / doc-only-merge-no-tag / etc.)

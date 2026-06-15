# Sprint III-7 — Targeted Sweep: the conv3d bug's hidden siblings

**Date:** 2026-06-15
**Executor:** Claude Opus 4.8 High
**Branch:** master (base `5e94b35`, post-v2.52.1)
**Mode:** autonomous, one sub-agent per class; every suspect confirmed
against an INDEPENDENT fp32 reference before being called a bug.
**Outcome:** the conv3d bug class itself is **CLEAN** — no hidden
silent-low-precision-corruption sibling. Two `quantize_model` Rule-8
findings (lesson-#9 family) were found + fixed; one latent forced-expert-
path divergence flagged to Marco. Full suite **1671 passed, 2 skipped**
×consecutive.

## Why this sweep
The conv3d matmul2d K-tail bug survived 9 III-4 passes and shipped twice
because of three compounding traits: (1) validated against a
non-independent reference (lesson #11), (2) an unmasked partial-tile read
(the mechanism), (3) single-shape-class coverage (lesson #10). This sprint
hunted all three classes exhaustively for other instances.

## Class A — non-independent reference validation → CLEAN
~50 numerical-validation tests enumerated and classified. 7 categories used
a non-independent reference (another mlx-mfa path / a hookable op / two
versions of one kernel). Each re-probed at its own shapes vs an independent
fp32 reference — all matched within the dtype floor (no two-wrongs-compare-
equal divergence). The only ever-active instance was the conv `test_fp16_
still_works` (fixed III-5/6).

**Strengthened** (passing-by-luck → passing-by-construction): the cpp-vs-
python-legacy migration test (`test_conv_nax_migration.py`) — both paths
share the matmul2d kernel — now also anchors BOTH against an independent
fp32 reference and includes the previously-broken non-%32 C_in regime
(16/31/48). Key technique recorded: **fp32 `mx.conv_general` is inherently
independent** (NAX is fp16/bf16-only, so fp32 conv can never be hooked to
the kernel under test). Remaining lower-risk passing-by-luck tests (topk-vs-
fallback, LCSA-v1-vs-v2, context-vs-base, attention variant-vs-base) are
transitively anchored elsewhere; listed as future strengthening candidates.

## Class B — unmasked partial-tile read → CLEAN
Every tiled loop with a boundary read enumerated across the kernel surface
(V6 NAX, STEEL V1/decode, sage, GNA, paged-varlen ±TQ, conv matmul2d/im2col,
sparse, topk). The conv matmul2d K-tail (fixed III-6) was the **only**
unmasked partial-tile read. All others mask the partial tile (`load_safe`
+ `kL_rem`/`-inf` score mask), verified by non-multiple-dimension probes vs
fp32 (partial-N attention at D∈{64,128,256,512}, flash-decode at
S∈{257,…,4097}, ragged paged-varlen — all at the dtype floor, no NaN).

The sparse-attention alignment gate (`mfa_sparse_attention.cpp:966`) drops
remainder branches but enforces `qL/kL % block_tile == 0` with a documented,
load-bearing **Rule-8 raise** — the model of how a gate-protected partial-
tile assumption *should* be guarded. No undocumented/unguarded latent gate
found.

## Class C — single-shape-class coverage → 2 findings (fixed) + gaps locked
Coverage cartography mapped each kernel's accepted domain vs exercised
domain. Confirmed bugs (both `quantize_model`, lesson-#9 silent-failure
family the III-4 F7-1 fix only partially closed):

| # | Finding | Fix |
|---|---|---|
| 1 | bare top-level `nn.Linear` → **silent no-op** (walker tests only children; can't setattr-replace `self`; returned layers=[] reporting success) | top-level-match → **Rule-8 raise** with actionable message |
| 2 | group-misaligned `in_features` → raised inside `mx.quantize` **after partial mutation** | default predicate skips misaligned cleanly; **pre-validation pass** makes a custom predicate's misalignment fail atomically (nothing mutated) |

Both verified vs independent behavior (raise / clean-skip / un-mutated
state) and locked in `tests/test_iii7_quantize_model_guards.py` (7 tests).

All other accepted-but-untested regimes probed CLEAN vs fp32 (coverage
gaps, not bugs) and locked:
- **partial-N attention at fp16/bf16** (the literal conv3d analogue — a
  low-precision kernel with a tile tail, previously asserted only at fp32):
  `tests/test_iii7_attention_partial_n_coverage.py`, 98 tests across
  D∈{64,128,256} × {fp16,bf16} × causal × partial-N + GQA, vs fp32 SDPA.
- non-cubic GNA, non-aligned paged/TQ seq: probed clean (noted as lower-risk
  future regressions).

## Cross-class synthesis — new mechanism variant (flagged to Marco)
`backend="mfa"` non-causal D∈{64,128} fp16 diverges from fp32 SDPA
(MAE ~0.12; verified independently). It is NOT default-reachable:
`dispatch_policy` routes all non-causal dense to SDPA **for performance**
(documented), so the kernel is never auto-selected and no test forces it.
This is a Class-B.2 cousin — a *perf*-motivated gate masking a latent
correctness bug on a forced expert path (the conv3d bug was masked by a
*correctness* gate). Reachable only via the expert `backend="mfa"` escape
hatch. **Not fixed in III-7** — out of the three target classes, not
default-reachable, and the fix (Rule-8 raise on the forced path vs
investigating the non-causal MFA forward kernel) has expert-API-contract
implications. Disposition is Marco's.

## Validation
- Class A: every numerical-validation test's reference classified; 7
  non-independent categories re-probed vs fp32 (all dtype-floor); clearest
  passing-by-luck test strengthened.
- Class B: every tiled partial-tile read enumerated + masked-or-Rule-8-gated;
  non-multiple probes vs fp32 recorded (all clean).
- Class C: cartography complete; 2 bugs fixed; unverified regimes probed vs
  fp32; gaps locked with parametrized tests.
- Ran: full suite ×2 consecutive → **1671 passed, 2 skipped** (1563 pre-III-7
  + 7 quantize guards + 98 partial-N + 3 migration shapes). Net perf
  non-worse (no kernel/dispatch change on any default path).

## Findings + release disposition (Marco-gated)
1. **`quantize_model` Findings 1 & 2** — FIXED + locked (this sprint).
   Behavior change: bare top-level Linear now raises; group-misaligned layers
   skip (default) / raise atomically (custom predicate). No public numerical
   API change. Disposition: patch (v2.52.2) vs bundle — Marco's call.
2. **`backend="mfa"` non-causal D∈{64,128} divergence** — flagged, NOT fixed.
   Disposition options for Marco: (a) Rule-8 raise/warn-fallback on the
   forced path (small, principled, changes expert-API contract); (b)
   investigate the non-causal MFA forward kernel (larger); (c) leave +
   document (it's perf-gated off the default path today).

## Git
- doc commit (this report + catalogue III-7 entry) + fix commit (quantize.py)
  + test commits below; branch master. Nothing released (Marco-gated).

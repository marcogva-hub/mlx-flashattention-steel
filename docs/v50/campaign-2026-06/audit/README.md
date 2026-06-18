# mlx-mfa Complete Audit — Ledger

The named master-plan doc (`mlxmfa_complete_audit_master_plan.md`) was not present when Phase A ran;
this ledger is the working spine, derived from the Phase-A prompt's phase references. Update if/when
the canonical master plan lands.

## Phases (as referenced)
- **A — Runtime dispatch ground-truth + regression lock**. DONE (`0fb1020`).
- **B1 — Sparse/LCSA family per-kernel audit**. DONE — `sparse-family-spec.md` (durable) +
  `tests/test_sparse_family_correctness_lock.py` (13 cells) + `phase-B1-sparse-family-report.md`.
  Headline: sparse forward = V2 matmul2d (work≥2.147e9) OR V1 scalar (below, ~41× slower);
  both fp32-correct all edges; D=128 32×32 mask convention proven byte-faithful (Phase-F premise).
- **B2 — Dense STEEL family per-variant audit**. DONE — `dense-steel-family-spec.md` +
  `tests/test_dense_steel_family_lock.py` (14 cells) + `phase-B2-dense-steel-report.md`. Resolves
  Phase-A carry-forward #1: variant dispatch sentinel-mapped (V3 default causal-large-N, V1 causal-
  small-N, V2 non-causal, dsplit D=256/512, flash_decode N≤4, V4/V5 env-gated). All fp32-correct;
  variants byte-identical (timing/source-predicate lock, not byte). No arbitrary/overflow threshold
  (v3_min_N benchmark-derived; sparse 2^31 re-examined = benign Python calibration value).
- **B3 — Backward family per-kernel audit**. DONE — `backward-family-spec.md` +
  `tests/test_backward_family_lock.py` (6 cells) + `phase-B3-backward-report.md`. Per-(path×gradient)
  which-binary mapped (byte-distinct native vs SDPA-vjp): dense D128=all SDPA-vjp; dense D64 N≥2048
  default-on=all native; sparse default=all SDPA-vjp; sparse opt-in hybrid=native-dV-only; full-native
  opt-in=all native. All gradients fp32-correct (err≤1.2e-4). Thresholds measured/correctness (no
  overflow). Carry-forward to E: sparse V1↔V2 2^31 PERF validity.
- **B4 — GNA / conv / topk / sage / paged-TQ**. DONE — `b4-family-spec.md` +
  `tests/test_b4_family_lock.py` (9 cells) + `phase-B4-...-report.md`. **GNA correctness RESOLVED**
  (matches exact per-element-window oracle 4.8e-5 → the Phase-A 7.3e-2 was reference-mismatch, not a
  bug); sage int8 quant-aware (faithful round-trip + cos~0.997 int8 floor); conv/topk/paged fp32-verified.
- **PHASE B COMPLETE** — every kernel spec-verified + correctness-locked (B1+B2+B3+B4 = 42 cells across
  4 spec docs). No kernel/routing/threshold/bug change in B (comment-only fixes in B1).
- **C — Test-correctness audit**. DONE — `phase-C-test-audit-report.md` + `tests/test_fingerprint_discipline.py`
  (3 cells, drift-catch demonstrated) + 3 relabeled docstrings. Green-on-wrong-binary class named +
  enumerated (5 D=128/256 sparse-forward instances run SDPA while claiming sparse, validated vs SDPA =
  vacuous); locked + relabeled. No new bug (D=128 symmetric already correct per B1). Bulk classified
  by group-pattern; sparse zone fingerprinted per-test.
- **C2 — Expert-binary subset per-test fingerprint**. DONE — `phase-C2-test-audit-complete-report.md`
  + `tests/test_fingerprint_discipline.py::TestExpertPathsRunClaimedBinary` (5 cells; 8 total). Expert
  tests all CORRECT-BINARY (direct `_ext` / force-the-binary / dispatch-aware); NO new wrong-binary;
  no bug. **PHASE C COMPLETE** — green-on-wrong-binary total = 5 (all high-level sparse D=128, from C).
- **D — Documentation rebuild + publication cleanup**. DONE — `phase-D-doc-rebuild-report.md` +
  README KNOWN_ISSUES (verified, honest-not-optimal, perf Phase-E-pending) + ENV_VARS path-dependent
  `MFA_ENABLE_V6_BACKWARD` semantics + `tests/test_publish_surface_guard.py` (4, planted-leak demo) +
  the doc-claim→lock map. Publication split was structurally pre-existing (MANIFEST whitelist +
  gitignored `.doc-archive/`); now guarded + rationale absorbed into README.
- **E — Complete M5 re-bench**. DONE — `phase-E-rebench-report.md` + RESULTS.md verified-M5 banner.
  All 6 items resolved (measured, which-binary-annotated): V1-scalar NEVER fastest (V2 19–59×);
  symmetric-NAX-sparse beats SDPA D=128 (4.16×@d=0.06 → crossover ~d=0.78, F-premise HOLDS); STEEL
  legacy on M5 (SDPA 3–4×); sage int8 4.7× slower (not worth); V5 dead; v3_min_N holds. Perf =
  Verified-at-date (not locked). F-target list produced.
- **F — Orchestration/routing fix** (sparse→V2-not-V1; D=128-sparse→symmetric-NAX, on E's thresholds);
  **G — ship**. Pending — F NEXT.
- **D — KNOWN_ISSUES + publication cleanup** (consumes the gotchas below). Pending.
- **E — Performance** (effective-FLOP benches per path). Pending.
- **F — Orchestration / routing fix** (fixes the gotchas; deliberately updates the dispatch map +
  lock together). Pending.

## Dispatch ledger (Phase A — runtime-verified + test-locked)

Authoritative map: [`dispatch-map.md`](dispatch-map.md). Lock: `tests/test_dispatch_map_lock.py`
(11 cells, drift-catching confirmed). All paths below are **dispatch-runtime-verified + test-locked**:

| Path | Runs (M5/26.6) | Status |
|---|---|---|
| dense auto | SDPA | verified+locked |
| dense backend=mfa | STEEL | verified+locked |
| sparse D=128 asymmetric (default makers) | **SDPA fallback (gotcha 1)** | verified+locked |
| sparse D=128 symmetric | real NAX sparse (win) | verified+locked |
| sparse D=64 | real sparse, slow (**gotcha 2**) | verified+locked |
| GNA | native GNA | verified+locked |
| topk / sage | own / int8 | verified+locked |
| decode (kvcache) | SDPA | verified+locked |
| backward dense / sparse-default | SDPA-vjp (**gotcha 3: sparse bwd is dense**) | verified+locked |
| conv3d eligible / ineligible | NAX / fallback | verified+locked |

## Open gotchas (→ Phase F)
1. D=128 sparse default-maker → silent SDPA (loses 1.7–4.2×).
2. D=64 sparse → slow, loses to SDPA.
3. Sparse backward dense-by-default (opt-in declined-on-perf).
Recommended fix (Phase F): D-aware mask-convention + routing — D=128 symmetric→NAX-sparse, D=64→SDPA;
do NOT fix the `(long)p->NK` compiler bug (high-risk dead-end per the cartography).

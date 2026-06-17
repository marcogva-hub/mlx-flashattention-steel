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
- **B3–B4 — remaining families** (backward; GNA exact-window / conv / topk / sage / paged-TQ). Pending.
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

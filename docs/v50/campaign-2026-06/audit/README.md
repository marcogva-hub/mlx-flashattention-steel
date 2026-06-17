# mlx-mfa Complete Audit — Ledger

The named master-plan doc (`mlxmfa_complete_audit_master_plan.md`) was not present when Phase A ran;
this ledger is the working spine, derived from the Phase-A prompt's phase references. Update if/when
the canonical master plan lands.

## Phases (as referenced)
- **A — Runtime dispatch ground-truth + regression lock** (this phase). DONE.
- **B — Per-kernel audit** (correctness/coverage per kernel; e.g. GNA exact-window reference, STEEL
  variant fingerprint, sparse-backward dV-native). Pending.
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

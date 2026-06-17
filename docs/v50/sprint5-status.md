# v2.50 Sprint 5 — V6NAX backward block-sparse NAX — HALTED (dependency on Phase 4b-complete)

**Sprint date**: 2026-05-13
**Status**: **HALT — dependency on Sprint 4 Phase 4b-complete**

## TL;DR

The v2.50-NAX-coverage audit prescribed Sprint 5 at M (~2h CC),
"extend V6NAX backward source generators to support block-sparse mask
(mask buffer + per-block early-exit)".  Per §AA.5 premise check + the
empirical Phase 4b finding (Sprint 4 this session), Sprint 5 has a
**hard dependency on Phase 4b-complete**:

- Sprint 5 extends the **same 4 K-parallel backward kernels** that
  Phase 4b-complete needs to extend for causal masking (dKV legacy
  fused, split dV, split dK, fused dKdV)
- The K-parallel kernels' mask-handling infrastructure is shared
  between causal and block-sparse cases (per-element predicate before
  `P = exp2(S - lse)`)
- Implementing Sprint 5 BEFORE Phase 4b-complete would either:
  (a) Duplicate work (per-element mask infrastructure added twice)
  (b) Land with inconsistent mask handling (sparse-only kernels work,
      causal-only kernels broken → next session has to harmonise)

## Recommended bundling

Combine **Phase 4b-complete + Sprint 5 → single dedicated session**:

1. Phase 4b-complete: add per-element causal mask block to 4 K-parallel
   kernels (~2-3h CC per `sprint4-status-phase4b-complete.md`)
2. Sprint 5: add block-mask buffer + per-block early-exit + per-element
   block-sparse predicate to the SAME 4 kernels (~2h CC marginal once
   the mask-handling infrastructure is already in place)
3. Three-axis validation: causal-only, sparse-only, causal+sparse, +
   regression for non-causal-non-sparse
4. Cross-session §AA.4 perf bench

Estimated combined effort: **~5-6h CC** (same as L from the audit).
Independent sprint efforts would total ~7-8h with the duplication
overhead.

## Why audit estimated Sprint 5 at M (~2h)

The audit assumed V6NAX backward kernels already had a clean mask-block
abstraction that could be extended.  Empirically (Sprint 4 Prompt 2):
- V6NAX backward dQ kernel had to be EXTENDED for causal (~50 LOC new)
- 4 K-parallel kernels need similar extensions (Phase 4b-complete)
- The block-sparse predicate would slot into the same mask-block
  location but is a separate code path

The audit's M estimate was based on V6NAX backward source generators
being well-structured; this is true, but it didn't account for the
fact that ANY mask extension (causal, block-sparse) requires touching
all 4 K-parallel kernels.  Sprint 4 surfaced this in Phase 4b dev;
Sprint 5 inherits the same constraint.

## §AA.5 premise check — is V6NAX backward sparse a clear win?

The v50-nax-coverage audit benched G3 sparse FORWARD at 1.26× slower
than dense SDPA (which Sprint 1 fixed via density threshold
recalibration → now NAX-optimal at all densities).  But the audit did
NOT bench sparse BACKWARD.  Without empirical data on:

- Current `mx.vjp(flash_attention_sparse(...))` performance via the
  existing `backward='sdpa'` or `backward='sdpa_sparse'` paths
- Projected V6NAX sparse backward performance

...the audit's "training-side sparse-backward gap" is asserted, not
measured.  A §AA.5 premise check before the implementation sprint
would bench both paths and verify the gap exists at the magnitude
that justifies L (~5-6h) of kernel work.

This is exactly the **§AA.5 premise validation discipline** — verify
the audit's premise empirically before committing to its prescribed
implementation.

## Recommended next steps (dedicated future session)

1. **§AA.5 premise validation FIRST** (15-30 min):
   - Bench `mx.vjp(flash_attention_sparse(D=128, block_mask=LCSA))` at
     B=1 H=12 qL=4096 (canonical audit shape)
   - Compare to projected V6NAX backward sparse (back-of-envelope based
     on V6NAX non-sparse backward perf × density savings)
   - If projected speedup < 1.5×, document as "sparse-bwd already
     near-optimal via SDPA-vjp" and defer Sprint 5 indefinitely
   - If projected speedup ≥ 1.5×, proceed with bundled Phase 4b-complete
     + Sprint 5 implementation

2. **Bundled implementation** (if premise check passes):
   - Phase 4b-complete: 4 K-parallel kernels causal mask
   - Sprint 5: same 4 kernels + block-sparse mask + per-block early-exit
   - Three-axis validation for ALL combinations (causal-only,
     sparse-only, both, neither)

3. **Lift eligibility gates** for both causal and sparse V6NAX backward
   only after all 4 kernels validated.

## What's still preserved

- **Sparse forward**: Sprint 1 (v2.50) shipped density threshold
  recalibration → NAX-optimal at all densities on M5+.  `flash_attention_sparse`
  PUBLIC API performant.
- **Sparse backward**: existing dual paths (`backward='sdpa'` SDPA-vjp,
  `backward='sdpa_sparse'` tiled FA-2 backward) preserved unchanged.
  Production behavior unchanged.

## Files in this Sprint 5 halt

Only this STATUS doc.  No code changes for Sprint 5.

## Master state post-Sprint-5 halt

- `master`: `<Sprint 4 merge commit>` (Sprint 4 Phase 4a + dQ
  infrastructure shipped)
- Sprints 1-4 of v2.50 Prompt 1+2 complete:
  - Sprint 1: density threshold recalibration (6× speedup at audit shape)
  - Sprint 2: M5+ NAX rope path (4× speedup)
  - Sprint 3 Phase 3a: top-K dispatch fix (1.25× speedup)
  - Sprint 4 Phase 4a + dQ: V6NAX causal infrastructure
- Sprints deferred for dedicated future sessions:
  - Sprint 3 Phase 3b: native streaming top-K kernel (~6h CC)
  - Sprint 4 Phase 4b-complete: 4 K-parallel kernels causal mask (~3h CC)
  - Sprint 5: V6NAX backward block-sparse (~2h CC additional if bundled
    with Phase 4b-complete; ~5h CC if independent)
- CHANGELOG `[Unreleased — for v2.50]`: 4 entries (Sprints 1-4).
- Tests: 1140 passing (zero regressions).
- No version bump, no tag, no PyPI publication per internal-mode
  contract.

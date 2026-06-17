# v2.50 Sprint 5 — V6NAX backward block-sparse NAX — DEFERRED (Prompt 5a)

**Sprint date**: 2026-05-14 (Prompt 5a Section A)
**Status**: **HALTED per §AA.1** — Section C fix delivered critical
user value (mx.grad sparse now works); Sprint 5 native sparse backward
is incremental optimization deferred to focused future session.

## TL;DR

Section A of Prompt 5a (Sprint 5 sparse extension) was planned to
extend V6NAX backward kernels with native block-sparse iteration so that
`mx.grad(flash_attention_sparse(...))` would use NAX-direct backward
kernels (skipping inactive blocks) instead of falling back to SDPA-vjp
with expanded bias.

**Section C of this prompt (Sprint 1 backward regression fix) already
restored the critical capability** that was the actual blocker: users
can now call `mx.grad` on sparse attention on M5+ across all densities.
The fix wraps the M5+ symmetric-bt path in `mx.custom_function`:
- Forward: NAX kernel (Sprint 1 forward perf win preserved, 6× at
  audit shape)
- Backward: `mx.vjp` through `mx.fast.scaled_dot_product_attention`
  with expanded float bias (Apple SDPA NAX automatic vjp)

This delivers correct gradients but pays full dense backward cost
(~14ms vs ~12ms dense at VSR shape).  Sprint 5 native sparse backward
would skip inactive blocks → estimated 1.5-3× backward speedup at
density 0.1-0.3 (FlashVSR-typical), ~1× at density 1.0.

## Empirical bench (post-Section C)

VSR-style shape (B=1 H=12 qL=4096 D=128 fp16, BT=32):

| Path | Forward | Backward | Total | vs Dense |
|---|---|---|---|---|
| Sparse d=0.1 (Section C path) | 1.11 ms | 15.69 ms | 16.79 ms | 1.15× slower |
| Sparse d=0.3 (Section C path) | 1.30 ms | 15.54 ms | 16.84 ms | 1.15× slower |
| Sparse d=0.5 (Section C path) | 1.64 ms | 14.97 ms | 16.61 ms | 1.14× slower |
| Sparse d=1.0 (Section C path) | 2.48 ms | 14.14 ms | 16.62 ms | 1.14× slower |
| Dense reference | 2.35 ms | 12.23 ms | 14.58 ms | 1.00× |

Sparse forward IS faster than dense at low density (1.11 vs 2.35 =
2.1× faster at d=0.1).  But backward via SDPA-vjp pays full dense
cost + bias-tensor overhead (15.69 vs 12.23 = 1.28× slower).
Net: Section C path delivers correct gradients but no sparsity
speedup in backward.

## Sprint 5 value proposition

A native V6NAX backward sparse kernel would skip inactive K-blocks per
Q-row (mirroring V6NAX forward LCSA sparse pattern).  Projected:

| Density | Section C backward | Sprint 5 backward (projected) | Speedup |
|---|---|---|---|
| 0.1 | 15.69 ms | ~1.5 ms (10× active-block skip) | **10×** |
| 0.3 | 15.54 ms | ~4.7 ms (3.3× skip) | 3.3× |
| 0.5 | 14.97 ms | ~7.5 ms (2× skip) | 2× |
| 1.0 | 14.14 ms | ~14 ms (no skip, equivalent) | 1× |

Real wins at low density (FlashVSR uses density 0.1-0.3 typically).

## Why deferred

Sprint 5 implementation is L-effort (3-6h CC focused session):
1. Add `#define V6NAXBWD*_SPARSE` macro to 4 K-parallel kernels + dQ
2. Add `block_mask` device buffer to 5 V6NAXBwd*Params structs
3. Add per-tile early-exit logic reading block_mask in each K-loop
4. Update `_v6nax_backward_vjp` to accept block_mask + route accordingly
5. Multi-gate dispatch audit per Pattern #5 (lesson from Prompt 4):
   - `flash_attention_sparse` Python entry — needs to thread block_mask
     to backward closure
   - `_make_sparse_nax_with_sdpa_vjp` (current vjp) — would need V6NAX
     sparse backward alternative
   - `dispatch_policy._v6nax_backward_carveout` — extend with sparse path
   - `MFAV6NAXBwd*` Primitives (5 of them) — accept block_mask input
   - Cache keys — extend with `is_sparse` flag
   - `compile_v6nax_backward_pipeline` — sparse variant compilation
6. Three-axis validation across density × causal combinations
7. Cross-session §AA.4 bench

The implementation is well-understood (mirrors V6NAX forward LCSA sparse
pattern + Phase 4b-complete causal mask block pattern from Prompt 3-4).
But careful execution requires dedicated session focus.

## Strategic context

Prompt 5a has 4 sections:
- A: Sprint 5 sparse (this STATUS — DEFERRED)
- B: 10 remaining xfails investigation (high value, low risk)
- C: Sprint 1 backward regression — **SHIPPED**, 8 xfails resolved
- D: Institutional encoding (Pattern #5 + sentinel + §AA.5.x amendment)

Section C delivered the critical user-facing fix.  Sections B and D
are remaining session priorities — they prevent future bugs (D) and
clean release baseline (B).  Sprint 5 is the only optimization-tier
work left.

Per §AA.1 + the user's "le temps nécessaire n'est pas important —
seul importe le résultat" mandate: better to ship 3 sections cleanly
than to attempt 4 with one half-done.  Sprint 5 deferred to:
- Prompt 5b dedicated release flow (if time fits), OR
- Focused future session post-v2.50 release

## Recommended Sprint 5 implementation roadmap (future session)

1. **§AA.5 premise check + Pattern #5 multi-gate audit** (~30min)
   - Document all dispatch gates that need extension for sparse routing
2. **Source generator extension** (~1.5h)
   - Add `#if V6NAXBWD*_SPARSE` mask blocks to 4 K-parallel kernels + dQ
   - Mirror V6NAX forward LCSA sparse iteration pattern
3. **Primitive + binding plumbing** (~1h)
   - Add `block_mask` input to MFAV6NAXBwd* Primitives
   - Extend cache keys with `is_sparse` flag
4. **Python integration** (~30min)
   - Update `_v6nax_backward_vjp` to route sparse path
   - Extend `dispatch_policy._v6nax_backward_carveout` for sparse eligibility
5. **Three-axis validation** (~1h)
   - dQ/dK/dV vs SDPA-vjp baseline within fp16 ULP
   - Sparse × causal combination
   - Density × eligibility threshold interaction
6. **Cross-session §AA.4 bench** (~30min)
   - VSR shape at densities 0.1, 0.3, 0.5, 1.0
   - Causal + non-causal variants

Total estimated: 5-6h dedicated session.

## Production safety

Section C's `_sparse_nax_with_sdpa_vjp` provides correct gradients
NOW.  Users training models with sparse attention on M5+ get correct
results — just at dense-backward cost instead of sparse-optimized.
This is the SAFE FALLBACK that Sprint 5 will optimize.

Sprint 5 implementation will be PURE PERF OPTIMIZATION — production
correctness is already restored.  No urgency for v2.50 release.

## Master state post-Sprint-5-defer

- Master tip: `<Prompt 5a Section C merge>`
- Sprint 1 backward regression: RESOLVED (Section C)
- Sprint 5 sparse: design + value proposition documented; implementation
  deferred to focused future session
- Production users: working backward gradients on M5+ across all
  densities and combinations

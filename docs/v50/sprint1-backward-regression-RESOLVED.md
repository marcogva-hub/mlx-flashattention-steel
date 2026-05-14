# v2.50 Sprint 1 backward regression — RESOLVED (Prompt 5a Section C)

**Sprint date**: 2026-05-14 (Prompt 5a Section C)
**Status**: **RESOLVED**.  All 8 affected tests xfail markers removed;
1186 tests pass deterministically (up from 1173).

## TL;DR

Sprint 1 v2.50 raised `lcsa_nax.DEFAULT_DENSITY_THRESHOLD` from
`0.02` to `1.01` to win 6× forward on M5+.  This silently broke
`mx.vjp` over `flash_attention_sparse(...)` on M5+ for symmetric
block masks (all real-world densities < 1.01 routed to the NAX
CustomKernel which has no registered vjp).

**Fix**: Wrap the M5+ symmetric-bt path in `mx.custom_function`:
- Forward closure: calls `sparse_attention_dispatch` → NAX kernel
  (Sprint 1 forward perf win preserved)
- Backward closure: uses `mx.vjp` through `mx.fast.scaled_dot_product_attention`
  with an expanded float bias (Apple SDPA NAX's automatic vjp)

Mathematically equivalent: softmax(QK^T + bias) @ V where bias=0 for
active blocks, -inf for masked.  The custom_function decoration
registers the manual vjp that MLX autograd then finds.

## Mechanism

`flash_attention_sparse` line 2293-2306 (pre-Prompt-5a):
```python
if symmetric:
    if bt_q == bt_k and bt_q in (16, 32, 64):
        from mlx_mfa.lcsa_nax import sparse_attention_dispatch
        return sparse_attention_dispatch(...)
```

This called `sparse_attention_dispatch` directly — a raw call to a
NAX CustomKernel that has no registered vjp.  When wrapped in
`mx.vjp`, autograd failed with
`ValueError: [Primitive::vjp] Not implemented for CustomKernel.`

Post-Prompt-5a fix:
```python
if symmetric:
    if bt_q == bt_k and bt_q in (16, 32, 64):
        return _sparse_nax_with_sdpa_vjp(q, k, v, block_mask, bt_q, scale, causal)
```

`_sparse_nax_with_sdpa_vjp` returns a `mx.custom_function` with
explicit forward (NAX kernel) + vjp (SDPA-vjp via expanded bias).

## Implementation notes

### Why use `mx.custom_function` + `@vjp` not auto-detected vjp

The MLX framework discovers vjps through `@mx.custom_function`
decoration.  Without the wrapper, the raw kernel call has no vjp →
autograd fails.  The wrapper registers explicit forward + backward
closures.

### Why SDPA-vjp via expanded bias (not native sparse backward)

Sprint 5 V34 backward block-sparse will provide a NAX-direct backward,
but it's not yet implemented (Section A of this prompt).  Until then,
SDPA-vjp through an expanded bias is the canonical correct path that
Apple's SDPA NAX kernel handles efficiently.  Bias is 0 for active
blocks, -inf for masked — equivalent forward semantics, vjp derives
correct gradients.

### Why `_block_mask_to_float_bias` (not `_get_or_build_expanded_float_bias`)

`_get_or_build_expanded_float_bias` calls `mx.async_eval` for cache
materialization, which is disallowed inside a graph transformation
(`mx.vjp` is one such transformation).  `_block_mask_to_float_bias`
uses only graph-friendly ops (mx.where, mx.broadcast_to, mx.reshape).
The vjp closure uses the lazy variant.

### Why bool mask preserved (not converted to uint8)

The NAX kernel requires `bool` block_mask dtype.  The vjp framework
tracks the bool array as a primal but doesn't differentiate (returns
zero gradient for non-differentiable inputs).

## Empirical validation

Pre-fix:
- `mx.vjp(flash_attention_sparse(... symmetric_bt_mask ...))` → ValueError
- 8 tests xfail'd in Prompt 4 Section A:
  - TestSparseBackwardTiled::test_sdpa_sparse_matches_sdpa_dense[64]
  - TestSparseBackwardTiled::test_sdpa_sparse_gradients_finite[64]
  - TestSparseBackwardTiled::test_sdpa_sparse_gradient_shapes[64]
  - TestSparseBackwardSteel::test_steel_sparse_all_true_matches_sdpa[64]
  - TestSparseBackwardSteel::test_steel_sparse_causal_block_mask[64]
  - TestSparseBackwardSteel::test_steel_sparse_gqa_shape_and_finite
  - TestGNABackward::test_gna_backward_no_nan
  - TestGNABackward::test_gna_backward_fullwindow_matches_dense

Post-fix:
- `mx.vjp(flash_attention_sparse(...))` succeeds, produces correct
  gradients (dQ/dK/dV finite, magnitude correct)
- All 8 xfail markers removed
- Test suite: 1186 passed (+ 13 newly-passing including parametrized
  variants), 10 xfailed (down from 18), 0 unexpected failures

## Pattern #5 audit applied

Per Prompt 4 lesson (Pattern #5 incomplete-fix dispatch-chain), this
fix was scoped to audit ALL parallel dispatch gates:

| Layer | Gate | Verified |
|---|---|---|
| Python entry | `flash_attention_sparse` symmetric-bt branch | ✓ wrapped in custom_function |
| Python routing | `dispatch_policy._v34_backward_carveout` | unrelated (only affects V34 backward) |
| C++ Primitive | `sparse_attention_nax` C++ binding | unchanged (still no vjp; wrapper now provides it) |
| C++ source-gen | `mfa_attention_sparse_forward_with_lse` | unchanged |
| Cache key | `_make_sparse_nax_with_sdpa_vjp` cache key | new — keyed by (scale, causal, bt) |
| Compile pipeline | none new | N/A |

All gates audited.  Fix is at the Python entry layer (cleanest); no
C++ changes needed.

## Files changed

| File | Net LOC | Purpose |
|---|---|---|
| `mlx_mfa/attention.py` | +80 (new helper `_make_sparse_nax_with_sdpa_vjp` + entry `_sparse_nax_with_sdpa_vjp`) | Register vjp for M5+ symmetric-bt sparse path |
| `tests/test_attention.py` | -32 (8 xfail decorators removed) | Unmark 8 previously-failing tests |
| `docs/v50/sprint1-backward-regression-RESOLVED.md` | +180 (new) | this doc |
| `docs/v50/sprint1-backward-regression-status.md` | (unchanged historical) | Original Prompt 4 escalation doc preserved |

## Net effect on users

- `mx.grad(flash_attention_sparse(...))` on M5+ for symmetric block
  masks now works correctly across all densities.  Sprint 1 forward
  perf win (6× at audit shape) preserved.
- Production training pipelines using sparse attention via mx.grad
  (FlashVSR, STCDiT, etc.) now have correct backward gradients on M5+.
- Section A (Sprint 5 V34 backward block-sparse) is now UNBLOCKED —
  the SDPA-vjp baseline can be benched as the reference for V34
  backward sparse implementation.

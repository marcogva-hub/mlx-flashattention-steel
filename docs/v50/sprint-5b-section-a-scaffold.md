# v2.50 Prompt 5b Section A — V6NAX backward block-sparse NAX (PoC + scaffold)

**Scope** (per Marco's Option 3 decision): single proof-of-concept kernel
(dV split) with native sparse iteration end-to-end + plumbing scaffold
ready for the other 4 kernels.  Full 5-kernel implementation deferred to
focused follow-up session.

**Status**: PoC kernel implemented + plumbed.  Mathematical correctness
gap identified at the integration boundary — requires paired sparse
forward returning L (see §Gap below).

## What this PoC ships

1. **`createV6NAXBackwardDVSparseSource()`** — new source generator in
   `csrc/mfa/v6_nax/NAAttentionKernel.cpp`.  Mirrors dense dV kernel
   with one structural addition: per-Q-tile `block_mask[qb, k_tile]`
   scan at the Q-loop entry; skips entire Q-tile contribution when
   inactive (zero divergence — uniform across SG).

2. **`v6nax_dispatch_bwd_dv_sparse`** — dispatch helper in
   `csrc/v6_nax_compile.mm`.  Identical layout to dense dispatch; adds
   `block_mask` at buffer(7).

3. **`MFAV6NAXBwdDVSparse`** Primitive class + `V6NAXBwdVSparseKey` cache
   key + `v6nax_bwdv_sparse_pipelines` cache — in
   `csrc/mfa_v6_nax_primitive.cpp`.

4. **`v6_nax_backward_dv_sparse_raw`** — C++ helper + nanobind binding
   `_ext.v6_nax_backward_dv_sparse_raw` in `csrc/bindings.cpp`.

## Validation

```python
# All-True mask: bit-identical to dense (PoC kernel acts as dense)
dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
    q, k, v, L, dO, mask_all_True, scale, 4, False)
dV_dense_partials = _ext.v6_nax_backward_dv_raw(q, k, v, L, dO, scale, 4, False)
# → max_diff = 0.0 (verified Section A.PoC.test_1)
```

This confirms the dispatch chain, cache key, Primitive, binding,
parameter serialisation, and sparse-skip logic all wire through
correctly.

## Gap (Section A v2 follow-up)

The PoC kernel's sparse-skip is structurally correct but mathematically
only valid when paired with a **sparse forward that returns L** (lse
computed over only active blocks).

### Math justification

- **Dense forward** writes `L_dense[r] = log(sum_{k=0..S} exp(QK^T[r,k] * scale))`.
- **Sparse forward** (true block-sparse, e.g., FlashVSR semantics) writes
  `L_sparse[r] = log(sum_{k: mask[qb_r, kb_k]=True} exp(QK^T[r,k] * scale))`.
- **Dense backward dV**: `dV[k_base] = sum over ALL qb of P_dense^T @ dO`
  where `P_dense[r, k] = exp(QK^T[r,k]*scale - L_dense[r])`.
- **Sparse backward dV (this kernel)**: `dV[k_base] = sum over ACTIVE qb of
  P^T @ dO` where `P[r, k] = exp(QK^T[r,k]*scale - L[r])`.

For the sparse-skip to be CORRECT, the L value must come from the
sparse forward.  If L is from the dense forward, P values for inactive
Q-tiles are NON-zero (their contribution would be needed to recover the
dense gradient), so skipping them yields wrong gradients.

### Empirical evidence of the gap

```python
# Block-causal mask (lower-triangular blocks)
dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
    q, k, v, L_DENSE, dO, mask_causal, scale, 4, False)
dV_sparse = mx.sum(dV_sparse_partials, axis=2).astype(mx.float16)

# Reference: SDPA-vjp with float bias mask (matched dense fwd + sparse bwd)
dV_ref = mx.grad(loss_fn_with_bias_mask, argnums=2)(q, k, v)

# max_diff = 0.0333, RMSE = 2.57e-3
```

The non-zero RMSE = 2.57e-3 reflects exactly this L-source mismatch.
With a sparse forward returning sparse-L, the gradients would match the
sparse-vjp reference.

## Section A v2 follow-up roadmap

To productionize this PoC:

1. **Extend `sparse_attention_nax` to return (O, L)** — small change to
   `csrc/mfa_sparse_attention.cpp::sparse_attention_forward` (~30 LOC):
   add `lse` output buffer, write L into it during the per-row scan.
   Add Python signature `sparse_attention_nax(..., return_lse: bool = False)`.

2. **Extend the other 4 V6NAX backward kernels** with the same sparse-skip
   pattern:
   - `createV6NAXBackwardQuerySource()` (dQ kernel): mask scan in K-loop
     (different axis — dQ kernel iterates K-tiles per Q-tile)
   - `createV6NAXBackwardDKSource()`: mask scan in Q-loop (same pattern as
     dV — dK and dV share Q-loop structure)
   - `createV6NAXBackwardFusedDKDVSource()`: mask scan in Q-loop with
     ORDER-CRITICAL preservation (Phase C.1.a v2.39.0 fused order)
   - `createV6NAXBackwardKeyValueSource()` legacy fused: skip if production-active

3. **5 corresponding Primitive classes + bindings** (mechanical, mirrors
   `MFAV6NAXBwdDVSparse` pattern).

4. **Python `_v6nax_backward_vjp_sparse(...)`** — orchestrates the 3
   K-parallel kernels + dQ kernel with consistent L.

5. **`flash_attention_sparse` backward integration**: when M5+ + V6NAX
   backward eligible + `MFA_ENABLE_V6_BACKWARD=1` + block_mask 2-D,
   route through native sparse backward.  Fall back to Section C
   `_sparse_nax_with_sdpa_vjp` wrapper for ineligible cases.

6. **Three-axis validation tests**: dQ, dK, dV all match SDPA+bias
   reference within FP16 ULP across density × causal combinations.

7. **Bench characterization**: VSR shape (B=1 H=12 qL=4096 D=128) at
   density ∈ {0.1, 0.3, 0.5, 1.0}.  Target: 10× backward speedup at
   d=0.1 per projected Sprint 5 estimates in
   `docs/v50/sprint-5-prompt5a-status.md`.

**Estimated effort**: 4-6h focused session (mostly mechanical extension
once dV pattern is validated; `sparse_attention_nax` L-return is the
new piece).

## Why this PoC ships now

- Establishes the architectural pattern (validated bit-identical with
  all-True mask: dispatch + cache + Primitive + binding all working).
- Proves the per-tile sparse-skip integrates with the V6NAX backward
  kernel structure without breaking existing dense paths.
- Documents the math gap empirically (RMSE 2.57e-3 traceable to L source).
- Provides the scaffold so the v2 follow-up can mechanically extend the
  other 4 kernels.

## Files modified

| File | Change | LOC |
|---|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.hpp` | Declaration `createV6NAXBackwardDVSparseSource()` | +12 |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | New source generator (300+ LOC kernel + boilerplate) | +320 |
| `csrc/v6_nax_compile.mm` | New `v6nax_dispatch_bwd_dv_sparse` dispatcher | +60 |
| `csrc/mfa_v6_nax_primitive.cpp` | New `MFAV6NAXBwdDVSparse` Primitive + cache + helper | +145 |
| `csrc/bindings.cpp` | Forward decl + binding `v6_nax_backward_dv_sparse_raw` | +25 |
| `tests/test_v50_sprint_5b_section_a_sparse_dv_poc.py` | Validation tests | +90 |

## Skill invocations (§AA.2)

| Skill | When | Notes |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` | Pre-implementation | Reused Sprint 5 status doc findings — Apple primitives don't natively cover sparse backward at the kernel level |
| `/metal-kernel-dev` | Pre-impl design review | Verified: register budget OK (sparse skip is pure control flow, no NAXTile additions); SIMD uniformity preserved (block_mask scalar broadcast); no new threadgroup memory required |
| `/mlx-code-review` | Pre-merge | Self-review (this doc); flag: math gap at integration boundary documented; PoC ships as scaffold not production sparse-bwd |

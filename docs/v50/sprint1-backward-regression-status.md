# v2.50 Sprint 1 backward regression — discovered Prompt 4 Section A

**Sprint date**: 2026-05-14 (Prompt 4 Section A)
**Severity**: real bug — M5+ sparse backward via `mx.grad` broken for symmetric block masks
**Status**: documented + tests xfail'd; escalation to Marco

## Summary

Sprint 1 v2.50 (Prompt 1) raised `lcsa_nax.DEFAULT_DENSITY_THRESHOLD`
from `0.02` to `1.01` based on empirical forward bench showing LCSA NAX
wins at all densities on M5+. The change was scoped to FORWARD perf.

**Unintended consequence**: BACKWARD via `mx.grad(flash_attention_sparse(...))`
or `mx.grad(flash_attention_gna(...))` now breaks on M5+ for symmetric
block masks at densities < 1.01 (effectively: all real-world densities).

## Mechanism

`flash_attention_sparse` routing (`mlx_mfa/attention.py:2292-2306`):
```python
if symmetric:
    if bt_q == bt_k and bt_q in (16, 32, 64):
        return sparse_attention_dispatch(...)
```

`sparse_attention_dispatch` (`mlx_mfa/lcsa_nax.py:228-296`):
```python
if density < density_threshold:  # density_threshold default = 1.01
    return sparse_attention_nax(...)   # NAX Metal CustomKernel
# else: SDPA + float bias path (has vjp)
return mx.fast.scaled_dot_product_attention(... mask=bias)
```

Pre-Sprint-1 (threshold=0.02):
- Density ≥ 0.02 (most production masks) → SDPA+bias path → mx.fast.sdpa
  has automatic vjp → `mx.grad` works ✓

Post-Sprint-1 (threshold=1.01):
- Density < 1.01 (all real-world masks) → NAX kernel → **no vjp** →
  `mx.grad` fails with `ValueError: [Primitive::vjp] Not implemented
  for CustomKernel.`

## Empirically affected tests (Prompt 4 Section A)

8 tests in `tests/test_attention.py` now xfail with this regression:
- `TestSparseBackwardTiled::test_sdpa_sparse_matches_sdpa_dense[64]`
- `TestSparseBackwardTiled::test_sdpa_sparse_gradients_finite[64]`
- `TestSparseBackwardTiled::test_sdpa_sparse_gradient_shapes[64]`
- `TestSparseBackwardSteel::test_steel_sparse_all_true_matches_sdpa[64]`
- `TestSparseBackwardSteel::test_steel_sparse_causal_block_mask[64]`
- `TestSparseBackwardSteel::test_steel_sparse_gqa_shape_and_finite`
- `TestGNABackward::test_gna_backward_no_nan`
- `TestGNABackward::test_gna_backward_fullwindow_matches_dense`

All use `mx.grad(... sparse path ...)` with symmetric block masks at
the bumped N=2048 (required by MLX mask_bytes >= 4096 constraint).
Pre-Prompt-4 they hit the original RuntimeError mask < 4096 bytes; the
new shape bump exposes the underlying Sprint 1 vjp regression.

## Production impact

Real impact for users on M5+ with `mx.grad(flash_attention_sparse(...))`:
- Workflow: any training pipeline that uses `mx.grad` over sparse
  attention (e.g., FlashVSR/STCDiT training) on M5+ hardware
- Error: `Primitive::vjp Not implemented for CustomKernel`
- Workaround (for the user): pass `backward='sdpa_sparse'` or
  `backward='steel_sparse'` explicitly, which short-circuits the
  symmetric-bt dispatch... but on M5+ this also goes through SDPA
  fallback (line 2347-2348). Need to verify.

Actually, looking again at the code:
```python
# Line 2347-2348 (M5+ branch):
if info.get("is_m5_plus"):
    return _sparse_fallback_sdpa_perhead(q, k, v, block_mask, scale, causal)
```

This is the M5+ fallback path that DOES use SDPA (has vjp). But it
only fires AFTER the symmetric-bt early return at line 2292-2306.
The symmetric-bt path goes straight to `sparse_attention_dispatch` →
NAX kernel without ever reaching the M5+ SDPA fallback.

So the workaround is: ensure the mask is NOT detected as symmetric-bt.
For an asymmetric mask (e.g., bt_q != bt_k from the natural BQ=32 BK=16
for D=64), the early return is bypassed and the M5+ SDPA fallback fires.

The audit shape uses D=64 (asymmetric BQ=32 BK=16) BUT
`make_causal_block_mask(N, D=64)` produces a symmetric (NQ, NQ) mask
where bt_q=bt_k=N/NQ=32. So `make_causal_block_mask` always trips
the symmetric-bt path.

## Proposed fixes (for a focused future session)

### Fix option 1 — add vjp to NAX sparse kernel
- Cleanest mathematically: NAX kernel becomes a full Primitive with
  forward + vjp registered
- Effort: significant — design the backward computation for sparse
  attention via NAX cooperative tensors
- Could leverage Sprint 5 (V6NAX backward sparse) which extends V6NAX
  backward kernels with block-sparse support; if Sprint 5 ships, the
  NAX forward could call into V6NAX backward

### Fix option 2 — auto-detect backward context in dispatch
- Detect that the caller is inside `mx.grad` → route to SDPA path even
  if density < threshold
- Effort: medium — MLX doesn't expose easy "is this a backward trace"
  signal; would require workarounds

### Fix option 3 — add `density_threshold` keyword to flash_attention_sparse
- Callers who need backward pass `density_threshold=0.02` to override
  Sprint 1's `1.01` default
- Effort: small — just plumb the kwarg through
- Downside: users must know about this footgun

### Fix option 4 — revert Sprint 1 threshold for symmetric path
- Keep Sprint 1's perf improvement for asymmetric/non-LCSA-bt masks
- Revert symmetric-bt routing to pre-Sprint-1 (threshold 0.02)
- Effort: small, but partially negates Sprint 1's win

### Recommended: combination of 3 + post-v2.50 4
- v2.50: option 3 (`density_threshold` kwarg) for explicit caller control
- post-v2.50: investigate option 1 (NAX vjp) for sustainable fix
- Tests in Prompt 4 Section A xfail'd with reference to this STATUS doc

## Cross-reference

- `docs/v50/sprint1-decisions.md` — original Sprint 1 design
- `docs/v50/audit-framing-inversions.md` — Sprint 1 FULL_INVERSION entry
- `mlx_mfa/lcsa_nax.py:DEFAULT_DENSITY_THRESHOLD` — the changed value
- `mlx_mfa/attention.py:2292-2306` — flash_attention_sparse symmetric-bt
  early return that bypasses M5+ SDPA fallback

## Decision for Prompt 4 Section A test cleanup

Per user's protocol "Tests cassés réels (si découverts) : STATUS doc +
escalation":
- 8 affected tests marked xfail with `reason="Sprint 1 backward
  regression — sparse via mx.grad → NAX CustomKernel no vjp on M5+;
  see docs/v50/sprint1-backward-regression-status.md"`
- This STATUS doc documents the regression for Marco's review and
  future fix sprint
- No production code change in this Prompt; production callers who hit
  this regression can work around via:
  - Pass non-symmetric masks (asymmetric bt_q != bt_k)
  - Use STEEL backward instead of mx.grad (backward='steel_sparse' arg)
  - Avoid M5+ via runtime monkeypatch (not recommended)

# Prompt 5d Section A — dispatch-chain audit refresh (Pattern #5)

**Mandate**: extend 3 V34 backward kernels (dQ + dK split + fused dKdV)
with native sparse iteration, replacing hybrid orchestrator from Prompt 5c.

## Multi-gate audit (Pattern #5 — full native extension scope)

| # | Gate | Location | Pre-Prompt 5d state | Action |
|---|---|---|---|---|
| G1 | `_v34_backward_carveout` | `dispatch_policy.py:373-380` | D ∈ {64,128} + qL≥2048 + fp16/bf16 (post-Prompt 5b Section D) | Verify; no change |
| G2 | `_v34_eligible` | `attention.py:3743` | PERMISSIVE D ∈ {64,128} | Verify; no change |
| G3 | `_v34_backward_vjp` routing | `attention.py:3827-3859` | AUTO D=64→fused, D=128→split | Verify; sparse variant orchestrator parallel |
| G4 | `MFAV6Forward::eval_gpu` causal-routing | `mfa_v6_nax_primitive.cpp:625` (post-Prompt 4) | PERMISSIVE | Verify; no change |
| G5 | `MFAV6Backward::eval_gpu` D-handling | `mfa_v6_nax_primitive.cpp` (Sprint B) | PERMISSIVE D=128 | Verify; no change |
| G6 | `compile_v34_backward_pipeline` cache keys | C++ Primitive cache | PERMISSIVE | Extend with sparse cache keys (3 new) |
| G7 | V34Bwd*Key cache structs | various | dV sparse SHIPPED (Prompt 5b) | Add dQ, dK, fused dKdV sparse keys |
| G8 | MFAV34Bwd* Primitive classes | various | dV sparse SHIPPED | Add 3 new sparse Primitives |
| G9 | Raw helpers + nanobind bindings | `bindings.cpp` | dV sparse SHIPPED | Add 3 new |
| G10 | Python `_v34_backward_vjp_sparse` | NEW | `_v34_sparse_hybrid_vjp` Prompt 5c | Replace orchestrator with full native |
| G11 | `flash_attention_sparse` dispatch | `attention.py:2495+` | Hybrid eligible → `_v34_sparse_hybrid_vjp` | Replace dispatch target |

## Sparse-LSE consistency check (Pattern #5 LSE convention)

V34 forward sparse (Prompt 5c Section A.1) writes natural-log sparse-LSE
to L output.  All 4 sparse backward kernels (dV PoC + dQ/dK/fused new)
must consume this L correctly:
- dV: works (Prompt 5c hybrid validated bit-identical for all-True mask,
  within FP16 ULP for sparse masks)
- dQ: NEW — must use sparse-L same as dense kernel uses dense-L
- dK: NEW — same
- Fused dKdV: NEW — ORDER-CRITICAL constraint preservation
  (sparse-skip applied BEFORE dV accumulation)

## Risk register

| Risk | Mitigation |
|---|---|
| dQ kernel K-loop pointer advance with `continue` | Pre-advance K and V before continue OR absolute offsets |
| Fused dKdV ORDER-CRITICAL break | Sparse-skip is `continue` — naturally preserves order |
| Register pressure regression | Sparse-skip is pure control flow, no NAXTile additions |
| Hybrid orchestrator backward compat | Preserve as fallback for shapes outside native sparse envelope |

## Conclusion

3 new C++ source generators + 3 new Primitives + 3 new dispatch funcs
+ 3 new bindings + Python full-native orchestrator + integration
replacement.  Hybrid orchestrator (Prompt 5c) deprecated to fallback
for ineligible shapes.

Test scope: 16+ tests per spec covering 3-axis validation across
density × D × causal combinations.

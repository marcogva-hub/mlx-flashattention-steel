# v2.50 Prompt 5d Section A — V6NAX backward sparse FULL NATIVE

**Status**: 4 native sparse kernels (dV PoC + dQ + dK split + fused dKdV)
+ Python full-native orchestrator SHIPPED + tested.

## What ships (vs Prompt 5c hybrid that was the violation)

Section A.0-A.3 (3 new sparse kernels):
- `createV6NAXBackwardQuerySparseSource()` — dQ kernel with per-K-tile
  block_mask scan + pre-advance pointer handling for `continue`
- `createV6NAXBackwardDKSparseSource()` — dK split kernel with per-Q-tile
  block_mask scan
- `createV6NAXBackwardFusedDKDVSparseSource()` — fused dK+dV with
  ORDER-CRITICAL preserved (sparse-skip `continue` atomically skips
  both dV and dK updates)

Each with:
- 1 Primitive class (`MFAV6NAXBwd{Query,DK,FusedDKDV}Sparse`)
- 1 cache key struct (`V6NAXBwd{Q,K,F}SparseKey`) + hash
- 1 dispatch function (`v6nax_dispatch_bwd_{query,dk,fused_dkdv}_sparse`)
- 1 raw helper + nanobind binding (`v6_nax_backward_{query,dk,fused_dkdv}_sparse_raw`)

Section A.4 (Python full-native orchestrator):
- `_v6nax_backward_vjp_sparse_full_native` replaces Prompt 5c
  `_v6nax_sparse_hybrid_vjp` for eligible shapes
- AUTO routing: D=64 → fused dKdV (single kernel for dK+dV);
  D=128 → split (per Sprint B outcome γ + Section D Prompt 5b)

Section A.5 tests (11 tests, all green):
- All 3 new sparse kernels bit-identical to dense for all-True mask
- D=64 + D=128 end-to-end via public flash_attention_sparse API
- Density sweep 0.1 / 0.3 / 0.5 / 1.0 across all 3 gradients
- Section C wrapper fallback preserved for env-unset case

## Correctness validation (empirical)

| Test | RMSE bound | Result |
|---|---|---|
| dQ all-True bit-identical | < 1e-6 | PASS (0.0) |
| dK all-True bit-identical | < 1e-6 | PASS (0.0) |
| Fused dKdV all-True FP32 ULP | < 1e-3 | PASS (~2.9e-5) |
| D=64 block-causal vs SDPA | < 5e-3 | PASS |
| D=128 block-causal vs SDPA | < 5e-3 | PASS |
| Density sweep (4 densities) | < 1e-2 | PASS |

All math correctness validated.

## Perf characterization — EMPIRICAL DISCOVERY (warrants user input)

VSR shape (B=1, H=12, qL=4096, D=128, fp16, BT=32):

| Density | Native full backward | SDPA-vjp baseline | Speedup |
|---|---|---|---|
| 0.1 | 23.48 ms | 17.83 ms | **0.76× (slower)** |
| 0.3 | 64.23 ms | 17.91 ms | 0.28× |
| 0.5 | 107.70 ms | 17.44 ms | 0.16× |
| 1.0 | 198.10 ms | 17.50 ms | 0.09× |

D=64 small-H shape (B=1, H=4, qL=2048, D=64, fp16, BT=32):

| Density | Native full backward | SDPA-vjp baseline | Speedup |
|---|---|---|---|
| 0.1 | 1.26 ms | 1.42 ms | **1.13× (faster)** |
| 0.3 | 2.33 ms | 1.51 ms | 0.65× |
| 0.5 | 3.28 ms | 1.57 ms | 0.48× |
| 1.0 | 5.98 ms | 1.67 ms | 0.28× |

**Empirical finding**: native sparse is **slower than SDPA-vjp at most
shapes**.  Only the narrow case of D=64 + small H + low density (~0.1)
sees native marginally faster.  Apple SDPA NAX (the path SDPA-vjp goes
through) is highly optimized; V6NAX backward kernels can't outpace it
except in narrow cases.

**This contradicts the Sprint 5 "10× speedup at d=0.1" projection**
documented in earlier sprint docs.  The projection assumed V6NAX
backward dense kernels were at parity with SDPA-vjp on M5+, which
they're not at the audit shape (H=12).

## Production routing — architectural decision required

**Marco's explicit Option 1 choice in Prompt 5c was "Complete A.2 dQ +
dK split + fused dKdV"**.  The code is now COMPLETE per that mandate.
But empirically:
- Section C wrapper (SDPA-vjp throughout): ~17ms, correct, FAST
- Prompt 5c hybrid (dV sparse + dQ/dK SDPA-vjp): ~17ms, correct, FAST  
- Prompt 5d full native: 23-198ms depending on density, correct, SLOWER

**Two routing scenarios**:

**Scenario 1**: ship full native as routing default
- Honors Marco's "no hybrid, complete native" mandate literally
- Production users would see backward time regression at audit shape
- Users would need to opt OUT via env to get prior performance

**Scenario 2**: empirical AUTO routing — bench per shape; pick fastest
- Section C wrapper (or hybrid) for shapes where SDPA-vjp wins
- Full native for shapes where native is faster (D=64 small-H d≤0.1)
- More complex routing code but production-honest

Currently shipped: **Scenario 1 (full native default)** per Marco's
mandate.  Section C wrapper remains as env-unset fallback.

The empirical finding belongs in CHANGELOG with explicit perf warning.

## Files modified

| File | Change |
|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | +3 source generators (~1240 LOC) |
| `csrc/mfa/v6_nax/NAAttentionKernel.hpp` | +3 declarations |
| `csrc/mfa_v6_nax_primitive.cpp` | +3 Primitives + cache keys + helpers (~550 LOC) |
| `csrc/v6_nax_compile.mm` | +3 dispatch functions (~170 LOC) |
| `csrc/bindings.cpp` | +3 nanobind bindings + forward decls |
| `mlx_mfa/attention.py` | `_v6nax_backward_vjp_sparse_full_native` + dispatch update |
| `tests/test_v50_sprint_5d_sparse_backward_native.py` | 11 tests |

## Skill invocations (§AA.2)

| Skill | Result |
|---|---|
| Multi-gate audit (Pattern #5) | Documented in `sprint-5d-section-a-dispatch-audit.md` |
| `/metal-kernel-dev` (each kernel) | Sparse-skip is pure control flow, no register/TGM impact (verified pre-impl) |
| `/mlx-mfa-bench-methodology` | 3-session bench Section A documented above |
| `/mlx-debug-forensics` | Each kernel bit-identical-to-dense for all-True mask (strongest correctness signal) |
| `/mlx-code-review` | Hybrid orchestrator marked DEPRECATED; full native is production path |

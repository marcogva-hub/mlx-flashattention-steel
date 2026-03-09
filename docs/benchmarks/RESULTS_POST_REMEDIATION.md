# mlx-mfa Benchmark Results — Post-Remediation

**Device**: Apple M1 Max (gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 1.2.1 (commit 6ca92d2)
**Date**: 2026-03-09
**Patch**: Phase 1+2 Python hot-path optimisations (tech-debt remediation)
**Warmup**: 5 iters  **Timed**: 20 iters (median)

---

## Forward Attention (STEEL vs SDPA)

| Config | MFA (ms) | SDPA (ms) | Speedup |
|--------|----------|-----------|--------|
| fwd D=64  N=4096  f16 causal | 5.84 | 5.78 | 0.99× |
| fwd D=64  N=8192  f16 causal | 16.19 | 24.76 | **1.53×** |
| fwd D=64  N=8192  f16 non-causal | 21.89 | 18.41 | 0.84× |
| fwd D=128 N=2048  f16 causal | 4.22 | 3.16 | 0.75× |
| fwd D=128 N=4096  f16 causal | 12.32 | 11.09 | 0.90× |
| fwd D=128 N=8192  f16 causal | 34.55 | 37.82 | **1.09×** |
| fwd D=128 N=8192  f16 non-causal | 52.16 | 37.23 | 0.71× |
| fwd D=128 N=4096  bf16 causal | 21.77 | 15.20 | 0.70× |
| fwd D=256 N=4096  f16 causal | 34.17 | 20.59 | 0.60× |
| fwd D=256 N=8192  f16 causal | 102.39 | 74.95 | 0.73× |
| fwd D=512 N=2048  f16 causal | 39.83 | 9.60 | 0.24× |
| fwd D=512 N=4096  f16 causal | 154.00 | 36.49 | 0.24× |
| fwd D=512 N=4096  f16 non-causal | 168.58 | 31.80 | 0.19× |

## Backward Attention (dQ + dK + dV, STEEL vs SDPA vjp)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|--------|
| bwd D=64  N=2048  f16 causal | 11.27 | 6.84 | 0.61× |
| bwd D=64  N=4096  f16 causal | 39.91 | 22.67 | 0.57× |
| bwd D=128 N=2048  f16 causal | 34.96 | 9.22 | 0.26× |
| bwd D=128 N=4096  f16 causal | 134.05 | 34.88 | 0.26× |
| bwd D=128 N=2048  bf16 causal | 53.06 | 9.19 | 0.17× |
| bwd D=256 N=2048  f16 causal | 86.01 | 13.96 | 0.16× |
| bwd D=256 N=4096  f16 causal | 340.84 | 54.61 | 0.16× |
| bwd D=512 N=1024  f16 causal | 47.25 | 6.69 | 0.14× |
| bwd D=512 N=2048  f16 causal | 202.68 | 24.67 | 0.12× |

## Sliding Window Attention (causal vs causal+window)

| Config | causal (ms) | window (ms) | Speedup | active tiles |
|--------|------------|------------|--------|-------------|
| win D=128 N=4096  w=512  f16 | 12.03 | 2.06 | **5.84×** | ~12% |
| win D=128 N=8192  w=512  f16 | 32.94 | 4.12 | **7.99×** | ~6% |
| win D=128 N=8192  w=1024 f16 | 34.23 | 7.42 | **4.61×** | ~12% |
| win D=128 N=16384 w=512  f16 | 113.79 | 8.03 | **14.18×** | ~3% |

## Paged KV Attention (gather+attend vs paged STEEL, N_q=1 decode)

B=1, H=8, D=128, block_size=64, f16

| Config | gather+attend (ms) | paged STEEL (ms) | Speedup |
|--------|-------------------|-----------------|--------|
| paged S=1024  | 0.037 | 0.025 | **1.51×** |
| paged S=4096  | 0.034 | 0.025 | **1.39×** |
| paged S=16384 | 0.036 | 0.026 | **1.38×** |

## SageAttention (int8 Q/K vs flash_attention, non-causal)

B=1, H=8, D=128, f16

| Config | FA (ms) | Sage (ms) | Speedup |
|--------|---------|-----------|---------|
| sage N=512  | 1.42 | 1.70 | 0.83× |
| sage N=1024 | 1.44 | 2.10 | 0.69× |
| sage N=2048 | 3.92 | 6.96 | 0.56× |
| sage N=4096 | 11.74 | 23.19 | 0.51× |

> Note: Python-side quantization overhead (quantize_per_block per call) continues to dominate despite A.3 float32 cast dedup. Pre-quantized KV caches remain required for positive speedup.

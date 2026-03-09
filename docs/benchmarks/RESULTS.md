# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 1.2.1
**Date**: 2026-03-09
**Warmup**: 5 iters  **Timed**: 20 iters (median)

---

## Forward Attention (STEEL vs SDPA)

| Config | MFA (ms) | SDPA (ms) | Speedup |
|--------|----------|-----------|--------|
| fwd D=64  N=4096  f16 causal | 6.1 | 5.8 | 0.94× |
| fwd D=64  N=8192  f16 causal | 15.8 | 22.1 | **1.40×** |
| fwd D=64  N=8192  f16 non-causal | 20.1 | 18.2 | 0.91× |
| fwd D=128 N=2048  f16 causal | 3.8 | 2.8 | 0.73× |
| fwd D=128 N=4096  f16 causal | 11.9 | 9.8 | 0.83× |
| fwd D=128 N=8192  f16 causal | 30.3 | 37.9 | **1.25×** |
| fwd D=128 N=8192  f16 non-causal | 44.3 | 36.4 | 0.82× |
| fwd D=128 N=4096  bf16 causal | 20.9 | 14.6 | 0.70× |
| fwd D=256 N=4096  f16 causal | 33.8 | 18.9 | 0.56× |
| fwd D=256 N=8192  f16 causal | 94.7 | 73.0 | 0.77× |
| fwd D=512 N=2048  f16 causal | 36.3 | 8.9 | 0.25× |
| fwd D=512 N=4096  f16 causal | 127.2 | 33.3 | 0.26× |
| fwd D=512 N=4096  f16 non-causal | 132.5 | 31.3 | 0.24× |

## Backward Attention (dQ + dK + dV, STEEL vs SDPA vjp)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|--------|
| bwd D=64  N=2048  f16 causal | 11.0 | 6.1 | 0.55× |
| bwd D=64  N=4096  f16 causal | 34.3 | 22.6 | 0.66× |
| bwd D=128 N=2048  f16 causal | 33.6 | 8.7 | 0.26× |
| bwd D=128 N=4096  f16 causal | 125.2 | 32.0 | 0.26× |
| bwd D=128 N=2048  bf16 causal | 50.7 | 9.3 | 0.18× |
| bwd D=256 N=2048  f16 causal | 80.9 | 13.1 | 0.16× |
| bwd D=256 N=4096  f16 causal | 312.2 | 50.2 | 0.16× |
| bwd D=512 N=1024  f16 causal | 48.5 | 6.0 | 0.12× |
| bwd D=512 N=2048  f16 causal | 182.7 | 22.5 | 0.12× |

## Sliding Window Attention (causal vs causal+window)

| Config | causal (ms) | window (ms) | Speedup | active tiles |
|--------|------------|------------|--------|-------------|
| win D=128 N=4096  w=512  f16 | 12.0 | 2.1 | **5.64×** | ~12% |
| win D=128 N=8192  w=512  f16 | 30.3 | 3.8 | **8.07×** | ~6% |
| win D=128 N=8192  w=1024 f16 | 30.2 | 6.8 | **4.46×** | ~12% |
| win D=128 N=16384 w=512  f16 | 101.8 | 7.7 | **13.24×** | ~3% |

## Paged KV Attention (gather+attend vs paged STEEL, N_q=1 decode)

B=1, H=8, D=128, block_size=64, f16

| Config | gather+attend (ms) | paged STEEL (ms) | Speedup |
|--------|-------------------|-----------------|--------|
| paged S=1024  | 0.039 | 0.025 | **1.54×** |
| paged S=4096  | 0.035 | 0.027 | **1.32×** |
| paged S=16384 | 0.037 | 0.023 | **1.63×** |

## SageAttention (int8 Q/K vs flash_attention, non-causal)

B=1, H=8, D=128, f16

| Config | FA (ms) | Sage (ms) | Speedup |
|--------|---------|-----------|---------|
| sage N=512  | 1.17 | 1.32 | 0.89× |
| sage N=1024 | 1.73 | 2.13 | 0.81× |
| sage N=2048 | 3.93 | 6.65 | 0.59× |
| sage N=4096 | 12.0  | 23.0 | 0.52× |

> Note: Current Python-side quantization overhead (quantize_per_block per call) offsets the int8 kernel speedup. Pre-quantized KV caches are needed for positive speedup.


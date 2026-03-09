# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 1.2.2
**Date**: 2026-03-09
**Warmup**: 5 iters  **Timed**: 20 iters (median)

---

## Forward Attention (STEEL vs SDPA)

| Config | MFA (ms) | SDPA (ms) | Speedup |
|--------|----------|-----------|--------|
| fwd D=64  N=4096  f16 causal | 6.0 | 5.9 | 0.98× |
| fwd D=64  N=8192  f16 causal | 15.8 | 21.7 | **1.37×** |
| fwd D=64  N=8192  f16 non-causal | 20.2 | 18.6 | 0.92× |
| fwd D=128 N=2048  f16 causal | 3.9 | 2.9 | 0.74× |
| fwd D=128 N=4096  f16 causal | 11.9 | 9.8 | 0.83× |
| fwd D=128 N=8192  f16 causal | 30.8 | 38.8 | **1.26×** |
| fwd D=128 N=8192  f16 non-causal | 44.9 | 37.2 | 0.83× |
| fwd D=128 N=4096  bf16 causal | 21.0 | 14.6 | 0.70× |
| fwd D=256 N=4096  f16 causal | 33.9 | 19.3 | 0.57× |
| fwd D=256 N=8192  f16 causal | 96.0 | 73.8 | 0.77× |
| fwd D=512 N=2048  f16 causal | 36.4 | 8.9 | 0.24× |
| fwd D=512 N=4096  f16 causal | 129.0 | 33.9 | 0.26× |
| fwd D=512 N=4096  f16 non-causal | 139.1 | 31.8 | 0.23× |

## Backward Attention (dQ + dK + dV, STEEL vs SDPA vjp)

| Config | MFA bwd (ms) | SDPA bwd (ms) | Speedup |
|--------|-------------|--------------|--------|
| bwd D=64  N=2048  f16 causal | 11.2 | 6.1 | 0.54× |
| bwd D=64  N=4096  f16 causal | 36.2 | 23.2 | 0.64× |
| bwd D=128 N=2048  f16 causal | 33.9 | 8.4 | 0.25× |
| bwd D=128 N=4096  f16 causal | 125.4 | 32.2 | 0.26× |
| bwd D=128 N=2048  bf16 causal | 50.0 | 9.5 | 0.19× |
| bwd D=256 N=2048  f16 causal | 79.2 | 13.2 | 0.17× |
| bwd D=256 N=4096  f16 causal | 315.0 | 50.2 | 0.16× |
| bwd D=512 N=1024  f16 causal | 48.4 | 5.9 | 0.12× |
| bwd D=512 N=2048  f16 causal | 183.2 | 22.5 | 0.12× |

## Sliding Window Attention (causal vs causal+window)

| Config | causal (ms) | window (ms) | Speedup | active tiles |
|--------|------------|------------|--------|-------------|
| win D=128 N=4096  w=512  f16 | 11.9 | 2.2 | **5.43×** | ~12% |
| win D=128 N=8192  w=512  f16 | 30.1 | 3.9 | **7.67×** | ~6% |
| win D=128 N=8192  w=1024 f16 | 30.1 | 6.8 | **4.46×** | ~12% |
| win D=128 N=16384 w=512  f16 | 101.8 | 7.7 | **13.17×** | ~3% |


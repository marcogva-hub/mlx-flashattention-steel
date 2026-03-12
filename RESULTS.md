# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (32 GPU cores, gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 2.7.0
**Date**: 2026-03-11
**Config**: B=2 H=8 f16, warmup=8, iters=20

---

## Forward Dense Causal — STEEL V2 vs SDPA

| Config | V2 ms | SDPA ms | V2/SDPA |
|--------|------:|--------:|--------:|
| D=64  N=2048  f16 causal | 1.9 | 2.6 | **1.36×** ★ |
| D=64  N=4096  f16 causal | 6.2 | 9.4 | **1.51×** ★ |
| D=64  N=8192  f16 causal | 19.6 | 35.8 | **1.82×** ★ |
| D=128 N=2048  f16 causal | 3.4 | 5.2 | **1.53×** ★ |
| D=128 N=4096  f16 causal | 11.5 | 18.4 | **1.60×** ★ |
| D=128 N=8192  f16 causal | 44.2 | 73.6 | **1.67×** ★ |
| D=128 N=16384 f16 causal | 167.7 | 293.6 | **1.75×** ★ |
| D=128 N=4096  f16 non-causal | 21.1 | 18.3 | 0.87× |
| D=128 N=8192  f16 non-causal | 81.4 | 73.7 | 0.90× |

★ = V2 exceeds SDPA by ≥2.5%

Notes:
- D=64/128 causal: STEEL V2 (sequential K/V phases, 2× BK vs V1).
- Non-causal: V2 slightly slower than SDPA (more K-tile work, no triangular skip).
- D=256/512: see D-split section below.

---

## D-split V2 — D=256/512

| Config | MFA ms | SDPA ms | MFA/SDPA |
|--------|-------:|--------:|---------:|
| D=256 N=1024  f16 causal (D-split) | 2.6 | 2.6 | 0.98× |
| D=256 N=4096  f16 causal (D-split) | 37.1 | 36.7 | 0.99× |
| D=256 N=8192  f16 causal (D-split) | 143.3 | 142.7 | 1.00× |
| D=256 N=4096  f16 non-causal (D-split) | 33.1 | 33.0 | 1.00× |
| D=512 N=1024  f16 causal (D-split) | 4.9 | 4.8 | 0.99× |
| D=512 N=4096  f16 causal (D-split) | 66.4 | 65.8 | 0.99× |
| D=512 N=8192  f16 causal (D-split) | 262.7 | 262.3 | 1.00× |
| D=512 N=4096  f16 non-causal (D-split) | 62.9 | 64.5 | 1.02× |

Notes:
- D=256/512 dense routes to SDPA by default (v2.6.0+): D-split V2 achieves
  ~1.00× SDPA with no speedup, so MFA adds only Python overhead for dense shapes.
- Window and sparse D=256/512 still route to MFA: tile-skip gives 5-20×
  regardless of head dimension.
- D-split prevents the 0.69× regression of the old V1 kernel at D=512.

---

## Sliding Window — MFA vs Full-SDPA

| Config | MFA ms | SDPA ms | MFA/SDPA |
|--------|-------:|--------:|---------:|
| D=64  N=4096  win=512  f16 causal | 1.7 | 10.6 | **6.27×** ★ |
| D=64  N=8192  win=512  f16 causal | 3.4 | 41.1 | **12.14×** ★ |
| D=128 N=4096  win=512  f16 causal | 3.2 | 18.7 | **5.87×** ★ |
| D=128 N=8192  win=512  f16 causal | 6.2 | 73.1 | **11.84×** ★ |
| D=128 N=4096  win=256  f16 causal | 2.0 | 18.8 | **9.53×** ★ |
| D=128 N=8192  win=256  f16 causal | 3.6 | 74.9 | **21.06×** ★ |

★ = MFA exceeds SDPA by ≥2.5%

---

## V2 Split-K — Small Grid (under-occupied)

| Config | V2 ms | SDPA ms | V2/SDPA |
|--------|------:|--------:|--------:|
| B=1 H=1 N=512  D=64  f16 causal | 0.4 | 0.4 | 0.99× |
| B=1 H=1 N=1024 D=64  f16 causal | 0.4 | 0.7 | 1.86× ★ |
| B=1 H=1 N=512  D=128 f16 causal | 0.4 | 0.4 | 1.13× |
| B=1 H=1 N=1024 D=128 f16 causal | 0.6 | 0.5 | 0.87× |
| B=1 H=2 N=512  D=128 f16 causal | 0.6 | 0.5 | 0.87× |
| B=1 H=4 N=512  D=128 f16 causal | 0.7 | 0.6 | 0.98× |

---

## Async Metallib — Hardware DMA Overlap

`async_v2.metallib` uses `simdgroup_async_copy` (private AIR intrinsic) to overlap
device→threadgroup DMA with ALU compute. **Requires Xcode ≤16 / macOS ≤15 to compile.**

**macOS 26 investigation (v2.6.0):**

The metallib loads and dispatches (valid MTLB, 30901 bytes, pipeline created).
macOS 26 runtime silently converts async_copy opcodes to synchronous loads.
Result: Async/Sync ≈ 1.00× (no DMA benefit). Correctness issue (max_abs_diff=3.86)
diagnosed and fixed: threadgroup_barrier added after simdgroup_event::wait.

| Path | D=64 N=4096 causal | vs Sync |
|------|-------------------:|--------:|
| Async metallib (macOS 26) | 5.5 ms | 1.14× |
| Sync V2 (MFA_DISABLE_ASYNC=1) | 6.2 ms | — |
| SDPA | 9.4 ms | — |

Expected throughput gain over sync V2 on macOS ≤15 (hardware DMA):
- D=64/128 causal: +20–40% (ALU fully hides DMA latency at long sequences)
- Non-causal: ~10–15%

Build on macOS 15 / Xcode 16:
```bash
bash scripts/build_async_metallib.sh
# → mlx_mfa/precompiled/async_v2.metallib
```

---

## V3 Kernel — Separate K_smem + V_smem (v2.7.0 experiment)

V3 reduces per-K-tile barriers from 4 (V2) to 2 by allocating K_smem and
V_smem as independent threadgroup buffers (vs V2's shared KV_smem).

**Benchmark** (M1 Max, B=2 H=8 f16, causal, 2026-03-12):

| Config | V2 ms | V3 ms | V3/V2 | V3/SDPA |
|--------|------:|------:|------:|--------:|
| D=64  N=1024  causal | 2.07 | 2.37 | 0.88× | 1.13× |
| D=64  N=2048  causal | 1.73 | 2.24 | 0.77× | 1.34× |
| D=64  N=4096  causal | 5.52 | 7.01 | 0.79× | 1.51× |
| D=64  N=8192  causal | 19.76 | 25.77 | 0.77× | 1.59× |
| D=128 N=1024  causal | 1.33 | 1.50 | 0.88× | 1.04× |
| D=128 N=2048  causal | 3.46 | 4.07 | 0.85× | 1.24× |
| D=128 N=4096  causal | 11.42 | 13.99 | 0.82× | 1.33× |
| D=128 N=8192  causal | 42.45 | 52.90 | 0.80× | 1.38× |

**Conclusion**: V3 regresses vs V2 (0.77–0.88×). Doubling TGP usage
(K+V separate, ~23 KB) vs V2 (shared max(K,V), ~14 KB) halves
occupancy from 2 TGs/CU to 1 TG/CU. The extra memory-stall latency
exceeds the savings from 2 fewer barriers per iteration.

**Status**: Kernel implemented and correct; disabled by default.
Enable via `MFA_ENABLE_V3=1` for research/benchmarking.

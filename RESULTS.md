# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 2.5.4
**Date**: 2026-03-11
**Config**: B=2 H=8, warmup=8, iters=20

---

## Forward Pass — V2 vs V1 vs SDPA

| Config | V2 ms | V1 ms | SDPA ms | V2/SDPA | V1/SDPA | V2/V1 |
|--------|------:|------:|--------:|--------:|--------:|------:|
| D=64 N=2048 f16 causal | 1.8 | 3.0 | 3.0 | **1.68×** ★ | 1.02× | 1.66× |
| D=64 N=4096 f16 causal | 5.7 | 9.0 | 10.7 | **1.88×** ★ | 1.18× | 1.60× |
| D=64 N=8192 f16 causal | 19.8 | 24.7 | 43.6 | **2.20×** ★ | 1.76× | 1.25× |
| D=64 N=8192 f16 non-causal | 36.8 | 38.9 | 36.2 | **0.98×** | 0.93× | 1.06× |
| D=128 N=2048 f16 causal | 3.3 | 6.0 | 5.0 | **1.50×** ★ | 0.83× | 1.81× |
| D=128 N=4096 f16 causal | 11.5 | 17.5 | 18.8 | **1.64×** ★ | 1.08× | 1.51× |
| D=128 N=8192 f16 causal | 44.1 | 51.8 | 77.9 | **1.77×** ★ | 1.51× | 1.18× |
| D=128 N=16384 f16 causal | 166.8 | 189.6 | 296.5 | **1.78×** ★ | 1.56× | 1.14× |
| D=128 N=4096 bf16 causal | 19.2 | 29.5 | 26.5 | **1.39×** | 0.90× | 1.54× |
| D=128 N=8192 f16 non-causal | 81.7 | 86.3 | 72.4 | **0.89×** | 0.84× | 1.06× |
| D=256 N=4096 f16 causal (D-split) | 36.0 | 48.4 | 36.5 | **1.01×** ★ | 0.78× | 1.34× |
| D=256 N=8192 f16 causal (D-split) | 137.9 | 158.8 | 145.5 | **1.06×** ★ | 0.91× | 1.14× |
| D=256 N=4096 f16 non-causal (D-split) | 61.8 | 68.0 | 33.1 | **0.54×** | 0.49× | 1.07× |
| D=512 N=4096 f16 causal (D-split) | 96.0 | 196.5 | 66.3 | **0.69×** | 0.34× | 2.05× |
| D=512 N=8192 f16 causal (D-split) | 374.8 | 588.0 | 258.4 | **0.69×** | 0.44× | 1.57× |

★ = V2 exceeds SDPA by ≥2.5% (at least 1.025×)

Notes:
- D=64/128: standard V2 (sequential K/V phases, 2× BK vs V1)
- D=256/512: V2 D-split (BD_HALF=128, D_SPLITS=2/4); 1.0–2.0× faster than V1 D-split

## Sliding Window — MFA vs Full-SDPA

| Config | MFA ms | SDPA ms | MFA/SDPA |
|--------|-------:|--------:|---------:|
| D=64 N=4096 win=512 f16 causal | 1.7 | 10.6 | **6.27×** ★ |
| D=64 N=8192 win=512 f16 causal | 3.4 | 41.1 | **12.14×** ★ |
| D=128 N=4096 win=512 f16 causal | 3.2 | 18.7 | **5.87×** ★ |
| D=128 N=8192 win=512 f16 causal | 6.2 | 73.1 | **11.84×** ★ |
| D=128 N=4096 win=256 f16 causal | 2.0 | 18.8 | **9.53×** ★ |
| D=128 N=8192 win=256 f16 causal | 3.6 | 74.9 | **21.06×** ★ |

## V2 Split-K — Small Grid

| Config | V2 ms | SDPA ms | V2/SDPA |
|--------|------:|--------:|--------:|
| B=1 H=1 N=512 D=64 f16 causal | 0.4 | 0.4 | 0.99× |
| B=1 H=1 N=1024 D=64 f16 causal | 0.4 | 0.7 | 1.86× ★ |
| B=1 H=1 N=512 D=128 f16 causal | 0.4 | 0.4 | 1.13× |
| B=1 H=1 N=1024 D=128 f16 causal | 0.6 | 0.5 | 0.87× |
| B=1 H=2 N=512 D=128 f16 causal | 0.6 | 0.5 | 0.87× |
| B=1 H=4 N=512 D=128 f16 causal | 0.7 | 0.6 | 0.98× |

## AOT Metallib — First-Call Latency

Metal device already initialized; measuring time to compile+dispatch a fresh kernel variant.

| Config | AOT ms | JIT ms | AOT speedup |
|--------|-------:|-------:|------------:|
| D=64 (Metal device warm) | 0.7 | 0.9 | 1.3× |
| D=128 (Metal device warm) | 0.8 | 0.8 | 1.0× |
| D=256 (Metal device warm) | 1.3 | 2.3 | **1.8×** |
| D=512 (Metal device warm) | 2.6 | 2.6 | 1.0× |

Note: macOS 26 `newLibraryWithSource:` is fast (~1-2ms for these kernels). AOT benefit is
most pronounced for D=256 where the JIT compile is longer. Both paths share the same
~25ms process startup overhead (Metal framework init).

## Async Metallib — CP4 Hardware DMA Overlap

`async_v2.metallib` uses `simdgroup_async_copy` (private AIR intrinsic) to overlap
device→threadgroup DMA with ALU compute. **Requires Xcode ≤16 / macOS ≤15 to compile.**

Expected throughput gain over sync V2 (hardware DMA vs software loads):
- D=64/128 causal: +20–40% estimated (ALU fully hides DMA latency at long sequences)
- Non-causal: smaller gain (~10–15%)

macOS 26 status: `xcrun metal` rejects `__asm("air.simdgroup_async_copy_2d...")` —
runtime fallback chain: async metallib → sync AOT metallib → JIT.

Build on macos-14 GitHub Actions runner (Xcode 16, macOS 15):
```bash
bash scripts/build_async_metallib.sh
# → mlx_mfa/precompiled/async_v2.metallib
```

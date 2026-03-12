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

---

## V4 Kernel — Direct Device K Reads (v2.8.0 experiment)

V4 eliminates K_smem: K fragments loaded directly from device memory per-simdgroup
in the GEMM loop. Reduces barriers from 4/tile (V2) to 2/tile. Gate: `MFA_ENABLE_V4=1`.
Measured on M1 Max with `MFA_FORCE_GEN=15` (simulates M3+ routing, not M3+ cache).

| Config | V2 ms | V4 ms | V4/V2 | V4/SDPA |
|--------|------:|------:|------:|--------:|
| D=64  N=4096  causal | 7.32 | 7.50 | 0.98× | 0.75× |
| D=64  N=8192  causal | 19.58 | 28.44 | 0.69× | 0.72× |
| D=128 N=4096  causal | 15.37 | 30.26 | 0.51× | 0.33× |
| D=128 N=8192  causal | 58.44 | 108.31 | 0.54× | 0.35× |
| D=64  N=4096  non-causal | 9.30 | 9.49 | 0.98× | 0.99× |
| D=128 N=4096  non-causal | 18.31 | 18.51 | 0.99× | 0.99× |

**Conclusion**: V4 regresses vs V2 on M1 (0.51–0.98×). The 4× redundant device
reads (WM=4 simdgroups each reading K independently) are not cached on M1's smaller
L2. M3+ has a larger, faster L2 cache expected to absorb the redundant reads —
validation pending real M3+ hardware. No RoPE support (K not staged in TGP).

**Status**: Kernel implemented and correct (9/9 tests pass); disabled by default.
Enable via `MFA_ENABLE_V4=1 MFA_FORCE_GEN=15` for M3+ simulation/benchmarking.

---

## SageAttention Benchmark (v2.8.0)

SageAttention uses INT8-quantized Q×K GEMMs. Current status: Python-side Q
quantization (`quantize_per_block`) adds significant per-call overhead.

**Benchmark** (M1 Max, B=2 H=8 f16, causal, 2026-03-12):

| Config | Flash ms | Sage ms | Sage/Flash |
|--------|--------:|--------:|-----------:|
| D=64  N=2048  causal | 1.76 | 3.33 | 0.53× |
| D=64  N=4096  causal | 5.48 | 10.68 | 0.51× |
| D=128 N=2048  causal | 3.42 | 7.35 | 0.46× |
| D=128 N=4096  causal | 11.57 | 24.37 | 0.47× |

**Conclusion**: Sage is ~2× slower than flash_attention due to Python-side Q
quantization per call. Speedup requires pre-quantized KV caches (KV quantized
once, Q quantized at decode time via CP2 fused path). `SageInferenceContext`
provides this: Q quantized in-kernel, KV cached as INT8.

---

## Padding Necessity Audit (v2.8.0)

`MFA_NO_PADDING=1` sets `padQ=padK=padV=0` in JIT kernels (V2/V3/V4).

**Performance with no padding** (M1 Max, B=2 H=8 N=4096 f16):

| D | causal | with_pad ms | no_pad ms | ratio |
|---|--------|----------:|----------:|------:|
| 128 | True  | 17.4 | 16.9 | 0.975× |
| 128 | False | 18.8 | 18.4 | 0.976× |
| 64  | True  | 9.4  | 8.7  | 0.929× |
| 64  | False | 9.3  | 9.3  | 0.994× |

**Correctness**: `MFA_NO_PADDING=1` causes 45/594 tests to produce NaN. Affected
features: RoPE, ALiBi, sliding window, sparse, per-batch seqlens. Root cause:
power-of-2 threadgroup strides (BK=64 for D=64, BK=32 for D=128) trigger bank
conflict write corruption on Apple Silicon — hardware produces NaN rather than
merely serializing writes.

**Conclusion**: The 2-7% padding cost is a correctness requirement.
`MFA_NO_PADDING=1` is for debugging only.

---

## STEEL V5 D-Blocked Benchmark (v2.9.0)

V5 uses BD_tile=32 D-chunks (BK=128), loading Q from device into registers — no
Q_smem — so TGP = WM×32 = 128B, enabling 3 TG/CU vs V2's 1 TG/CU.

**Benchmark** (M1 Max, B=2 H=8 f16, 2026-03-12):

| D | N | Mode | SDPA ms | V2 ms | V5 ms | V5/SDPA | V5/V2 |
|---|---|------|--------:|------:|------:|--------:|------:|
| 64 | 1024 | causal | 2.14 | 2.09 | 1.79 | 1.20× | 1.16× |
| 64 | 2048 | causal | 3.06 | 2.33 | 2.33 | 1.32× | 1.00× |
| 64 | 4096 | causal | 10.62 | 5.51 | 6.24 | 1.70× | 0.88× |
| 64 | 8192 | causal | 41.10 | 19.57 | 22.19 | 1.85× | 0.88× |
| 64 | 1024 | dense | 1.14 | 1.95 | 2.25 | 0.51× | 0.87× |
| 64 | 4096 | dense | 9.35 | 9.60 | 10.85 | 0.86× | 0.88× |
| 128 | 2048 | causal | 4.99 | 3.31 | 4.91 | 1.02× | 0.67× |
| 128 | 4096 | causal | 20.40 | 11.51 | 16.86 | 1.21× | 0.68× |
| 128 | 8192 | causal | 75.27 | 42.68 | 63.08 | 1.19× | 0.68× |
| 128 | 4096 | dense | 18.41 | 20.76 | 28.66 | 0.64× | 0.72× |

**Conclusion**: V5 regresses on M1 Max vs V2.
Root cause: 16 threadgroup barriers per K-tile (4 D-chunks × 4 barriers each)
dominate over the 3× TG/CU occupancy gain from smaller TGP.
V5 **not dispatched by default**; enabled via `MFA_ENABLE_V5=1`.
Intended as a foundation for M3+ hardware where device reads replace smem
loads entirely, reducing to 0 barriers per K-tile.

## STEEL V5 Post-Fix Benchmark (post-v2.9.0, commit c115b50)

Full grid: D=64/128, N=512–16384, causal+dense. B=2 H=8 f16, M1 Max.
V5 built with: padding removed (8,192B → 4 TG/CU) + vectorized O store + M3+
direct-reads path (TGP path tested here; MFA_FORCE_GEN not set).

| D | N | Mode | SDPA ms | V2 ms | V5 ms | V5/SDPA | V5/V2 |
|---|---|------|--------:|------:|------:|--------:|------:|
| 64 | 512 | causal | 1.02 | 0.96 | 1.11 | 0.92× | **0.86×** |
| 64 | 1024 | causal | 1.02 | 1.41 | 1.33 | 0.77× | **1.06×** |
| 64 | 2048 | causal | 3.07 | 2.25 | 2.15 | 1.43× | **1.05×** |
| 64 | 4096 | causal | 10.98 | 5.58 | 6.98 | 1.57× | **0.80×** |
| 64 | 8192 | causal | 41.73 | 19.64 | 25.10 | 1.66× | **0.78×** |
| 64 | 16384 | causal | 166.50 | 75.26 | 95.33 | 1.75× | **0.79×** |
| 64 | 512 | dense | 0.90 | 0.87 | 0.97 | 0.92× | 0.90× |
| 64 | 1024 | dense | 1.83 | 1.31 | 1.67 | 1.10× | 0.79× |
| 64 | 2048 | dense | 2.58 | 2.84 | 3.46 | 0.75× | 0.82× |
| 64 | 4096 | dense | 9.30 | 9.81 | 12.38 | 0.75× | 0.79× |
| 64 | 8192 | dense | 35.70 | 36.87 | 46.76 | 0.76× | 0.79× |
| 64 | 16384 | dense | 141.86 | 145.37 | 183.98 | 0.77× | 0.79× |
| 128 | 512 | causal | 0.80 | 1.45 | 0.92 | 0.87× | **1.58×** |
| 128 | 1024 | causal | 2.02 | 1.73 | 2.15 | 0.94× | 0.80× |
| 128 | 2048 | causal | 5.23 | 3.52 | 5.65 | 0.93× | 0.62× |
| 128 | 4096 | causal | 18.83 | 11.60 | 19.24 | 0.98× | 0.60× |
| 128 | 8192 | causal | 78.15 | 48.73 | 72.34 | 1.08× | 0.67× |
| 128 | 16384 | causal | 334.81 | 186.94 | 287.38 | 1.17× | 0.65× |
| 128 | 512 | dense | 0.69 | 1.21 | 1.19 | 0.58× | **1.02×** |
| 128 | 1024 | dense | 1.48 | 2.07 | 2.77 | 0.53× | 0.75× |
| 128 | 2048 | dense | 4.88 | 5.52 | 9.18 | 0.53× | 0.60× |
| 128 | 4096 | dense | 20.53 | 22.49 | 33.60 | 0.61× | 0.67× |
| 128 | 8192 | dense | 83.59 | 92.87 | 134.08 | 0.62× | 0.69× |
| 128 | 16384 | dense | 323.02 | 355.77 | 524.71 | 0.62× | 0.68× |

**Dispatch decision**: V5 remains opt-in (`MFA_ENABLE_V5=1` gate unchanged).
On M1 Max (TGP path, 17 barriers/K-tile), V5 is generally slower than V2:
- D=64 causal: 0.78–1.06× V2 (wins only at N=1024–2048, under-occupied grid)
- D=64 dense: 0.79–0.90× V2 (consistent regression)
- D=128 causal: 0.60–1.58× V2 (wins only at N=512 where V2 is severely under-occupied)
- D=128 dense: 0.60–1.02× V2 (regression at N≥1024)

The padding-removal (CP7 of v2.9.0) worsened D=64 causal at large N (0.88×→0.78×)
because power-of-2 LDK=128 causes bank-conflict read serialization that more than
offsets the +1 TG/CU gain.

**Expected gain on M3+**: M3+ direct reads (MFA_DIRECT_READS=1, commit c115b50)
eliminate all 17 barriers/K-tile. With 0 barriers and 3× occupancy over V2's 1 TG/CU,
V5 should significantly outperform V2 on M3+ for all N≥1024. Benchmark pending M3+
hardware.

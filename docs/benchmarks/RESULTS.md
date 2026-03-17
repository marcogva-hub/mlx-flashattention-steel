# Benchmark Results — v2.11.0

**Date**: 2026-03-17
**Hardware**: Apple M1 Max (32 GPU cores, gen 13) · Apple M4 Max (40 GPU cores, gen 16)
**Software**: MLX 0.31.1 · macOS 26.3-26.4 · Python 3.11
**Suite**: 24/24 pass on both chips

---

## 1) Dispatch Matrix (B=1 H=8, f16)

The dispatch matrix determines when `flash_attention()` auto-routes to the MFA
kernel vs falling back to MLX SDPA. Ratios >1× mean MFA is faster.

### Causal

| D | N | M1 Max | M4 Max | Notes |
|--:|----:|-------:|-------:|-------|
| 64 | 512 | 0.59× | 0.88× | SDPA default on both |
| 64 | 1024 | 0.94× | **1.11×** | M4 wins from N=1024 |
| 64 | 2048 | **1.23×** | **1.40×** | Both win |
| 64 | 4096 | **1.53×** | **1.76×** | |
| 64 | 8192 | **1.69×** | **2.07×** | M4 peak: V1 routing |
| 128 | 512 | 0.99× | 0.93× | SDPA default |
| 128 | 1024 | 0.87× | **1.23×** | M4 wins from N=1024 |
| 128 | 2048 | **1.17×** | **1.38×** | |
| 128 | 4096 | **1.46×** | **1.52×** | |
| 128 | 8192 | **1.58×** | **1.62×** | |
| 256 | 512 | 0.76× | **1.27×** | M4 wins all N for D=256 |
| 256 | 1024 | 0.60× | **1.44×** | |
| 256 | 2048 | 0.77× | **1.64×** | |
| 256 | 4096 | 0.95× | **1.78×** | |
| 256 | 8192 | 0.99× | **1.81×** | |
| 512 | 512 | 0.50× | **1.18×** | M4 wins all N for D=512 |
| 512 | 1024 | 0.37× | **1.21×** | |
| 512 | 2048 | 0.52× | **1.32×** | |
| 512 | 4096 | 0.65× | **1.32×** | |
| 512 | 8192 | 0.68× | **1.35×** | |

### Non-causal

| D | N | M1 Max | M4 Max | Notes |
|--:|----:|-------:|-------:|-------|
| 64 | 512 | 0.90× | 0.67× | SDPA default on both |
| 64 | 1024 | 0.85× | 0.71× | |
| 64 | 2048 | **1.06×** | 0.74× | M1 wins, M4 loses |
| 64 | 4096 | **1.33×** | 0.70× | |
| 64 | 8192 | **1.43×** | 0.60× | |
| 128 | 512 | 0.91× | 0.77× | |
| 128 | 1024 | 0.81× | 0.74× | |
| 128 | 2048 | **1.12×** | 0.77× | M1 wins, M4 loses |
| 128 | 4096 | **1.40×** | 0.77× | |
| 128 | 8192 | **1.51×** | 0.68× | |
| 256 | 512 | 0.72× | 0.92× | SDPA default on both |
| 256 | 1024 | 0.59× | 0.89× | |
| 256 | 2048 | 0.50× | 0.83× | |
| 256 | 4096 | 0.51× | 0.81× | |
| 256 | 8192 | 0.52× | 0.77× | |
| 512 | 512 | 0.47× | 0.74× | |
| 512 | 1024 | 0.33× | 0.76× | |
| 512 | 2048 | 0.33× | 0.71× | |
| 512 | 4096 | 0.34× | 0.69× | |
| 512 | 8192 | 0.34× | 0.66× | |

**Dispatch summary**: M1 Max 12/40 MFA wins (30%) · M4 Max 18/40 MFA wins (45%).
Non-causal D=64/128 is enabled on M1/M2 (threshold N≥2048), disabled on M3+.

---

## 2) Dense Forward — Production Shapes (B=2 H=8, f16)

These are the shapes most relevant for LLM inference (multi-batch, multi-head).

### M4 Max

| Config | SDPA ms | V1 ms | MFA ms | MFA/SDPA | Kernel |
|--------|--------:|------:|-------:|---------:|--------|
| D=64 N=2048 causal | 1.99 | 1.19 | 1.21 | **1.65×** | V1 (guard) |
| D=64 N=4096 causal | 7.28 | 3.78 | 3.78 | **1.93×** | V1 (guard) |
| D=64 N=8192 causal | 28.84 | 14.08 | 14.22 | **2.03×** | V1 (guard) |
| D=64 N=8192 non-causal | 25.25 | 26.18 | 47.57 | 0.53× | V2 (no guard) |
| D=128 N=2048 causal | 3.62 | 2.48 | 2.68 | **1.35×** | V1 (guard) |
| D=128 N=4096 causal | 13.77 | 9.08 | 9.05 | **1.52×** | V1 (guard) |
| D=128 N=8192 causal | 54.95 | 34.69 | 34.72 | **1.58×** | V1 (guard) |
| D=128 N=16384 causal | 225.04 | 137.47 | 137.19 | **1.64×** | V1 (guard) |
| D=128 N=4096 bf16 causal | 13.95 | 9.35 | 9.27 | **1.50×** | V1 (guard) |
| D=128 N=8192 non-causal | 51.58 | 67.98 | 90.35 | 0.57× | V2 |
| D=256 N=4096 causal | 29.50 | 20.12 | 17.96 | **1.64×** | V2 D-split |
| D=256 N=8192 causal | 118.32 | 77.79 | 69.03 | **1.71×** | V2 D-split |
| D=256 N=4096 non-causal | 26.10 | 38.25 | 34.87 | 0.75× | SDPA default |

Note: On M4 Max, V1≈V2 for D≤128 causal because the dispatch guard routes both
through `flash_attention()` to the same V1 kernel. The V2 column shows V2's
actual performance only for non-causal (where the guard doesn't apply).

### M1 Max

| Config | SDPA ms | V1 ms | V2 ms | Best/SDPA | Kernel |
|--------|--------:|------:|------:|----------:|--------|
| D=64 N=2048 causal | 2.88 | 2.85 | 2.14 | **1.35×** | V2 |
| D=64 N=4096 causal | 10.55 | 8.85 | 6.66 | **1.58×** | V2 |
| D=64 N=8192 causal | 40.94 | 23.84 | 24.49 | **1.72×** | V1 |
| D=64 N=8192 non-causal | 35.60 | 42.59 | 25.05 | **1.42×** | V2 |
| D=128 N=2048 causal | 5.53 | 6.28 | 3.79 | **1.46×** | V2 |
| D=128 N=4096 causal | 18.63 | 17.40 | 12.59 | **1.48×** | V2 |
| D=128 N=8192 causal | 78.19 | 50.70 | 45.95 | **1.70×** | V2 |
| D=128 N=16384 causal | 293.19 | 187.06 | 177.94 | **1.65×** | V2 |
| D=128 N=4096 bf16 causal | 26.30 | 30.18 | 19.26 | **1.37×** | V2 |
| D=128 N=8192 non-causal | 72.70 | 85.14 | 46.59 | **1.56×** | V2 |
| D=256 N=4096 causal | 36.72 | 48.19 | 37.02 | 0.99× | V2 D-split |
| D=256 N=8192 causal | 143.70 | 162.67 | 142.79 | **1.01×** | V2 D-split |
| D=256 N=4096 non-causal | 34.05 | 70.78 | 63.95 | 0.53× | SDPA default |

Note: On M1 Max, V1 and V2 compete at D=64/128 — V2 wins at medium N (BK=64
advantage), V1 catches up at large N. The dispatch routes to V2 on M1/M2.

---

## 3) Window Masking (B=2 H=8, f16 causal)

Sliding window attention with tile-skip optimization. This is MFA's strongest
feature — tiles outside the window are skipped entirely.

| Config | M1 Max | | M4 Max | |
|--------|-------:|-------:|-------:|-------:|
| | SDPA ms | MFA/SDPA | SDPA ms | MFA/SDPA |
| D=64 N=4096 win=512 | 10.96 | **6.08×** | 7.35 | **6.70×** |
| D=64 N=8192 win=512 | 42.13 | **14.16×** | 29.37 | **14.35×** |
| D=128 N=4096 win=512 | 19.34 | **6.12×** | 14.29 | **6.05×** |
| D=128 N=8192 win=512 | 75.26 | **12.37×** | 57.14 | **12.33×** |
| D=128 N=4096 win=256 | 19.00 | **9.92×** | 14.35 | **9.98×** |
| D=128 N=8192 win=256 | 76.09 | **18.44×** | 56.69 | **20.79×** |

Window masking scales with N/window_size: the larger the ratio, the more tiles
skipped, the bigger the speedup.

---

## 4) D=256 Detailed

### D=256 Decision Pass (B=2 H=8, f16)

| Config | M1 Max | | M4 Max | |
|--------|-------:|-------:|-------:|-------:|
| | V2-ds ms | V2/SDPA | V2-ds ms | V2/SDPA |
| N=4096 causal | 37.27 | 0.98× | 17.77 | **1.68×** |
| N=8192 causal | 142.29 | **1.01×** | 70.80 | **1.65×** |
| N=16384 causal | 552.40 | **1.07×** | 274.23 | **1.75×** |
| N=4096 non-causal | 63.42 | 0.52× | 34.35 | 0.76× |
| N=8192 non-causal | 249.11 | 0.52× | 146.63 | 0.73× |
| N=16384 non-causal | 996.05 | 0.52× | 586.44 | 0.71× |

### D=256 Design Matrix — f16 causal (best MFA route vs SDPA)

| Config | M1 Max | M4 Max |
|--------|-------:|-------:|
| B=2 H=8 N=2048 | 0.93× | **1.65×** |
| B=2 H=8 N=4096 | 1.00× | **1.68×** |
| B=2 H=8 N=8192 | **1.03×** | **1.69×** |
| B=2 H=8 N=16384 | **1.07×** | **1.69×** |
| B=1 H=1 N=2048 | 0.96× | **1.18×** |
| B=1 H=1 N=4096 | **1.23×** | **1.81×** |
| B=1 H=1 N=8192 | **1.27×** | **2.03×** |
| B=1 H=1 N=16384 | **1.19×** | **2.16×** |

### D=256 Design Matrix — bf16 causal

| Config | M1 Max | M4 Max | Notes |
|--------|-------:|-------:|-------|
| B=2 H=8 N=2048 | 0.68× | **1.63×** | M1 emulation cost |
| B=2 H=8 N=4096 | 0.65× | **1.66×** | |
| B=2 H=8 N=8192 | 0.76× | **1.68×** | |
| B=2 H=8 N=16384 | 0.88× | **1.58×** | |
| B=1 H=1 N=2048 | 0.67× | **1.36×** | Low-occupancy |
| B=1 H=1 N=4096 | 0.75× | **1.73×** | |
| B=1 H=1 N=8192 | 0.86× | **1.97×** | |
| B=1 H=1 N=16384 | 0.76× | **2.18×** | M4 peak bf16 |

bf16 D=256 is enabled on M3+ (N≥2048) and disabled on M1/M2.

### D=256 non-causal (f16 + bf16)

| Config | M1 Max f16 | M4 Max f16 | M1 Max bf16 | M4 Max bf16 |
|--------|--------:|--------:|--------:|--------:|
| B=2 H=8 N=2048 | 0.52× | 0.77× | 0.66× | 0.74× |
| B=2 H=8 N=4096 | 0.52× | 0.77× | 0.42× | 0.74× |
| B=2 H=8 N=8192 | 0.52× | 0.74× | 0.40× | 0.71× |
| B=2 H=8 N=16384 | 0.52× | 0.70× | 0.41× | 0.74× |

D=256 non-causal loses on both chips — SDPA default.

---

## 5) D=512 (f16 causal)

D=512 remains SDPA-default in the dispatch policy. Shown for reference.

### B=2 H=8

| Config | M1 Max | M4 Max |
|--------|-------:|-------:|
| N=1024 f16 | 0.45× | **1.30×** |
| N=2048 f16 | 0.63× | **1.35×** |
| N=4096 f16 | 0.69× | **1.22×** |
| N=8192 f16 | 0.69× | **1.30×** |
| N=1024 bf16 | 0.39× | **1.26×** |
| N=2048 bf16 | 0.50× | **1.28×** |
| N=4096 bf16 | 0.54× | **1.30×** |
| N=8192 bf16 | 0.56× | **1.28×** |

### B=1 H=1 (low-occupancy)

| Config | M1 Max f16 | M4 Max f16 |
|--------|--------:|--------:|
| N=2048 causal | **1.07×** | **3.18×** |
| N=4096 causal | 0.58× | **2.38×** |
| N=8192 causal | 0.66× | **1.45×** |

D=512 shows clear wins on M4 Max but losses on M1 Max. Not promoted because
the policy currently applies uniformly. Low-occupancy shapes (B=1 H=1) see
extreme M4 wins up to 3.18× due to better hardware utilization.

---

## 6) Backward Pass (f16/bf16 causal)

Native STEEL backward kernels. Not promoted — shown for future reference.

| Config | M1 Max native/SDPA | M4 Max native/SDPA |
|--------|-------------------:|-------------------:|
| f16 D=64 N=2048 | 0.63× | **1.29×** |
| f16 D=64 N=4096 | 0.64× | **1.37×** |
| f16 D=64 N=8192 | 0.60× | **1.40×** |
| f16 D=64 N=16384 | 0.72× | **1.45×** |
| f16 D=128 N=2048 | 0.22× | 0.62× |
| f16 D=128 N=4096 | 0.24× | 0.76× |
| f16 D=128 N=8192 | 0.24× | 0.78× |
| f16 D=128 N=16384 | 0.24× | 0.78× |
| bf16 D=64 N=2048 | 0.48× | **1.27×** |
| bf16 D=64 N=4096 | 0.73× | **1.33×** |
| bf16 D=64 N=8192 | 0.56× | **1.38×** |
| bf16 D=64 N=16384 | 0.61× | **1.44×** |
| bf16 D=128 N=2048 | 0.18× | 0.71× |
| bf16 D=128 N=4096 | 0.19× | 0.73× |
| bf16 D=128 N=8192 | 0.18× | 0.75× |
| bf16 D=128 N=16384 | 0.20× | 0.76× |

D=64 backward is promising on M4 Max (1.27-1.45×). D=128 backward loses on
both chips. Investigation deferred.

---

## 7) Softcap & ALiBi (B=2 H=8 N=4096 D=128 f16 causal)

| Variant | M1 Max ms | M1 vs SDPA | M4 Max ms | M4 vs SDPA |
|---------|----------:|-----------:|----------:|-----------:|
| sdpa_ref | 18.60 | 1.00× | 13.79 | 1.00× |
| sdpa_softcap | 30.69 | 0.61× | 28.06 | 0.49× |
| mfa_plain | 12.42 | **1.50×** | 8.90 | **1.55×** |
| mfa_softcap | 13.53 | **1.37×** | 10.26 | **1.34×** |

ALiBi:

| Variant | M1 Max ms | vs SDPA | M4 Max ms | vs SDPA |
|---------|----------:|--------:|----------:|--------:|
| sdpa_alibi_ref | 40.07 | 0.46× | 38.10 | 0.36× |
| mfa_plain | 12.29 | **1.51×** | 9.04 | **1.52×** |

---

## 8) V5 Experimental (B=2 H=8, f16)

V5 is an experimental kernel variant. Not promoted.

### M4 Max

| D | N | Mode | V5/SDPA | V5/V2 |
|--:|----:|------|--------:|------:|
| 64 | 512 | causal | 1.04× | 0.92× |
| 64 | 1024 | causal | **1.50×** | 0.80× |
| 64 | 2048 | causal | 1.06× | 0.69× |
| 64 | 4096 | causal | 1.09× | 0.56× |
| 64 | 8192 | causal | 1.07× | 0.52× |
| 128 | 512 | causal | **1.62×** | 1.41× |
| 128 | 1024 | causal | 1.02× | 0.83× |
| 128 | 2048 | causal | 1.00× | 0.68× |
| 128 | 4096 | causal | 1.01× | 0.66× |
| 128 | 8192 | causal | 0.95× | 0.59× |
| 64 | 8192 | dense | 0.49× | 0.98× |
| 128 | 8192 | dense | 0.49× | 0.80× |

### M1 Max

| D | N | Mode | V5/SDPA | V5/V2 |
|--:|----:|------|--------:|------:|
| 64 | 512 | causal | **1.87×** | 1.79× |
| 64 | 1024 | causal | 0.91× | 0.93× |
| 64 | 2048 | causal | **1.51×** | 0.94× |
| 64 | 4096 | causal | **1.50×** | 0.98× |
| 64 | 8192 | causal | **1.58×** | 0.98× |
| 128 | 512 | causal | **1.14×** | 0.81× |
| 128 | 1024 | causal | 0.73× | 0.68× |
| 128 | 2048 | causal | 0.91× | 0.68× |
| 128 | 4096 | causal | **1.22×** | 0.65× |
| 128 | 8192 | causal | 1.04× | 0.64× |
| 64 | 8192 | dense | 0.77× | 0.47× |
| 128 | 8192 | dense | 0.54× | 0.35× |

V5 shows small-N causal wins but loses to V2 at scale and loses broadly on
dense. Not a viable replacement for V2.

---

## 9) Split-K Small Grid (B=1 H=1-4, f16 causal)

Low-occupancy shapes where the GPU grid is small.

| Config | M1 Max V2/SDPA | M4 Max V2/SDPA |
|--------|----------:|----------:|
| B=1 H=1 N=512 D=64 | **1.16×** | 0.93× |
| B=1 H=1 N=1024 D=64 | **1.10×** | 0.73× |
| B=1 H=1 N=512 D=128 | **1.08×** | 0.75× |
| B=1 H=1 N=1024 D=128 | 0.85× | 0.62× |
| B=1 H=2 N=512 D=128 | **1.20×** | 0.78× |
| B=1 H=4 N=512 D=128 | **1.03×** | 0.78× |

M1 Max has marginal wins at very low occupancy. M4 Max loses — SDPA is more
efficient with small thread grids on M4.

---

## 10) Architecture Notes

### M1/M2 GPU
- GPU L1: 8KB (tiny) → threadgroup memory (TGP, 32KB) is critical for K/V reuse
- TGP bandwidth is high → V2 shared-KV (BK=64, fewer iterations) outperforms V1
  double-buffer (BK=16-32, more iterations but fewer barriers)
- V2 is the optimal kernel for D=64/128 on M1/M2
- Non-causal V2 benefits from high TGP bandwidth (1.06-1.56×)
- 128 GPR max per simdgroup (static allocation)

### M3/M4 GPU
- Dynamic register allocation (register file acts as cache)
- Unified L1 cache for registers, TGP, tile, and stack
- TGP bandwidth reduced → barrier cost increased
- V1 double-buffer (2 barriers/tile) beats V2 shared-KV (3-4 barriers/tile)
- Dispatch guard automatically routes D≤128 causal to V1 on M3+
- Non-causal loses on M3+ for both V1 and V2 (SDPA is faster)
- D=256/512 D-split unaffected (different memory access pattern)
- FP16/FP32/INT parallel ALU pipelines (up to 2× ALU perf with high occupancy)

### Dispatch Policy Summary

| Shape | M1/M2 Kernel | M1/M2 Threshold | M3+ Kernel | M3+ Threshold |
|-------|-------------|-----------------|------------|---------------|
| D=64 causal | V2 | N≥1024 | V1 (guard) | N≥512 |
| D=64 non-causal | V2 | N≥2048 | SDPA | — |
| D=128 causal | V2 | N≥2048 | V1 (guard) | N≥1024 |
| D=128 non-causal | V2 | N≥2048 | SDPA | — |
| D=256 f16 causal | V2 D-split | N≥4096 | V2 D-split | N≥2048 |
| D=256 bf16 causal | SDPA | — | V2 D-split | N≥2048 |
| D=256 non-causal | SDPA | — | SDPA | — |
| D=512 | SDPA | — | SDPA | — |
| Window masking | Always MFA | — | Always MFA | — |

Environment variable overrides:
- `MFA_FORCE_V2=1`: force V2 even on M3+ D≤128 causal (for benchmarking)
- `MFA_FORCE_SDPA=1`: force SDPA fallback everywhere
- `MLX_MFA_VERBOSE_DISPATCH=1`: log dispatch decisions

### Sources
- Chips and Cheese: Apple M1 GPU microarchitecture analysis
- Dougall Johnson: Apple GPU instruction set and register file documentation
- philipturner: Metal Performance Shaders and TGP bandwidth measurements
- Apple Tech Talk (2023): M3 GPU dynamic caching
- Apple ML Research (Nov 2025): M5 Neural Accelerators via Metal 4 TensorOps

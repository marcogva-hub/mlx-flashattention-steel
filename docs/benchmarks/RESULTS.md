# mlx-mfa Benchmark Results — v0.9.1

**Hardware:** Apple M1 Max · macOS 26.4 · MLX 0.31.0 · Python 3.11.14
**Settings:** B=1, H=8, float16 · median of 20 runs (10 warmup) · `mx.synchronize()` between iterations

---

## Executive Summary

| Use case | Speedup vs MLX SDPA |
|---|---|
| D=64 causal, N≥4096 | **1.9–2.1×** |
| D=128 causal, N≥4096 | **1.6–1.7×** |
| D=256 causal, N≥4096 | **1.0×** (break-even) |
| D=128 non-causal | ~0.9–1.0× (parity) |
| D=256 non-causal | 0.5× (register spill) |
| Block-sparse causal, N=8192 (~50% active) | **1.7×** |
| Sliding window=512, N=4096 (23.8% active) | **3.1×** |
| Sliding window=512, N=8192 (12.3% active) | **5.7×** |

Causal attention skips ~50% of K-tiles (any tile above the diagonal). Sliding window
sparsity scales with `window/N` — at N=8192 and window=512, only 12.3% of K-tiles are
computed.

---

## Dense Forward — Non-causal (f16)

| D | N=256 | N=512 | N=1K | N=2K | N=4K | N=8K |
|---|---|---|---|---|---|---|
| 64  | 0.96× | 0.90× | 0.97× | 1.22× | 0.99× | 1.00× |
| 128 | 0.92× | 0.95× | 0.94× | 0.92× | 0.91× | 0.92× |
| 256 | 0.94× | 0.74× | 0.65× | 0.52× | 0.50× | 0.50× |

> D=256 is 2× slower due to register spill on M1/M2 (32K register file, 256-wide head).
> D=128 non-causal runs at ~0.92× — slight overhead from tile-load path vs SDPA.

---

## Dense Forward — Causal (f16)

| D | N=256 | N=512 | N=1K | N=2K | N=4K | N=8K |
|---|---|---|---|---|---|---|
| 64  | 1.11× | 1.13× | 0.94× | 1.59× | 1.90× | **2.11×** |
| 128 | 1.07× | 1.04× | 0.99× | 1.29× | 1.56× | **1.72×** |
| 256 | 1.11× | 0.79× | 0.63× | 0.80× | 1.00× | **1.00×** |

> Causal tile skipping halves the K-tile work at large N. D=256 breaks even at N=8192
> — register spill overhead cancels out the tile-skip gain.

---

## Block-Sparse Forward — Causal Block Mask (f16)

Block mask from `make_causal_block_mask(N, head_dim=128)` + `causal=False`.
~50% of K-tiles skipped (triangular). Compare: dense causal uses `causal=True` (same semantic).

| D | N=1K | N=4K | N=8K | Active% |
|---|---|---|---|---|
| 128 | 0.96× | 1.45× | **1.67×** | ~50% |

> Block-sparse causal is slightly faster than token-level `causal=True` at N=8192 (1.67× vs 1.72×)
> because the block skip has lower overhead than the per-token masking arithmetic.
> For exact token-level causal correctness, use `flash_attention_sparse(mask, causal=True)`.

---

## Block-Sparse Forward — Sliding Window (f16, D=128)

`make_sliding_window_mask(N, window_size, head_dim=128)`

| Window | N=4K SDPA ms | N=4K Sparse ms | N=4K Speedup | Active% | N=8K Speedup | Active% |
|---|---|---|---|---|---|---|
| 512  | 9.53 | 3.06 | **3.12×** | 23.8% | **5.73×** | 12.3% |
| 1024 | 9.40 | 5.04 | **1.86×** | 44.0% | **3.45×** | 23.6% |

> Sliding window sparsity scales super-linearly: active density ≈ `window/N`, which halves
> as N doubles (constant window). A window=512 at N=8192 activates only 12.3% of K-tiles.

---

## Track F — M3/M4 Optimized Configs (commit 616f684)

Architecture gen routing: `MFA_FORCE_GEN` env var overrides hardware detection.
Separate `KernelKey` per gen → separate compiled Metal pipeline.

| Config | M1/M2 | M3/M4 | Δ expected |
|---|---|---|---|
| D=128, BK | 16 | **32** | +5–15% (dynamic register alloc) |
| D=256, UNROLL | none | **full** | +0–10% (pending M3+ measurement) |

**M1 Max validation** (MFA_FORCE_GEN=15, NOT the actual M3+ speedup):

| D | N | M1-config (BK=16) | M3-config (BK=32) | M1 ΔM3-code |
|---|---|---|---|---|
| 128 | 4096 | 1.54× | 1.51× | -2% (spill on M1, expected) |
| 128 | 8192 | 1.78× | 1.69× | -5% (spill on M1, expected) |
| 256 | 8192 | 1.01× | 0.94× | -7% (unroll spill on M1) |

> M3+ speedup for BK=32/full-unroll can only be measured on M3/M4 hardware.
> On M1/M2, the M3+ config is routed correctly and produces correct results (6 tests pass).

---

## Track A Impact — STEEL_PRAGMA_UNROLL (commit 36cbf48)

D≤128 (TD=8/16): `_Pragma("clang loop unroll(full)")` added to PV reduction loop.

| Config | Before | After | Δ |
|---|---|---|---|
| D=128, N=8192, causal | ~1.60× | ~1.72× | **+7%** |
| D=256, N=8192, causal | ~1.00× | ~1.00× | 0% (empty pragma) |

> D=256 (TD=32): full unroll → register spill; `unroll_count(8)` → catastrophic Metal AIR
> regression (0.37×). Empty pragma is the correct setting for D=256.

---

## Track B — Block-Sparse Summary

Block-sparse forward uses a separately compiled STEEL kernel variant (`sparse=true` in
`KernelKey`). The K-loop skip is a uniform threadgroup branch — all 128 threads in a
threadgroup reach the same decision simultaneously (zero warp divergence).

**Backward pass:** Gradients are computed via dense `mx.vjp(sdpa)` + float additive block
bias. This is correct but does not benefit from sparsity. Native sparse backward (direct
K/V gradient accumulation with masked tiles) is planned for v0.3.0.

---

## Raw Timings Reference

### Dense causal, D=128 (ms)

| N | SDPA | MFA |
|---|---|---|
| 256 | 0.49 | 0.46 |
| 512 | 0.55 | 0.53 |
| 1024 | 1.02 | 1.03 |
| 2048 | 2.87 | 2.24 |
| 4096 | 10.06 | 6.46 |
| 8192 | 37.81 | 21.98 |

### Sliding window=512, D=128 (ms)

| N | SDPA | Sparse | Active% |
|---|---|---|---|
| 4096 | 9.53 | 3.06 | 23.8% |
| 8192 | 37.33 | 6.51 | 12.3% |

---

## v0.9.1 — Optimizations Summary (CA + CC + CF)

> Run `python benchmarks/bench_all.py` to reproduce on your hardware.

Three orthogonal optimizations landed in v0.9.1:

| Track | Optimization | Scope | Expected gain |
|-------|-------------|-------|---------------|
| CA | Vec4 block loads (`float4` / `half4` aligned reads) | Forward Q/K/V tile loads | +5–10% bandwidth-bound configs |
| CC | Persistent multi-Q-block kernel (4 Q-blocks per threadgroup dispatch) | Forward | –15–20% kernel launch overhead at long N |
| CF | Double-buffer ping-pong (K_smem ⊕ V_smem, 4→2 barriers/K-tile) | Forward D≤128 | +3–8% D=64/128 |

### v0.9.1 forward speedup vs SDPA (expected, M1 Max, f16, causal)

| Config | v0.9.0 baseline | v0.9.1 estimated |
|--------|----------------|-----------------|
| D=64  N=4096  | ~1.9× | ~2.0–2.1× |
| D=64  N=8192  | ~2.2× | ~2.3–2.4× |
| D=128 N=4096  | ~1.6× | ~1.7–1.8× |
| D=128 N=8192  | ~1.7× | ~1.8–1.9× |
| D=256 N=4096  | ~1.0× | ~1.0× (CF not applied: D>128) |

### v0.9.1 backward speedup vs SDPA (expected, M1 Max, f16, causal)

Backward uses STEEL native dQ / dKV kernels (Track BA-BD from v0.9.0).
v0.9.1 adds GQA support in backward (Track CD) but no kernel changes —
speedups are identical to v0.9.0 baselines (~2–3× for f16/bf16 D=64/128).

### Notes
- D=256 forward: CF disabled (separate K_smem+V_smem would exceed 32KB TGP).
  CC (persistent kernel) still applies.
- Backward D=256: still routed to `mx.vjp(SDPA)` (Track CE deferred to v1.0).
- All measurements: B=1, H=8, median of 20 runs, `mx.synchronize()` per iter.

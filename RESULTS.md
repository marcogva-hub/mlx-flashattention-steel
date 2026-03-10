# mlx-mfa Benchmark Results

> **Current version: v2.4.0** (STEEL V2 + Gen-Aware BK + Auto-Calibration + RoPE/ALiBi in V2)
> Device: Apple M1 Max (gen=13, 32 cores)  
> `mlx_mfa.flash_attention` vs `mx.fast.scaled_dot_product_attention` with explicit causal mask  
> Warmup=8, iters=20 per measurement. Values are means; GPU variance ±5–15%.

---

## v2.2.0 — Dense Forward Pass (B=2 H=8, f16/bf16)

SDPA = `_fallback_sdpa` (explicit upper-triangular mask).  
V1 = STEEL V1 (`MFA_DISABLE_V2=1`).  V2 = STEEL V2 (default path).

| Config | SDPA ms | V1 ms | V2 ms | V1/SDPA | V2/SDPA | V2/V1 |
|--------|--------:|------:|------:|--------:|--------:|------:|
| D=64  N=2048  f16 causal | 2.93 | 2.91 | 1.70 | 1.01× | **1.73×** | 1.72× |
| D=64  N=4096  f16 causal | 10.63 | 9.46 | 6.03 | 1.12× | **1.76×** ★ | 1.57× |
| D=64  N=8192  f16 causal | 42.61 | 24.27 | 20.72 | 1.76× ★ | **2.06×** ★ | 1.17× |
| D=64  N=8192  f16 non-causal | 36.22 | 39.50 | 37.46 | 0.92× | 0.97× | 1.05× |
| D=128 N=2048  f16 causal | 5.34 | 6.81 | 3.84 | 0.78× | **1.39×** | 1.77× |
| D=128 N=4096  f16 causal | 19.79 | 17.32 | 11.70 | 1.14× | **1.69×** ★ | 1.48× |
| D=128 N=8192  f16 causal | 74.87 | 51.49 | 43.16 | 1.45× | **1.73×** ★ | 1.19× |
| D=128 N=16384 f16 causal | 297.38 | 190.84 | 171.98 | 1.56× ★ | **1.73×** ★ | 1.11× |
| D=128 N=4096  bf16 causal | 27.40 | 29.12 | 19.79 | 0.94× | **1.38×** | 1.47× |
| D=128 N=8192  f16 non-causal | 77.21 | 86.21 | 81.50 | 0.90× | 0.95× | 1.06× |
| D=256 N=4096  f16 causal | 37.63 | 49.88 | 49.48 | 0.75× | 0.76× | 1.01× |
| D=256 N=8192  f16 causal | 145.22 | 159.84 | 161.25 | 0.91× | 0.90× | 0.99× |
| D=256 N=4096  f16 non-causal | 34.45 | 70.62 | 69.82 | 0.49× | 0.49× | 1.01× |

★ = ≥1.5× speedup.

**Key wins**: D=64 N=8192 causal **2.06×**, D=128 N=4096+ causal **1.69–1.76×**.  
D=256 dense is slower than SDPA (both V1 and V2): needs 3D blocking not yet supported.

---

## v2.2.0 — Window Masking (B=2 H=8, f16 causal)

Window masking always routes to MFA (tile-skip benefit regardless of D).

| Config | SDPA ms | MFA ms | MFA/SDPA |
|--------|--------:|-------:|---------:|
| D=64  N=4096 win=512 | 10.75 | 1.70 | **6.3×** ★ |
| D=64  N=8192 win=512 | 45.97 | 3.20 | **14.4×** ★ |
| D=128 N=4096 win=512 | 20.08 | 3.30 | **6.1×** ★ |
| D=128 N=8192 win=512 | 82.54 | 7.13 | **11.6×** ★ |
| D=128 N=4096 win=256 | 19.90 | 2.25 | **8.9×** ★ |
| D=128 N=8192 win=256 | 74.63 | 3.69 | **20.2×** ★ |
| D=256 N=4096 win=512 | 37.25 | 10.02 | **3.7×** ★ |
| D=256 N=8192 win=512 | 147.21 | 20.74 | **7.1×** ★ |
| D=256 N=8192 win=256 | 147.05 | 12.42 | **11.8×** ★ |

All window configs achieve dramatic speedups: 3.7×–20.2× SDPA.  
D=256 window also routes to MFA via tile-skip (V1 sparse path).

---

## v2.2.0 — V2 Split-K (Small-Grid, f16 causal)

Split-K activates for `total_tgs < 0.8 × gpu_cores` (32 on M1 Max).

| Config | SDPA ms | V2 ms | V2/SDPA |
|--------|--------:|------:|--------:|
| B=1 H=1 N=512  D=64  | 0.50 | 0.42 | 1.20× |
| B=1 H=1 N=1024 D=64  | 0.62 | 0.38 | **1.63×** ★ |
| B=1 H=1 N=512  D=128 | 0.41 | 0.35 | 1.15× |
| B=1 H=1 N=1024 D=128 | 0.49 | 0.47 | 1.04× |
| B=1 H=2 N=512  D=128 | 0.39 | 0.39 | 1.00× |
| B=1 H=4 N=512  D=128 | 0.41 | 0.40 | 1.01× |

Split-K helps most at B=1 H=1 (single under-occupied tile).

---

## v2.2.0 — BK=64 Evaluation for D=128 (Reverted)

BK=64 for D=128 was evaluated as a barrier-reduction strategy (TK=8, 27,136B TGP).
Results (M1 Max, B=2 H=8 f16 causal, V2/SDPA):

| N | BK=32 (current) | BK=64 (reverted) |
|---|----------------:|-----------------:|
| 4096 | 1.63× | 1.15× |
| 8192 | 1.73× | 1.25× |

BK=64 regresses because TK=8 doubles K/P accumulator registers, causing spill
alongside the pinned Q accumulators (BQ×D=4096 elements per simdgroup).
BK=32 remains the default.

---

## Dispatch Thresholds (M1/M2)

Dense attention routes to MFA only when N ≥ threshold:

| D | causal | threshold N |
|---|--------|------------|
| 64  | True  | 2048 |
| 64  | False | 8192 |
| 128 | True  | 2048 |
| 128 | False | 8192 |
| 256 | True  | never (V1 is ≤SDPA for dense) |

Window/sparse attention always routes to MFA regardless of N and D.

---

## Historical Context

| Version | Key change | D=128 N=8192 causal |
|---------|-----------|---------------------|
| v1.3.0  | STEEL V1 (no V2) | 1.22× SDPA |
| v2.2.0  | STEEL V2 (sequential KV_smem, 2× BK) | **1.73×** SDPA |

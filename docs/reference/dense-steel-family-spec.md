# Dense STEEL Family — Verified Per-Variant Spec (audit B2, durable reference)

The `backend="mfa"` expert dense path (default `flash_attention` → Apple SDPA; STEEL is the
forced/expert path, Δ=1.9e-6 vs SDPA = a real STEEL kernel). RUNTIME-verified on M5/26.6. Correctness
vs an INDEPENDENT manual fp32 oracle (not SDPA, not another variant — lesson #11), locked by
`tests/test_dense_steel_family_lock.py`. Variant SELECTION by SENTINEL (env-toggle timing) — byte-
identity CANNOT distinguish (all variants are **byte-identical, Δ=0.0**). Labels: [V]erified/[D]educed.

## Variants + distinguishing feature (all faithful FlashAttention, no inspired-by deviation [V])

| Variant | Distinguishing feature | KernelType |
|---|---|---|
| **V1** (`SteelForward`) | baseline simdgroup-matrix FA-2; the fallback | SteelForward |
| **V2** (`SteelForwardV2`) | sequential K/V phases, 2× BK | SteelForwardV2 |
| **V3** (`SteelForwardV3`) | separate K_smem+V_smem, 2 barriers/tile (vs V2's 4) | SteelForwardV3 |
| **V4** | K loaded direct from device (no K_smem) | SteelForwardV4 |
| **V5** | D-blocked (BD_tile=32, BK=128), register-Q | SteelForwardV5 |
| **split-K** | K-dim reduction split for under-occupied grids | SteelV2SplitKPartial |
| **dsplit** | D-split (D=256→2, D=512→4 sub-tiles) | SteelV2DSplit256/512 |
| **flash_decode** | two-phase split-KV decode (N_q≤4) | flash_decode partial/reduce |

## Variant-level dispatch map (M5/26.6, backend="mfa", sentinel-confirmed)

| Input class | Variant that runs | Evidence |
|---|---|---|
| N≤4 & S≥256, f16/bf16, no mask/rope | **flash_decode** | gate `N<=4 && S>=256`; fp32-correct [V] |
| D≤128 **causal**, N≥v3_min_N (D64:4096/D128:2048), B·H≥4 | **V3** | env-toggle: D64 default 2.73ms vs DISABLE_V3(V2) 3.13ms = 1.14× [V]; D128 V3≈V2 parity (toggle masked, predicate holds) [D] |
| D≤128 **causal**, below V3 shape | **V1** | env-toggle: D128 N512 default 0.42ms ≠ FORCE_V2 0.34ms [V] |
| D≤128 **non-causal** | **V2** (split-K if under-occupied) | V3 needs causal; fp32-correct [V] |
| D=256 / 512 | **dsplit** | only D=256/512 path; fp32-correct [V] |
| (any) + `MFA_ENABLE_V4=1` eligible | **V4** | env-toggle changes dispatch (12.29→11.54ms) → eligible, NOT default [V] |
| (any) + `MFA_ENABLE_V5=1` | **V5** | ineligible at tested shapes (no timing change) — env-gated, rarely fires [D] |

**Constraints:** D∈{64,128} for V1-V5/split-K; D∈{256,512} for dsplit; fp16/bf16; split-K excludes
block_mask/attn_bias; V3 excludes block_mask; V4/V5 env-gated (V4 `MFA_ENABLE_V4`, V5 `MFA_ENABLE_V5`),
"disabled by default pending benchmarks". GQA supported. [V]

## Correctness (fp32 oracle, all forced cells) — LOCKED
V1, V2, V3 (D64+D128), V4, V5, split-K, dsplit, flash_decode, GQA: max_abs_err ≤ 7.7e-5, all finite
(14 cells incl. 3 threshold locks). [V]

## Selection-threshold audit (the B1 `2^31` lesson, generalized)

| Threshold | Value | Verdict |
|---|---|---|
| `v3_min_N` | (D==64)?4096:2048 | **MEASURED** — documented + benchmark-derived + M5-re-validated (Queue Closure 2026-06-17, 3-session §4-strict). NOT arbitrary. [V] |
| flash_decode gate | N≤4 & S≥256 | sensible decode bounds; not overflow [V] |
| split-K occupancy | total_tgs < 0.8·gpu_cores | occupancy tuning; `total_tgs=NQ·H·B` int, bounded (no overflow at real shapes) [V] |
| split-K `calibrated_max_n` | from dispatch_policy config | data-driven [V] |

**No arbitrary/overflow-looking threshold in the dense dispatch** (unlike the sparse path's flag).
Re-examination of the sparse `_V2_DEFAULT_WORK_THRESHOLD = 2_147_483_648` (B1): it equals `4096·4096·128`
(the calibration shape) which coincidentally = 2³¹; it is computed in **Python (unbounded int)** in
`decide_auto_version`, so it is **NOT an overflow bug** — just a calibration value sitting at 2³¹. [V]

## M5-optimal vs M1–M4-legacy (flagged for Phase E)
Default dense → SDPA on M5; STEEL is the expert/forced path. The V3 re-validation (2026-06-17) shows
V3 faster-or-parity vs V2 on M5, but **whether ANY STEEL variant beats SDPA on M5 is a Phase-E
question** — do not assume M5-optimal or M1–M4-legacy. [D — flagged, not concluded]

## Comment sweep
Dense STEEL comments are **fresh** (V3 re-validated 2026-06-17; "pending benchmarks" for V4/V5 and the
scratch-lifetime "Phase 1/2 pending under lazy eval" are accurate, not stale). No correction needed.

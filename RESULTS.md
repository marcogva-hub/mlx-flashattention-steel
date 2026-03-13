# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (32 GPU cores, gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 2.9.2
**Date**: 2026-03-12
**Config**: B=2 H=8 f16, warmup=8, iters=20

---

## Production Interpretation (v2.9.2)

- **V2 remains the production default** for dense causal small-D (`D=64/128`).
- **Native dense backward remains non-default** (`mx.vjp(SDPA)` fallback in auto mode).
- **D=256 is narrow-policy only** (causal f16 long-N on M1/M2).
- **D=512 remains SDPA-default** after dedicated decision pass (`0/32` wins).
- **Sage remains specialized decode** with narrow benchmark-backed auto routing.
- **Paged decode remains explicit-only** while shared-prefix/splitfuse stay opt-in runtime helpers.
- **Paged KV + packed varlen query** is now supported via explicit API/runtime
  bridge (`flash_attention_paged_varlen`, `DecodeRuntime(..., query_layout="packed")`).
- **Paged continuous batching remap** is now supported via explicit API/runtime
  mapping (`cache_batch_idx` in paged APIs; runtime batched paged remap helpers).
- **V3/V4/V5 remain experimental** unless a future decision pass establishes a narrow winning regime.

---

## v2.9.2 Decision Addendum — Split-K + D=256 + D=512

### Split-K composability status

| Feature family | V2 split-K status | Validation |
|---|---|---|
| RoPE | ✅ supported | split-K vs non-split parity tests |
| ALiBi | ✅ supported | split-K vs non-split parity tests |
| window `(left,right)` | ✅ supported | split-range/window intersection parity tests |
| RoPE + window | ✅ supported | split-K vs non-split parity tests |
| RoPE + ALiBi | 🚫 gated | explicit API-level guard (unchanged) |
| sparse/block-mask | 🚫 excluded from split-K | remains routed to non-split sparse path |

Split-K dispatch is now calibrated per family (`dense`, `ALiBi`, `window=256`, `window=512`)
and persisted in `dispatch_table.json` as `splitk_thresholds`. Debug override:
`MFA_FORCE_SPLITK=0|1`.

### D=256 decision pass (M1 Max, B=2 H=8)

| N | dtype | causal | SDPA ms | Auto ms | SDPA/Auto |
|---:|:---:|:---:|--------:|--------:|----------:|
| 4096  | f16 | ✅ | 38.49 | 38.24 | 1.01× |
| 8192  | f16 | ✅ | 153.08 | 144.26 | 1.06× |
| 16384 | f16 | ✅ | 653.99 | 564.79 | 1.16× |
| 4096  | bf16 | ✅ | 43.97 | 46.05 | 0.95× |
| 8192  | bf16 | ✅ | 177.33 | 176.88 | 1.00× |
| 16384 | bf16 | ✅ | 728.42 | 735.21 | 0.99× |

Dispatch decision from this pass:
- Promote only `D=256`, `causal=True`, `dtype=f16`, `N>=4096` (M1/M2) to MFA V2 D-split.
- Keep SDPA default for D=256 `bf16`, D=256 non-causal, and conservative M3+ until measured.
- D=512 now has a dedicated decision pass (below): keep dense D=512 on SDPA.
- Post-backward refresh matrix (`notes/d256_design_matrix_post_bwd_latest.json`):
  32 cases -> `maybe_win=8`, `neutral=0`, `losing=24`; wins are concentrated in
  causal `f16` only. Policy remains unchanged after refresh.

### D=512 decision pass (M1 Max, post-runtime unification)

Dedicated matrix artifact: `notes/d512_decision_matrix_latest.json`.

Scope:
- `N in {1024, 2048, 4096, 8192}`
- `causal in {False, True}`
- `dtype in {f16, bf16}`
- profiles: `prod_b2h8` and `under_b1h1`

Outcome:
- `maybe_win=0`, `no_win=0`, `losing=32`
- best MFA/SDPA in matrix: `0.81x`
- `backend="auto"` routed dense D=512 to SDPA on all rows (`0/32` MFA routes)
- Narrow candidate check (`MFA_V2_FORCE_BK_D256=64` for D-split) improved some
  rows slightly but remained below SDPA (best observed `0.77x` on tested long
  causal production shapes)

---

## Native Backward Targeted Pass (v2.9.2)

Targeted dense-backward sweep (M1 Max, `B=2 H=8`, causal, `D={64,128}`,
`N={2048,4096,8192,16384}`, `f16/bf16`) compared direct native STEEL backward
against SDPA VJP.

| Family | Native/SDPA range | Outcome |
|---|---:|---|
| f16 D=64 causal | 0.60–0.86× | losing |
| f16 D=128 causal | 0.19–0.32× | losing |
| bf16 D=64 causal | 0.41–0.71× | losing |
| bf16 D=128 causal | 0.21–0.24× | losing |

Classification: **0 promising / 0 neutral / 16 losing**.

Decision:
- Keep dense backward default as `mx.vjp(SDPA)` in auto mode.
- Keep native dense backward available only via explicit debug override
  (`MFA_FORCE_NATIVE_BWD=1`) while no benchmark-backed winning regime exists.

Raw artifact: `notes/native_backward_targeted_latest.json`

---

## Sage Decode Productionization (v2.9.2, post-backward)

Decode-only matrix (`benchmarks/bench_sage_decode_matrix.py`) swept:
`N_q ∈ {1,2,4}`, `N_cache ∈ {512,1024,2048,4096,8192}`, `D ∈ {64,128}`,
window `None` and `(256,0)`, production-like GQA and under-occupied profiles.

| total rows | sage_win | maybe | losing |
|-----------:|---------:|------:|-------:|
| 240 | 13 | 4 | 223 |

Decision:
- Keep STEEL V2 as default production decode path.
- Keep Sage as specialized decode backend, primarily useful with
  `QuantizedKVCache` reuse and windowed decode.
- Add a strict auto-route only for benchmark-backed slices:
  - `D=128`, `causal=True`, window enabled, `H_q/H_kv=2`,
    `N_cache=4096`, and `N_q=4` (f16) or `N_q=1` (bf16).
- Add debug override: `MFA_FORCE_SAGE_DECODE=0|1`.

Policy quality check on the recorded matrix:
- auto-selected rows: 2
- `sage_win`: 2, `maybe`: 0, `losing`: 0

Selective AOT decision:
- Broad Sage AOT metallib coverage is deferred in this pass.
- Current AOT focus remains STEEL V2 / V2 D-split while Sage winning
  regimes remain narrow and highly parameter-specific.

Artifacts:
- `notes/sage_decode_matrix_post_bwd_latest.json`
- `notes/sage_decode_productionization_task1.md`
- `notes/sage_decode_productionization_task2_policy.md`
- `notes/sage_decode_productionization_task4_aot.md`

---

## Runtime Unification Pass (v2.9.2)

Added lightweight runtime surface:
- `DecodeRuntime`
- `create_decode_runtime(...)`

The runtime wraps existing dense/paged/Sage contexts (no kernel changes) and
exposes shared-prefix, splitfuse, and speculative-verify helpers from the same
API surface.

### Microbenchmark — legacy context vs unified runtime (separate process)

Shape: `B=1, H_q=4, H_kv=4, N_pre=64, D=64, steps=32, f16`

| Metric | Legacy (`create_inference_context`) | Unified (`create_decode_runtime`) | Ratio |
|---|---:|---:|---:|
| Decode loop mean | 23.02 ms | 22.81 ms | 0.991× |
| Factory overhead | 306.17 µs | 311.94 µs | 1.019× |

Decision:
- No decode-loop regression from runtime unification.
- Small factory overhead increase is acceptable given reduced orchestration
  branching and unified helper access.

Artifacts:
- `benchmarks/bench_runtime_decode_overhead.py`
- `notes/runtime_unification_overhead_latest.json`
- `notes/runtime_unification_perf.md`

---

## Paged / Shared-Prefix Productionization (v2.9.2)

Focused runtime matrix:
- script: `benchmarks/bench_paged_sharedprefix_matrix.py`
- artifact: `notes/paged_sharedprefix_matrix_latest.json`
- profile: `B=1, H_q=8, H_kv=4` (GQA 2:1), `D={64,128}`, causal decode

### Matrix classification counts

| Family | clear_win | maybe_win | no_win | losing |
|---|---:|---:|---:|---:|
| paged_step (`flash_attention_paged` vs dense kvcache) | 0 | 1 | 1 | 28 |
| paged_setup (paged runtime prefill vs dense runtime prefill) | 0 | 0 | 0 | 10 |
| shared_prefix (reuse flow) | 4 | 0 | 3 | 1 |
| splitfuse (`flash_attention_splitfuse`) | 3 | 0 | 0 | 5 |

Decision from this pass:
- Keep paged decode explicit-only in runtime auto mode (no benchmark-backed
  stable winning regime in this matrix).
- Keep shared-prefix and splitfuse available through unified runtime helpers,
  with docs explicitly calling out shape/workload sensitivity.

Supporting note:
- `notes/paged_sharedprefix_productionization_task1.md`
- `notes/paged_sharedprefix_productionization_task3_policy.md`

---

## Paged KV + Packed Varlen Query Unification (v2.9.2)

Capability focus:
- Packed queries: `q=[1,H_q,total_q,D]` with `cu_seqlens_q`
- Paged KV: `k_pages/v_pages`, `block_table`, `seq_lens_kv`
- Public API: `flash_attention_paged_varlen(...)`
- Unified runtime path: `DecodeRuntime(..., query_layout="packed").paged_varlen(...)`

Current implementation strategy:
- Uniform query lengths: one batched paged dispatch
- Heterogeneous query lengths: per-sequence paged dispatch + packed concat

Benchmark matrix (M1 Max, f16):
- script: `benchmarks/bench_paged_varlen.py`
- artifact: `notes/paged_varlen_matrix_latest.json`

| Scenario | D | varlen ms | padded paged ms | seq-loop ms | varlen/padded |
|---|---:|---:|---:|---:|---:|
| GQA `B=8 H_q=8 H_kv=4` hetero | 64 | 3.536 | 3.770 | 4.263 | 1.07× |
| GQA `B=8 H_q=8 H_kv=4` hetero | 128 | 4.495 | 4.529 | 4.394 | 1.01× |
| MQA `B=8 H_q=16 H_kv=1` hetero | 64 | 4.267 | 1.561 | 4.220 | 0.37× |
| MQA `B=8 H_q=16 H_kv=1` hetero | 128 | 4.191 | 3.380 | 4.236 | 0.81× |

Decision:
- Treat this pass as a correctness/runtime-unification milestone.
- Keep selection explicit for now (no broad auto-promotion from this matrix).

---

## Paged Continuous Batching Support (v2.9.2)

Scope in this pass:
- Added explicit request-slot remapping for paged APIs:
  - `flash_attention_paged(..., cache_batch_idx=...)`
  - `flash_attention_paged_varlen(..., cache_batch_idx=...)`
- Added runtime helpers for scheduler-friendly batched paged flows:
  - `DecodeRuntime.paged_prefill_batch(...)`
  - `DecodeRuntime.paged_step_batch(...)`
  - remap-aware `DecodeRuntime.paged_varlen(...)`

Benchmark matrix (M1 Max, f16):
- script: `benchmarks/bench_paged_continuous_batching.py`
- artifact: `notes/paged_continuous_batching_latest.json`

| Scenario | D | manual ms | runtime-remap ms | manual/runtime |
|---|---:|---:|---:|---:|
| paged_step_batch reorder active sets | 64 | 28.279 | 27.648 | 1.02× |
| paged_varlen remap reorder active sets | 64 | 22.004 | 24.529 | 0.90× |
| paged_step_batch reorder active sets | 128 | 49.385 | 47.097 | 1.05× |
| paged_varlen remap reorder active sets | 128 | 34.899 | 35.533 | 0.98× |

Correctness:
- All rows reported `max_err = 0.0` against manual reference paths.

Decision:
- Treat this as an operational/runtime capability milestone for continuous
  batching semantics in paged mode.
- Keep policy explicit; do not claim broad speedup from this pass.

---

## Experimental Path Triage + Selective AOT Evaluation (v2.9.2)

Primary artifact: `notes/experimental_path_triage_latest.json`  
Status matrix + recommendations: `notes/experimental_path_status_matrix.md`

### Experimental forward path triage (M1 Max)

| Path | ineligible | clear_win | neutral | losing | Status |
|---|---:|---:|---:|---:|---|
| V3 | 0 | 3 | 0 | 13 | Experimental only (narrow wins; mostly losing) |
| V4 | 16 | 0 | 0 | 0 | Hardware-specific (M3+); parked on M1/M2 |
| V5 | 0 | 1 | 0 | 15 | Experimental opt-in only |

Simulated M3 V4 probe (`MFA_FORCE_GEN=15`), D=128 N=4096 causal:
- `v4_sim_m3 / v2 = 0.39x` (losing)

### Selective advanced-kernel AOT decision

Targeted cold-start probes (separate processes) were run in two modes:
- JIT-only (`MFA_DISABLE_PRECOMPILED=1`)
- Precompiled mode (`python -m mlx_mfa.compile_metallib --force`)

| Candidate | JIT first-call (ms) | Precompiled first-call (ms) | Decision |
|---|---:|---:|---|
| `sage_decode_d128_gqa2` | 6.33 | 152.74 | Defer AOT (regression) |
| `paged_gather_d128` | 4.96 | 83.73 | Defer AOT (regression) |
| `paged_steel_d128` | 121.72 | 198.81 | Defer AOT (regression) |

Decision from this pass:
- Keep AOT production focus on STEEL V2 / V2 D-split.
- Do not promote selective advanced-kernel AOT until cold-start behavior is favorable.
- Notes hygiene check: no stale `notes/` root artifacts older than 24h were
  found in this pass to move into `notes/archive/`.

Reference note: `notes/experimental_aot_evaluation.md`

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
| D=512 N=1024  f16 causal (D-split, prod_b2h8) | 10.6 | 5.1 | 0.48× |
| D=512 N=4096  f16 causal (D-split, prod_b2h8) | 98.8 | 72.1 | 0.73× |
| D=512 N=8192  f16 causal (D-split, prod_b2h8) | 383.2 | 281.0 | 0.73× |
| D=512 N=4096  f16 non-causal (D-split, prod_b2h8) | 186.5 | 67.4 | 0.36× |

Notes:
- D=256 dense now uses a narrow promotion (`causal=True`, `dtype=f16`,
  `N>=4096` on M1/M2) from the design-track pass; bf16, non-causal, and
  shorter causal remain SDPA-default.
- D=512 dense routes to SDPA by default: decision pass found no benchmark-backed
  dense winning regime (`0/32` matrix wins vs SDPA).
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

Notes:
- The split-K kernel now composes with ALiBi and windowed attention in the production path.
- Sparse/block-mask remains intentionally excluded from split-K in v2.9.2.

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

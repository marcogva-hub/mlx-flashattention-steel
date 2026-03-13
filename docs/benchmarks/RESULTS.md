# mlx-mfa Benchmark Results

**Device**: Apple M1 Max (32 GPU cores, gen 13, M3+: False)
**MLX version**: 0.31.0
**mlx-mfa version**: 2.9.2
**Date**: 2026-03-13
**Config**: mixed per section/script; dense decision rows use B=2 H=8 f16,
while serving refresh matrices use per-script scenarios (typically
warmup=1, iters=2-4)

---

## v2.9.2 Decision Addendum — Split-K + D=256 + D=512 + Paged Varlen

### Split-K composability status

| Feature family | V2 split-K status |
|---|---|
| RoPE | ✅ supported |
| ALiBi | ✅ supported |
| window `(left,right)` | ✅ supported |
| RoPE + window | ✅ supported |
| RoPE + ALiBi | 🚫 explicitly gated |
| sparse/block-mask | 🚫 excluded from split-K |

Split-K selection is now calibrated per shape family and persisted as
`splitk_thresholds`; `MFA_FORCE_SPLITK=0|1` overrides policy for debugging.

### D=256 decision pass

| N | causal | SDPA ms | V2 D-split ms | V2/SDPA |
|---:|:------:|--------:|--------------:|--------:|
| 4096  | ✅ | 36.55 | 37.35 | 0.98× |
| 8192  | ✅ | 143.24 | 141.78 | 1.01× |
| 16384 | ✅ | 685.77 | 578.13 | 1.19× |
| 4096  | ❌ | 36.56 | 66.66 | 0.55× |
| 8192  | ❌ | 144.52 | 267.52 | 0.54× |
| 16384 | ❌ | 611.60 | 1108.43 | 0.55× |

Dispatch outcome: promote only `D=256`, causal, `N>=8192`; keep SDPA default
for shorter causal and all non-causal D=256.

---

## Paged KV + Packed Varlen Query (vLLM-oriented)

Matrix script: `benchmarks/bench_paged_varlen.py`  
Artifact: `notes/paged_varlen_matrix_latest.json`

| Scenario | D | varlen ms | padded paged ms | seq-loop ms | varlen/padded |
|---|---:|---:|---:|---:|---:|
| GQA `B=8 H_q=8 H_kv=4` hetero | 64 | 3.548 | 6.190 | 3.610 | 1.75× |
| GQA `B=8 H_q=8 H_kv=4` hetero | 128 | 4.274 | 9.675 | 4.811 | 2.26× |
| MQA `B=8 H_q=16 H_kv=1` hetero | 64 | 2.037 | 1.318 | 1.941 | 0.65× |
| MQA `B=8 H_q=16 H_kv=1` hetero | 128 | 2.996 | 2.892 | 3.240 | 0.97× |

Interpretation:
- New capability is now available through explicit API/runtime:
  `flash_attention_paged_varlen(...)` and
  `DecodeRuntime(..., query_layout="packed").paged_varlen(...)`.
- Current heterogeneous-query path is a correctness-first runtime bridge
  (per-sequence paged dispatch + packed concat), not a fully fused single-pass
  kernel.

---

## Paged Continuous Batching Remap (Scheduler-Oriented)

Matrix script: `benchmarks/bench_paged_continuous_batching.py`  
Artifact: `notes/paged_continuous_batching_latest.json`

| Scenario | D | manual ms | runtime-remap ms | manual/runtime |
|---|---:|---:|---:|---:|
| paged_step_batch reorder active sets | 64 | 26.101 | 21.114 | 1.24× |
| paged_varlen remap reorder active sets | 64 | 16.840 | 16.022 | 1.05× |
| paged_step_batch reorder active sets | 128 | 32.533 | 31.296 | 1.04× |
| paged_varlen remap reorder active sets | 128 | 27.090 | 26.847 | 1.01× |

Interpretation:
- Primary win is capability/usability: explicit scheduler-friendly remap
  semantics for paged batched and packed-varlen runtime flows.
- Performance is mixed and near parity in this matrix; no broad auto policy
  promotion is implied.

---

## Chunked Prefill (Serving-Oriented)

Matrix script: `benchmarks/bench_chunked_prefill.py`  
Artifact: `notes/chunked_prefill_matrix_latest.json`

| Group | D | chunk_size | monolithic ms | chunked ms | chunked/mono |
|---|---:|---:|---:|---:|---:|
| dense (`B=1, N=8192`) | 64 | 256 | 12.61 | 33.30 | 2.64× |
| dense (`B=1, N=8192`) | 128 | 256 | 28.46 | 57.74 | 2.03× |
| paged batched (`B=2, N=4096`) | 64 | 256 | 84.48 | 110.76 | 1.31× |
| paged batched (`B=2, N=4096`) | 128 | 256 | 172.14 | 220.92 | 1.28× |
| paged packed (`total_q=6144`) | 64 | 256 | 58.10 | 68.33 | 1.18× |
| paged packed (`total_q=6144`) | 128 | 256 | 109.59 | 143.44 | 1.31× |

Interpretation:
- Chunked prefill is added as an explicit runtime/scheduler capability via
  `DecodeRuntime.chunked_prefill(...)`.
- In this matrix, monolithic prefill remains faster; chunked mode trades total
  throughput for interleavable per-chunk units and bounded chunk latency.
- This pass should be interpreted as a serving integration milestone, not a
  broad performance promotion.

---

## Runtime-Integrated Prefix Caching

Matrix script: `benchmarks/bench_prefix_caching_runtime.py`  
Artifact: `notes/prefix_caching_runtime_matrix_latest.json`

| Scenario | D | no-reuse ms | explicit-helper ms | runtime-managed ms | no-reuse/runtime |
|---|---:|---:|---:|---:|---:|
| dense prefix reuse chunked | 64 | 2.830 | 6.241 | 6.589 | 0.43× |
| dense prefix reuse chunked | 128 | 4.716 | 7.072 | 7.268 | 0.65× |
| paged prefix reuse chunked | 64 | 41.231 | 19.370 | 19.468 | 2.12× |
| paged prefix reuse chunked | 128 | 53.641 | 32.141 | 31.583 | 1.70× |

Interpretation:
- Runtime-managed prefix path tracks explicit helper orchestration closely
  (`max_err_runtime_vs_explicit = 0.0` in this matrix) while reducing runtime
  orchestration fragmentation.
- Paged serving-style flows show clear gains vs no-reuse baseline.
- Dense rows in this setup remain dominated by chunked suffix overhead, so the
  main win there is cleaner runtime integration rather than raw speedup.

---

## Runtime-Integrated Speculative Decode

Matrix script: `benchmarks/bench_speculative_decode_runtime.py`  
Artifact: `notes/speculative_decode_runtime_matrix_latest.json`

| Scenario | Mode | manual helper ms | runtime `speculative_step` ms | manual/runtime |
|---|---|---:|---:|---:|
| dense_short (D=64, cache=1024, N_draft=4) | full_accept | 1.436 | 1.287 | 1.12× |
| dense_short (D=64, cache=1024, N_draft=4) | partial_accept | 1.224 | 1.170 | 1.05× |
| dense_short (D=64, cache=1024, N_draft=4) | reject_all | 1.289 | 1.431 | 0.90× |
| dense_micro (D=128, cache=2048, N_draft=8) | full_accept | 1.289 | 1.211 | 1.06× |
| dense_micro (D=128, cache=2048, N_draft=8) | partial_accept | 1.177 | 1.185 | 0.99× |
| dense_micro (D=128, cache=2048, N_draft=8) | reject_all | 1.128 | 1.164 | 0.97× |
| paged_short (D=64, cache=1024, N_draft=4) | full_accept | 1.035 | 1.183 | 0.88× |
| paged_short (D=64, cache=1024, N_draft=4) | partial_accept | 1.006 | 1.056 | 0.95× |
| paged_short (D=64, cache=1024, N_draft=4) | reject_all | 1.208 | 1.027 | 1.18× |

Interpretation:
- This pass is a runtime capability milestone for draft/verify orchestration.
- Manual and runtime-integrated paths match acceptance outputs in all measured
  rows.
- Performance deltas are mixed; this matrix supports integration clarity, not
  a broad throughput-promotion claim.

---

## KV Cache Abstraction Smoke Matrix

Matrix script: `benchmarks/bench_cache_abstraction_smoke.py`  
Artifact: `notes/cache_abstraction_smoke_latest.json`

| Scenario | D | baseline ms | abstraction/runtime ms | ratio |
|---|---:|---:|---:|---:|
| dense cache append+view (direct vs adapter) | 64 | 1.186 | 0.918 | 0.77× |
| dense cache append+view (direct vs adapter) | 128 | 0.391 | 0.753 | 1.93× |
| paged cache append+tables (direct vs adapter) | 64 | 0.459 | 0.892 | 1.94× |
| paged cache append+tables (direct vs adapter) | 128 | 1.646 | 0.528 | 0.32× |
| dense prefill+step (InferenceContext vs DecodeRuntime) | 64 | 1.251 | 2.509 | 2.01× |
| dense prefill+step (InferenceContext vs DecodeRuntime) | 128 | 2.252 | 1.693 | 0.75× |
| speculative flow (helper vs runtime step) | 64 | 4.333 | 3.321 | 0.77× |
| speculative flow (helper vs runtime step) | 128 | 1.221 | 3.235 | 2.65× |

Interpretation:
- Structural/maintainability milestone: cache interactions now route through a
  capability adapter layer.
- Smoke timings are mixed; this section is a regression guard, not a speedup
  claim.

---

## Hybrid KV Cache Behavior Smoke Matrix

Matrix script: `benchmarks/bench_hybrid_kv_cache.py`  
Artifact: `notes/hybrid_kv_cache_bench_latest.json`

| Scenario | D | baseline ms | hybrid ms | hybrid/baseline |
|---|---:|---:|---:|---:|
| dense prefill + 8 decode steps | 64 | 4.062 | 3.807 | 0.94× |
| paged batch (2 seq, 16 decode steps) | 64 | 17.007 | 16.160 | 0.95× |
| dense prefix reuse prefill_with_prefix | 64 | 1.970 | 1.596 | 0.81× |
| dense prefill + 8 decode steps | 128 | 4.083 | 3.385 | 0.83× |
| paged batch (2 seq, 16 decode steps) | 128 | 14.263 | 14.537 | 1.02× |
| dense prefix reuse prefill_with_prefix | 128 | 1.261 | 1.258 | 1.00× |

Interpretation:
- Hybrid behavior is now a runtime/cache capability milestone: local hot/cold
  residency transitions, promotion/demotion, and prefetch/warmup hooks are
  operational.
- Performance impact is mixed in this matrix; the primary value is structural
  and serving-oriented cache control rather than broad speedup.

---

## Final Serving Completion (Offload + Splitfuse Deepening + Page-Native Runtime)

Supporting artifacts:
- `notes/final_serving_capabilities_summary.md`
- `notes/splitfuse_runtime_matrix_latest.json`
- `notes/paged_page_native_runtime_latest.json`
- `notes/minimal_kv_offloading_design.md`

Key rows:

| Scenario | D | baseline/bridge ms | runtime path ms | runtime/baseline |
|---|---:|---:|---:|---:|
| splitfuse decode-step (dense) | 64 | 0.945 (helper) | 1.047 (`splitfuse_step`) | 1.11× |
| splitfuse decode-step (dense) | 128 | 1.002 (helper) | 1.040 (`splitfuse_step`) | 1.04× |
| paged decode-only splitfuse | 64 | 1.305 (manual bridge) | 1.101 (page-native runtime) | 0.84× |
| paged decode-only splitfuse | 128 | 1.106 (manual bridge) | 1.131 (page-native runtime) | 1.02× |

Interpretation:
- Runtime and cache control surfaces are now substantially more serving-native.
- Offload is now a real local behavior milestone (hot/cold/offloaded with
  reload/promotion) via external-adapter extension points.
- Performance remains shape-sensitive; this section is a capability milestone,
  not a broad throughput-promotion claim.

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
- D=256 dense now uses a narrow promotion (`causal=True`, `N>=8192`) from the
  v2.9.2 decision pass; non-causal and shorter causal remain SDPA-default.
- D=512 dense remains SDPA-default (parity-only in current measurements).
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
- Split-K now composes with ALiBi and windowed attention in the production path.
- Sparse/block-mask remains intentionally excluded from split-K.

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

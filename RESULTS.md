# mlx-mfa Results Summary

Version: **2.23.0**
Benchmark hardware: **Apple M1 Max** · **Apple M4 Max**

For complete benchmark tables and architectural notes, see
`docs/benchmarks/RESULTS.md`.

## 1) Production Interpretation

- **V2 dense** remains the production default for causal D=64/128 on M1/M2.
- **V1 double-buffer** is the production default for causal D≤128 on M3+.
- **D=256** promoted for f16 causal (both chips) and bf16 causal (M3+ only).
- **D=512** remains SDPA-default.
- **Non-causal D=64/128** enabled on M1/M2 only (1.06-1.56×); M3+ stays SDPA.
- **Native dense backward** was benchmarked and not promoted.
- **Sage** remains a specialized decode backend (narrow policy).
- **V3/V4/V5** remain experimental/hardware-dependent.
- **TurboQuant** KV cache compression (Phase 1–3) production-ready.

## 2) Representative Results

| Scenario | M1 Max | M4 Max |
|---|---:|---:|
| D=64 N=8192 causal | 1.69× | **2.07×** |
| D=128 N=8192 causal | 1.58× | **1.62×** |
| D=256 N=8192 causal f16 | 1.01× | **1.81×** |
| D=256 N=8192 causal bf16 | SDPA default | **1.68×** |
| D=64/128 non-causal | up to 1.51× | SDPA default |
| Sliding-window D=128 N=8192 win=256 | **18.4×** | **20.8×** |
| D=64 backward | 0.60-0.72× | **1.29-1.45×** |
| Softcap D=128 | **1.37×** | **1.34×** |

## 3) Serving/Runtime Capability Outcomes

### Paged + packed varlen queries

**PagedVarlenForward fused kernel** (v2.14.1): single dispatch for heterogeneous
query lengths + paged KV. Benchmark results (M1 Max, H_q=32 H_kv=8 D=128 f16):

| Config | Fused | Bridge | Speedup |
|--------|------:|-------:|--------:|
| B=4 decode kv hetero | 0.09ms | 0.43ms | 4.7× |
| B=8 decode kv hetero | 0.16ms | 0.77ms | 4.8× |
| B=16 decode kv hetero | 0.27ms | 1.48ms | 5.6× |
| B=4 prefill q+kv hetero | 0.03ms | 0.37ms | 10.9× |
| B=8 mixed q hetero | 0.04ms | 0.90ms | 25.6× |

Full data: `devnotes/paged_varlen_fused_bench.json`

Previous artifact: `devnotes/paged-packed-varlen-unification/paged_varlen_matrix_latest.json`

- GQA hetero rows show strong wins (up to ~2.26x vs padded paged baseline).
- MQA hetero rows are mixed/near parity.
- Treated as capability closure with workload-sensitive performance.

### Paged continuous batching remap

Artifact: `devnotes/paged-continuous-batching/paged_continuous_batching_latest.json`

- Runtime remap path is operational and parity-to-win in sampled rows
  (`~1.01x` to `~1.24x` vs manual baseline).

### Chunked prefill

Artifact: `devnotes/chunked-prefill/chunked_prefill_matrix_latest.json`

- Operational scheduling capability.
- Monolithic prefill remains faster in current M1 Max rows.

### Prefix caching runtime integration

Artifact: `devnotes/prefix-caching-automation/prefix_caching_runtime_matrix_latest.json`

- Runtime-managed path tracks explicit helper orchestration closely.
- Strongest practical benefit appears in paged shared-prefix scenarios.

### Speculative runtime integration

Artifact: `devnotes/speculative-decode-runtime/speculative_decode_runtime_matrix_latest.json`

- Runtime `speculative_step` path is parity-level with manual helper wiring in
  aggregate.
- Interpreted as capability/integration milestone.

### Hybrid/offload + splitfuse/page-native final pass

Artifacts:
- `devnotes/hybrid-kv-cache-behavior/hybrid_kv_cache_bench_latest.json`
- `devnotes/final-serving-completion/splitfuse_runtime_matrix_latest.json`
- `devnotes/final-serving-completion/paged_page_native_runtime_latest.json`
- `devnotes/final-serving-completion/final_serving_capabilities_summary.md`

Interpretation:
- `HybridKVCache` now has real local offload-capable behavior
  (hot/cold/offloaded residency + reload/promotion).
- External adapter extension point exists with concrete local backend
  (`LocalHostKVStoreAdapter`).
- Splitfuse/runtime deepening and page-native reductions are real but
  shape-sensitive.

## 4) TurboQuant KV Cache Compression (v2.21.0–v2.23.0)

Training-free, data-oblivious KV compression based on Google's TurboQuant (ICLR 2026).
Three phases: Phase 1 (non-fused decompress), Phase 2 (K fused in kernel),
Phase 3 (K+V fused in kernel).

### Benchmark matrix (M1 Max, f16)

| Config | fp16 (ms) | P2/K-only (ms) | P3/K+V (ms) | P3/fp16 | cos(P2) | cos(P3) | fp16 MB | KV-TQ MB |
|--------|----------:|----------------:|-------------:|--------:|--------:|--------:|--------:|---------:|
| Llama-8B 1seq 2K | 0.59 | 4.04 | 4.36 | 7.4× | 0.9825 | 0.9654 | 8.0 | 2.1 |
| Llama-8B 1seq 8K | 1.40 | 11.29 | 12.23 | 8.7× | 0.9811 | 0.9647 | 32.0 | 8.5 |
| Llama-8B 4seq 2K | 1.24 | 3.33 | 3.90 | 3.1× | 0.9811 | 0.9652 | 32.0 | 8.5 |
| Llama-8B 8seq 4K | 3.86 | 13.53 | 14.94 | 3.9× | 0.9824 | 0.9656 | 128.0 | 34.0 |
| Llama-8B prefill 512 | 3.68 | 3.06 | 5.00 | 1.4× | 0.9875 | 0.9707 | 2.0 | 0.5 |
| Llama-8B prefill 2K | 40.33 | 25.40 | 37.06 | 0.9× | 0.9863 | 0.9694 | 8.0 | 2.1 |
| Qwen-7B 1seq 8K | 0.90 | 11.68 | 12.92 | 14.4× | 0.9826 | 0.9647 | 16.0 | 4.3 |
| Qwen-7B 4seq 4K | 1.51 | 6.30 | 7.55 | 5.0× | 0.9822 | 0.9650 | 32.0 | 8.5 |
| Mixed 8seq hetero | 4.05 | 12.49 | 14.52 | 3.6× | 0.9818 | 0.9649 | 60.0 | 15.9 |

**Interpretation:**
- Memory savings: ~3.8× with both K+V compressed (Phase 3).
- Quality: cosine similarity 0.9647–0.9707 vs fp16 (Phase 3), 0.9811–0.9875 (Phase 2 K-only).
- Latency: Phase 2/3 are currently slower than fp16 due to Python-side pack/unpack overhead.
  The fused Metal kernel avoids decompression but packing still happens in Python.
- Primary value is memory savings for long-context serving, not latency reduction.

Full data: `devnotes/turboquant_full_bench.json`

## 5) Practical Ceiling Statement (Current Hardware)

For this architecture on M1 Max and M4 Max:
- dense kernel family appears close to practical ceiling in the main production
  design space;
- M3/M4 optimizations (V1 routing, direct reads) recover hardware-specific wins;
- remaining improvements are mostly serving/runtime integration and future
  hardware-dependent opportunities (M5+ tensor API).

## 6) Artifact Map

- Dense/backward/large-D decisions: `devnotes/native-backward-pass/`,
  `devnotes/d256-design-track/`, `devnotes/d512-decision-pass/`
- Sage/runtime/paged passes: `devnotes/sage-decode-productionization/`,
  `devnotes/runtime-unification/`,
  `devnotes/paged-shared-prefix-productionization/`
- Serving completion tracks:
  - `devnotes/paged-packed-varlen-unification/`
  - `devnotes/paged-continuous-batching/`
  - `devnotes/chunked-prefill/`
  - `devnotes/prefix-caching-automation/`
  - `devnotes/speculative-decode-runtime/`
  - `devnotes/hybrid-kv-cache-abstraction/`
  - `devnotes/hybrid-kv-cache-behavior/`
  - `devnotes/final-serving-completion/`
- Experimental triage: `devnotes/experimental-triage/`
- TurboQuant: `devnotes/turboquant_full_bench.json`

## 7) Related Documentation

- Benchmark details: `docs/benchmarks/RESULTS.md`
- API reference: `docs/API_MANUAL.md`
- Architecture: `docs/ARCHITECTURE.md`
- Historical archive index: `devnotes/README.md`

# mlx-mfa Results Summary

Version: **2.20.0**
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

## 4) Practical Ceiling Statement (Current Hardware)

For this architecture on M1 Max and M4 Max:
- dense kernel family appears close to practical ceiling in the main production
  design space;
- M3/M4 optimizations (V1 routing, direct reads) recover hardware-specific wins;
- remaining improvements are mostly serving/runtime integration and future
  hardware-dependent opportunities (M5+ tensor API).

## 5) Artifact Map

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

## 6) Related Documentation

- Benchmark details: `docs/benchmarks/RESULTS.md`
- API reference: `docs/API_MANUAL.md`
- Architecture: `docs/ARCHITECTURE.md`
- Historical archive index: `devnotes/README.md`

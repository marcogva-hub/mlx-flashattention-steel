# mlx-mfa Results Summary (Freeze Prep)

Version target: **2.10.0**  
Primary benchmark hardware: **Apple M1 Max**

This file is the concise top-level benchmark interpretation. Detailed matrices
and historical artifacts are linked under `devnotes/` and
`docs/benchmarks/RESULTS.md`.

## 1) Production Interpretation

- **V2 dense** remains the production default for causal D=64/128 regimes.
- **D=256** remains narrow benchmark-backed only.
- **D=512** remains SDPA-default.
- **Native dense backward** was benchmarked and not promoted.
- **Sage** remains a specialized decode backend (narrow policy).
- **V3/V4/V5** remain experimental/hardware-dependent.

## 2) Representative Dense Outcomes (M1 Max)

| Scenario | Outcome |
|---|---:|
| V2 causal D=64 N=8192 vs SDPA | ~1.82x |
| V2 causal D=128 N=16384 vs SDPA | ~1.75x |
| Sliding-window tile-skip regimes | up to ~21x vs full SDPA |
| D=256 causal long-N (narrow rows) | up to ~1.16x |
| D=512 decision pass | no broad wins (SDPA-default) |

## 3) Serving/Runtime Capability Outcomes

### Paged + packed varlen queries

Artifact: `devnotes/paged-packed-varlen-unification/paged_varlen_matrix_latest.json`

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

For this architecture on M1 Max:
- dense kernel family appears close to practical ceiling in the main production
  design space;
- remaining improvements are mostly serving/runtime integration and future
  hardware-dependent opportunities.

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

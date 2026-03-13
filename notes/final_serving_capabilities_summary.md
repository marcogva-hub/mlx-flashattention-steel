# Final Serving Capabilities Summary (Final Serving Completion Branch)

Date: 2026-03-13  
Version: 2.9.2  
Device: Apple M1 Max

## Scope of this milestone

This branch closes the remaining serving-oriented gaps with a minimal-real approach:
- real local offload behavior in `HybridKVCache` (hot/offloaded residency + reload)
- external-cache adapter extension point with a concrete local host backend
- deeper runtime splitfuse integration (`splitfuse_step`, metadata)
- one page-native runtime improvement for paged decode-only splitfuse
- refreshed serving benchmark artifacts for all major serving features

## Support level by capability

| Capability | Status in this branch | Notes |
|---|---|---|
| Hybrid cache tiering (hot/cold/offloaded) | Implemented | Real residency transitions with promotion/demotion and reload path |
| Minimal real offload behavior | Implemented (local-only) | Offloaded tier is local host memory via adapter; no remote/distributed backend |
| External/LMCache-like adapter contract | Implemented | `ExternalKVCacheAdapter` + `LocalHostKVStoreAdapter` provide real put/fetch/prefetch/evict path |
| Splitfuse runtime integration | Implemented | Reachable via runtime (`splitfuse`, `splitfuse_step`) with metadata |
| Paged runtime page-native improvement | Implemented (narrow) | Paged decode-only splitfuse path avoids dense bridge materialization |
| Paged packed-varlen query | Implemented (bridge for hetero q_len) | Equal q_len uses batched fast path; hetero path remains per-seq bridge |
| Paged continuous batching remap | Implemented | Explicit request-slot remap path for batched and paged_varlen runtime flows |
| Chunked prefill | Implemented | Runtime capability; currently a serving-scheduling feature, not a throughput optimization |
| Runtime prefix caching | Implemented | Runtime-managed registration/seed/reuse with chunked/paged integration |
| Runtime speculative decode | Implemented | Runtime verify/step integration with explicit accept/reject outputs |

## Benchmark highlights (refreshed artifacts)

Sources:
- `notes/hybrid_kv_cache_bench_latest.json`
- `notes/splitfuse_runtime_matrix_latest.json`
- `notes/paged_page_native_runtime_latest.json`
- `notes/speculative_decode_runtime_matrix_latest.json`
- `notes/prefix_caching_runtime_matrix_latest.json`
- `notes/chunked_prefill_matrix_latest.json`
- `notes/paged_varlen_matrix_latest.json`
- `notes/paged_continuous_batching_latest.json`

Observed summary (ratio > 1.0 means the new/runtime path is faster than baseline where noted in each artifact):
- Hybrid cache vs baseline: mixed; several rows show parity or better, one paged D=128 row is near parity/slightly slower.
- Splitfuse runtime path: integration overhead is modest; runtime splitfuse-step is close to helper path (~1.04–1.11x helper time in current matrix).
- Page-native paged improvement: decode-only splitfuse D=64 improved (`runtime_vs_manual` ~0.84); other touched rows are mixed.
- Speculative runtime: near parity with manual orchestration overall (dense mean ~1.015x, paged mean ~1.002x).
- Prefix runtime-managed vs explicit helper: near parity (`~0.95–1.02x`); paged reuse remains materially better than no-reuse baseline.
- Chunked prefill: slower than monolithic on this hardware matrix; main value remains scheduler/interleaving control.
- Paged varlen packed: clearly workload-dependent; strong wins in GQA rows, mixed/near-parity in MQA rows.
- Continuous batching remap runtime path: parity-to-win across current rows (`~1.01–1.24x` vs manual baseline).

## Honest limits

- Offload is now real but local-only; this is not yet a remote/distributed KV cache system.
- Some serving paths still use correctness-first bridge logic in heterogeneous layouts.
- This branch is a serving-runtime capability milestone, not a claim of universal throughput wins.

## Recommended next continuation point (future hardware / future pass)

1. Keep this branch as a stable serving-capability baseline.  
2. If resumed later, prioritize remote/off-process adapter backends under `ExternalKVCacheAdapter` without changing runtime API semantics.  
3. Revisit deeper page-native fusion only after profiling on newer hardware.

# Hybrid KV Cache Abstraction Branch Summary

Date: 2026-03-13
Branch: `codex/hybrid-kv-cache-abstraction`

## What abstraction was introduced

- Added a new cache abstraction module: `mlx_mfa/kv_cache.py`.
- Added a capability model and adapters:
  - `KVCacheCapabilities`
  - `KVCacheAdapter`
  - `DenseKVCacheAdapter`
  - `PagedKVCacheAdapter`
  - `QuantizedKVCacheAdapter`
- Added explicit unsupported-operation signaling:
  - `KVCacheOperationUnsupported`
- Added context helpers:
  - `resolve_context_cache(...)`
  - `resolve_context_cache_adapter(...)`
  - `adapt_kv_cache(...)`

## Runtime flows now using the abstraction

Serving-oriented `DecodeRuntime` paths now use adapter capability calls in key
cache interaction points:

- prefix seeding (`_seed_dense_prefix`, `_seed_paged_prefixes`)
- packed chunked prefill cache append/reset path
- paged varlen path (`paged_varlen`) for block-table/seq-lens/pool access
- paged batched prefill/decode helpers
  (`paged_prefill_batch`, `paged_step_batch`)
- speculative verify fallback cache lookup (dense + paged)
- metadata now includes:
  - `cache_kind`
  - `cache_capabilities`

`InferenceContext`, `PagedInferenceContext`, and `SageInferenceContext` now
also expose `cache_adapter` for explicit adapter access.

## Future-facing hybrid/offload scaffold

- Added `HybridKVCache` and `HybridKVCacheAdapter` as structural groundwork.
- Current behavior delegates to a primary cache adapter.
- Future hooks are intentionally explicit and unimplemented:
  - `offload_seq(...)`
  - `prefetch_seq(...)`
  - `promote_seq(...)`
- `ready_for_production` is currently `False`.

## Benchmark / smoke outcome

Smoke matrix added:
- script: `benchmarks/bench_cache_abstraction_smoke.py`
- artifact: `devnotes/cache_abstraction_smoke_latest.json`

Observed outcome:
- mixed micro-level timing deltas across scenarios
- no broad regression signal from this smoke matrix
- no speedup claim; primary value is architectural clarity and extension points

## Remaining work for true hybrid/offload cache support

- define concrete tiering/offload policy semantics and lifecycle
- add explicit eviction/promotion behavior
- add asynchronous prefetch/offload scheduling hooks
- add end-to-end serving traces for hybrid policy validation
- keep production routing conservative until hybrid implementation is complete

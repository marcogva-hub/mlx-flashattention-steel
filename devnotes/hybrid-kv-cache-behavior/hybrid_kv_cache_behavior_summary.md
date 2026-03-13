# Hybrid KV Cache Behavior Summary (Branch: codex/hybrid-kv-cache-behavior)

## What Was Implemented

- `HybridKVCache` is now a real local tiered cache behavior layer (not scaffold-only).
- Implemented deterministic tier policy with:
  - hot/cold residency tracking per `seq_id`
  - promotion on access
  - demotion/eviction under hot-tier capacity pressure
  - pinned-sequence protection in victim selection
- Added explicit prefetch/warmup control surfaces:
  - `mark_for_prefetch(...)`
  - `prefetch_seq(...)`
  - `prefetch(...)`
  - `prepare_hot_window(...)`
  - `clear_prefetch_intent(...)`
- Added compatibility/runtime views through hybrid methods so existing flows can consume:
  - attention views
  - paged pool/tables
  - quantized view (when primary tier supports it)

## Runtime Integration

- Added runtime-level hybrid integration in `create_decode_runtime(...)`:
  - `hybrid_cache=True`
  - `hybrid_policy=...`
  - `hybrid_hot_seq_capacity=...`
  - `hybrid_with_secondary=...`
- Added runtime helpers:
  - `DecodeRuntime.hybrid_prefetch(...)`
  - `DecodeRuntime.hybrid_mark_for_prefetch(...)`
  - `DecodeRuntime.hybrid_state`
  - `DecodeRuntime.hybrid_cache_enabled`
- Integrated hybrid behavior hooks into supported serving flows where low-risk:
  - prefix seeding paths
  - paged prefill/decode batch paths
  - chunked prefill paths (dense/paged-supported variants)
- Runtime metadata now includes:
  - `hybrid_cache_active`
  - `hybrid_state`

## Correctness Coverage

Added/extended tests validate:
- append/update transitions
- cold->hot promotion behavior
- demotion/eviction under pressure
- pinned capacity guard behavior
- attention-view correctness after promotion
- runtime dense/paged integration
- speculative runtime compatibility
- unsupported combinations fail clearly (e.g. Sage backend hybrid wrapping)

## Benchmark / Smoke Outcome

- Script: `benchmarks/bench_hybrid_kv_cache.py`
- Artifact: `devnotes/hybrid_kv_cache_bench_latest.json`
- Summary (M1 Max): mixed overhead/benefit across scenarios.
  - Dense rows show near-parity to modest win/loss depending on shape.
  - Paged row shows modest overhead in this smoke matrix.

Interpretation:
- This branch is a **real cache behavior milestone** for serving/runtime control.
- It is **not** a broad throughput-promotion claim.

## Remaining Future Work

- No remote/offloaded tier implementation yet.
- No LMCache/distributed adapter yet.
- Sage backend hybrid wrapping remains intentionally unsupported in runtime factory.
- Further policy tuning (e.g., workload-aware pinning/prefetch heuristics) can be layered on this foundation.

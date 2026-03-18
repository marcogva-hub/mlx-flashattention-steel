# mlx-mfa Architecture (Freeze Prep)

Version target: **2.10.0**

## 1) System Overview

`mlx-mfa` is a hybrid Python + C++/Metal package that integrates with MLX:

```text
Python API/runtime (mlx_mfa.*)
  -> nanobind extension (csrc/bindings.cpp)
    -> MFA Primitive dispatch (csrc/mfa_attention.cpp)
      -> JIT shader generation + pipeline cache
        -> Metal kernels (STEEL V2/V3/V4/V5, Sage, paged helpers)
```

Key principle: keep dense production routing conservative and benchmark-backed,
while exposing advanced serving functionality through explicit runtime APIs.

## 2) Production vs Research Paths

- Production default: **V2 dense** where policy shows wins.
- Conservative fallback: **MLX SDPA** where wins are not established.
- Narrow promotion: D=256 causal long-N regimes only.
- Non-promotion outcomes retained:
  - D=512 remains SDPA-default
  - native dense backward remains non-default
- Experimental families remain opt-in: V3/V4/V5.

## 3) Core Runtime Surface

Main runtime entry point:
- `create_decode_runtime(...) -> DecodeRuntime`

`DecodeRuntime` unifies:
- dense decode/prefill
- paged decode/prefill
- packed query layout handling for paged varlen
- chunked prefill
- prefix reuse integration
- speculative draft/verify integration
- splitfuse helper integration
- runtime metadata for selected backend/cache/query-layout state

Lower-level context factory remains available:
- `create_inference_context(...)`

## 4) Attention API Families

Primary callable families:
- Dense/general: `flash_attention(...)`
- KV-cache/decode: `flash_attention_kvcache(...)`
- Paged: `flash_attention_paged(...)`
- Paged + packed varlen query: `flash_attention_paged_varlen(...)`
- Varlen training packed tensors: `flash_attention_varlen(...)`
- Sparse/window masks: `flash_attention_sparse(...)`
- Splitfuse helper path: `flash_attention_splitfuse(...)`
- Speculative verify helpers:
  - `flash_attention_speculative_verify(...)`
  - `flash_attention_speculative_verify_paged(...)`

## 5) Serving-Oriented Flow Architecture

### 5.1 Paged + packed varlen

- Supports packed queries with per-sequence boundaries (`cu_seqlens_q`) over
  paged KV pools.
- Equal query lengths can use batched paged fast path.
- Heterogeneous query lengths currently use a correctness-first bridge path.

### 5.2 Paged continuous batching/remap

- Explicit remap via `cache_batch_idx` for scheduler-controlled active order.
- Runtime helpers support batched prefill/step and remapped packed-varlen calls.

### 5.3 Chunked prefill

- `DecodeRuntime.chunked_prefill(...)` provides explicit chunk boundaries for
  interleaving/scheduling.
- Designed as serving control capability; not assumed to improve total
  throughput on M1 Max.

### 5.4 Runtime-managed prefix reuse

- Prefix state can be registered/seeded/reused via runtime methods.
- Intended to reduce orchestration fragmentation vs helper-only wiring.

### 5.5 Runtime speculative decode

- `DecodeRuntime.speculative_step(...)` wraps verify output into explicit
  accept/reject bookkeeping (mask + accepted prefix length + accepted/rejected
  ids).
- Dense runtime path is primary; paged support is narrower and explicit.

### 5.6 Splitfuse runtime deepening

- Splitfuse is available via runtime helpers, including decode-step focused
  path (`splitfuse_step(...)`).
- Includes a narrow page-native paged decode-only route to reduce bridge glue.

## 6) Cache Architecture

## 6.1 Concrete cache classes

- `DenseKVCache`
- `PagedKVCache`
- `QuantizedKVCache`

## 6.2 Adapter/capability layer

`mlx_mfa.kv_cache` introduces cache abstraction components:
- `KVCacheCapabilities`
- `KVCacheAdapter` + concrete adapters
- `adapt_kv_cache(...)`
- `resolve_context_cache(...)`
- `resolve_context_cache_adapter(...)`

Goal: runtime code relies on capabilities rather than concrete cache internals.

## 6.3 Hybrid cache behavior

`HybridKVCache` now has real behavior:
- hot/cold/offloaded residency state
- promotion on access
- demotion/eviction under pressure
- reload/promotion back into hot tier
- prefetch intent hooks and runtime-visible metadata

This is a **minimal local offload milestone**, not distributed offload.

## 6.4 External cache adapter groundwork

`mlx_mfa.external_cache` provides extension points:
- `ExternalKVCacheAdapter`
- `ExternalKVCacheCapabilities`
- `LocalHostKVStoreAdapter` (first local backend)

This defines a future LMCache-like integration surface without claiming full
remote backend support in the freeze state.

## 7) Native Extension Architecture

Core native files:
- `csrc/mfa_attention.cpp`: primitive dispatch and routing
- `csrc/mfa_steel_fwd.cpp` + `csrc/mfa_steel_fwd_v2.cpp`: dense forward family
- `csrc/mfa_steel_bwd.cpp`: native backward kernels (gated non-default)
- `csrc/mfa_sage_fwd.cpp`: Sage path
- `csrc/mfa_paged_gather.cpp` / `csrc/mfa_scatter.cpp`: paged helpers
- `csrc/shader_cache.mm`: pipeline compilation/cache

Kernels are generated/selected by policy; active production behavior is policy
and benchmark constrained.

## 8) Documentation and Historical Separation

Active references:
- `README.md`
- `docs/API_MANUAL.md`
- `docs/benchmarks/RESULTS.md`
- `RESULTS.md`

Historical branch/track artifacts:
- `devnotes/` (organized by pass/track)

This separation is intentional for freeze-readability.

## 9) Deferred Work

Deferred until future continuation (likely newer hardware generation):
- deeper fused page-native heterogeneous paths
- remote/distributed offload backends via external adapter contract
- broader speculative scheduler integration
- new hardware-family kernel redesign work

## 10) LLM Serving Layer Status (v2.14.0)

The serving layer is considered production-ready for local inference.
See `docs/SERVING_GUIDE.md` for usage guide.

| Component | Status |
|---|---|
| Dense decode runtime | Production |
| Paged KV (batched/packed) | Production (bridge for heterogeneous) |
| HybridKVCache (hot/cold/offloaded) | Production (local offload only) |
| Prefix caching | Production |
| Speculative decode | Production (narrow) |
| Chunked prefill (batched) | Production |
| Chunked prefill (packed) | Not supported |
| Splitfuse | Narrow/conditional |
| mlx-lm patch | Production |
| Remote/distributed offload | Deferred (M5+) |

# mlx-mfa API Manual (Freeze Prep)

Version target: **2.10.0**  
Public exports: **73 + `__version__`**

This manual documents the retained public API surface for the freeze-prep
state. It emphasizes current usage and serving/runtime integration behavior.

## 1) Core Attention APIs

### `flash_attention(...)`

General dense attention entry point with backend routing.

```python
flash_attention(
    q, k, v,
    *,
    scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    window_size=None,
    return_lse=False,
    return_attn_weights=False,
    attn_bias=None,
    backend="auto",
    stream=None,
)
```

Notes:
- Primary production API.
- Supports GQA (`H_q % H_kv == 0`).
- `backend="auto"` is policy-driven and benchmark-conservative.

### `flash_attention_kvcache(...)`

Unified dense/paged KV-cache attention entry point.

```python
flash_attention_kvcache(
    q,
    k_cache,
    v_cache,
    *,
    cache_seqlens=0,
    block_table=None,
    seq_lens=None,
    block_size=16,
    cache_batch_idx=None,
    causal=True,
    scale=None,
    stream=None,
)
```

### `flash_attention_paged(...)`

Paged KV attention for batched query layout.

```python
flash_attention_paged(
    q,
    k_pages,
    v_pages,
    block_table,
    seq_lens,
    *,
    block_size=16,
    cache_batch_idx=None,
    scale=None,
    causal=True,
    return_lse=False,
    stream=None,
)
```

### `flash_attention_paged_varlen(...)`

Paged KV + packed varlen query path.

```python
flash_attention_paged_varlen(
    q_packed,
    k_pages,
    v_pages,
    block_table,
    seq_lens_kv,
    cu_seqlens_q,
    *,
    block_size=16,
    cache_batch_idx=None,
    max_seqlen_q=None,
    scale=None,
    causal=True,
    stream=None,
)
```

### `flash_attention_varlen(...)`

Packed varlen training/inference attention over contiguous packed tensors.

### `flash_attention_sparse(...)`

Block-mask sparse/window attention path.

### `flash_attention_splitfuse(...)`

Helper path for combined prefill/decode orchestration.

### `flash_attention_speculative_verify(...)`
### `flash_attention_speculative_verify_paged(...)`

Low-level speculative verify helpers (dense and paged-oriented variants).

## 2) Runtime API (Preferred Serving Surface)

### `create_decode_runtime(...)`

Primary serving-oriented constructor.

```python
create_decode_runtime(
    *,
    backend="auto",           # "auto"|"dense"|"paged"|"sage"
    paged=False,
    quantized_kv=False,
    query_layout="batched",   # "batched"|"packed"
    B=None,
    H_q=None,
    H_kv=None,
    D=None,
    max_seq_len=None,
    default_seq_id=0,
    hybrid_cache=False,
    hybrid_hot_seq_capacity=2,
    hybrid_enable_offload=False,
    hybrid_external_adapter=None,
    hybrid_secondary_cache=None,
    hybrid_policy="lru",
    **kwargs,
) -> DecodeRuntime
```

### `DecodeRuntime` key methods

Core decode/prefill:
- `prefill(...)`
- `step(...)`
- `reset(...)`
- `seq_length(...)`

Paged/serving helpers:
- `paged_prefill_batch(...)`
- `paged_step_batch(...)`
- `paged_varlen(...)`

Scheduling helpers:
- `chunked_prefill(...)`

Prefix reuse helpers:
- `register_prefix(...)`
- `seed_prefix(...)`
- `prefill_with_prefix(...)`
- `drop_prefix(...)`
- `clear_registered_prefixes(...)`

Splitfuse/speculative helpers:
- `splitfuse(...)`
- `splitfuse_step(...)`
- `speculative_verify(...)`
- `speculative_step(...)`

Hybrid controls:
- `hybrid_mark_for_prefetch(...)`
- `hybrid_prefetch(...)`
- `hybrid_state` property

Introspection:
- `metadata` property (backend/cache/query-layout/flow-state snapshot)

### `create_inference_context(...)`

Lower-level context constructor retained for direct context usage.

## 3) Context and Cache Classes

Context classes:
- `InferenceContext`
- `PagedInferenceContext`
- `SageInferenceContext`

Cache protocol/classes:
- `KVCacheProtocol`
- `DenseKVCache`
- `PagedKVCache`
- `QuantizedKVCache`

## 4) Cache Abstraction Layer

Adapter/capability exports:
- `KVCacheCapabilities`
- `KVCacheOperationUnsupported`
- `KVCacheAdapter`
- `DenseKVCacheAdapter`
- `PagedKVCacheAdapter`
- `QuantizedKVCacheAdapter`
- `HybridKVCacheAdapter`
- `HybridKVCache`
- `adapt_kv_cache(...)`
- `resolve_context_cache(...)`
- `resolve_context_cache_adapter(...)`

External adapter groundwork:
- `ExternalKVCacheCapabilities`
- `ExternalKVCacheAdapter`
- `LocalHostKVStoreAdapter`

Notes:
- Hybrid offload behavior is local-only in this freeze state.
- External adapter contract is real, but remote/distributed backends are future work.

## 5) SageAttention APIs

- `sage_attention(...)`
- `sage_attention_prequantized(...)`
- `sage_attention_kvcache(...)`
- `sage_block_sizes(...)`
- `smooth_k(...)`
- `sage_output_correction(...)`

## 6) Mask/RoPE/Utility APIs

Mask builders:
- `make_causal_block_mask`
- `make_sliding_window_mask`
- `make_spatial_2d_mask`
- `make_spatial_3d_mask`
- `make_topk_spatial_mask`
- `make_segment_mask`
- `make_causal_segment_mask`
- `make_adaptive_window_mask`
- `make_lcsa_mask`
- `make_axial_spatial_mask`
- `make_axial_temporal_mask`
- `make_dilated_temporal_mask`
- `make_sink_window_mask`
- `make_reference_frame_mask`
- `make_cross_stream_mask`

RoPE utilities:
- `flash_attention_rope(...)`
- `flash_attention_rope_unified(...)`
- `flash_attention_kvcache_rope_append(...)`
- `make_rope_3d_tables(...)`

Quantization utilities:
- `quantize_per_block(...)`
- `dequantize(...)`

Introspection/system:
- `is_mfa_available()`
- `get_device_info()`
- `get_supported_configs()`
- `warmup_kernels(...)`
- `compile_metallib(...)`
- `calibrate_dispatch(...)`

Integrations:
- `patch_mlx_lm(...)`
- `unpatch_mlx_lm(...)`

## 7) Behavioral Guidance (Current Freeze State)

- Dense production default: `flash_attention(..., backend="auto")`.
- Serving orchestration: prefer `create_decode_runtime(...)`.
- Paged/packed/chunked/prefix/speculative flows are explicit runtime capabilities.
- D=256 and Sage remain narrow-policy domains.
- D=512 remains SDPA-default.

## 8) Export Index

Current exported symbols (`mlx_mfa.__all__`) are listed below for quick
cross-checking:

`DecodeRuntime`, `DenseKVCache`, `DenseKVCacheAdapter`, `DispatchPolicy`,
`ExternalKVCacheAdapter`, `ExternalKVCacheCapabilities`, `HybridKVCache`,
`HybridKVCacheAdapter`, `InferenceContext`, `KVCacheAdapter`,
`KVCacheCapabilities`, `KVCacheOperationUnsupported`, `KVCacheProtocol`,
`LocalHostKVStoreAdapter`, `PagedInferenceContext`, `PagedKVCache`,
`PagedKVCacheAdapter`, `QuantizedKVCache`, `QuantizedKVCacheAdapter`,
`SageInferenceContext`, `adapt_kv_cache`, `calibrate_dispatch`,
`compile_metallib`, `create_decode_runtime`, `create_inference_context`,
`dequantize`, `flash_attention`, `flash_attention_kv_packed`,
`flash_attention_kvcache`, `flash_attention_kvcache_rope_append`,
`flash_attention_paged`, `flash_attention_paged_varlen`,
`flash_attention_qkv_packed`, `flash_attention_rope`,
`flash_attention_rope_unified`, `flash_attention_sparse`,
`flash_attention_speculative_verify`, `flash_attention_speculative_verify_paged`,
`flash_attention_splitfuse`, `flash_attention_varlen`,
`flash_attention_varlen_kv_packed`, `flash_attention_varlen_qkv_packed`,
`get_device_info`, `get_supported_configs`, `is_mfa_available`,
`make_adaptive_window_mask`, `make_axial_spatial_mask`,
`make_axial_temporal_mask`, `make_causal_block_mask`,
`make_causal_segment_mask`, `make_cross_stream_mask`,
`make_dilated_temporal_mask`, `make_lcsa_mask`, `make_reference_frame_mask`,
`make_rope_3d_tables`, `make_segment_mask`, `make_shared_prefix_cache`,
`make_sink_window_mask`, `make_sliding_window_mask`, `make_spatial_2d_mask`,
`make_spatial_3d_mask`, `make_topk_spatial_mask`, `quantize_per_block`,
`resolve_context_cache`, `resolve_context_cache_adapter`, `sage_attention`,
`sage_attention_kvcache`, `sage_attention_prequantized`, `sage_block_sizes`,
`sage_output_correction`, `smooth_k`, `warmup_kernels`.

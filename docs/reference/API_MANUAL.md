# API manual

Version: **2.62.0**
Public exports: **103**

The definitive export list is `mlx_mfa.__all__`. Signatures below describe the
primary surfaces; auxiliary names are grouped at the end.

## Dense attention

```python
flash_attention(q, k, v, scale=None, causal=False, softcap=0.0,
                alibi_slopes=None, dropout_p=0.0,
                return_attn_weights=False, window_size=None,
                return_lse=False, stream=None, attn_bias=None,
                backend="auto")
```

Q/K/V are BHND arrays. Q and K share their head dimension and K/V share their
sequence/head counts. A V dimension different from Q/K is accepted through the
fallback path. `backend` is `auto`, `mfa` or `sdpa`.

RoPE wrappers are `flash_attention_rope` and
`flash_attention_rope_unified`. Packed dense wrappers are
`flash_attention_qkv_packed` and `flash_attention_kv_packed`.

## Sparse and neighborhood attention

```python
flash_attention_sparse(q, k, v, block_mask, scale=None,
                       causal=False, stream=None, backward="sdpa")

flash_attention_gna(q, k, v, seq_shape, window_size, stride,
                    scale=None, stream=None)

flash_attention_topk(q, k, v, ...)
```

Sparse masks may be rank 2, 3 or 4. Native mask dtype requirements depend on
the selected path; public validation/fallback behavior is defined in
`sparse-family-spec.md`. GNA accepts logical sequence/window/stride tuples of
matching dimensionality.

Mask constructors exported from the package are:

`make_causal_block_mask`, `make_sliding_window_mask`,
`make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask`,
`make_segment_mask`, `make_causal_segment_mask`,
`make_adaptive_window_mask`, `make_lcsa_mask`, `make_axial_spatial_mask`,
`make_axial_temporal_mask`, `make_dilated_temporal_mask`,
`make_sink_window_mask`, `make_reference_frame_mask`,
`make_cross_stream_mask`, `make_gna_mask`, `make_diagonal_mask`,
`make_strided_mask`, `make_temporal_group_mask`,
`make_temporal_distance_bias`, and `temporal_distance_bias_to_mask`.

## Packed-varlen

```python
flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k,
                       scale=None, causal=False, block_mask=None,
                       stream=None)
```

The storage batch is one. `cu_seqlens_*` begin at zero, are monotonic and end
at the packed total. Each segment attends independently. Packed wrappers are
`flash_attention_varlen_qkv_packed` and
`flash_attention_varlen_kv_packed`.

## KV cache and paged attention

Primary functions are `flash_attention_kvcache`,
`flash_attention_kvcache_rope_append`, `flash_attention_paged`,
`flash_attention_paged_varlen`, and
`flash_attention_paged_varlen_turboquant`.

`DenseKVCache`, `PagedKVCache`, `QuantizedKVCache` and `KVCacheProtocol` are
the direct cache types. Adapter-level APIs are `KVCacheCapabilities`,
`KVCacheOperationUnsupported`, `KVCacheAdapter`, `DenseKVCacheAdapter`,
`PagedKVCacheAdapter`, `QuantizedKVCacheAdapter`, `HybridKVCache`,
`HybridKVCacheAdapter`, `adapt_kv_cache`, `resolve_context_cache`, and
`resolve_context_cache_adapter`.

External storage uses `ExternalKVCacheCapabilities`,
`ExternalKVCacheAdapter`, and `LocalHostKVStoreAdapter`.

## Serving runtime

```python
create_inference_context(*, backend="auto", paged=False,
                         quantized_kv=False, B=None, H_q=None,
                         H_kv, D, max_seq_len=8192, ...)

create_decode_runtime(*, backend="auto", paged=False,
                      quantized_kv=False, turboquant=False,
                      hybrid_cache=False, H_kv, D, ...)
```

Returned classes are `InferenceContext`, `PagedInferenceContext`,
`SageInferenceContext`, `TurboQuantPagedInferenceContext`, and
`DecodeRuntime`. Higher-level helpers include `flash_attention_splitfuse`,
`flash_attention_speculative_verify`,
`flash_attention_speculative_verify_paged`, and
`make_shared_prefix_cache`.

## Quantization

- `quantize_per_block`, `dequantize`, `smooth_k`,
  `sage_output_correction`, `sage_block_sizes`
- `sage_attention`, `sage_attention_kvcache`,
  `sage_attention_prequantized`
- `turboquant_compress`, `turboquant_decompress`, `TurboQuantKVCache`,
  `pack_k_for_metal`, `pack_v_for_metal`, `build_tq_paged_k_pool`,
  `build_tq_paged_v_pool`, `pack_3bit_optimal`, `unpack_3bit_optimal`
- `SVDQuantLinear`, `quantize_model`

## Introspection and control

- `get_device_info`, `get_supported_configs`, `is_mfa_available`, `has_nax`
- `NaxUnavailable`, `warmup_kernels`, `calibrate_dispatch`, `DispatchPolicy`
- `enable`, `disable`, `hooks_status`, `get_hook_stats`, `reset_hook_stats`
- `diagnostics`, `compile_metallib`, `sparse_attention_dispatch`

`enable` and `disable` control transparent hooks, not every direct API route.
`has_nax` reports runtime capability; it does not guarantee that a particular
shape will select NAX.

## Error and fallback contract

Malformed tensors, inconsistent cumulative lengths and unsupported direct
expert dimensions raise. Public high-level calls may delegate valid inputs to
MLX. D512 dense and varlen correction coverage is delegation coverage, not a
native D512 kernel claim.

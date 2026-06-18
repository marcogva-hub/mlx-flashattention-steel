# mlx-mfa API Manual

Version: **2.27.0**
Public exports: **94 + `__version__`**

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

**`attn_bias` parameter** (v2.27.0):
- Additive bias added to attention logits before softmax.
- Supported broadcast shapes (native Metal kernel, no SDPA fallback):
  - Mode 1: `[1, 1, 1, N_kv]` — per-KV position (token merging, conditioning)
  - Mode 2: `[1, H, 1, N_kv]` — per-head per-KV (temporal distance, custom ALiBi)
- SDPA fallback shapes (materializes full score matrix):
  - Mode 3: `[1, H, N_q, N_kv]` — per-head full
  - Mode 0: `[B, H, N_q, N_kv]` — full
- Must be same dtype as Q/K/V (float16 or bfloat16 for native path).
- Incompatible with `alibi_slopes` (raises ValueError if both set).
- Backward pass uses SDPA recomputation (native kernel is forward-only for bias).
- Use cases: token merging (`log(merge_count)`), temporal distance bias,
  cross-attention conditioning, custom ALiBi variants.

**V6NAX backward NAX-direct (v2.37.0, SHIP_OPT_IN)**:
- Set `MFA_ENABLE_V6_BACKWARD=1` to route `flash_attention` backward
  through V6NAX NAX-direct kernels on M5+.  Default off (SDPA-vjp).
- Eligible shapes: D ∈ {64, 128}, FP16/BF16, no causal/window/softcap,
  V6NAX-forward-routing-eligible (D=128 always; D=64 with Nk > 8000).
- Current perf: V6NAX backward 2.2-2.4× slower than SDPA-vjp on M5 Max
  (architectural floor — dK kernel inherently 2× heavier than dV due to
  required dO@V^T matmul).  Apple's NAX backward is NYI in MLX framework,
  so V6NAX backward is the only path for NAX-accelerated backward on M5+.
- See `ENV_VARS.md` `MFA_V6BWD*` group for tile-tuning knobs.

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

### `flash_attention_paged_varlen_turboquant(...)`

Paged varlen + TurboQuant fused kernel. Reads K (and optionally V) directly
from packed uint8 pools with centroid lookup in the Metal kernel.

```python
flash_attention_paged_varlen_turboquant(
    q_packed, k_pool_tq, v_pool, block_table, seq_lens_kv,
    cu_seqlens_q, centroids, k_scales,
    *,
    block_size=64, max_seqlen_q=None, scale=None, causal=True,
    tq_bits=3, tq_v_enabled=False,
    v_pool_tq=None, v_centroids=None, v_scales=None,
    stream=None,
)
```

Notes:
- Phase 2 (K-only): `tq_v_enabled=False`, V is fp16 in `v_pool`.
- Phase 3 (K+V): `tq_v_enabled=True`, V read from `v_pool_tq` with centroid lookup.
  Output is un-rotated automatically (WHT inverse).

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
    backend="auto",           # "auto"|"dense"|"paged"|"sage"|"turboquant"
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
    turboquant=False,         # enable TurboQuant KV compression
    tq_bits=3,                # 2, 3, or 4
    tq_v=True,                # compress V as well as K
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
- `TurboQuantPagedInferenceContext` — paged KV with TQ compression on append

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
- `make_gna_mask`
- `make_diagonal_mask`
- `make_strided_mask`
- `make_temporal_group_mask`
- `make_temporal_distance_bias`
- `temporal_distance_bias_to_mask`

GNA / sparse attention (v2.12.0+):
- `flash_attention_gna(q, k, v, seq_shape, window_size, stride)` — multi-dimensional
  windowed attention. As of v2.26.0, tries the native Metal GNA kernel first (D=128,
  3D `seq_shape`, f16/bf16, forward-only). Falls back to `make_gna_mask()` +
  `flash_attention_sparse()` for other configs or when a backward pass is needed.
  The native kernel applies exact per-element window masking (more precise than
  tile-level sparse approximation).
- `flash_attention_topk(q, k, v, topk_ratio)` — per-query top-k attention
  (Python reference, O(N^2) memory). Composable with block masks.

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

Current exported symbols (`mlx_mfa.__all__`) — **94 symbols** (+ `__version__`) — are listed below:

`DecodeRuntime`, `DenseKVCache`, `DenseKVCacheAdapter`, `DispatchPolicy`,
`ExternalKVCacheAdapter`, `ExternalKVCacheCapabilities`, `HybridKVCache`,
`HybridKVCacheAdapter`, `InferenceContext`, `KVCacheAdapter`,
`KVCacheCapabilities`, `KVCacheOperationUnsupported`, `KVCacheProtocol`,
`LocalHostKVStoreAdapter`, `PagedInferenceContext`, `PagedKVCache`,
`PagedKVCacheAdapter`, `QuantizedKVCache`, `QuantizedKVCacheAdapter`,
`SVDQuantLinear`, `SageInferenceContext`, `TurboQuantPagedInferenceContext`,
`adapt_kv_cache`, `calibrate_dispatch`,
`compile_metallib`, `create_decode_runtime`, `create_inference_context`,
`dequantize`, `flash_attention`, `flash_attention_gna`,
`flash_attention_kv_packed`,
`flash_attention_kvcache`, `flash_attention_kvcache_rope_append`,
`flash_attention_paged`, `flash_attention_paged_varlen`,
`flash_attention_paged_varlen_turboquant`,
`flash_attention_qkv_packed`, `flash_attention_rope`,
`flash_attention_rope_unified`, `flash_attention_sparse`,
`flash_attention_speculative_verify`, `flash_attention_speculative_verify_paged`,
`flash_attention_splitfuse`, `flash_attention_topk`,
`flash_attention_varlen`,
`flash_attention_varlen_kv_packed`, `flash_attention_varlen_qkv_packed`,
`get_device_info`, `get_supported_configs`, `is_mfa_available`,
`make_adaptive_window_mask`, `make_axial_spatial_mask`,
`make_axial_temporal_mask`, `make_causal_block_mask`,
`make_causal_segment_mask`, `make_cross_stream_mask`,
`make_diagonal_mask`, `make_dilated_temporal_mask`,
`make_gna_mask`, `make_lcsa_mask`, `make_reference_frame_mask`,
`make_rope_3d_tables`, `make_segment_mask`, `make_shared_prefix_cache`,
`make_sink_window_mask`, `make_sliding_window_mask`, `make_spatial_2d_mask`,
`make_spatial_3d_mask`, `make_strided_mask`,
`make_temporal_distance_bias`, `make_temporal_group_mask`,
`make_topk_spatial_mask`,
`pack_3bit_optimal`, `pack_k_for_metal`, `pack_v_for_metal`,
`build_tq_paged_k_pool`, `build_tq_paged_v_pool`,
`quantize_model`, `quantize_per_block`,
`resolve_context_cache`, `resolve_context_cache_adapter`, `sage_attention`,
`sage_attention_kvcache`, `sage_attention_prequantized`, `sage_block_sizes`,
`sage_output_correction`, `smooth_k`,
`temporal_distance_bias_to_mask`,
`turboquant_compress`, `turboquant_decompress`, `TurboQuantKVCache`,
`unpack_3bit_optimal`, `warmup_kernels`.

---

## TurboQuant KV Cache Compression (v2.21.0–v2.24.0)

Training-free, data-oblivious KV cache compression based on Google's
TurboQuant (ICLR 2026). Three phases of increasing kernel fusion.

### Phase 1 — Non-fused (v2.21.0)

#### `turboquant_compress(x, bits=3, *, use_qjl=True, rotation="wht", seed=42)`

Compress a `[B, H, S, D]` KV tensor to 2-4 bits per coordinate.

**Parameters:**
- `x` — fp16/bf16/f32 tensor `[B, H, S, D]`
- `bits` — 2, 3, or 4 (default: 3)
- `use_qjl` — apply QJL 1-bit residual correction (default: True)
- `rotation` — `"wht"` (Walsh-Hadamard, faster) or `"qr"` (random orthogonal)
- `seed` — random seed for deterministic rotation and QJL projection

**Returns:** dict with packed indices, scales, and optional QJL signs.

#### `turboquant_decompress(compressed)`

Decompress back to fp16/bf16 for use with existing attention kernels.

#### `TurboQuantKVCache`

Drop-in KV cache that stores K (and optionally V) in compressed format.
Transparent decompression on attention access.

```python
cache = TurboQuantKVCache(bits=3, use_qjl=True, compress_v=True)
cache.append(k_new, v_new)
k_fp16 = cache.k_decompressed()
v_fp16 = cache.v_decompressed()
print(cache.compression_ratio)  # ~4.1× for K+V at 3-bit
```

### Phase 2 — K fused in kernel (v2.22.0)

#### `pack_k_for_metal(k, centroids, bits=3)`

Pack K tensor into uint8 for direct Metal kernel consumption (2 indices/byte).

#### `build_tq_paged_k_pool(k_pool_fp16, centroids, bits=3, block_size=64)`

Build a paged TQ K pool from an existing fp16 pool.

### Phase 3 — K+V fused in kernel (v2.23.0)

#### `pack_v_for_metal(v, centroids, bits=3)`

Pack V tensor into uint8 for direct Metal kernel consumption.

#### `build_tq_paged_v_pool(v_pool_fp16, centroids, bits=3, block_size=64)`

Build a paged TQ V pool from an existing fp16 pool.

#### `TurboQuantPagedInferenceContext`

Stateful paged KV cache with automatic TQ compression on append.
Auto-rotates Q with WHT, calls fused TQ kernel, un-rotates output.

```python
from mlx_mfa import create_decode_runtime

rt = create_decode_runtime(
    turboquant=True, tq_bits=3, tq_v=True,
    H_q=32, H_kv=8, D=128, max_seq_len=8192,
    dtype=mx.float16,
)
out = rt.prefill(q, k, v)
out = rt.step(q_step, k_step, v_step)
```

### Compression quality (3-bit, cosine similarity vs fp16)

| Phase | cos(K) | cos(V) | Memory savings |
|-------|-------:|-------:|---------------:|
| Phase 2 (K-only) | 0.98 | — | ~1.6× |
| Phase 3 (K+V) | 0.98 | 0.97 | ~3.8× |

### Note on QJL

QJL (Quantized Johnson-Lindenstrauss) 1-bit residual correction is a Phase 1
feature only. The fused Metal kernels (Phase 2/3) use PolarQuant centroids
without QJL correction, which is sufficient for serving quality.

---

## SVDQuant (v2.25.0)

Post-training weight quantization with optional SVD low-rank FP16 correction.
Both symbols are in `mlx_mfa.svdquant` and re-exported from `mlx_mfa`.

#### `SVDQuantLinear(weight_q, scales, biases, group_size, bits, U=None, V=None)`

W4A16 linear layer. `weight_q`, `scales`, and `biases` are the quantized weight
tensor and its per-group scale/bias. `U` and `V` (optional) are FP16 low-rank
factors `[out_features, rank]` and `[rank, in_features]`; when present the
forward pass computes `y = dequant(W)*x + U*(V*x)` for improved accuracy on
outlier-heavy layers.

#### `quantize_model(model, group_size=64, bits=4, rank=0, calibration_data=None)`

Replace all `nn.Linear` layers in `model` with `SVDQuantLinear` in-place.
If `rank > 0`, SVD decomposition of each layer's quantization residual is
computed and stored as FP16 correction factors. Providing `calibration_data`
(a list of input tensors) enables activation-aware rank allocation.

---

## GNA Native Metal Kernel (v2.26.0)

As of v2.26.0, `flash_attention_gna()` dispatches a native Metal kernel when
all of the following conditions are met:

- `D == 128`
- `len(seq_shape) == 3` (3D spatial sequence)
- `dtype` is `float16` or `bfloat16`
- No backward pass is required (forward-only call)

The native kernel applies **exact per-element GNA window masking** directly in
the Metal shader, which is more precise than the tile-level block-mask
approximation used by the sparse fallback path. All other configurations
(D ≠ 128, 2D sequences, float32, or when gradients are needed) continue to
route through `make_gna_mask()` + `flash_attention_sparse()`.

**KernelType:** `GNAForward = 24`

**Backward:** no VJP on the native kernel; backward always uses the sparse path
(equivalent numerical result for supported configs).

# mlx-mfa API Manual

Version: **2.61.0**
Public exports: **101** (the `mlx_mfa.__all__` surface)

This manual documents the retained public API surface for the freeze-prep
state. It emphasizes current usage and serving/runtime integration behavior.

## 1) Core Attention APIs

### `flash_attention(...)`

General dense attention entry point with backend routing.

```python
flash_attention(
    q, k, v,
    scale=None,
    causal=False,
    softcap=0.0,
    alibi_slopes=None,
    dropout_p=0.0,
    return_attn_weights=False,
    window_size=None,
    return_lse=False,
    stream=None,
    attn_bias=None,
    backend="auto",
)
```

(Signature verified against `mlx_mfa.flash_attention`; arguments are positional-or-keyword,
not keyword-only.)

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

**Backward gradients: split-V6 NAX-direct is the public default for D=64 (≥2048) — VERIFIED
2.16–3.05× faster than SDPA-vjp; D=128 + small/other shapes use SDPA-vjp.** (Verified
2026-06-19, M5 Max, MLX 0.31.2, full-backward, fresh-input, gold which-binary + fp32 oracle.)

- **D=64, qL ≥ 2048, fp16/bf16, default scale → split-V6 backward (DEFAULT-ON)**, opt out via
  `MFA_DISABLE_V6_BACKWARD=1`. Per-regime (B=1 H=4, unit-scale, median-30, full dQ+dK+dV;
  SDPA-vjp/V6): **non-causal 2.16× @qL4096 / 2.21× @qL8192; causal 2.77× / 3.05×**. Engagement
  proven by which-binary trace (`_dispatch_trace` → `v6_split_backward`) + byteΔ vs SDPA-vjp;
  fp32-oracle-correct (non-causal ≈9.8e-4, causal ≈3.9e-3, fp16-bwd floor). Kernels:
  `v6_nax_backward_query` (dQ) + split `v6_nax_backward_dv_raw` / `dk_raw` (the **split** path
  at BK=32 — NOT the fused kernel below). Locked by the backward routing snapshot
  (`tests/test_backward_routing_snapshot.py`).
- **D=128 → opt-in only** (`MFA_ENABLE_V6_BACKWARD=1`) and **SLOWER** (0.54× nc / 0.57× causal
  @qL4096) → correctly NOT default; SDPA-vjp is the D=128 default. **D=64 qL<2048** also stays
  SDPA-vjp (V6 regresses below 2048).
- **Fused dKdV BK=16 (TK=1) is numerically CORRECT** (corrected 2026-06-20). II-6 found it
  corrupt (paired 16×32×16 MMA, TK=BK/16=1 → unpaired 2nd fragment OOB, dV err ≈ 35.9);
  **Phase II-8 item 3 fixed it** with a TK=1 odd-tail scratch path (the unpaired fragment loads
  `tail_lim ≤ 16` rows + writes a throwaway `scratch` fragment — no OOB), and the dense fused
  dispatch passes `generator_handles_odd_tk=true` so the `BK%32` guard admits BK=16 ONLY through
  that validated tail. Verified 2026-06-20 (M5/MLX-0.31.2): oracle-correct dq/dk/dv at unit scale,
  causal + non-causal, which-binary `V6NAXBWDF_TK=1`. **But it is NOT faster than split**:
  BK=16-fused vs split@BK=32 = **1.00–1.03×** (parity-to-noise, far below split's 2.18–3.06× vs
  SDPA-vjp) — the old "1.01–1.12× edge" was corrupt-*math* but the magnitude was ~real register
  relief, just within noise. **Split@BK=32 remains the default** (correct + the real win above);
  BK=16-fused is correct-but-not-advantageous. The `BK%32` guard still blocks the *forward* and
  the *sparse* fused generators (no odd-tail there → they raise, not corrupt). Locks:
  `tests/test_v6_fused_bk16_tk1_lock.py` (BK=16 oracle-correct) +
  `tests/test_v6_bwd_paired_mma_guard.py` (forward BK=16 still raises).
- Prior withdrawn ratios (artifacts): "1.4–1.85×" (retracted), "1.91–2.00×" (fused-BK16 corrupt
  math), cc4fd10 "parity"/3f92d96 "unverified" (shared-input graph-cache → byteΔ=0 masked
  engagement), b01e40d "2.55–5.75×" (dQ-only timing). The number above supersedes all — it is
  full-backward, engagement-proven (trace), oracle-correct, fresh-input, MLX-stamped.
- **⚠ Sign-off / net:** D=64 split-V6 is already the shipped default (no routing change this
  pass); the verified number + the backward routing snapshot are the evidence package. Any
  future change to this default must keep the snapshot GREEN. M1/M4 tier number is unmeasured
  (flag — needs Marco's golden, as with the forward net).
- See `ENV_VARS.md` `MFA_V6BWD*` group for the experimental tile-tuning knobs.

### `flash_attention_kvcache(...)`

Unified dense/paged KV-cache attention entry point.

```python
flash_attention_kvcache(
    q,
    k_cache,
    v_cache,
    *,
    k_new=None,
    v_new=None,
    block_table=None,
    seq_lens=None,
    block_size=16,
    scale=None,
    causal=True,
    softcap=0.0,
    alibi_slopes=None,
    window_size=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=0,
    interleaved=True,
    rotary_dim=None,
    cache_batch_idx=None,
    stream=None,
)
```

(Signature verified against `mlx_mfa.flash_attention_kvcache` — supports fused
`k_new`/`v_new` append and inline RoPE; returns `mx.array` or a tuple.)

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
    scale=None,
    causal=False,
    block_size=16,
    cache_batch_idx=None,
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
    max_seqlen_q=None,
    scale=None,
    causal=False,
    block_size=16,
    cache_batch_idx=None,
    stream=None,
)
```

### `flash_attention_paged_varlen_turboquant(...)`

Paged varlen + TurboQuant fused kernel. Reads K (and optionally V) directly
from packed uint8 pools with centroid lookup in the Metal kernel.

```python
flash_attention_paged_varlen_turboquant(
    q, k_pool_tq, v_pages, block_table, seq_lens_kv,
    cu_seqlens_q, centroids, k_scales,
    *,
    scale=None, causal=False, block_size=16,
    tq_bits=3, tq_v_enabled=False, tq_wht_enabled=False,
    v_pool_tq=None, v_centroids=None, v_scales=None,
    stream=None,
)
```

(Signature verified against `mlx_mfa.flash_attention_paged_varlen_turboquant`:
`causal` defaults to `False`, `block_size` to `16`, and `tq_wht_enabled` is present.)

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
    turboquant=False,         # enable TurboQuant KV compression
    tq_bits=3,                # 2, 3, or 4
    tq_v=True,                # compress V as well as K
    hybrid_cache=False,
    hybrid_policy="lru",
    hybrid_hot_seq_capacity=1,
    hybrid_with_secondary=True,
    hybrid_enable_offload=False,
    hybrid_external_adapter=None,
    query_layout="batched",   # "batched"|"packed"
    B=None,
    H_q=None,
    H_kv,                     # required
    D,                        # required
    max_seq_len=8192,
    decode_nq=1,
    expected_cache_len=0,
    causal=True,
    window_size=None,
    num_blocks=None,
    block_size=16,
    dtype=mx.float16,
    stream=None,
    default_seq_id=0,
) -> DecodeRuntime
```

(Signature verified against `mlx_mfa.create_decode_runtime`: `H_kv` and `D` are
required; `hybrid_with_secondary` replaces the older `hybrid_secondary_cache`
keyword; `max_seq_len` defaults to `8192`, `hybrid_hot_seq_capacity` to `1`.)

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

Auto-hook control / telemetry:
- `enable(...)` / `disable(...)` — install / remove the auto-routing `mx.*` hooks
- `hooks_status()` — current hook installation state
- `get_hook_stats()` / `reset_hook_stats()` — per-surface dispatch counters
- `diagnostics()` — environment + hook + device summary dict
- `sparse_attention_dispatch(...)` — sparse/LCSA NAX dispatcher

Integrations:
- mlx-lm integration hooks live in `mlx_mfa.integrations.mlx_lm`
  (import directly from that submodule; they are not re-exported on the top-level
  `mlx_mfa` namespace / `__all__`).

## 7) Behavioral Guidance (Current Freeze State)

- Dense production default: `flash_attention(..., backend="auto")`.
- Serving orchestration: prefer `create_decode_runtime(...)`.
- Paged/packed/chunked/prefix/speculative flows are explicit runtime capabilities.
- D=256 and Sage remain narrow-policy domains.
- D=512 remains SDPA-default.

## 8) Export Index

Current exported symbols (`mlx_mfa.__all__`) — **101 symbols** including
`__version__` (100 callables/classes listed below + `__version__`):

`DecodeRuntime`, `DenseKVCache`, `DenseKVCacheAdapter`,
`DispatchPolicy`, `ExternalKVCacheAdapter`,
`ExternalKVCacheCapabilities`, `HybridKVCache`,
`HybridKVCacheAdapter`, `InferenceContext`, `KVCacheAdapter`,
`KVCacheCapabilities`, `KVCacheOperationUnsupported`,
`KVCacheProtocol`, `LocalHostKVStoreAdapter`, `PagedInferenceContext`,
`PagedKVCache`, `PagedKVCacheAdapter`, `QuantizedKVCache`,
`QuantizedKVCacheAdapter`, `SVDQuantLinear`, `SageInferenceContext`,
`TurboQuantKVCache`, `TurboQuantPagedInferenceContext`,
`adapt_kv_cache`, `build_tq_paged_k_pool`, `build_tq_paged_v_pool`,
`calibrate_dispatch`, `compile_metallib`, `create_decode_runtime`,
`create_inference_context`, `dequantize`, `diagnostics`, `disable`,
`enable`, `flash_attention`, `flash_attention_gna`,
`flash_attention_kv_packed`, `flash_attention_kvcache`,
`flash_attention_kvcache_rope_append`, `flash_attention_paged`,
`flash_attention_paged_varlen`,
`flash_attention_paged_varlen_turboquant`,
`flash_attention_qkv_packed`, `flash_attention_rope`,
`flash_attention_rope_unified`, `flash_attention_sparse`,
`flash_attention_speculative_verify`,
`flash_attention_speculative_verify_paged`,
`flash_attention_splitfuse`, `flash_attention_topk`,
`flash_attention_varlen`, `flash_attention_varlen_kv_packed`,
`flash_attention_varlen_qkv_packed`, `get_device_info`,
`get_hook_stats`, `get_supported_configs`, `hooks_status`,
`is_mfa_available`, `make_adaptive_window_mask`,
`make_axial_spatial_mask`, `make_axial_temporal_mask`,
`make_causal_block_mask`, `make_causal_segment_mask`,
`make_cross_stream_mask`, `make_diagonal_mask`,
`make_dilated_temporal_mask`, `make_gna_mask`, `make_lcsa_mask`,
`make_reference_frame_mask`, `make_rope_3d_tables`,
`make_segment_mask`, `make_shared_prefix_cache`,
`make_sink_window_mask`, `make_sliding_window_mask`,
`make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_strided_mask`,
`make_temporal_distance_bias`, `make_temporal_group_mask`,
`make_topk_spatial_mask`, `pack_3bit_optimal`, `pack_k_for_metal`,
`pack_v_for_metal`, `quantize_model`, `quantize_per_block`,
`reset_hook_stats`, `resolve_context_cache`,
`resolve_context_cache_adapter`, `sage_attention`,
`sage_attention_kvcache`, `sage_attention_prequantized`,
`sage_block_sizes`, `sage_output_correction`, `smooth_k`,
`sparse_attention_dispatch`, `temporal_distance_bias_to_mask`,
`turboquant_compress`, `turboquant_decompress`, `unpack_3bit_optimal`,
`warmup_kernels`.

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

#### `SVDQuantLinear(in_features, out_features, bias=True, group_size=64, bits=4, rank=0)`

W4A16 `nn.Module` linear layer (module-style ctor — construct by dimensions, then
load/quantize weights). `group_size`/`bits` set the W4A16 quantization; `rank>0`
adds an FP16 low-rank SVD correction (`U [out_features, rank]`, `V [rank, in_features]`),
so the forward computes `y = dequant(W)·x + U·(V·x)` for improved accuracy on
outlier-heavy layers. Typically produced via `quantize_model(...)` rather than
constructed directly. (Signature verified against `mlx_mfa/svdquant/linear.py`.)

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

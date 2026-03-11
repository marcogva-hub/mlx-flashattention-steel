# mlx-mfa API Manual — v2.6.0

Complete reference for all 53 public exports in `mlx_mfa.__all__` (plus `__version__`).

## Table of Contents

1. [Core Attention](#1-core-attention)
2. [Packed Input Variants](#2-packed-input-variants)
3. [Sparse Attention](#3-sparse-attention)
4. [Variable-Length Attention](#4-variable-length-attention)
5. [KV Cache & Decode](#5-kv-cache--decode)
6. [Paged Attention](#6-paged-attention)
7. [LLM Inference Helpers](#7-llm-inference-helpers)
8. [SageAttention — int8 Q/K](#8-sageattention--int8-qk)
9. [Mask Builders](#9-mask-builders)
10. [RoPE Utilities](#10-rope-utilities)
11. [Inference Contexts](#11-inference-contexts)
12. [KV Cache Classes](#12-kv-cache-classes)
13. [Dispatch & Calibration](#13-dispatch--calibration)
14. [Introspection](#14-introspection)
15. [mlx-lm Integration](#15-mlx-lm-integration)
16. [AOT Compilation](#16-aot-compilation)

---

## 1. Core Attention

### `flash_attention`

```python
flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    dropout_p: float = 0.0,
    window_size: Optional[tuple] = None,
    return_lse: bool = False,
    return_attn_weights: bool = False,
    attn_bias: Optional[mx.array] = None,
    backend: str = "auto",
    stream: Optional[mx.Stream] = None,
) -> mx.array  # or tuple[mx.array, mx.array] when return_lse=True
```

Drop-in replacement for `mx.fast.scaled_dot_product_attention`.  Routes to the
STEEL V2 Metal kernel when it is faster than SDPA, otherwise falls back to SDPA.

**Tensor layout:** `[B, H, N, D]` row-major.  GQA: K/V may have `H_kv < H_q`
(must divide evenly).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q, k, v` | `mx.array` | — | Query/Key/Value. fp16, bf16, or fp32 |
| `scale` | `float` | `1/√D` | Attention scale factor |
| `causal` | `bool` | `False` | Apply causal (upper-triangular) mask |
| `softcap` | `float` | `0.0` | Gemma-2/Grok: `tanh(S/cap)×cap` before softmax |
| `alibi_slopes` | `[H] float32` | `None` | Per-head ALiBi linear position slopes |
| `dropout_p` | `float` | `0.0` | Attention dropout probability (training only) |
| `window_size` | `(int, int)` | `None` | `(left, right)` sliding window radii. `-1` disables a side |
| `return_lse` | `bool` | `False` | Return `(output, lse)` where lse is `[B,H,N]` log-sum-exp |
| `return_attn_weights` | `bool` | `False` | Return softmax weights (slow Python path, debug only) |
| `attn_bias` | `[B,H,N,S]` | `None` | Additive bias tensor (forces SDPA fallback) |
| `backend` | `str` | `"auto"` | `"auto"`, `"mfa"`, `"sdpa"`, or `"sage"` |

**Backend routing (`"auto"`):**

| Condition | Backend |
|-----------|---------|
| D=64 causal N≥4096 | STEEL V2 |
| D=128 causal N≥8192 (M1) or N≥2048 (M3+) | STEEL V2 |
| Any window or sparse | STEEL V2 (tile-skip always wins) |
| D≥256, non-causal, short N | SDPA |
| Mixed dtypes (q fp32 + k/v fp16) | STEEL V2 |
| `backend="sage"` | `sage_attention()` — inference-only |

**Returns:** `[B, H, N, D]` or `(output, lse)` if `return_lse=True`.

**Autograd:** fully supported via STEEL backward (D≤512, f16/bf16) or
`mx.vjp(SDPA)` fallback.

```python
import mlx.core as mx
from mlx_mfa import flash_attention

q = mx.random.normal((2, 8, 4096, 128)).astype(mx.float16)
k = mx.random.normal((2, 8, 4096, 128)).astype(mx.float16)
v = mx.random.normal((2, 8, 4096, 128)).astype(mx.float16)

# Standard causal attention
out = flash_attention(q, k, v, causal=True)

# GQA (8 Q heads, 2 KV heads)
k2 = mx.random.normal((2, 2, 4096, 128)).astype(mx.float16)
v2 = mx.random.normal((2, 2, 4096, 128)).astype(mx.float16)
out = flash_attention(q, k2, v2, causal=True)

# Sliding window (attend left 512, right-unbounded)
out = flash_attention(q, k, v, causal=True, window_size=(512, -1))

# Get log-sum-exp for Flash Decoding / log-prob computation
out, lse = flash_attention(q, k, v, causal=True, return_lse=True)
# lse: [2, 8, 4096]
```

**See also:** `flash_attention_kvcache` (decode), `flash_attention_rope_unified` (RoPE),
`flash_attention_sparse` (sparse), `DispatchPolicy`

---

### `flash_attention_rope`

```python
flash_attention_rope(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    **kwargs,           # same as flash_attention
) -> mx.array
```

Legacy wrapper. Applies RoPE to Q and K, then calls `flash_attention`.
Prefer `flash_attention_rope_unified` for new code.

---

### `flash_attention_rope_unified`

```python
flash_attention_rope_unified(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    *,
    cache_seqlens: Union[int, mx.array] = 0,
    scale: Optional[float] = None,
    causal: bool = True,
    interleaved: bool = True,
    rotary_dim: Optional[int] = None,
    backend: str = "auto",
    stream: Optional[mx.Stream] = None,
) -> mx.array
```

Single entry point for attention + RoPE. Handles both prefill (new sequence)
and incremental decode (extending an existing cache) via `cache_seqlens`.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `rotary_cos, rotary_sin` | `[max_seq_len, D/2]` or `[max_seq_len, rotary_dim/2]` |
| `cache_seqlens` | Scalar or `[B]` int — positions already in cache. RoPE applied at offset `cache_seqlens + i` for query position `i`. `0` = prefill from scratch |
| `interleaved` | `True` = GPT-J/Qwen style, `False` = Llama/GPT-NeoX style |
| `rotary_dim` | Apply RoPE only to first `rotary_dim` dims. Default: all `D` |

```python
from mlx_mfa import flash_attention_rope_unified

# Prefill
out = flash_attention_rope_unified(q, k, v, cos, sin, causal=True)

# Decode step (100 tokens already in cache)
out = flash_attention_rope_unified(
    q_tok, k_tok, v_tok, cos, sin,
    cache_seqlens=100, causal=True
)
```

---

## 2. Packed Input Variants

Convenience wrappers that split packed tensors and call `flash_attention`.

### `flash_attention_qkv_packed`

```python
flash_attention_qkv_packed(
    qkv: mx.array,   # [B, H, N, 3, D]  or  [B, N, 3*H*D]
    **kwargs
) -> mx.array
```

Q, K, V interleaved in a single tensor. Splits along dim 3 (or last).

### `flash_attention_kv_packed`

```python
flash_attention_kv_packed(
    q:  mx.array,    # [B, H, N, D]
    kv: mx.array,    # [B, H, S, 2, D]
    **kwargs
) -> mx.array
```

Q separate; K and V interleaved in `kv`.

---

## 3. Sparse Attention

### `flash_attention_sparse`

```python
flash_attention_sparse(
    q: mx.array,               # [B, H, N, D]  f16 or bf16
    k: mx.array,               # [B, H, S, D]
    v: mx.array,               # [B, H, S, D]
    block_mask: mx.array,      # bool, various shapes (see below)
    scale: Optional[float] = None,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
    backward: str = "sdpa",
) -> mx.array
```

Block-sparse attention. Only computes (Q-tile, K-tile) pairs where
`block_mask == True`. Masked-out pairs contribute zero (equivalent to −∞
before softmax). Tile-skip is implemented as a uniform threadgroup branch with
zero warp divergence.

**`block_mask` shapes** (all equivalent; more specific shapes take precedence):

| Shape | Meaning |
|-------|---------|
| `[NQ_tiles, NK_tiles]` | Same mask for all batches and heads |
| `[H, NQ_tiles, NK_tiles]` | Per-head mask |
| `[B, H, NQ_tiles, NK_tiles]` | Per-batch per-head mask |

Tile dimensions: `BQ=32`, `BK` depends on D (use `sage_block_sizes(D)` or
`make_causal_block_mask(N, S, D)` to get correctly-sized masks).

**`backward` parameter:**

| Value | Behavior |
|-------|----------|
| `"sdpa"` (default) | Dense `mx.vjp(SDPA)` — correct, ignores sparsity |
| `"sdpa_sparse"` | Tiled sparse backward — skips inactive tiles, saves memory |
| `"steel_sparse"` | STEEL native sparse backward (faster, D≤128 f16/bf16) |

```python
from mlx_mfa import flash_attention_sparse, make_sliding_window_mask

# Sliding window: attend only within 256 tokens
mask = make_sliding_window_mask(N=4096, window_size=256, head_dim=128)
out = flash_attention_sparse(q, k, v, mask, causal=True)

# Spatial attention on 32×32 image tokens (radius=4)
from mlx_mfa import make_spatial_2d_mask
mask = make_spatial_2d_mask(H=32, W=32, radius=4, head_dim=128)
out = flash_attention_sparse(q, k, v, mask)
```

---

## 4. Variable-Length Attention

### `flash_attention_varlen`

```python
flash_attention_varlen(
    q: mx.array,               # [1, H, total_q, D]
    k: mx.array,               # [1, H, total_k, D]
    v: mx.array,               # [1, H, total_k, D]
    cu_seqlens_q: mx.array,    # [num_seqs+1] int32
    cu_seqlens_k: mx.array,    # [num_seqs+1] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: Optional[float] = None,
    causal: bool = False,
    block_mask: Optional[mx.array] = None,
    stream: Optional[mx.StreamOrDevice] = None,
) -> mx.array                  # [1, H, total_q, D]
```

Attention over multiple packed variable-length sequences. Sequences do not
attend to each other — each `[i, i+1)` slice attends only within itself.

`cu_seqlens_q[0] = 0`, `cu_seqlens_q[-1] = total_q`.

```python
import mlx.core as mx
from mlx_mfa import flash_attention_varlen

# 3 sequences: lengths [256, 512, 128]
total = 896
q = mx.random.normal((1, 8, total, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, total, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, total, 128)).astype(mx.float16)
cu_q = mx.array([0, 256, 768, 896], dtype=mx.int32)
out = flash_attention_varlen(q, k, v, cu_q, cu_q, 512, 512, causal=True)
```

### `flash_attention_varlen_qkv_packed`

```python
flash_attention_varlen_qkv_packed(qkv, cu_seqlens_q, cu_seqlens_k, ...)
```

Varlen with interleaved Q/K/V tensor `[1, H, total, 3, D]`.

### `flash_attention_varlen_kv_packed`

```python
flash_attention_varlen_kv_packed(q, kv, cu_seqlens_q, cu_seqlens_k, ...)
```

Varlen with separate Q and interleaved K/V `[1, H, total, 2, D]`.

---

## 5. KV Cache & Decode

### `flash_attention_kvcache`

```python
flash_attention_kvcache(
    q: mx.array,                          # [B, H_q, N_q, D]
    k_cache: Optional[mx.array],          # [B, H_kv, S, D] — dense cache
    v_cache: Optional[mx.array],
    *,
    k_new: Optional[mx.array] = None,     # append mode: new tokens
    v_new: Optional[mx.array] = None,
    block_table: Optional[mx.array] = None,  # paged mode
    seq_lens: Optional[mx.array] = None,
    block_size: int = 16,
    scale: Optional[float] = None,
    causal: bool = True,
    softcap: float = 0.0,
    alibi_slopes: Optional[mx.array] = None,
    window_size: Optional[tuple] = None,
    rotary_cos: Optional[mx.array] = None,   # applies RoPE to q (and k_new)
    rotary_sin: Optional[mx.array] = None,
    cache_seqlens: Union[int, mx.array, Sequence[int]] = 0,
    interleaved: bool = True,
    stream: Optional[mx.Stream] = None,
) -> mx.array
```

Unified KV-cache attention covering four modes:

| Mode | What to pass |
|------|-------------|
| Dense read | `k_cache, v_cache` |
| Dense append | `k_cache, v_cache, k_new, v_new` |
| Paged read | `k_cache=pool, v_cache=pool, block_table=bt, seq_lens=sl` |
| Paged append | above + `k_new, v_new` |

RoPE is applied when `rotary_cos` is provided (to Q always; to `k_new` when appending).

```python
from mlx_mfa import flash_attention_kvcache, DenseKVCache

cache = DenseKVCache(B=1, H=8, D=128, max_seq_len=2048)
cache.append(k_prefill, v_prefill)

# Decode step
cache.append(k_tok, v_tok)
out = flash_attention_kvcache(
    q_tok, cache.k, cache.v, cache_seqlens=cache.seqlen, causal=True
)
```

**See also:** `DenseKVCache`, `PagedKVCache`, `InferenceContext`

---

### `flash_attention_kvcache_rope_append`

```python
flash_attention_kvcache_rope_append(
    q: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    k_cache: Optional[mx.array],
    v_cache: Optional[mx.array],
    rotary_cos: mx.array,
    rotary_sin: mx.array,
    cache_seqlens: int = 0,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    interleaved: bool = True,
    stream: Optional[mx.Stream] = None,
) -> tuple[mx.array, mx.array, mx.array]  # (output, k_appended, v_appended)
```

Append new tokens to the KV cache with fused RoPE rotation, then attend.
Returns updated `(output, k_cache_updated, v_cache_updated)`.

Keys are stored **pre-rotated** — the RoPE is applied once at write time.
This is the pattern used by mlx-lm for model serving.

```python
out, k_cache, v_cache = flash_attention_kvcache_rope_append(
    q_tok, k_tok, v_tok, k_cache, v_cache, cos, sin,
    cache_seqlens=seqlen, causal=True
)
seqlen += q_tok.shape[2]
```

---

## 6. Paged Attention

### `flash_attention_paged`

```python
flash_attention_paged(
    q: mx.array,             # [B, H_q, N_q, D]
    k_pages: mx.array,       # [num_blocks, block_size, H_kv, D]
    v_pages: mx.array,       # [num_blocks, block_size, H_kv, D]
    block_table: mx.array,   # [B, max_blocks_per_seq] int32
    seq_lens: mx.array,      # [B] int32
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    block_size: int = 16,
    stream: Optional[mx.StreamOrDevice] = None,
) -> mx.array
```

Low-level paged KV attention. Gathers K/V from the page pool into contiguous
tensors via a single Metal dispatch, then runs `flash_attention`.

`block_table[b, i]` maps the i-th logical block of sequence b to a physical
block index. Use `-1` to pad unused entries.

Supports autograd: gradients are scattered back to the page pool.

```python
from mlx_mfa import flash_attention_paged, PagedKVCache

cache = PagedKVCache(num_blocks=256, block_size=16, H=8, D=128)
cache.append(k_prefill, v_prefill, seq_id=0)
cache.append(k_tok, v_tok, seq_id=0)

out = flash_attention_paged(
    q_tok,
    cache.k_pool, cache.v_pool,
    cache.get_block_table(), cache.get_seq_lens(),
    block_size=cache.block_size, causal=True,
)
```

**See also:** `PagedKVCache`, `flash_attention_kvcache` (paged mode)

---

## 7. LLM Inference Helpers

### `flash_attention_speculative_verify`

```python
flash_attention_speculative_verify(
    q_target: mx.array,   # [B, H, N_draft, D]
    k_cache: mx.array,    # [B, H_kv, S, D]
    v_cache: mx.array,
    draft_ids: mx.array,  # [B, N_draft] int32
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    temperature: float = 1.0,
    stream: Optional[mx.Stream] = None,
) -> tuple[mx.array, mx.array]  # (target_logprobs, accepted_mask)
```

Verify draft tokens from a speculative decoder against the target model's KV
cache. Returns per-draft-token log-probabilities under the target distribution
and an acceptance mask (True = accept).

---

### `flash_attention_splitfuse`

```python
flash_attention_splitfuse(
    q_prefill: Optional[mx.array],      # [B_p, H, N_prefill, D]
    k_prefill: Optional[mx.array],
    v_prefill: Optional[mx.array],
    q_decode:  Optional[mx.array],      # [B_d, H, N_decode, D]
    k_cache_decode: Optional[mx.array], # [B_d, H_kv, S, D]
    v_cache_decode: Optional[mx.array],
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    stream: Optional[mx.Stream] = None,
) -> tuple[Optional[mx.array], Optional[mx.array]]  # (prefill_out, decode_out)
```

Process prefill and decode requests together. Both batches are independent;
there is no cross-attention between them. Decode activates Flash Decode when
`N_decode ≤ 4` and `S ≥ 256`. Pass `None` to skip a sub-batch.

---

### `make_shared_prefix_cache`

```python
make_shared_prefix_cache(
    k_prefix: mx.array,   # [1, H, N_prefix, D]
    v_prefix: mx.array,
) -> tuple[mx.array, mx.array]  # (k_cache, v_cache) ready for multi-sequence reuse
```

Build a shared prefix KV cache. Multiple decode requests can attend to the
same prompt without duplicating the KV cache in memory.

---

## 8. SageAttention — int8 Q/K

SageAttention quantizes Q and K to int8 per-block before Metal dispatch,
reducing Q/K memory bandwidth by 2×. V is always fp16/bf16.

### `sage_attention`

```python
sage_attention(
    q: mx.array,              # [B, H, N, D]  fp16 or bf16
    k: mx.array,              # [B, H_kv, S, D]
    v: mx.array,              # [B, H_kv, S, D]
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    apply_smooth_k: bool = True,
    window_size: Optional[tuple] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array
```

Inference-only (no autograd). Quantizes Q and K on every call. Use
`QuantizedKVCache` + `sage_attention_prequantized` to avoid re-quantizing the
full K cache on every decode step.

**`apply_smooth_k`**: When `True`, subtracts the per-channel mean of K before
quantizing (SageAttention K-smoothing). Reduces quantization error at negligible
cost. Set `False` only for benchmarking.

**`window_size`**: Same semantics as `flash_attention`.

```python
from mlx_mfa import sage_attention

out = sage_attention(q, k, v, causal=True)
# or via flash_attention:
out = flash_attention(q, k, v, causal=True, backend="sage")
```

---

### `sage_attention_kvcache`

```python
sage_attention_kvcache(
    q: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    *,
    scale: Optional[float] = None,
    causal: bool = True,
    cache_seqlens: int = 0,
    window_size: Optional[tuple] = None,
) -> mx.array
```

Thin wrapper around `sage_attention` for decode. Slices the cache to
`[:, :, :cache_seqlens + q.shape[2], :]`, then calls `sage_attention`.

**Note:** Re-quantizes the full K slice on every call — O(seqlen × D) overhead.
For long sequences, use `QuantizedKVCache` + `sage_attention_prequantized`.

---

### `sage_attention_prequantized`

```python
sage_attention_prequantized(
    q_int8: mx.array,      # [B, H, N_q, D]  int8
    k_int8: mx.array,      # [B, H_kv, S, D] int8
    v: mx.array,           # [B, H_kv, S, D] fp16/bf16
    q_scale: mx.array,     # [B, H, n_q_blocks] float32
    k_scale: mx.array,     # [B, H_kv, n_k_blocks] float32
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    window_size: Optional[tuple] = None,
    stream: Optional[mx.Stream] = None,
) -> mx.array
```

Calls the Sage Metal kernel directly with pre-quantized int8 inputs.
Bypasses all Python-side quantization. The primary use case is
`QuantizedKVCache` which pre-stores K as int8.

All inputs must be contiguous — the function calls `.flatten().reshape()` as
needed to guarantee canonical strides before C++ dispatch.

---

### Quantization utilities

#### `quantize_per_block`

```python
quantize_per_block(
    x: mx.array,          # [B, H, N, D] fp16 or bf16
    block_size: int,      # tokens per quantization block
) -> tuple[mx.array, mx.array]  # (x_int8, x_scale)
```

Quantize each `[block_size, D]` tile to int8 with a per-tile float32 scale.
`x_scale` shape: `[B, H, ceil(N / block_size)]`.

#### `dequantize`

```python
dequantize(
    x_int8: mx.array,     # [B, H, N, D] int8
    x_scale: mx.array,    # [B, H, n_blocks] float32
    block_size: int,
) -> mx.array             # float32
```

Reconstruct fp32 from int8 + per-block scale. The inverse of `quantize_per_block`.

#### `smooth_k`

```python
smooth_k(
    k: mx.array,          # [B, H, S, D] fp16/bf16
) -> tuple[mx.array, mx.array]  # (k_smoothed, k_mean)
```

Subtract per-channel (D) mean from K. Returns smoothed K and the channel mean
`[B, H, 1, D]`. The mean subtraction is exactly cancelled in the softmax ratio
(no bias in the final output), but dramatically reduces int8 quantization error.

#### `sage_output_correction`

```python
sage_output_correction(
    O: mx.array,
    q: mx.array,
    k_mean: mx.array,
    v: mx.array,
    L: mx.array,
    scale: float,
) -> mx.array
```

Legacy: computes the output correction term for K-smoothing bias. This is a
mathematical no-op (correction = 0) and is **not called** by `sage_attention`.
Provided for reference.

#### `sage_block_sizes`

```python
sage_block_sizes(head_dim: int) -> tuple[int, int]  # (BQ, BK)
```

Return the quantization block sizes for the given head dimension. BQ and BK
are the Q and K tile sizes used by the Sage Metal kernel.

| D | BQ | BK |
|---|----|----|
| 64 | 32 | 64 |
| 128 | 32 | 32 |
| 256 | 16 | 16 |

---

## 9. Mask Builders

All mask builders return boolean tensors compatible with `flash_attention_sparse`.
The output shape is `[NQ_tiles, NK_tiles]` or `[B, H, NQ_tiles, NK_tiles]`
depending on the function.

Tile dimensions are fixed at `BQ=32, BK=32` for D≤128; `BQ=16, BK=16` for D=256.

| Function | Shape | Description |
|----------|-------|-------------|
| `make_causal_block_mask(N, S, D)` | `[NQ, NK]` | Lower-triangular block mask |
| `make_sliding_window_mask(N, S, D, w)` | `[NQ, NK]` | Retain tokens within `w` positions |
| `make_spatial_2d_mask(H, W, radius, D)` | `[NQ, NK]` | 2D grid neighborhood (image) |
| `make_spatial_3d_mask(T, H, W, r, D)` | `[NQ, NK]` | 3D spatiotemporal neighborhood (video) |
| `make_topk_spatial_mask(H, W, k, D)` | `[NQ, NK]` | Top-K nearest spatial tokens |
| `make_segment_mask(seg_ids, D)` | `[NQ, NK]` | Same-segment non-causal mask |
| `make_causal_segment_mask(seg_ids, D)` | `[NQ, NK]` | Causal + same-segment |
| `make_adaptive_window_mask(widths, D)` | `[NQ, NK]` | Variable window width per position |
| `make_lcsa_mask(N, w, g, D)` | `[NQ, NK]` | Local-context sparse (FlashVSR: local window + global tokens) |
| `make_axial_spatial_mask(H, W, D)` | `[NQ, NK]` | Row + column axial attention |
| `make_axial_temporal_mask(T, H, W, D)` | `[NQ, NK]` | Temporal + spatial axial |
| `make_dilated_temporal_mask(T, stride, D)` | `[NQ, NK]` | Dilated strided temporal windows |
| `make_sink_window_mask(N, w, sinks, D)` | `[NQ, NK]` | Sink tokens + local window (StreamingLLM style) |
| `make_reference_frame_mask(T, N, D)` | `[NQ, NK]` | First-frame reference + local window (video) |
| `make_cross_stream_mask(N, S, D)` | `[NQ, NK]` | Bidirectional cross-stream attention |

---

## 10. RoPE Utilities

### `make_rope_3d_tables`

```python
make_rope_3d_tables(
    dim: int,
    T: int,
    H: int,
    W: int,
    theta: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> tuple[mx.array, mx.array]  # (cos, sin)  shape: [T*H*W, dim/2]
```

Build 3D rotary position tables for video models. Temporal and spatial
frequencies are split across `dim` using standard RoPE frequency allocation.
Pass the resulting `cos, sin` to `flash_attention_rope_unified`.

---

## 11. Inference Contexts

Stateful wrappers that manage the KV cache lifecycle across prefill and decode steps.

### `InferenceContext`

```python
InferenceContext(
    B: int,
    H_kv: int,
    D: int,
    max_seq_len: int = 4096,
    dtype: mx.Dtype = mx.float16,
)
```

Dense KV cache context for autoregressive generation. Wraps `DenseKVCache`.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `prefill` | `(q, k, v, *, scale, causal)` | Full sequence forward; populates cache |
| `step` | `(q_tok, k_tok, v_tok, *, scale)` | Single-token decode; appends to cache |
| `reset` | `()` | Clear cache; returns `self` for chaining |
| `seqlen` | property | Current number of tokens in cache |
| `k_cache, v_cache` | property | Raw cache arrays (read-only) |

Implements the context manager protocol: `with ctx: ...` auto-resets on exit.

```python
from mlx_mfa import InferenceContext

ctx = InferenceContext(B=1, H_kv=8, D=128, max_seq_len=4096)
out = ctx.prefill(q, k, v, scale=1/11.31)
for step in range(max_new_tokens):
    out = ctx.step(q_tok, k_tok, v_tok)
ctx.reset()

# or as context manager:
with ctx:
    out = ctx.prefill(q, k, v)
    for step in range(steps):
        out = ctx.step(q_tok, k_tok, v_tok)
# auto-reset here
```

---

### `PagedInferenceContext`

```python
PagedInferenceContext(
    num_blocks: int,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 128,
    dtype: mx.Dtype = mx.float16,
)
```

Paged KV cache context for multi-request inference. Multiple sequences share
the block pool; blocks are dynamically allocated and freed.

**Key methods:** `prefill(q, k, v, seq_id)`, `step(q_tok, k_tok, v_tok, seq_id)`,
`reset(seq_id=None)`, `seq_length(seq_id)`.

---

### `SageInferenceContext`

```python
SageInferenceContext(
    B: int,
    H_kv: int,
    D: int,
    max_seq_len: int = 4096,
    dtype: mx.Dtype = mx.float16,
)
```

KV cache context using SageAttention for decode. Prefill uses exact fp16
attention; decode uses `sage_attention_prequantized` with a `QuantizedKVCache`
that re-quantizes only newly appended tokens.

**Same interface as `InferenceContext`:** `prefill`, `step`, `reset`, context manager.

---

## 12. KV Cache Classes

### `KVCacheProtocol`

Abstract base class (protocol) for all KV cache implementations. Implementations:
`DenseKVCache`, `PagedKVCache`, `QuantizedKVCache`.

**Required interface:**

| Method | Description |
|--------|-------------|
| `append(k_new, v_new, seq_id=0)` | Append new tokens to cache |
| `reset(seq_id=None)` | Clear cache (all sequences or one) |
| `k` property | Active K slice `[B, H, seqlen, D]` |
| `v` property | Active V slice `[B, H, seqlen, D]` |
| `seqlen` property | Current token count |

---

### `DenseKVCache`

```python
DenseKVCache(
    B: int,
    H: int,
    D: int,
    max_seq_len: int = 4096,
    dtype: mx.Dtype = mx.float16,
)
```

Pre-allocated dense KV cache. Solves the O(N²) graph-accumulation problem of
repeated `mx.concatenate` by using `mx.slice_update` into a fixed buffer.

**Attributes:** `k`, `v`, `seqlen`, `max_seq_len`
**Methods:** `append(k_new, v_new)`, `reset()`

---

### `QuantizedKVCache`

```python
QuantizedKVCache(
    B: int,
    H: int,
    D: int,
    max_seq_len: int = 8192,
    dtype: mx.Dtype = mx.float16,
    block_size: Optional[int] = None,   # default: sage_block_sizes(D)[1]
)
```

Pre-stores K as int8 for use with `sage_attention_prequantized`. On each
`append`, only the newly added block is quantized — O(block_size × D) per step,
regardless of total sequence length.

**Attributes:**
- `k_int8`: Active K as int8 `[B, H, seqlen, D]` (contiguous)
- `k_scale`: Block scales `[B, H, n_blocks]` float32
- `v`: Active V slice `[B, H, seqlen, D]` (contiguous)
- `seqlen`: Current token count

**Methods:** `append(k_new, v_new)`, `reset()`

```python
from mlx_mfa import QuantizedKVCache, sage_attention_prequantized
from mlx_mfa import quantize_per_block, sage_block_sizes

cache = QuantizedKVCache(B=1, H=8, D=128, max_seq_len=4096)

# Prefill
cache.append(k_prefill, v_prefill)

# Decode loop (O(block_size × D) quantize per step)
_, BK = sage_block_sizes(128)
for _ in range(steps):
    cache.append(k_tok, v_tok)
    q_int8, q_scale = quantize_per_block(q_tok, BK)
    out = sage_attention_prequantized(
        q_int8, cache.k_int8, cache.v,
        q_scale, cache.k_scale,
        causal=True
    )
```

---

### `PagedKVCache`

```python
PagedKVCache(
    num_blocks: int,
    block_size: int = 16,
    H: int = 8,
    D: int = 128,
    dtype: mx.Dtype = mx.float16,
)
```

Paged block allocator for multi-sequence inference. Dual K/V page pools.

**Key attributes:** `k_pool`, `v_pool`, `block_size`

**Key methods:**
- `append(k_new, v_new, seq_id=0)` — allocate blocks and write tokens
- `get_block_table(seq_ids=None)` → `[B, max_blocks_per_seq]` int32
- `get_seq_lens(seq_ids=None)` → `[B]` int32
- `free_seq(seq_id)` — release all blocks for a sequence
- `reset(seq_id=None)` — free all sequences (or one)

---

## 13. Dispatch & Calibration

### `DispatchPolicy`

String constants for the `backend=` parameter of `flash_attention`.

| Constant | Value | Description |
|----------|-------|-------------|
| `DispatchPolicy.AUTO` | `"auto"` | Shape-aware routing (default) |
| `DispatchPolicy.MFA` | `"mfa"` | Force STEEL V2 Metal kernel |
| `DispatchPolicy.SDPA` | `"sdpa"` | Force `mx.fast.scaled_dot_product_attention` |
| `DispatchPolicy.SAGE` | `"sage"` | Route to `sage_attention` (int8 Q/K, inference-only) |

---

### `calibrate_dispatch`

```python
calibrate_dispatch(
    head_dims: list[int] = [64, 128],
    warmup: int = 5,
    n_iters: int = 20,
    save_path: Optional[str] = None,   # default: ~/.mlx_mfa/dispatch_table.json
    calibrate_kernel_configs: bool = True,  # also calibrate BK for D=128
    verbose: bool = True,
) -> dict
```

Benchmark MFA vs SDPA on the current device at several (D, N, causal) points.
Saves optimal crossover thresholds to JSON. The file is loaded automatically
at import time via `_load_calibrated_kernel_config()`.

```bash
# From CLI
python -m mlx_mfa calibrate
```

**Environment variables:**

| Variable | Effect |
|----------|--------|
| `MFA_V2_FORCE_BK=32` or `64` | Override D=128 K-tile size |
| `MFA_DISABLE_V2=1` | Force V1 kernel (disable V2) |
| `MFA_FORCE_GEN=15` | Pretend M3+ for testing (gen 15 = M3) |
| `MLX_MFA_DISPATCH_TABLE=path` | Load custom thresholds JSON |
| `MLX_MFA_VERBOSE_DISPATCH=1` | Log dispatch decisions to stdout |

---

## 14. Introspection

### `is_mfa_available`

```python
is_mfa_available() -> bool
```

Returns `True` if the C++ extension is compiled and the Metal GPU is available.
When `False`, all functions fall back to `mx.fast.scaled_dot_product_attention`.

---

### `get_device_info`

```python
get_device_info() -> dict
```

Returns device metadata:

```python
{
    "device_name":    "Apple M1 Max",
    "chip_name":      "M1 Max",
    "gpu_family_gen": 13,      # applegpu_g13s → 13
    "gpu_cores":      32,
    "is_m3_plus":     False,   # gen >= 15
    "is_m5_plus":     False,   # gen >= 17 (Metal 4 tensor API)
    "extension_available": True,
}
```

GPU gen mapping: 13=M1, 14=M2, 15=M3, 16=M4. (Note: this is NOT the
`MTLGPUFamilyApple` enum value — it is the numeric suffix of the architecture
string `applegpu_gNs`.)

---

### `get_supported_configs`

```python
get_supported_configs() -> dict
```

Returns the full feature matrix:

```python
{
    "head_dims":   [64, 128, 256, 512],
    "dtypes":      ["f16", "bf16", "f32"],
    "extension_available": True,
    "features": {
        "causal": True,
        "window": True,
        "sparse": True,
        "gqa":    True,
        "rope":   True,
        "alibi":  True,
        "softcap": True,
        "varlen": True,
        "paged":  True,
        "flash_decode": True,
        "sage_attention": True,
        "sage_attention_kvcache": True,
        "quantized_kvcache": True,
        # ... 16 total flags
    },
    "kernel_types": 12,
}
```

---

### `warmup_kernels`

```python
warmup_kernels(
    head_dims: Optional[list[int]] = None,
    dtypes: Optional[list] = None,
) -> None
```

Pre-compile all STEEL kernel variants to avoid first-call JIT latency.
Call once at application startup.

```python
from mlx_mfa import warmup_kernels
warmup_kernels()  # compiles all variants for all supported (D, dtype) pairs
```

---

## 15. mlx-lm Integration

### `patch_mlx_lm`

```python
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm, unpatch_mlx_lm, is_patched

patch_mlx_lm(
    verbose: bool = False,
    dispatch_policy: str = "auto",
)
```

Monkey-patches mlx-lm's attention function to use `flash_attention`.
Compatible with mlx-lm 0.30+.

**Routing within patch:**
- `mask="causal"` (prefill) → `flash_attention(causal=True)`
- `mask=None` (decode) → `flash_attention(causal=False)` or Flash Decode
- Quantized caches, sink tokens, array masks → falls back to original

**Functions:**

| Function | Description |
|----------|-------------|
| `patch_mlx_lm(verbose=False)` | Replace mlx-lm attention globally |
| `unpatch_mlx_lm()` | Restore original attention |
| `is_patched()` → `bool` | Check if patch is active |

```python
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
import mlx_lm

patch_mlx_lm(verbose=True)
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
response = mlx_lm.generate(model, tokenizer, prompt="Hello!")
```

---

## 16. AOT Compilation

### `compile_metallib`

```python
from mlx_mfa import compile_metallib

compile_metallib(
    output_dir: Optional[str] = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> dict
```

Pre-compile common STEEL V2 kernel configurations to precompiled AIR metallibs
(`~/.mlx_mfa/metallib/` by default).  The C++ `ShaderCache` loads them on
subsequent runs, reducing cold-start JIT compilation latency from ~50 ms to
~5 ms per kernel.

**How it works:** Launches subprocesses with `MFA_DEBUG_SHADERS=1` to capture
the JIT-generated Metal source strings, then compiles each to a `.metallib`
via `xcrun metal` / `xcrun metallib`.  The C++ JIT generator remains the
single source of truth — this function only intercepts what the C++ emits.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str \| None` | `~/.mlx_mfa/metallib` | Directory to write `.metallib` files |
| `force` | `bool` | `False` | Recompile even if the metallib already exists |
| `verbose` | `bool` | `True` | Print progress messages per kernel config |

**Returns:** `dict` mapping filename → `True` (compiled/cached) or `False` (failed).

**Compiled configs — standard V2:**

| Config | Filename pattern |
|--------|-----------------|
| D=64  BK=64  f16/bf16  causal/non-causal | `v2_D64_BK64_M{0\|1}_dtype{0\|1}_causal{0\|1}.metallib` |
| D=128 BK=32  f16/bf16  causal/non-causal (M1/M2) | `v2_D128_BK32_M0_dtype{0\|1}_causal{0\|1}.metallib` |
| D=128 BK=64  f16/bf16  causal/non-causal (M3+) | `v2_D128_BK64_M1_dtype{0\|1}_causal{0\|1}.metallib` |

**Compiled configs — V2 D-split:**

| Config | Filename pattern |
|--------|-----------------|
| D=256 BK=32/64 f16/bf16  causal/non-causal | `v2_dsplit_D256_BK{BK}_M{0\|1}_dtype{0\|1}_causal{0\|1}.metallib` |
| D=512 BK=32/64 f16/bf16  causal/non-causal | `v2_dsplit_D512_BK{BK}_M{0\|1}_dtype{0\|1}_causal{0\|1}.metallib` |

**Filename fields:** `dtype0` = float16, `dtype1` = bfloat16; `M1` = M3+ device.

**Requirements:** MFA C++ extension + `xcrun metal` (Xcode Command Line Tools).
Returns `{}` silently if either is absent.

**CLI usage:**

```bash
python -m mlx_mfa.compile_metallib
python -m mlx_mfa.compile_metallib --output-dir /path/to/cache
python -m mlx_mfa.compile_metallib --force   # recompile all
```

**Example:**

```python
from mlx_mfa import compile_metallib

results = compile_metallib(verbose=True)
# [compile_metallib] Compiling: v2_D64_BK64_M0_dtype0_causal1.metallib ... ok
# [compile_metallib] Compiling: v2_D128_BK32_M0_dtype0_causal1.metallib ... ok
# ...
# [compile_metallib] 16/16 configs compiled -> /Users/you/.mlx_mfa/metallib

failed = [k for k, v in results.items() if not v]
print(f"Failed: {failed}")  # [] on success
```

**Notes:**
- Metallibs are loaded automatically by `ShaderCache` on the next run — no
  explicit loading call is needed.
- The fallback chain is: AOT metallib → JIT compilation.  If the metallib is
  absent or incompatible, JIT compilation proceeds transparently.
- Thread-safe: `ShaderCache` uses a mutex around the metallib path lookup.
- The `async_v2.metallib` (hardware DMA) is a separate file in
  `mlx_mfa/precompiled/` and is NOT generated by `compile_metallib`.
  It must be compiled with `bash scripts/build_async_metallib.sh` on macOS ≤15.

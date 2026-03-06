# mlx-mfa

[![PyPI version](https://img.shields.io/pypi/v/mlx-mfa.svg)](https://pypi.org/project/mlx-mfa/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mlx-mfa.svg)](https://pypi.org/project/mlx-mfa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![macOS](https://img.shields.io/badge/macOS-14%2B-blue.svg)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M4-orange.svg)](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/)

**Metal Flash Attention for MLX** — causal attention 1.5–2.9× faster than `mx.fast.scaled_dot_product_attention` on Apple Silicon.

A drop-in replacement for `mx.fast.scaled_dot_product_attention` powered by the **STEEL** (Structured Tiled Execution Engine Layer) kernel: Q loaded once into registers, K/V streamed tile-by-tile, causal tiles skipped entirely.

## Performance (M1 Max, float16, B=1, H=8)

| head_dim | N | non-causal | causal | sliding window 512 |
|:--------:|:-:|:----------:|:------:|:------------------:|
| 64 | 8192 | 1.00× | **2.11×** | — |
| 128 | 4096 | 0.91× | **1.56×** | **3.1×** |
| 128 | 8192 | 0.92× | **1.72×** | **5.7×** |
| 256 | 8192 | 0.50× | 1.00× | — |

Causal speedup is fundamental to STEEL: ~half the K-tiles are skipped (all keys after
the current query position), halving effective work while SDPA still pays full cost.
Sliding-window sparsity scales with `window/N` — at N=8192 and window=512, only 12%
of K-tiles are active.
Full results: [`docs/benchmarks/RESULTS.md`](docs/benchmarks/RESULTS.md).

## Features

- **Drop-in replacement** for `mx.fast.scaled_dot_product_attention`
- **Full autograd** — dQ, dK, dV via custom gradient checkpointing backward
- **All head dims**: 64, 128, 256
- **All dtypes**: float16, bfloat16, float32
- **Causal and non-causal** attention
- **GQA / MQA** — Native Grouped Query Attention (no K/V expansion)
- **Block-sparse attention** — `flash_attention_sparse()` with causal or sliding-window masks
- **Flash Decoding** — split-KV parallelism for single-token decode (N≤4, S≥256); Phase 1 dispatches KV splits in parallel, Phase 2 reduces via log-sum-exp
- **Cross-attention** — N_q != N_kv supported
- **M5+ detection** — `is_m5_plus` flag in `get_device_info()`, reserved stub for Metal 4 tensor API (A19+)
- **Unified KV-cache API** — `flash_attention_kvcache()` consolidates dense, paged, RoPE, ALiBi, sliding-window and continuous batching in one call (v1.0.0)
- **Native sliding window in STEEL** — `flash_attention(..., window_size=(left, right))` applies boundary masking inside the Metal kernel without materializing a mask tensor (v1.0.0)
- **Kernel-level paged KV** — `flash_attention_kvcache(q, pool_k, pool_v, block_table=..., seq_lens=...)` reads K/V tiles directly from the page pool inside the STEEL forward kernel, no separate gather dispatch (v1.0.0)
- **Fused RoPE cache append** — `flash_attention_kvcache_rope_append` rotates new keys before cache append; O(1) rotation cost per decode step (v1.0.0)
- **Return LSE** — `flash_attention(..., return_lse=True)` → `(output, lse [B,H,N])` for speculative decoding and custom reducers (v1.0.0)
- **Graceful fallback** to `mx.fast.scaled_dot_product_attention` when the extension is unavailable or head_dim is unsupported
- **RoPE fusion** — `flash_attention_rope()` with 1D or 3D rotary embeddings (`make_rope_3d_tables`)
- **Variable-length batching** — `flash_attention_varlen()` for packed sequences with `cu_seqlens`
- **Video/VSR mask builders** — `make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask`, `make_segment_mask`, `make_causal_segment_mask`, `make_adaptive_window_mask`
- **Softcap** — `flash_attention(..., softcap=50.0)` applies `tanh(S/cap)*cap` before softmax (Gemma-style)
- **ALiBi** — `flash_attention_alibi(q, k, v, slopes, ...)` for linear position biases without RoPE
- **RoPE non-interleaved** — `flash_attention_rope(..., interleaved=False)` for GPT-NeoX split-halves layout
- **Per-batch cache offsets** — `cache_seqlens` accepts `list[int]` or `mx.array` for heterogeneous batches
- **D_v ≠ D_qk** — graceful fallback when value head_dim differs from query head_dim
- **KV cache append** — `flash_attention_with_kv_cache(q, k_new, v_new, k_cache, v_cache)` → `(out, k, v)`
- **Attention dropout** — `flash_attention(..., dropout_p=0.1)` for training
- **Return attention weights** — `flash_attention(..., return_attn_weights=True)` → `(out, weights [B,H,N,S])`
- **Differentiable varlen** — `flash_attention_varlen()` supports `mx.grad()` via `mx.custom_function` (v0.9.3)
- **Paged attention backward** — `flash_attention_paged()` computes dQ correctly via Metal gather + per-seq vjp (v0.9.3)
- **Varlen packed formats** — `flash_attention_varlen_qkv_packed()` / `flash_attention_varlen_kv_packed()` for fused-tensor varlen (v0.9.3)
- **D=256 D-split backward** — STEEL dQ/dK/dV with BD_HALF=128 sub-tiles fits D=256 in registers (v0.9.2)

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | 14+ (Sonoma) with Metal |
| Python | 3.10+ |
| MLX | >= 0.18.0 |
| nanobind | >= 2.0 |
| Apple Silicon | M1, M2, M3, M4 (M5+ stub) |

## Installation

```bash
pip install mlx-mfa
```

The wheel includes the pre-compiled Metal C++ extension for Apple Silicon (macOS 14+, Python 3.10+).

**From source** (for development):

```bash
# 1. Install build dependencies
pip install mlx nanobind scikit-build-core

# 2. Validate your environment
python scripts/check_env.py

# 3. Install with C++ build
pip install -e .
```

## Quick Start

```python
import mlx.core as mx
from mlx_mfa import flash_attention

B, H, N, D = 1, 8, 2048, 128
q = mx.random.normal((B, H, N, D)).astype(mx.float16)
k = mx.random.normal((B, H, N, D)).astype(mx.float16)
v = mx.random.normal((B, H, N, D)).astype(mx.float16)

# Drop-in: identical API to mx.fast.scaled_dot_product_attention
out = flash_attention(q, k, v, scale=None, causal=True)
mx.eval(out)
```

### Training (autograd)

```python
def loss_fn(q, k, v):
    return mx.sum(flash_attention(q, k, v, causal=True) ** 2)

grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
dq, dk, dv = grad_fn(q, k, v)
mx.eval(dq, dk, dv)
```

### Grouped Query Attention (GQA / MQA)

```python
Hq, Hkv = 8, 2
q  = mx.random.normal((1, Hq,  N, D))
k  = mx.random.normal((1, Hkv, N, D))
v  = mx.random.normal((1, Hkv, N, D))

out = flash_attention(q, k, v)  # native GQA — no K/V expansion needed
```

## API Reference

### `flash_attention(q, k, v, scale=None, causal=False, softcap=0.0, dropout_p=0.0, return_attn_weights=False, stream=None)`

Compute scaled dot-product attention.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H, N, D]` | Query tensor |
| `k` | `mx.array [B, Hkv, S, D]` | Key tensor (GQA: Hkv divides H) |
| `v` | `mx.array [B, Hkv, S, Dv]` | Value tensor (Dv may differ from D → SDPA fallback) |
| `scale` | `float or None` | Attention scale. Defaults to `1/sqrt(D)` |
| `causal` | `bool` | Apply causal masking |
| `softcap` | `float` | Tanh softcapping factor `cap` (0.0 = disabled) |
| `dropout_p` | `float` | Softmax dropout probability (0.0 = disabled; training only) |
| `return_attn_weights` | `bool` | If True, returns `(output, attn_weights)` tuple |
| `stream` | `mx.Stream or None` | MLX stream (honoured on fallback path) |

Returns `mx.array [B, H, N, D]` normally, or `(mx.array, mx.array [B, H, N, S])` when `return_attn_weights=True`.

Raises `ValueError` if inputs are not 4-D, if `q/k` have mismatched `head_dim`, or if the GQA ratio is non-integer.

---

### `flash_attention_rope(q, k, v, rotary_cos=None, rotary_sin=None, scale=None, causal=False, cache_seqlens=0, rope_3d=None, interleaved=True, stream=None)`

Flash Attention with in-kernel RoPE fusion. Applies rotary position embeddings inside the Metal kernel, eliminating a separate elementwise pass over Q/K.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q, k, v` | `mx.array [B, H, N, D]` | Standard attention inputs |
| `rotary_cos` | `mx.array [N, D/2]` | Cosine table from `make_rope_3d_tables` or precomputed 1D tables |
| `rotary_sin` | `mx.array [N, D/2]` | Sine table |
| `cache_seqlens` | `int or list[int] or mx.array` | KV cache offsets (for decode; 0 = prefill) |
| `rope_3d` | `dict or None` | 3D RoPE tables: `{"cos": ..., "sin": ..., "grid_shape": (T,H,W)}` |
| `interleaved` | `bool` | `True` = adjacent pair rotation (LLaMA); `False` = split-halves (GPT-NeoX) |

Returns `mx.array [B, H, N, D]`.

---

### `flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale=None, causal=False, stream=None)`

Variable-length batched attention. Multiple sequences of different lengths packed into a single `B=1` tensor; each sequence attends independently. **Differentiable** via `mx.custom_function` — supports `mx.grad()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q, k, v` | `mx.array [1, H, total_tokens, D]` | Packed tensors |
| `cu_seqlens_q` | `mx.array int32 [num_seqs+1]` | Cumulative Q lengths; `[0, ..., total_q]` |
| `cu_seqlens_k` | `mx.array int32 [num_seqs+1]` | Cumulative KV lengths |
| `max_seqlen_q` | `int` | Maximum Q sequence length |
| `max_seqlen_k` | `int` | Maximum KV sequence length |

Returns `mx.array [1, H, total_tokens, D]`.

---

### `flash_attention_kvcache(q, k_cache, v_cache, *, block_table=None, seq_lens=None, block_size=16, scale=None, causal=True, softcap=0.0, alibi_slopes=None, window_size=None, rotary_cos=None, rotary_sin=None, cache_seqlens=0, interleaved=True, rotary_dim=None, cache_batch_idx=None, stream=None)`

**Unified KV-cache attention** — recommended entry point for all inference workloads.  Supports dense and paged KV caches, RoPE, ALiBi, softcap, and sliding window in a single call.

**Dense mode** (complete accumulated cache as positional args):

```python
# Full KV sequence — grow via concatenation each decode step
out = flash_attention_kvcache(q, k_full, v_full, scale=scale, causal=True)
```

**Paged mode** (pool arrays as k_cache/v_cache, plus block_table/seq_lens):

```python
# k_cache / v_cache = page pool [num_pages, block_size, H_kv, D]
out = flash_attention_kvcache(
    q, pool_k, pool_v,
    block_table=block_table,   # int32 [B, max_pages_per_seq]; -1 = padding
    seq_lens=seq_lens,         # int32 [B] — true KV length per sequence
    block_size=64,
    scale=scale, causal=True,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H_q, N_q, D]` | Query tensor |
| `k_cache, v_cache` | `mx.array` | Dense: `[B, H_kv, S, D]`. Paged: pool `[num_pages, block_size, H_kv, D]` |
| `block_table` | `mx.array int32 [B, max_pages]` or `None` | Activates paged mode; `-1` = unused slot |
| `seq_lens` | `mx.array int32 [B]` or `None` | True KV length per sequence in paged mode |
| `block_size` | `int` | Tokens per page (must match pool shape) |
| `scale` | `float or None` | Attention scale. Defaults to `1/sqrt(D)` |
| `causal` | `bool` | Causal masking |
| `softcap` | `float` | Tanh softcap factor (0.0 = disabled) |
| `alibi_slopes` | `mx.array [H]` or `None` | ALiBi per-head slopes |
| `window_size` | `tuple (left, right)` or `None` | Sliding-window attention |
| `rotary_cos/sin` | `mx.array [N, D/2]` or `None` | RoPE tables applied to Q |
| `cache_seqlens` | `int, list[int], or mx.array` | KV cache offsets for decode |
| `interleaved` | `bool` | RoPE rotation layout (`True` = LLaMA, `False` = GPT-NeoX) |
| `rotary_dim` | `int or None` | Partial RoPE: rotate only first `rotary_dim` dimensions |
| `cache_batch_idx` | `mx.array int32 [B]` or `None` | Continuous batching: maps batch → cache slot |

Returns `mx.array [B, H_q, N_q, D]`.

---

### `flash_attention_with_kv_cache(q, k_new, v_new, k_cache=None, v_cache=None, scale=None, causal=True, softcap=0.0, stream=None)`

Compute attention and update the KV cache in a single call. Concatenates new K/V tokens onto the cache, runs `flash_attention`, and returns `(output, k_full, v_full)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H, N_new, D]` | Query for new tokens |
| `k_new, v_new` | `mx.array [B, H, N_new, D]` | New token KV |
| `k_cache, v_cache` | `mx.array [B, H, S_cache, D] or None` | Existing cache (pass `None` for first step) |

Returns `(output [B,H,N_new,D], k_full [B,H,S_cache+N_new,D], v_full)`.

---

### `flash_attention_paged(q, k_pages, v_pages, block_table, seq_lens, *, scale=None, causal=False, block_size=16, stream=None)`

Paged KV cache attention with Metal gather. Gathers K/V from a block pool via `mfa_paged_kv_gather` Metal kernel, then runs `flash_attention`. Supports autograd: dQ is correct; dK/dV pages are zeros (caches are not trainable parameters).

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H_q, N_q, D]` | Query tensor |
| `k_pages, v_pages` | `mx.array [num_blocks, block_size, H_kv, D]` | Block pool |
| `block_table` | `mx.array int32 [B, max_blocks_per_seq]` | Logical→physical block map; `-1` = padding |
| `seq_lens` | `mx.array int32 [B]` | Actual KV token count per sequence |
| `block_size` | `int` | Tokens per page (must match pool shape) |

Returns `mx.array [B, H_q, N_q, D]`.

---

### `flash_attention_qkv_packed(qkv, *, scale=None, causal=False, num_heads=None, num_kv_heads=None, stream=None)`

Attention from a fused QKV tensor. Accepts `[B, N, 3*H*D]` (flat) or `[B, H, N, 3, D]` (head-first). Returns `[B, H, N, D]`. `num_heads` is required for flat layout.

### `flash_attention_kv_packed(q, kv, *, scale=None, causal=False, num_kv_heads=None, stream=None)`

Attention from a fused KV tensor. Accepts `[B, S, 2*H_kv*D]` (flat) or `[B, H_kv, S, 2, D]` (head-first). Returns `[B, H_q, N, D]`. `num_kv_heads` is required for flat layout.

### `flash_attention_varlen_qkv_packed(qkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, scale=None, causal=False, num_heads=None, num_kv_heads=None, stream=None)`

Varlen attention from a packed QKV tensor. Unpacks into Q/K/V then calls `flash_attention_varlen`. Layouts: `[1, H, total, 3, D]` (head-first) or `[1, total, 3*H*D]` (flat). Returns `[1, H_q, total, D]`.

### `flash_attention_varlen_kv_packed(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, scale=None, causal=False, num_kv_heads=None, stream=None)`

Varlen attention from a packed KV tensor. Unpacks K/V then calls `flash_attention_varlen`. Layouts: `[1, H_kv, total_kv, 2, D]` (head-first) or `[1, total_kv, 2*H_kv*D]` (flat). Returns `[1, H_q, total_q, D]`.

---

### `flash_attention_sparse(q, k, v, block_mask, scale=None, causal=False, stream=None)`

Block-sparse Flash Attention — only computes tiles where `block_mask[q_tile, k_tile] == True`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H, N, D]` | Query. **float16 or bfloat16 only.** |
| `k` | `mx.array [B, H, S, D]` | Key |
| `v` | `mx.array [B, H, S, D]` | Value |
| `block_mask` | `mx.array[bool] [NQ, NK]` | Active tile map. Use `make_causal_block_mask` or `make_sliding_window_mask`. |
| `scale` | `float or None` | Attention scale. Defaults to `1/sqrt(D)` |
| `causal` | `bool` | Additional token-level causal masking within active blocks |

`NQ = ceil(N / BQ)`, `NK = ceil(S / BK)` where `(BQ, BK)` comes from `_steel_block_config(D)`.

Raises `ValueError` for float32 input or wrong `block_mask` shape.

> **Backward pass limitation:** Gradients are computed via dense `mx.fast.sdpa` with a float additive bias (correct, but no sparsity speedup in the backward). A native sparse backward is planned.

---

### `PagedKVCache(num_blocks, block_size, H, D, dtype=mx.float16)`

Fixed-size block pool for paged KV cache management. Eliminates padding waste when batch sequences have different lengths. Pool layout: `[num_blocks, block_size, H_kv, D]`.

```python
cache = PagedKVCache(num_blocks=64, block_size=16, H=4, D=128)
k_new = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
v_new = mx.random.normal((1, 4, 32, 128)).astype(mx.float16)
cache.append(k_new, v_new, seq_id=0)
k_seq, v_seq = cache.gather(seq_id=0)   # [1, 4, 32, 128]

# Feed into flash_attention_paged:
bt, sl = cache.block_table_and_seq_lens([0])
out = flash_attention_paged(q, cache.k_pool, cache.v_pool, bt, sl)
```

Methods: `append(k, v, seq_id)`, `gather(seq_id) → (k, v)`, `block_table_and_seq_lens(seq_ids) → (block_table, seq_lens)`, `free(seq_id)`.

---

### Mask builders

`make_causal_block_mask` and `make_sliding_window_mask` are the most common; the full set of 15 mask builders is listed below.

#### `make_causal_block_mask(seq_len, head_dim=128) -> mx.array`

Returns a lower-triangular block mask `[NQ, NK]` (dtype `bool`) matching the STEEL tile size for `head_dim`. Combine with `causal=True` for exact token-level causal masking:

```python
mask = make_causal_block_mask(N, head_dim=128)
out  = flash_attention_sparse(q, k, v, mask, causal=True)
```

#### `make_sliding_window_mask(seq_len, window_size, head_dim=128, causal=False) -> mx.array`

Returns a sliding-window block mask. Each Q-tile attends only to K-tiles within `window_size` tokens.

```python
mask = make_sliding_window_mask(4096, window_size=512)
out  = flash_attention_sparse(q, k, v, mask)
```

#### Remaining mask builders

| Function | Key parameters | Use case |
|----------|---------------|----------|
| `make_spatial_2d_mask(height, width, spatial_radius, head_dim, patch_size)` | `spatial_radius` in patch units | Image/frame Chebyshev locality |
| `make_spatial_3d_mask(height, width, num_frames, spatial_radius, temporal_radius, ...)` | Both radii | Video spatio-temporal locality |
| `make_topk_spatial_mask(q, k, top_k, head_dim)` | `top_k` K-tiles per Q-tile | Content-aware top-k scoring |
| `make_segment_mask(segment_lengths, head_dim)` | `segment_lengths: list[int]` | Block-diagonal; each segment isolated |
| `make_causal_segment_mask(segment_lengths, head_dim)` | Same as above | Block-diagonal + causal within each segment |
| `make_adaptive_window_mask(height, width, num_frames, base_window_h/w/t, train_resolution, inference_resolution, ...)` | Scales window with resolution ratio | SeedVR2-style RoPE aliasing prevention |
| `make_lcsa_mask(q, k, height, width, spatial_radius, top_k, ...)` | `spatial_radius` + `top_k` | FlashVSR LCSA (spatial window ∩ top-k) |
| `make_axial_spatial_mask(height, width, num_frames, head_dim, ...)` | Optional `spatial_radius` | Same-frame attention (spatial axis only) |
| `make_axial_temporal_mask(height, width, num_frames, head_dim, ...)` | Optional `temporal_radius`, `causal` | Same-position across frames (temporal axis) |
| `make_dilated_temporal_mask(height, width, num_frames, dilation_rate, local_window, ...)` | `dilation_rate` | Dilated long-range temporal |
| `make_sink_window_mask(seq_len, window_size, num_sink_tokens, head_dim, causal)` | `num_sink_tokens` | StreamingLLM: sinks + sliding window |
| `make_reference_frame_mask(height, width, num_frames, reference_frames, ...)` | `reference_frames: list[int]` | Global reference frames + local context |
| `make_cross_stream_mask(n_tokens_q, n_tokens_kv, head_dim, pattern, ...)` | `pattern`: "full"/"temporal"/"segment" | Rectangular Q≠KV cross-attention (LTX-2) |

All mask builders return `mx.array[bool] [NQ_tiles, NK_tiles]` for use with `flash_attention_sparse`.

---

### `make_rope_3d_tables(grid_h, grid_w, num_frames, d_h=None, d_w=None, d_t=None, head_dim=128, theta=10000.0) -> tuple[mx.array, mx.array]`

Build 3D RoPE cosine/sine tables for video attention. Returns `(cos, sin)` of shape `[N, D/2]` where `N = grid_h * grid_w * num_frames`. Sub-bands are allocated proportionally across height/width/temporal axes.

```python
cos, sin = make_rope_3d_tables(grid_h=16, grid_w=16, num_frames=8, head_dim=128)
out = flash_attention_rope(q, k, v, cos, sin, rope_3d={"cos": cos, "sin": sin, "grid_shape": (8, 16, 16)})
```

---

### `is_mfa_available() -> bool`

Returns `True` if the MFA C++ extension compiled and loaded successfully.

### `get_device_info() -> dict`

Returns Metal GPU hardware information.

```python
from mlx_mfa import get_device_info
info = get_device_info()
# {
#   'device_name': 'Apple M1 Max',
#   'gpu_family_gen': 13,
#   'is_m3_plus': False,
#   'is_m5_plus': False,
#   'chip_name': 'M1',
#   'extension_available': True
# }
```

### `get_supported_configs() -> dict`

Returns the set of (head_dim, dtype) configurations that use the MFA kernel.

---

## mlx-lm Integration

Use STEEL attention with any mlx-lm model in two lines:

```python
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
patch_mlx_lm()

# All subsequent mlx-lm models automatically use STEEL attention
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
generate(model, tokenizer, prompt="Hello world", verbose=True)
```

The patch transparently routes `mask="causal"` (prefill) and `mask=None` (decode) through
the STEEL kernel. It falls back to the original mlx_lm SDPA for quantized KV caches,
attention sinks, and unsupported configs. Call `unpatch_mlx_lm()` to restore.

**Expected speedup:** 1.5–2.1× on causal prefill (D=128, f16); decode step is memory-bound
so speedup is minimal there.

## Testing

```bash
# All tests
pytest tests/ -v

# Fallback path only (no C++ build required)
pytest tests/ -v -k "Fallback or PublicAPI"

# MFA kernel tests
pytest tests/ -v -k "MFAKernel"

# Backward pass tests
pytest tests/ -v -k "Backward"

# Edge case tests (GQA, N=1, cross-attention)
pytest tests/ -v -k "EdgeCase or BackwardEdge"
```

Expected: **307 pytest runs** across 42 test classes (~45 skip without C++ build).

## Supported Configurations

| head_dim | float16 | bfloat16 | float32 | Causal | GQA |
|----------|---------|----------|---------|--------|-----|
| 64       | yes     | yes      | yes     | yes    | yes |
| 128      | yes     | yes      | yes     | yes    | yes |
| 256      | yes     | yes      | yes     | yes    | yes |
| Other    | fallback| fallback | fallback| yes    | yes |

## GPU Generation Notes

| Chip | Silicon gen | Block params |
|------|-------------|--------------|
| M1 family | 13 | `preferAsyncLoad` |
| M2 family | 14 | `preferAsyncLoad` |
| M3 family | 15 | `preferAsyncCache` |
| M4 family | 16 | `preferAsyncCache` |

The silicon generation is derived from MLX's architecture string (e.g. `applegpu_g13s` -> 13).

## Roadmap

| Track | Description | Status |
|-------|-------------|--------|
| 1.1 | Project scaffold | Done |
| 1.2 | Extract MFA kernels from ccv | Done |
| 1.3 | Decouple from ccv types | Done |
| 1.4 | Forward pass (all D, dtypes, causal) | Done |
| 1.5 | Backward pass (full autograd) | Done |
| 4   | Production-ready: GQA, public API, CI | Done |
| 5   | STEEL forward kernel (1.5–2.9× causal) | **Done (v0.1.0)** |
| B   | Block-sparse attention (`flash_attention_sparse`) | **Done (v0.2.0)** |
| C   | Native GQA kernel (gqa_factor in STEEL, no mx.repeat) | **Done (v0.3.0)** |
| D   | mlx-lm integration (`patch_mlx_lm`) | **Done (v0.3.0)** |
| F   | M3+ architecture routing (BK=32 for M3+) | **Done (v0.4.0)** |
| G   | Sparse backward (tiled FA-2 dQ/dK/dV) | **Done (v0.4.0)** |
| H   | Flash Decoding (split-KV, N≤4 decode) | **Done (v0.5.0)** |
| I   | M5+ detection stub (gen≥17, is_m5_plus) | **Done (v0.5.0)** |
| K   | Quantized KV Cache (Q4/Q8 dequantize before STEEL) | **Done (v0.6.0)** |
| L   | RoPE Fusion (in-kernel rotary embeddings, `flash_attention_rope`) | **Done (v0.6.0)** |
| M   | Paged Attention design document (`docs/PAGED_ATTENTION_DESIGN.md`) | **Done (v0.6.0)** |
| N1  | STEEL native backward kernel (dQ/dK/dV in Metal) | **Done (v0.9.0)** |
| N2  | Native sparse backward (block-sparse dQ/dK/dV) | Planned (v1.0) |
| O   | Spatial 2D/3D block masks + segment masks + adaptive window | **Done (v0.7.0)** |
| P   | Variable-length batching (`flash_attention_varlen`, split-concat) | **Done (v0.7.0)** |
| R   | 3D RoPE table construction + `flash_attention_rope(rope_3d=...)` | **Done (v0.7.0)** |
| U   | LCSA composite mask (FlashVSR) | **Done (v0.8.0)** |
| V   | Axial / factored attention masks | **Done (v0.8.0)** |
| W   | Dilated temporal mask | **Done (v0.8.0)** |
| X   | Sink tokens + reference frame masks | **Done (v0.8.0)** |
| Y   | Cross-stream mask (LTX-2 dual-stream DiT) | **Done (v0.8.0)** |
| AA  | Softcapping (Gemma 2 / Grok) | **Done (v0.8.0)** |
| AB  | ALiBi (Falcon, MPT, BLOOM) | **Done (v0.8.0)** |
| AC  | RoPE non-interleaved (GPT-NeoX) | **Done (v0.8.0)** |
| AD  | Per-batch cache_seqlens (list/array) | **Done (v0.8.0)** |
| AE  | D_v ≠ D_qk graceful fallback | **Done (v0.8.0)** |
| AF  | Fused KV cache append (`flash_attention_with_kv_cache`) | **Done (v0.8.0)** |
| AG  | Attention dropout (training) | **Done (v0.8.0)** |
| AH  | Return attention weights | **Done (v0.8.0)** |
| BA  | STEEL native backward + varlen Metal kernel | **Done (v0.9.0)** |
| BB  | Paged KV decode (`PagedKVCache`), packed QKV/KV formats | **Done (v0.9.0)** |
| CA  | Vec4 block loads (float4/half4 aligned tile reads) | **Done (v0.9.1)** |
| CB  | `mx.compile` for Python fallback paths | **Done (v0.9.1)** |
| CC  | Persistent multi-Q-block kernel (4× Q-blocks/threadgroup) | **Done (v0.9.1)** |
| CD  | GQA support in STEEL backward (`gqa_factor` baked as `#define`) | **Done (v0.9.1)** |
| CF  | Double-buffer ping-pong (K_smem⊕V_smem, 4→2 barriers/K-tile, D≤128) | **Done (v0.9.1)** |
| CE  | D=256 backward D-split (BD_HALF=128) | **Done (v0.9.2)** |
| DA  | Fix GQA backward Python guard | **Done (v0.9.2)** |
| DC  | `mx.compile` for `_apply_rope_mlx` | **Done (v0.9.2)** |
| EA  | Differentiable `flash_attention_varlen` (`mx.custom_function`) | **Done (v0.9.3)** |
| EB  | Metal paged KV gather kernel + `flash_attention_paged` backward | **Done (v0.9.3)** |
| EC  | `flash_attention_varlen_qkv_packed` + `flash_attention_varlen_kv_packed` | **Done (v0.9.3)** |
| FA  | Unified KV-cache API (`flash_attention_kvcache`) | **Done (v1.0.0)** |
| FB  | Native sliding-window in STEEL kernel | **Done (v1.0.0)** |
| FC  | Fused RoPE cache append (`flash_attention_kvcache_rope_append`) | **Done (v1.0.0)** |
| FD  | Kernel-level paged KV STEEL forward + Flash Decode path | **Done (v1.0.0)** |
| FX  | `return_lse`, `cache_batch_idx`, `rotary_dim` additions | **Done (v1.0.0)** |
| Q   | Metal 4 tensor API (cooperative tensors, M5+/A19+ only) | Planned (v1.1+) |

## References

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) - Algorithm, blocking tables, pseudocode
- [liuliu/ccv mfa subtree](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) - C++ source (production-grade)
- [MLX C++ extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) - MLX extension API

## License

MIT

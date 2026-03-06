# mlx-mfa

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
# 1. Install build dependencies
pip install mlx nanobind scikit-build-core

# 2. Validate your environment
python scripts/check_env.py

# 3. Install (builds the C++ extension)
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
#   'chip_name': 'M1',
#   'extension_available': True
# }
```

### `get_supported_configs() -> dict`

Returns the set of (head_dim, dtype) configurations that use the MFA kernel.

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

### `make_causal_block_mask(seq_len, head_dim=128) -> mx.array`

Returns a lower-triangular block mask `[NQ, NK]` (dtype `bool`) matching the STEEL tile size for `head_dim`. Combine with `causal=True` for exact token-level causal masking:

```python
mask = make_causal_block_mask(N, head_dim=128)
out  = flash_attention_sparse(q, k, v, mask, causal=True)
# Identical to flash_attention(q, k, v, causal=True) but skips upper triangle tiles.
```

### `make_sliding_window_mask(seq_len, window_size, head_dim=128, causal=False) -> mx.array`

Returns a sliding-window block mask. Each Q-tile attends only to K-tiles within `window_size` tokens.

```python
mask = make_sliding_window_mask(4096, window_size=512)
out  = flash_attention_sparse(q, k, v, mask)
```

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

Expected: 152 tests collected (45 active + 107 extension-gated/skipped without C++ build).

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
| CE  | D=256 backward multi-pass tiling | Deferred to v1.0 |
| PG  | Paged KV decode — native Metal kernel (full block-table gather) | Planned (v1.0) |
| Q   | Metal 4 tensor API (cooperative tensors, M5+/A19+ only) | Planned (v1.0+) |

## References

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) - Algorithm, blocking tables, pseudocode
- [liuliu/ccv mfa subtree](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) - C++ source (production-grade)
- [MLX C++ extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) - MLX extension API

## License

MIT

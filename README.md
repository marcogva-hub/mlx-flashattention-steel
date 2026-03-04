# mlx-mfa

**Metal Flash Attention for MLX** — causal attention 1.5–2.9× faster than `mx.fast.scaled_dot_product_attention` on Apple Silicon.

A drop-in replacement for `mx.fast.scaled_dot_product_attention` powered by the **STEEL** (Structured Tiled Execution Engine Layer) kernel: Q loaded once into registers, K/V streamed tile-by-tile, causal tiles skipped entirely.

## Performance (M1 Max, float16, B=1, H=8)

| head_dim | N | non-causal | causal |
|:--------:|:-:|:----------:|:------:|
| 64 | 4096 | 0.99× | **1.96×** |
| 64 | 8192 | 0.91× | **2.13×** |
| 128 | 4096 | 0.91× | **1.54×** |
| 128 | 8192 | 0.94× | **2.62×** |
| 256 | 4096 | 0.49× | **1.46×** |
| 256 | 8192 | 0.41× | **1.16×** |

Causal speedup is fundamental to STEEL: ~half the K-tiles are skipped (all keys after
the current query position), halving effective work while SDPA still pays full cost.
Full results: [`docs/benchmarks/RESULTS.md`](docs/benchmarks/RESULTS.md).

## Features

- **Drop-in replacement** for `mx.fast.scaled_dot_product_attention`
- **Full autograd** — dQ, dK, dV via custom gradient checkpointing backward
- **All head dims**: 64, 128, 256
- **All dtypes**: float16, bfloat16, float32
- **Causal and non-causal** attention
- **GQA / MQA** — Grouped Query Attention via repeat-KV fallback
- **Cross-attention** — N_q != N_kv supported
- **Graceful fallback** to `mx.fast.scaled_dot_product_attention` when the extension is unavailable or head_dim is unsupported

## Requirements

| Requirement | Version |
|-------------|---------|
| macOS | 14+ (Sonoma) with Metal |
| Python | 3.10+ |
| MLX | >= 0.18.0 |
| nanobind | >= 2.0 |
| Apple Silicon | M1, M2, M3, M4 |

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

out = flash_attention(q, k, v)  # automatically tiles k/v to match Hq
```

## API Reference

### `flash_attention(q, k, v, scale=None, causal=False, stream=None)`

Compute scaled dot-product attention.

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | `mx.array [B, H, N, D]` | Query tensor |
| `k` | `mx.array [B, Hkv, S, D]` | Key tensor |
| `v` | `mx.array [B, Hkv, S, D]` | Value tensor |
| `scale` | `float or None` | Attention scale. Defaults to `1/sqrt(D)` |
| `causal` | `bool` | Apply causal masking |
| `stream` | `mx.Stream or None` | MLX stream (honoured on fallback path) |

Returns `mx.array [B, H, N, D]` in the same dtype as `q`.

Raises `ValueError` if inputs are not 4-D, if `q/k/v` have mismatched `head_dim`, or if the GQA ratio is non-integer.

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

Expected: 57 tests total (6 fallback + 5 public API + 10 forward + 8 backward + 8 edge + 4 backward edge + 6 sparse API + 10 sparse kernel).

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

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | Project scaffold | Done |
| 1.2 | Extract MFA kernels from ccv | Done |
| 1.3 | Decouple from ccv types | Done |
| 1.4 | Forward pass (all D, dtypes, causal) | Done |
| 1.5 | Backward pass (full autograd) | Done |
| 4   | Production-ready: GQA, public API, CI | Done |
| 5   | STEEL forward kernel (1.5–2.9× causal) | **Done (v0.1.0)** |
| B   | Block-sparse attention (`flash_attention_sparse`) | **Done (v0.2.0)** |
| 6   | STEEL backward kernel | Planned |
| 6   | Native sparse backward (no dense fallback) | Planned |
| 6   | Native GQA kernel (no tiling) | Planned |
| 6   | Flash Decoding for long contexts | Planned |

## References

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) - Algorithm, blocking tables, pseudocode
- [liuliu/ccv mfa subtree](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) - C++ source (production-grade)
- [MLX C++ extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html) - MLX extension API

## License

MIT

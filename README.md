# mlx-mfa

**Metal Flash Attention for MLX** — high-performance FlashAttention on Apple Silicon.

Based on [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) kernels (via [liuliu/ccv](https://github.com/liuliu/ccv) C++ port).

## Status

**Phase 1.1** — Project scaffold complete. Metal kernel integration in progress.

Currently falls back to `mx.fast.scaled_dot_product_attention` for all operations.

## Installation

```bash
# Prerequisites
pip install mlx nanobind scikit-build-core

# Check environment
python scripts/check_env.py

# Install (editable)
pip install -e .
```

## Usage

```python
from mlx_mfa import flash_attention

# Drop-in replacement for mx.fast.scaled_dot_product_attention
out = flash_attention(q, k, v, scale=None, causal=False)
```

**Shapes**: `q`, `k`, `v` are `[batch, heads, seq_len, head_dim]`.

**Supported head_dim**: 64, 128, 256 (falls back to MLX SDPA otherwise).

**Supported dtypes**: float16, bfloat16, float32.

## Architecture

```
Python API (mlx_mfa.flash_attention)
    |
    +-- fallback --> mx.fast.scaled_dot_product_attention
    |
    +-- MFA path --> C++ Extension (nanobind)
                         |
                         +-- MFAttention Primitive (eval_gpu / vjp)
                         |
                         +-- ShaderCache (JIT Metal compilation)
                         |
                         +-- Metal GPU Kernels
```

## Development

```bash
# Run tests (fallback path works without compilation)
pytest tests/ -v

# Run benchmarks
python benchmarks/bench_attention.py --head-dim 128 --seq-len 1024 4096
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | Project scaffold | Done |
| 1.2 | Extract MFA kernels from ccv | TODO |
| 1.3 | Decouple from ccv types | TODO |
| 1.4 | Forward pass integration | TODO |
| 2 | Benchmarks & tuning | TODO |
| 3 | Backward pass | TODO |
| 4 | Production-ready | TODO |

## License

MIT

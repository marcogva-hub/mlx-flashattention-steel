# mlx-mfa — Metal Flash Attention for MLX

[![PyPI version](https://img.shields.io/pypi/v/mlx-mfa.svg)](https://pypi.org/project/mlx-mfa/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast attention for Apple Silicon. Drop-in replacement for
`mx.fast.scaled_dot_product_attention` with automatic STEEL V2 routing,
sliding window, block-sparse, GQA, RoPE, ALiBi, and int8 SageAttention.

---

## Quick Start

```bash
pip install mlx-mfa
python -m mlx_mfa info        # verify installation + device info
python -m mlx_mfa calibrate   # calibrate thresholds for your GPU (~2 min)
```

```python
import mlx.core as mx
from mlx_mfa import flash_attention

q = mx.random.normal((1, 8, 4096, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, 4096, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 4096, 128)).astype(mx.float16)

out = flash_attention(q, k, v, causal=True)  # STEEL V2: ~1.7× SDPA at N=4096
```

---

## Performance (v2.5.0, M1 Max, B=2 H=8 f16)

> Regenerate: `python benchmarks/bench_v2_final.py --save`

### Dense Causal — STEEL V2 vs SDPA

| Config | V2 (ms) | SDPA (ms) | Speedup |
|--------|--------:|----------:|--------:|
| D=64  N=4096  causal | 5.5 | 10.9 | **2.0×** |
| D=64  N=8192  causal | 19.7 | 42.0 | **2.1×** |
| D=128 N=4096  causal | 11.7 | 19.1 | **1.6×** |
| D=128 N=8192  causal | 43.5 | 74.7 | **1.7×** |
| D=128 N=16384 causal | 186  | 309  | **1.7×** |
| D=256 N=8192  causal | 157  | 146  | 0.9× (SDPA wins → auto-routes to SDPA) |

### Sliding Window — MFA vs Full SDPA

| Config | MFA (ms) | SDPA (ms) | Speedup |
|--------|----------:|----------:|--------:|
| D=64  N=8192  win=512  | 3.0 | 40.9 | **13.7×** |
| D=128 N=8192  win=512  | 6.2 | 73.9 | **12.0×** |
| D=128 N=8192  win=256  | 3.5 | 72.9 | **20.9×** |
| D=128 N=4096  win=256  | 2.4 | 19.0 |  **7.8×** |

### SageAttention (int8 Q/K)

Sage halves Q/K memory bandwidth. Speedup is meaningful at long sequences
where bandwidth dominates; at short sequences the Python quantization overhead
dominates. Use `QuantizedKVCache` to amortize this cost across decode steps.

---

## Features

- **STEEL V2 forward** — sequential K/V phases share threadgroup memory, halving K-tile
  iterations; gen-aware config (M3+/M4 uses BK=64 for D=128)
- **V2 split-K** — under-occupied grids (single-batch decode) are split across more threadgroups
- **Flash Decode** — two-phase split-KV for N_q ≤ 4 decode steps
- **Causal tile-skip** — skips upper-triangular K-tiles entirely (no masking overhead)
- **Sliding window** — `window_size=(left, right)` skips out-of-window K-tiles; 5–20× vs SDPA
- **Block-sparse attention** — `block_mask` tensor skips entire tiles at zero warp cost
- **Native GQA** — kernel computes Q-head-to-KV-head mapping without repeat; no copy
- **RoPE fusion** — RoPE applied in-kernel for prefill and decode with `flash_attention_rope_unified`
- **ALiBi** — per-head linear position bias
- **Softcap** — tanh softcapping (Gemma-2, Grok) fused in-kernel
- **SageAttention** — int8 quantized Q/K with smooth-K and sliding window
- **QuantizedKVCache** — O(1) per-step quantize; pre-stores K as int8 for incremental decode
- **Variable-length** — packed sequences with `cu_seqlens` (training)
- **Paged KV** — page-pool KV cache with `block_table` (multi-request serving)
- **Autograd** — full `mx.vjp` support via STEEL backward kernels (D≤512, f16/bf16)
- **Smart dispatch** — `backend="auto"` routes to STEEL V2 only when faster than SDPA
- **Auto-calibration** — `calibrate_dispatch()` benchmarks your device and saves thresholds
- **mlx-lm integration** — `patch_mlx_lm()` replaces attention in Llama/Mistral/Qwen models

---

## Installation

```bash
# From PyPI (pure Python + JIT-compiled Metal shaders, no pre-built binary)
pip install mlx-mfa

# From source (builds C++ extension)
git clone https://github.com/marcogva-hub/mlx-flashattention-steel.git mlx-mfa-v2
cd mlx-mfa-v2
pip install -e .
```

**Requirements:** macOS arm64 · Python 3.10+ · mlx ≥ 0.18.0 · Xcode Command Line Tools

---

## Usage

### Standard attention

```python
from mlx_mfa import flash_attention, DispatchPolicy

# auto: STEEL V2 when faster, SDPA otherwise
out = flash_attention(q, k, v, causal=True)

# GQA: q has H_q heads, k/v have H_kv heads (H_q must divide H_q)
out = flash_attention(q_32h, k_8h, v_8h, causal=True)

# Sliding window
out = flash_attention(q, k, v, causal=True, window_size=(512, -1))

# Softcap (Gemma-2 style)
out = flash_attention(q, k, v, causal=True, softcap=50.0)

# Force backend
out = flash_attention(q, k, v, backend=DispatchPolicy.MFA)
out = flash_attention(q, k, v, backend=DispatchPolicy.SDPA)
out = flash_attention(q, k, v, backend=DispatchPolicy.SAGE)   # int8 Q/K
```

### RoPE attention

```python
from mlx_mfa import flash_attention_rope_unified

out = flash_attention_rope_unified(q, k, v, cos, sin, causal=True)
# append mode (decode + cache extend)
out = flash_attention_rope_unified(q, k, v, cos, sin,
                                   cache_seqlens=cache.seqlen, causal=True)
```

### Block-sparse attention

```python
from mlx_mfa import flash_attention_sparse, make_sliding_window_mask, make_spatial_2d_mask

mask = make_sliding_window_mask(N=4096, window_size=256, head_dim=128)
out = flash_attention_sparse(q, k, v, mask, causal=True)

mask = make_spatial_2d_mask(H=16, W=16, radius=4, head_dim=128)
out = flash_attention_sparse(q, k, v, mask)
```

### Stateful decode (InferenceContext)

```python
from mlx_mfa import InferenceContext

ctx = InferenceContext(B=1, H_kv=8, D=128, max_seq_len=4096)
out = ctx.prefill(q, k, v, scale=scale)     # full sequence
for _ in range(100):
    out = ctx.step(q_tok, k_tok, v_tok)     # single token decode
ctx.reset()
```

### SageAttention decode (QuantizedKVCache)

```python
from mlx_mfa import QuantizedKVCache, sage_attention_prequantized

cache = QuantizedKVCache(B=1, H_kv=8, D=128, max_seq_len=2048)
cache.append(k_new, v_new)

out = sage_attention_prequantized(
    q_int8, cache.k_int8, cache.v,
    q_scale, cache.k_scale,
    causal=True
)
```

### Autograd

```python
import mlx.core as mx
from mlx_mfa import flash_attention

def attn(q, k, v):
    return flash_attention(q, k, v, causal=True)

output, grads = mx.vjp(attn, [q, k, v], [cotangent])
dq, dk, dv = grads
```

### mlx-lm integration

```python
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm

patch_mlx_lm(verbose=True)   # all mlx-lm models now use STEEL V2
# ... run model.generate() as usual
```

---

## Dispatch Policy

`backend="auto"` activates STEEL V2 only when empirically faster than SDPA:

| D | causal | Activated at N ≥ | Note |
|---|--------|-----------------|------|
| 64 | yes | 4096 (M1), 4096 (M3+) | V2 ~2× |
| 128 | yes | 8192 (M1), 2048 (M3+) | V2 ~1.7× |
| 256+ | any | never (SDPA wins) | register spill |
| any | window | always | tile-skip guarantee |
| any | sparse | always | tile-skip guarantee |

Run `python -m mlx_mfa calibrate` to measure crossover points on your device
and automatically save optimal thresholds.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/API_MANUAL.md](docs/API_MANUAL.md) | Complete function reference — all 52 exports |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kernel architecture and design decisions |
| [docs/benchmarks/RESULTS.md](docs/benchmarks/RESULTS.md) | Latest benchmark results |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Supported Configurations

| Property | Values |
|----------|--------|
| Head dims | 64, 128, 256, 512 |
| Dtypes | float16, bfloat16, float32 |
| Layout | `[B, H, N, D]` row-major |
| Platform | macOS arm64 |
| Python | 3.10+ |
| MLX | ≥ 0.18.0 |

---

## License

MIT — see [LICENSE](LICENSE).

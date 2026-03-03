# CLAUDE.md

## What is this project?

mlx-mfa is a Python/C++ library that brings Metal Flash Attention (MFA) kernels to Apple's MLX framework. It exposes `flash_attention(q, k, v)` as a drop-in replacement for `mx.fast.scaled_dot_product_attention()`.

## Architecture

```
Python API (mlx_mfa.flash_attention)
    |
    +-- fallback --> mx.fast.scaled_dot_product_attention
    |
    +-- MFA path --> C++ Extension (nanobind, scikit-build-core)
                         |
                         +-- MFAttention : mlx::core::Primitive
                         |     eval_gpu()  -> forward Metal dispatch
                         |     vjp()       -> backward (Phase 3)
                         |
                         +-- ShaderCache (singleton)
                         |     JIT Metal shader source strings
                         |     Compile via MTLDevice newLibraryWithSource
                         |     Cache MTLComputePipelineState by param hash
                         |
                         +-- Metal GPU Kernels (JIT-generated, NOT static .metal)
```

## Key technical decisions

1. **C++ extension (nanobind)** over `mx.fast.metal_kernel()`: backward pass needs 2 separate kernel dispatches (dQ and dK/dV), JIT generation too complex for inline API, need autograd via `Primitive::vjp()`.

2. **shader_cache.mm is Objective-C++**: uses native Metal API directly instead of metal-cpp wrapper. Interface uses `void*` with `__bridge_retained` for ARC-safe pipeline management.

3. **Kernel source**: from C++ port in [liuliu/ccv](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa), NOT Swift reference (philipturner/metal-flash-attention). The ccv code is proven in production (Draw Things).

4. **JIT shader generation**: Metal shaders are source strings generated at runtime, parameterized by head_dim, dtype, block sizes, causal flag, device caps. NOT pre-compiled .metal files.

5. **Backward uses 7 GEMMs** (vs standard 5) to avoid FP32 atomics not natively supported on Apple Silicon. Two kernels: dQ (parallel rows), dK/dV (parallel cols).

## Build system

- scikit-build-core + CMake + nanobind
- MLX detected via Python (no find_package)
- Requires: macOS arm64, Python 3.10+, mlx >= 0.18.0, nanobind >= 2.0
- `pip install -e .` builds everything
- `python scripts/check_env.py` validates env

## Source layout

```
csrc/
  mfa_attention.hpp/.cpp  -- MFAttention Primitive
  shader_cache.hpp        -- Cache interface (pure C++)
  shader_cache.mm         -- Obj-C++ Metal compilation
  bindings.cpp            -- nanobind module
  kernels/                -- Placeholder .metal (real kernels are JIT)
mlx_mfa/
  __init__.py             -- Public API
  attention.py            -- flash_attention() + fallback
tests/test_attention.py   -- Fallback + extension-gated tests
benchmarks/               -- MFA vs MLX SDPA comparison
scripts/check_env.py      -- Pre-build validation
```

## MFA kernel source (ccv)

Extract from:
```
liuliu/ccv  (branch: unstable)
  lib/nnc/mfa/
    ccv_nnc_mfa_attention.cpp/.hpp  -- Param resolution, dispatch
    ccv_nnc_mfa_hash.hpp            -- Kernel key hashing
    ccv_nnc_mfa_gemm.cpp/.hpp       -- GEMM primitive
    v2/                             -- JIT shader generation (current)
    3rdparty/                       -- metal-cpp headers (skip)
```

ccv type mapping:

| ccv | MLX |
|-----|-----|
| `ccv_nnc_tensor_t` | `mlx::core::array` |
| ccv Metal device | `mlx::core::metal::device()` |
| ccv allocator | `mlx::core::allocator::malloc_or_wait()` |
| ccv command buffer | MLX compute encoder |
| ccv stream | `mlx::core::Stream` |

## MFA blocking parameters

Deformed tile aspect ratios:
- Parallelization dim: 16-32 (small, many parallel tiles)
- Traversal dim: 80-128 (large, amortize register spilling)
- D=256: 3D blocking splits head_dim into sub-tiles of 128

Block params vary by generation (M1/M2 vs M3/M4). Lookup tables in ccv source.

## MLX Primitive pattern

```cpp
class MFAttention : public mlx::core::Primitive {
  void eval_gpu(inputs, outputs) override;
  std::vector<array> vjp(...) override;
};

auto outputs = array::make_arrays(
    {out_shape, lse_shape}, {q.dtype(), float32},
    std::make_shared<MFAttention>(stream, params), {q, k, v});
```

## Current status

- Phase 1.1 DONE: scaffold, Python API with fallback, C++ skeleton
- Phase 1.2 TODO: extract MFA kernels from ccv
- Phase 1.3 TODO: decouple from ccv, wire into MLX
- Phase 1.4 TODO: full forward (all D, dtypes, causal)

## Performance targets

- Forward D=128 N=4096: >= 20% faster than MLX SDPA
- Forward D=256 N=8192: >= 30% faster
- ALU utilization >= 70%
- Max abs error < 1e-5 (f32), < 1e-2 (f16)

## Testing

```bash
pytest tests/ -v                     # Fallback (no build needed)
pytest tests/ -v -k "MFAKernel"     # Extension tests (needs build)
python benchmarks/bench_attention.py
```

## References

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) -- Algorithm, blocking tables, pseudocode
- [liuliu/ccv mfa subtree](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) -- C++ source
- [Draw Things blog](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18) -- Production validation
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX C++ extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)

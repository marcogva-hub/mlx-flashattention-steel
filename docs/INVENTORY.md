# mlx-mfa Repository Inventory

_Auto-regenerated at v0.7.0 (2026-03-05)._

## Project structure

```
.github/workflows/ci.yml         — GitHub Actions CI (test-fallback + test-mfa jobs)
CHANGELOG.md                     — Version history
CLAUDE.md                        — AI assistant instructions
README.md                        — Project documentation
RESULTS.md                       — Benchmark results (appended by bench_*.py scripts)
pyproject.toml                   — Build config (scikit-build-core, version)

benchmarks/
  bench_attention.py             — Core forward/backward speedup vs SDPA
  bench_mlx_lm.py                — mlx-lm integration throughput benchmark
  bench_rope_3d.py               — 3D RoPE table build + fwd timing
  bench_segment.py               — Segment mask: sparse vs per-segment vs dense
  bench_spatial_masks.py         — 2D/3D/segment/adaptive mask build + attention
  bench_varlen.py                — Varlen vs padded vs sequential attention

csrc/                            — C++ / Objective-C++ extension
  bindings.cpp                   — nanobind Python bindings
  kernels/attention_forward.metal — Placeholder .metal (real kernels are JIT)
  mfa_attention.cpp/.hpp         — MFAttention Primitive (MLX custom op)
  mfa_shader_gen.cpp/.hpp        — ccv MFA JIT shader generation (f32 path)
  mfa_steel_fwd.cpp/.hpp         — STEEL forward kernel + flash decode
  shader_cache.hpp               — ShaderCache interface (pure C++)
  shader_cache.mm                — Obj-C++ Metal pipeline compilation
  mfa/                           — ccv-derived kernel builders
    AttentionKernel.cpp/.hpp
    AttentionKernelDescriptor.cpp/.hpp
    AttentionKernelType.hpp
    AttentionOperand.hpp
    CodeWriter.cpp/.hpp
    DeviceProperties.hpp
    GEMMHeaders.cpp/.hpp
    GEMMOperandPrecision.hpp
    mfa_compat.h

docs/
  ARCHITECTURE.md                — Architecture notes (kernel paths, blocking, etc.)
  INVENTORY.md                   — This file
  PAGED_ATTENTION_DESIGN.md      — Paged KV cache design document (v1.0 planned)
  benchmarks/RESULTS.md          — Historical benchmark archive

mlx_mfa/                         — Python package
  __init__.py                    — Public API (exports, __version__)
  attention.py                   — flash_attention, flash_attention_sparse,
                                   flash_attention_varlen, flash_attention_rope,
                                   make_rope_3d_tables + fallback helpers
  masks.py                       — Spatial/segment/adaptive block-mask builders
  integrations/
    __init__.py
    mlx_lm.py                    — mlx-lm patch_mlx_lm() / unpatch_mlx_lm()

scripts/
  check_env.py                   — Pre-build env validation

tests/
  __init__.py
  test_attention.py              — All unit tests (~152 collected)
  test_mlx_lm_integration.py     — mlx-lm integration tests
```

## Public API (mlx_mfa)

```python
# Core attention
flash_attention(q, k, v, scale=None, causal=False, stream=None)
flash_attention_sparse(q, k, v, block_mask, scale=None, causal=False, stream=None)
flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k,
                       scale=None, causal=False, block_mask=None, stream=None)
flash_attention_rope(q, k, v, rotary_cos=None, rotary_sin=None,
                     scale=None, causal=False, cache_seqlens=0,
                     rope_3d=None, stream=None)

# Mask builders (mlx_mfa/masks.py)
make_spatial_2d_mask(height, width, spatial_radius, head_dim=128, patch_size=1)
make_spatial_3d_mask(height, width, num_frames, spatial_radius, temporal_radius,
                     head_dim=128, patch_size=1, temporal_patch_size=1)
make_topk_spatial_mask(q, k, top_k, head_dim=128)
make_segment_mask(segment_lengths, head_dim=128)
make_causal_segment_mask(segment_lengths, head_dim=128)
make_adaptive_window_mask(height, width, num_frames=1, base_window_h=16,
                          base_window_w=16, base_window_t=4,
                          train_resolution=(256,256), inference_resolution=(512,512),
                          head_dim=128, patch_size=1)
make_causal_block_mask(N, head_dim)
make_sliding_window_mask(N, window_size, head_dim)

# RoPE tables
make_rope_3d_tables(grid_h, grid_w, num_frames, d_h=None, d_w=None, d_t=None,
                    head_dim=128, theta=10000.0)

# Device / capability queries
is_mfa_available() -> bool
get_device_info() -> dict
get_supported_configs() -> dict
```

# mlx-mfa Repository Inventory

_Auto-regenerated at v0.8.0 (2026-03-05)._

## Project structure

```
mlx-mfa-v2/
├── mlx_mfa/               Python package
│   ├── __init__.py        Public API (25 exports, version=0.8.0)
│   ├── attention.py       Core attention + fallback paths (1699 lines)
│   ├── masks.py           Mask builders — 13 functions (1129 lines)
│   └── integrations/
│       └── mlx_lm.py      mlx-lm patch/unpatch
├── csrc/                  C++ extension (nanobind)
│   ├── bindings.cpp       Python bindings
│   ├── mfa_attention.hpp  MFAttention primitive declarations
│   ├── mfa_attention.cpp  Primitive eval_gpu + free functions
│   ├── mfa_steel_fwd.hpp  STEEL forward params + decls
│   ├── mfa_steel_fwd.cpp  STEEL forward JIT Metal generator
│   ├── mfa_shader_gen.*   ccv-derived shader gen (f32 fallback)
│   ├── shader_cache.hpp   KernelKey + ShaderCache interface
│   ├── shader_cache.mm    Obj-C++ Metal pipeline cache + routing
│   ├── mfa/               ccv kernel infrastructure (AttentionKernel, GEMMHeaders…)
│   └── kernels/           Placeholder .metal files (real kernels are JIT)
├── tests/
│   ├── test_attention.py  198 tests
│   └── test_mlx_lm_integration.py  11 tests
├── benchmarks/            7 benchmark scripts
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INVENTORY.md       This file
│   └── PAGED_ATTENTION_DESIGN.md
├── scripts/check_env.py
├── pyproject.toml         version=0.8.0
└── CMakeLists.txt
```

---

## Public API (`mlx_mfa.__all__` — 25 symbols)

### Core attention (5)
| Function | Signature highlights |
|----------|---------------------|
| `flash_attention` | `(q,k,v, scale, causal, softcap, dropout_p, return_attn_weights)` |
| `flash_attention_rope` | `(q,k,v, cos,sin, scale, causal, cache_seqlens, rope_3d, interleaved)` |
| `flash_attention_sparse` | `(q,k,v, block_mask, scale, causal, backward)` |
| `flash_attention_varlen` | `(q,k,v, cu_seqlens_q, cu_seqlens_k, max_q, max_k, scale, causal)` |
| `flash_attention_with_kv_cache` | `(q, k_new, v_new, k_cache, v_cache, scale, causal, softcap)` |

### Mask builders (15)
| Group | Functions |
|-------|-----------|
| Dense-sparse | `make_causal_block_mask`, `make_sliding_window_mask` |
| Spatial | `make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask` |
| Segment | `make_segment_mask`, `make_causal_segment_mask` |
| Adaptive | `make_adaptive_window_mask` |
| Video/VSR | `make_lcsa_mask`, `make_axial_spatial_mask`, `make_axial_temporal_mask` |
| Temporal | `make_dilated_temporal_mask` |
| Special | `make_sink_window_mask`, `make_reference_frame_mask`, `make_cross_stream_mask` |

### Utilities (5)
- `make_rope_3d_tables` — build 3D rotary tables for video
- `is_mfa_available()` — True when C++ extension loaded
- `get_device_info()` — device_name, gpu_family_gen, is_m3_plus, is_m5_plus
- `get_supported_configs()` — head_dims, dtypes, extension_available
- `__version__` — "0.8.0"

---

## C++ Metal kernel variants (v0.8.0)

| KernelType | Dtype | Pass | Description |
|-----------|-------|------|-------------|
| `SteelForward` | f16/bf16 | Fwd | Main path; cooperative threadgroup loads |
| `SteelForwardSparse` | f16/bf16 | Fwd | Block-sparse: skip inactive K tiles |
| `SteelForwardALiBi` | f16/bf16 | Fwd | ALiBi per-head linear bias |
| `SteelForwardRoPE` | f16/bf16 | Fwd | In-kernel RoPE fusion |
| `FlashDecodePartial` | f16/bf16 | Fwd | Split-KV decode (N_q ≤ 4) |
| `FlashDecodeReduce` | f16/bf16 | Fwd | Phase-2 log-sum-exp reduction |
| `AttentionForward` (ccv) | f32 | Fwd | f32 fallback |
| `AttentionBackwardDQ` (ccv) | all | Bwd | dQ kernel |
| `AttentionBackwardDKV` (ccv) | all | Bwd | dK/dV kernel |

**Backward strategy (v0.8.0)**: f16/bf16 backward uses `mx.vjp(SDPA)` — correct but dense. STEEL native backward (dQ + dK/dV) is v0.9.0 target.

---

## Tests (209 total)

| Class | Count | What |
|-------|------:|------|
| TestFallback | 4 | SDPA fallback |
| TestMFAKernel | 16 | All D/dtype/causal combos |
| TestMFABackward | 5 | Autograd dQ/dK/dV |
| TestBlockSparse | 16 | Sparse kernel + masks |
| TestSparseBackwardTiled | 10 | FA-2 sparse backward |
| TestGQA | 9 | Grouped query attention |
| TestFlashDecode | 11 | Split-KV decode |
| TestQuantizedKV | 6 | Q4/Q8 dequantize |
| TestRoPEFusion | 8 | 1D + 3D RoPE |
| TestVarlen | 10 | Variable-length batching |
| TestSpatialMasks + video masks | ~30 | 13 mask builder classes |
| TestSoftcap | 5 | Tanh softcapping |
| TestALiBi | 5 | Per-head position biases |
| TestRoPENonInterleaved | 6 | GPT-NeoX RoPE |
| TestPerBatchCacheSeqlens | 3 | list/array cache offsets |
| TestHeadDimVMismatch | 4 | D_v != D_qk fallback |
| TestKVCacheAppend | 4 | flash_attention_with_kv_cache |
| TestAttentionDropout | 4 | Training dropout |
| TestReturnAttnWeights | 4 | return_attn_weights=True |
| TestPublicAPI / TestEdgeCases / etc. | ~35 | API, M3+ routing, edge cases |
| test_mlx_lm_integration.py | 11 | mlx-lm integration |
| **Total** | **209** | |

---

## Benchmarks (7 scripts)

| Script | Scenarios |
|--------|-----------|
| `bench_attention.py` | Dense + sparse MFA vs SDPA; D=64/128/256, N up to 16384 |
| `bench_mlx_lm.py` | mlx-lm tokens/sec with/without MFA patch |
| `bench_rope_3d.py` | 3D RoPE overhead |
| `bench_segment.py` | Segment mask attention throughput |
| `bench_softcap_alibi.py` | Softcap + ALiBi overhead vs baseline (NEW v0.8.0) |
| `bench_spatial_masks.py` | Spatial mask attention throughput |
| `bench_varlen.py` | Variable-length batching throughput |

---

## v0.9.0 planned additions

- `csrc/mfa_steel_bwd.hpp/.cpp` — STEEL native backward (dQ + dK/dV kernels)
- STEEL varlen forward kernel (cu_seqlens in Metal)
- `mlx_mfa/paged_kv.py` — PagedKVCache allocator
- `mlx_mfa/packed.py` — QKV/KV packed format utilities
- `benchmarks/bench_backward.py` — STEEL vs SDPA backward benchmark

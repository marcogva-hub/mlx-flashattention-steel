# mlx-mfa Repository Inventory

_Auto-regenerated at v0.9.2 (2026-03-06)._

## Project structure

```
mlx-mfa-v2/
├── mlx_mfa/               Python package
│   ├── __init__.py        Public API (25 exports, version=0.9.1)
│   ├── attention.py       Core attention + fallback paths (2187 lines)
│   ├── masks.py           Mask builders — 13 functions (1129 lines)
│   └── integrations/
│       └── mlx_lm.py      mlx-lm patch/unpatch
├── csrc/                  C++ extension (nanobind)
│   ├── bindings.cpp       Python bindings
│   ├── mfa_attention.hpp  MFAttention primitive declarations
│   ├── mfa_attention.cpp  Primitive eval_gpu + free functions
│   ├── mfa_steel_fwd.hpp  STEEL forward params + decls
│   ├── mfa_steel_fwd.cpp  STEEL forward JIT Metal generator
│   ├── mfa_steel_bwd.hpp  STEEL backward params + decls
│   ├── mfa_steel_bwd.cpp  STEEL backward JIT Metal generator (dQ + dK/dV)
│   ├── mfa_shader_gen.*   ccv-derived shader gen (f32 fallback)
│   ├── shader_cache.hpp   KernelKey + ShaderCache interface
│   ├── shader_cache.mm    Obj-C++ Metal pipeline cache + routing
│   ├── mfa/               ccv kernel infrastructure (AttentionKernel, GEMMHeaders…)
│   └── kernels/           Placeholder .metal files (real kernels are JIT)
├── tests/
│   ├── test_attention.py  198 tests
│   └── test_mlx_lm_integration.py  11 tests
├── benchmarks/            8 benchmark scripts
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INVENTORY.md       This file
│   └── PAGED_ATTENTION_DESIGN.md
├── scripts/check_env.py
├── pyproject.toml         version=0.9.1
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
- `__version__` — "0.9.1"

---

## C++ Metal kernel variants (v0.9.1)

| KernelType | Dtype | Pass | Description |
|-----------|-------|------|-------------|
| `SteelForward` | f16/bf16 | Fwd | Main path; cooperative threadgroup loads |
| `SteelForwardSparse` | f16/bf16 | Fwd | Block-sparse: skip inactive K tiles |
| `SteelForwardALiBi` | f16/bf16 | Fwd | ALiBi per-head linear bias |
| `SteelForwardRoPE` | f16/bf16 | Fwd | In-kernel RoPE fusion |
| `FlashDecodePartial` | f16/bf16 | Fwd | Split-KV decode (N_q ≤ 4) |
| `FlashDecodeReduce` | f16/bf16 | Fwd | Phase-2 log-sum-exp reduction |
| `AttentionForward` (ccv) | f32 | Fwd | f32 fallback |
| `AttentionBackwardDQ` (ccv) | f32 | Bwd | dQ kernel (f32 only) |
| `AttentionBackwardDKV` (ccv) | f32 | Bwd | dK/dV kernel (f32 only) |
| `SteelBackwardDQ` | f16/bf16 | Bwd | STEEL dQ kernel |
| `SteelBackwardDKV` | f16/bf16 | Bwd | STEEL dK/dV kernel |

**Backward strategy (v0.9.2)**: f16/bf16 D≤256 dispatches native STEEL Metal kernels
(`MFASteelBwdDQ`, `MFASteelBwdDKV`) via `_make_mfa_custom._backward`. D=256 uses D-split
(BD_HALF=128) to stay within 32 KB TGP. f32 stays on ccv path.
Buffer aliasing fix: `_sever_lazy_graph(cotangent)` before gradient-checkpointing re-run of forward.

---

## Tests (241 pytest runs / 203 test functions)

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
| TestSteelBackwardGQA | 3 | STEEL backward GQA (DA) |
| TestSteelBackwardD256 | 6 | D=256 D-split backward (CE) |
| TestPublicAPI / TestEdgeCases / etc. | ~35 | API, M3+ routing, edge cases |
| test_mlx_lm_integration.py | 11 | mlx-lm integration |
| **Total** | **241** | |

---

## Benchmarks (9 scripts)

| Script | Scenarios |
|--------|-----------|
| `bench_attention.py` | Dense + sparse MFA vs SDPA; D=64/128/256, N up to 16384 |
| `bench_mlx_lm.py` | mlx-lm tokens/sec with/without MFA patch |
| `bench_rope_3d.py` | 3D RoPE overhead |
| `bench_segment.py` | Segment mask attention throughput |
| `bench_softcap_alibi.py` | Softcap + ALiBi overhead vs baseline |
| `bench_spatial_masks.py` | Spatial mask attention throughput |
| `bench_varlen.py` | Variable-length batching throughput |
| `bench_all.py` | Consolidated fwd+bwd suite (v0.9.1) |
| `bench_compile.py` | `mx.compile` overhead: softcap/alibi/rope compiled vs raw (v0.9.2) |

---

## v0.9.2 additions (DA–DE)

| Track | File(s) | Description |
|-------|---------|-------------|
| DA | `mlx_mfa/attention.py` | Fix GQA backward Python guard (was blocking STEEL dispatch for GQA) |
| DB | `CHANGELOG.md`, `docs/INVENTORY.md` | Fix doc inaccuracies (track CB scope, test counts) |
| DC | `mlx_mfa/attention.py`, `benchmarks/bench_compile.py` | `mx.compile` for `_apply_rope_mlx` (shape-keyed cache) + compile benchmark |
| CE | `csrc/mfa_steel_bwd.cpp`, `mlx_mfa/attention.py` | D=256 D-split STEEL backward (BD_HALF=128); widen guard to D≤256 |
| DD | `docs/INVENTORY.md`, `docs/ARCHITECTURE.md` | Documentation refresh (v0.9.2 additions, test count 241, bench count 9) |
| DE | `pyproject.toml`, `mlx_mfa/__init__.py`, `CHANGELOG.md` | Version bump → 0.9.2, tag |

---

## v0.9.1 additions (CA–CI)

| Track | File(s) | Description |
|-------|---------|-------------|
| CA | `csrc/mfa_steel_fwd.cpp` | Vec4 aligned block loads (float4/half4) |
| CB | `mlx_mfa/attention.py` | `mx.compile` for fallback paths |
| CC | `csrc/mfa_steel_fwd.cpp` | Persistent multi-Q-block kernel (4× Q-blocks/dispatch) |
| CD | `csrc/mfa_steel_bwd.cpp`, `shader_cache.hpp/.mm` | GQA in STEEL backward (bake `gqa_factor` as `#define`) |
| CE | `csrc/mfa_steel_bwd.cpp` | D=256 backward D-split (completed in v0.9.2) |
| CF | `csrc/mfa_steel_fwd.cpp` | Double-buffer ping-pong (K_smem⊕V_smem, 4→2 barriers/K-tile) |
| CG | `benchmarks/bench_all.py`, `docs/benchmarks/RESULTS.md` | Consolidated benchmark + v0.9.1 results |
| CH | `docs/INVENTORY.md`, `docs/ARCHITECTURE.md`, `README.md` | Documentation refresh |
| CI | `pyproject.toml`, `mlx_mfa/__init__.py`, `CHANGELOG.md` | Version bump → 0.9.1, tag |

## v0.9.0 additions (BA–BH, for reference)

- `csrc/mfa_steel_bwd.hpp/.cpp` — STEEL native backward (dQ + dK/dV kernels)
- STEEL varlen forward kernel (cu_seqlens in Metal)
- `mlx_mfa/paged_kv.py` — PagedKVCache allocator
- `mlx_mfa/packed.py` — QKV/KV packed format utilities
- `benchmarks/bench_backward.py` — STEEL vs SDPA backward benchmark
- `benchmarks/bench_varlen.py` — varlen STEEL kernel note

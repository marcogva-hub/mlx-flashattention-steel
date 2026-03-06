# mlx-mfa Repository Inventory

_Regenerated at v0.9.3 (2026-03-06). All line counts verified with `wc -l`._

---

## Project structure

```
mlx-mfa-v2/
├── mlx_mfa/               Python package
│   ├── __init__.py        Public API (31 exports, version=0.9.3)  [102 lines]
│   ├── attention.py       Core attention + fallback paths         [2469 lines]
│   ├── masks.py           Mask builders — 15 functions            [1129 lines]
│   └── integrations/
│       └── mlx_lm.py      mlx-lm patch/unpatch                   [191 lines]
├── csrc/                  C++ extension (nanobind)
│   ├── bindings.cpp       Python bindings                         [304 lines]
│   ├── mfa_attention.hpp  MFAttention + 5 other Primitive decls   [366 lines]
│   ├── mfa_attention.cpp  Primitive eval_gpu + free functions    [1322 lines]
│   ├── mfa_steel_fwd.hpp  STEEL forward params + decls            [178 lines]
│   ├── mfa_steel_fwd.cpp  STEEL forward JIT Metal generator      [2205 lines]
│   ├── mfa_steel_bwd.hpp  STEEL backward params + decls            [68 lines]
│   ├── mfa_steel_bwd.cpp  STEEL backward JIT Metal generator     [1372 lines]
│   ├── mfa_paged_gather.hpp  MFAPagedKVGather Primitive decl       [84 lines]
│   ├── mfa_paged_gather.cpp  Metal paged KV gather kernel          [241 lines]
│   ├── mfa_shader_gen.hpp ccv shader gen interface                  [59 lines]
│   ├── mfa_shader_gen.cpp ccv-derived shader gen (f32 fallback)    [305 lines]
│   ├── shader_cache.hpp   KernelKey enum (10 types) + ShaderCache   [89 lines]
│   ├── shader_cache.mm    Obj-C++ Metal pipeline cache + routing   [224 lines]
│   ├── mfa/               ccv kernel infrastructure
│   └── kernels/           Placeholder .metal files (real kernels are JIT)
├── tests/
│   ├── test_attention.py  196 test functions / 40 test classes    [4017 lines]
│   └── test_mlx_lm_integration.py  16 test functions              [402 lines]
├── benchmarks/            10 benchmark scripts
├── docs/
│   ├── ARCHITECTURE.md    Architecture + algorithm documentation
│   ├── INVENTORY.md       This file
│   └── PAGED_ATTENTION_DESIGN.md
├── scripts/check_env.py
├── pyproject.toml         version=0.9.3
└── CMakeLists.txt
```

---

## Public API (`mlx_mfa.__all__` — 31 symbols)

### Core attention (11)

| Symbol | Signature highlights |
|--------|---------------------|
| `flash_attention` | `(q,k,v, scale, causal, softcap, dropout_p, return_attn_weights)` |
| `flash_attention_rope` | `(q,k,v, cos,sin, scale, causal, cache_seqlens, rope_3d, interleaved)` |
| `flash_attention_sparse` | `(q,k,v, block_mask, scale, causal, backward)` |
| `flash_attention_varlen` | `(q,k,v, cu_q,cu_k, max_q,max_k, scale, causal)` — differentiable (EA) |
| `flash_attention_with_kv_cache` | `(q, k_new,v_new, k_cache,v_cache, scale, causal, softcap)` |
| `flash_attention_paged` | `(q, k_pages,v_pages, block_table,seq_lens, scale, causal)` — Metal gather (EB) |
| `flash_attention_qkv_packed` | `(qkv, scale, causal, num_heads, num_kv_heads)` |
| `flash_attention_kv_packed` | `(q, kv, scale, causal, num_kv_heads)` |
| `flash_attention_varlen_qkv_packed` | `(qkv, cu_q,cu_k, max_q,max_k, num_heads, ...)` — (EC) |
| `flash_attention_varlen_kv_packed` | `(q, kv, cu_q,cu_k, max_q,max_k, num_kv_heads, ...)` — (EC) |
| `PagedKVCache` | `(num_blocks, block_size, H, D, dtype)` — allocator class |

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

### RoPE helpers (1)

- `make_rope_3d_tables` — build 3D rotary tables for video models

### Utilities (3 + `__version__`)

- `is_mfa_available()` — `True` when C++ extension loaded
- `get_device_info()` — `device_name`, `gpu_family_gen`, `is_m3_plus`, `is_m5_plus`, `chip_name`
- `get_supported_configs()` — `head_dims`, `dtypes`, `extension_available`
- `__version__` — `"0.9.3"`

---

## C++ Metal kernel variants (v0.9.3)

10 `KernelType` entries in `csrc/shader_cache.hpp`:

| Value | KernelType | Dtype | Pass | Description |
|:-----:|-----------|-------|------|-------------|
| 0 | `AttentionForward` | f32 | Fwd | ccv f32 forward |
| 1 | `AttentionBackwardDQ` | f32 | Bwd | ccv dQ (compiled; unused — SDPA vjp used for f32) |
| 2 | `AttentionBackwardDKV` | f32 | Bwd | ccv dK/dV (compiled; unused) |
| 3 | `SteelForward` | f16/bf16 | Fwd | Main STEEL path; cooperative threadgroup loads |
| 4 | `FlashDecodePartial` | f16/bf16 | Fwd | Split-KV decode Phase 1 (N_q ≤ 4) |
| 5 | `FlashDecodeReduce` | f16/bf16 | Fwd | Split-KV decode Phase 2 (LSE reduce) |
| 6 | `SteelBackwardDQ` | f16/bf16 | Bwd | STEEL dQ kernel |
| 7 | `SteelBackwardDKV` | f16/bf16 | Bwd | STEEL dK/dV kernel |
| 8 | `SteelVarlenForward` | f16/bf16 | Fwd | Varlen: `(total_q_tiles, H, 1)` grid |
| 9 | `PagedKVGather` | f16/bf16 | Gather | Pool `[NB,BS,H,D]` → `[B,H,max_kv,D]` (EB) |

**Backward routing (v0.9.3)**: f16/bf16 D≤256 softcap==0 → STEEL kernels 6+7.
f32 → `mx.vjp(_fallback_sdpa)`. softcap/alibi → `mx.vjp` over compiled reference.

---

## C++ Primitive classes

| Class | File | Role |
|-------|------|------|
| `MFAttention` | `mfa_attention.cpp` | Main forward (ccv + STEEL dispatch) |
| `MFABackwardQuery` | `mfa_attention.cpp` | ccv dQ (compiled; routing superseded by STEEL) |
| `MFABackwardKeyValue` | `mfa_attention.cpp` | ccv dK/dV (same) |
| `MFASteelBwdDQ` | `mfa_attention.cpp` | STEEL dQ backward |
| `MFASteelBwdDKV` | `mfa_attention.cpp` | STEEL dK/dV backward |
| `MFAVarlenAttention` | `mfa_attention.cpp` | STEEL varlen forward |
| `MFAPagedKVGather` | `mfa_paged_gather.cpp` | Metal paged KV gather (EB) |

---

## Tests (257 pytest runs / 212 test functions)

| Class | Count | What |
|-------|------:|------|
| TestFallbackPath | 4 | SDPA fallback |
| TestMFAKernel | 16 | All D/dtype/causal combos |
| TestMFABackward | 5 | Autograd dQ/dK/dV |
| TestPublicAPI | ~6 | is_mfa_available, get_device_info, etc. |
| TestEdgeCases | ~8 | GQA ratios, N=1, non-multiple seq |
| TestBackwardEdge | ~5 | N=1 backward, value_and_grad, partial argnums |
| TestNativeGQA | 9 | GQA ratio 2/4/8 × D=64/128, causal, backward |
| TestSparseAttentionAPI | 6 | Sparse kernel API (shapes, dtypes) |
| TestSparseAttentionKernel | 10 | Sparse kernel correctness |
| TestM3M4Path | 6 | M3+/M1 routing via MFA_FORCE_GEN env var |
| TestSparseBackwardTiled | 10 | FA-2 sparse backward (tiled) |
| TestFlashDecode | 11 | Split-KV decode (N_q ≤ 4, S ≥ 256) |
| TestM5Detection | 2 | M5+ detection stub (gen ≥ 17) |
| TestRoPEFusion | 8 | 1D + 3D RoPE kernel |
| TestSpatialMasks | ~6 | 2D/3D spatial mask builders |
| TestSegmentMask | ~4 | Segment + causal-segment masks |
| TestAdaptiveWindowMask | ~2 | Adaptive window mask |
| TestVarlenAttention | 10 | Variable-length batching correctness |
| TestSteelVarlen | ~8 | STEEL varlen kernel (requires ext) |
| TestRoPE3D | ~4 | 3D RoPE table construction |
| TestLCSAMask | ~2 | LCSA composite mask (FlashVSR) |
| TestAxialMasks | ~4 | Axial spatial + temporal masks |
| TestDilatedTemporalMask | ~2 | Dilated temporal mask |
| TestSinkAndReferenceFrameMasks | ~4 | Sink + reference frame masks |
| TestCrossStreamMask | ~2 | Cross-stream mask (LTX-2) |
| TestSoftcap | 5 | Tanh softcapping (Gemma 2) |
| TestALiBi | 5 | Per-head position biases (Falcon) |
| TestRoPENonInterleaved | 6 | GPT-NeoX RoPE |
| TestPerBatchCacheSeqlens | 3 | list/array cache offsets |
| TestHeadDimVMismatch | 4 | D_v != D_qk fallback |
| TestKVCacheAppend | 4 | flash_attention_with_kv_cache |
| TestAttentionDropout | 4 | Training dropout |
| TestReturnAttnWeights | 4 | return_attn_weights=True |
| TestPagedKVCache | ~12 | PagedKVCache allocator + paged attention |
| TestPackedFormats | ~8 | QKV/KV packed tensor formats |
| TestSteelBackwardGQA | 3 | STEEL backward GQA correctness (DA) |
| TestSteelBackwardD256 | 6 | D=256 D-split backward (CE) |
| TestVarlenBackward | 6 | Differentiable varlen backward (EA) |
| TestPagedBackward | 6 | Metal paged gather + dQ backward (EB) |
| TestVarlenPacked | 4 | Varlen QKV/KV packed formats (EC) |
| test_mlx_lm_integration.py | 16 | mlx-lm patch/unpatch, correctness, GQA |
| **Total collected** | **257** | (196 + 16 = 212 functions; 257 with parametrize) |

---

## Benchmarks (10 scripts)

| Script | Scenarios |
|--------|-----------|
| `bench_attention.py` | Dense + sparse MFA vs SDPA; D=64/128/256, N up to 16384 |
| `bench_backward.py` | STEEL vs SDPA backward throughput |
| `bench_mlx_lm.py` | mlx-lm tokens/sec with/without MFA patch |
| `bench_rope_3d.py` | 3D RoPE overhead |
| `bench_segment.py` | Segment mask attention throughput |
| `bench_softcap_alibi.py` | Softcap + ALiBi overhead vs baseline |
| `bench_spatial_masks.py` | Spatial mask attention throughput |
| `bench_varlen.py` | Variable-length batching throughput |
| `bench_all.py` | Consolidated fwd+bwd suite (v0.9.1) |
| `bench_compile.py` | `mx.compile` overhead: softcap/alibi/rope compiled vs raw (v0.9.2) |

---

## v0.9.3 additions (EA–EE)

| Track | File(s) | Description |
|-------|---------|-------------|
| EA | `mlx_mfa/attention.py`, `tests/test_attention.py` | Differentiable `flash_attention_varlen` via `mx.custom_function`; STEEL forward, per-sequence backward |
| EB | `csrc/mfa_paged_gather.hpp/.cpp`, `csrc/shader_cache.*`, `csrc/bindings.cpp`, `CMakeLists.txt`, `mlx_mfa/attention.py` | `MFAPagedKVGather` Metal Primitive; `flash_attention_paged` with `mx.custom_function`; per-seq slicing fix |
| EC | `mlx_mfa/attention.py`, `mlx_mfa/__init__.py` | `flash_attention_varlen_qkv_packed` + `flash_attention_varlen_kv_packed` |
| ED | `CHANGELOG.md`, `docs/ARCHITECTURE.md`, `docs/INVENTORY.md`, `README.md` | Documentation maintenance |
| EE | `pyproject.toml`, `mlx_mfa/__init__.py`, `csrc/bindings.cpp`, `CHANGELOG.md` | Version bump → 0.9.3, tag v0.9.3 |

---

## v0.9.2 additions (DA–DE)

| Track | File(s) | Description |
|-------|---------|-------------|
| DA | `mlx_mfa/attention.py` | Fix GQA backward Python guard |
| DB | `CHANGELOG.md`, `docs/INVENTORY.md` | Fix doc inaccuracies |
| DC | `mlx_mfa/attention.py`, `benchmarks/bench_compile.py` | `mx.compile` for `_apply_rope_mlx` + benchmark |
| CE | `csrc/mfa_steel_bwd.cpp`, `mlx_mfa/attention.py` | D=256 D-split STEEL backward (BD_HALF=128) |
| DD | `docs/INVENTORY.md`, `docs/ARCHITECTURE.md` | Documentation refresh |
| DE | `pyproject.toml`, `mlx_mfa/__init__.py`, `CHANGELOG.md` | Version bump → 0.9.2, tag |

---

## v0.9.1 additions (CA–CI)

| Track | File(s) | Description |
|-------|---------|-------------|
| CA | `csrc/mfa_steel_fwd.cpp` | Vec4 aligned block loads (float4/half4) |
| CB | `mlx_mfa/attention.py` | `mx.compile` for fallback paths |
| CC | `csrc/mfa_steel_fwd.cpp` | Persistent multi-Q-block kernel (4× Q-blocks/dispatch) |
| CD | `csrc/mfa_steel_bwd.cpp`, `shader_cache.hpp/.mm` | GQA in STEEL backward (`gqa_factor` baked as `#define`) |
| CE | `csrc/mfa_steel_bwd.cpp` | D=256 backward D-split (completed in v0.9.2) |
| CF | `csrc/mfa_steel_fwd.cpp` | Double-buffer ping-pong (K_smem⊕V_smem, 4→2 barriers/K-tile, D≤128) |
| CG | `benchmarks/bench_all.py` | Consolidated benchmark + v0.9.1 results |
| CH | `docs/INVENTORY.md`, `docs/ARCHITECTURE.md`, `README.md` | Documentation refresh |
| CI | `pyproject.toml`, `mlx_mfa/__init__.py`, `CHANGELOG.md` | Version bump → 0.9.1, tag |

---

## v0.9.0 additions (BA–BH, for reference)

- `csrc/mfa_steel_bwd.hpp/.cpp` — STEEL native backward (dQ + dK/dV kernels)
- `MFAVarlenAttention` — STEEL varlen forward kernel (cu_seqlens in Metal)
- `PagedKVCache` allocator + `flash_attention_paged` (Python-loop gather; upgraded in EB)
- `flash_attention_qkv_packed` / `flash_attention_kv_packed`
- `benchmarks/bench_backward.py` — STEEL vs SDPA backward benchmark

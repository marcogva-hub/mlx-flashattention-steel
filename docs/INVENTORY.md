# mlx-mfa Code Inventory — v1.0.5

All numbers verified by running shell commands against the source tree.
Regenerated: 2026-03-08.

---

## Version

| Key | Value |
|-----|-------|
| `pyproject.toml` | `1.0.5` (pending bump in Track 5) |
| `mlx_mfa/__init__.py` | `1.0.4` (pending) |
| Latest git tag | `v1.0.4` |

---

## Source files

### Python (`mlx_mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `mlx_mfa/__init__.py` | 133 | Public API re-exports, ABI check, `__version__` |
| `mlx_mfa/attention.py` | 3425 | All attention functions + helpers |
| `mlx_mfa/masks.py` | 1129 | 15 mask builders |
| `mlx_mfa/integrations/mlx_lm.py` | 351 | `patch_mlx_lm` / `unpatch_mlx_lm` |
| `mlx_mfa/integrations/__init__.py` | 0 | Package marker |
| **Python total** | **5038** | |

### C++ / Objective-C++ (`csrc/`)

| File | Lines | Purpose |
|------|-------|---------|
| `csrc/bindings.cpp` | 410 | nanobind module + Python bindings |
| `csrc/mfa_attention.cpp` | 1554 | `MFAttention` Primitive: `eval_gpu`, `vjp` |
| `csrc/mfa_attention.hpp` | 437 | Primitive header |
| `csrc/mfa_steel_fwd.cpp` | 3123 | STEEL forward: JIT source gen, dispatch |
| `csrc/mfa_steel_fwd.hpp` | 235 | STEEL forward header |
| `csrc/mfa_steel_bwd.cpp` | 1295 | STEEL backward dQ + dKV kernels |
| `csrc/mfa_steel_bwd.hpp` | 68 | STEEL backward header |
| `csrc/mfa_paged_gather.cpp` | 242 | Paged KV gather Metal kernel |
| `csrc/mfa_paged_gather.hpp` | 84 | Paged gather header |
| `csrc/mfa_shader_gen.cpp` | 305 | ccv-based shader generator (legacy) |
| `csrc/mfa_shader_gen.hpp` | 59 | Shader gen header |
| `csrc/shader_cache.hpp` | 91 | `KernelType` enum, cache interface |
| `csrc/shader_cache.mm` | 230 | Objective-C++ Metal pipeline compilation |
| `csrc/mfa/AttentionKernel.cpp` | 3324 | ccv MFA kernel generation |
| `csrc/mfa/AttentionKernel.hpp` | 134 | ccv kernel header |
| `csrc/mfa/AttentionKernelDescriptor.cpp` | 42 | ccv kernel descriptor |
| `csrc/mfa/AttentionKernelDescriptor.hpp` | 78 | ccv kernel descriptor header |
| `csrc/mfa/AttentionKernelType.hpp` | 50 | ccv kernel type enum |
| `csrc/mfa/AttentionOperand.hpp` | 309 | ccv operand definitions |
| `csrc/mfa/CodeWriter.cpp` | 51 | ccv code writer |
| `csrc/mfa/CodeWriter.hpp` | 65 | ccv code writer header |
| `csrc/mfa/DeviceProperties.hpp` | 8 | ccv device properties |
| `csrc/mfa/GEMMHeaders.cpp` | 786 | ccv GEMM headers |
| `csrc/mfa/GEMMHeaders.hpp` | 36 | ccv GEMM header |
| `csrc/mfa/GEMMOperandPrecision.hpp` | 89 | ccv GEMM precision |
| **C++ total** | **13305** | |

**Total source (Python + C++)**: 18343 lines across 29 files.

---

## Public API (`mlx_mfa.__all__` — 32 symbols)

### Core attention (11 functions + 1 class)

| Symbol | Brief description |
|--------|------------------|
| `flash_attention` | Standard BHND attention; MFA or SDPA backend |
| `flash_attention_rope` | RoPE-fused attention (3D RoPE, interleaved/split, rotary_dim) |
| `flash_attention_sparse` | Block-sparse attention with `block_mask` |
| `flash_attention_varlen` | Variable-length (jagged) sequences, cu_seqlens |
| `flash_attention_kvcache` | Unified KV-cache: dense, paged, append (k_new/v_new) |
| `flash_attention_kvcache_rope_append` | Fused RoPE + KV-cache append |
| `flash_attention_paged` | Paged KV pool attention (block_table) |
| `flash_attention_qkv_packed` | Fused QKV tensor input |
| `flash_attention_kv_packed` | Fused KV tensor input |
| `flash_attention_varlen_qkv_packed` | Varlen + packed QKV |
| `flash_attention_varlen_kv_packed` | Varlen + packed KV |
| `PagedKVCache` | Python KV block allocator (dual-pool design) |

### Mask builders (15)

| Symbol | Description |
|--------|-------------|
| `make_causal_block_mask` | Block-level lower-triangular mask |
| `make_sliding_window_mask` | Sliding window (left-only boundary) |
| `make_spatial_2d_mask` | 2D grid neighbourhood |
| `make_spatial_3d_mask` | 3D volumetric neighbourhood |
| `make_topk_spatial_mask` | Top-K nearest spatial tokens |
| `make_segment_mask` | Same-segment non-causal mask |
| `make_causal_segment_mask` | Same-segment causal mask |
| `make_adaptive_window_mask` | Adaptive sliding window |
| `make_lcsa_mask` | Local-global cross-stream attention |
| `make_axial_spatial_mask` | Axial (row + column) spatial attention |
| `make_axial_temporal_mask` | Axial temporal attention |
| `make_dilated_temporal_mask` | Dilated temporal strided attention |
| `make_sink_window_mask` | Sink token + local window |
| `make_reference_frame_mask` | Reference-frame video attention |
| `make_cross_stream_mask` | Cross-stream (bidirectional) attention |

### RoPE helpers (1)

| Symbol | Description |
|--------|-------------|
| `make_rope_3d_tables` | Build 3D rotary frequency tables for video |

### Utilities (3 + `__version__`)

| Symbol | Returns | Description |
|--------|---------|-------------|
| `is_mfa_available` | `bool` | True when C++ ext + Metal GPU present |
| `get_device_info` | `dict` | device_name, gpu_family_gen, is_m3_plus, is_m5_plus, chip_name |
| `get_supported_configs` | `dict` | head_dims, dtypes, extension_available, features (22 flags), kernel_types |
| `__version__` | `str` | Package version string |

---

## Metal kernel types (`csrc/shader_cache.hpp` — 11 active)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `AttentionForward` | ccv MFA forward (legacy) |
| 1 | `AttentionBackwardDQ` | ccv MFA backward dQ |
| 2 | `AttentionBackwardDKV` | ccv MFA backward dKV |
| 3 | `SteelForward` | STEEL cooperative forward (all D; d-split for D=512) |
| 4 | `FlashDecodePartial` | Flash Decode Phase 1: partial attn per KV split |
| 5 | `FlashDecodeReduce` | Flash Decode Phase 2: LSE reduce over splits |
| 6 | `SteelBackwardDQ` | STEEL native backward dQ (f16/bf16, D<=512) |
| 7 | `SteelBackwardDKV` | STEEL native backward dKV (f16/bf16, D<=512) |
| 8 | `SteelVarlenForward` | STEEL varlen forward (D<=256; D=512 -> SDPA fallback) |
| 9 | `PagedKVGather` | Paged KV gather: pool to contiguous BHND |
| 10 | `PagedSteelForward` | STEEL forward with kernel-level paged KV (D<=256) |
| — | `TensorOpsForward` | Reserved: Metal 4 cooperative tensors (M5+/A19+ only) |

---

## Tests

**Total: 385 pytest-collected tests** (330 test methods; 55 parametrized expansions)

| File | Classes | Methods | Collected |
|------|---------|---------|-----------|
| `tests/test_attention.py` | 53 | 297 | ~352 |
| `tests/test_mlx_lm_integration.py` | 7 | 33 | ~33 |

### Test classes — `test_attention.py` (53 classes, 297 methods)

| Class | Methods | What it tests |
|-------|---------|---------------|
| TestFallbackPath | 6 | SDPA fallback (no extension needed) |
| TestMFAKernel | 4 | Forward pass via MFA extension |
| TestMFABackward | 5 | Backward: dQ, dK, dV correctness |
| TestPublicAPI | 7 | is_mfa_available, get_device_info, get_supported_configs (feature matrix) |
| TestEdgeCases | 8 | GQA, N=1, non-multiple seq, cross-attention, D mismatches |
| TestBackwardEdge | 4 | Backward at edge shapes, partial argnums |
| TestFlashAttentionAPI | 16 | API params: scale, causal, window_size (right>0 guard), backend, attn_bias |
| TestNativeGQA | 3 | Native GQA ratios 2/4/8 |
| TestSparseAttentionAPI | 6 | flash_attention_sparse: shapes, mask properties, dtype rejection |
| TestSparseAttentionKernel | 5 | Sparse STEEL kernel: causal-block, sliding window, all-false rows |
| TestM3M4Path | 2 | M3+ BK=32 routing via MFA_FORCE_GEN env override |
| TestSparseBackwardTiled | 7 | Tiled Python sparse backward correctness, GQA, value_and_grad |
| TestSparseBackwardSteel | 4 | Native STEEL sparse backward |
| TestFlashDecode | 8 | Flash Decode (N_q<=4, 2-phase split-KV) |
| TestM5Detection | 3 | is_m5_plus flag, gen>=17 threshold |
| TestRoPEFusion | 5 | flash_attention_rope: correctness, cache_seqlens, rope_3d |
| TestSpatialMasks | 9 | make_spatial_2d/3d/topk masks |
| TestSegmentMask | 6 | make_segment_mask, make_causal_segment_mask |
| TestAdaptiveWindowMask | 4 | make_adaptive_window_mask |
| TestVarlenAttention | 5 | flash_attention_varlen: basic + GQA |
| TestSteelVarlen | 7 | STEEL varlen kernel incl. D=512 fallback path |
| TestRoPE3D | 8 | 3D RoPE tables + flash_attention_rope(rope_3d=True) |
| TestLCSAMask | 5 | make_lcsa_mask |
| TestAxialMasks | 5 | make_axial_spatial_mask, make_axial_temporal_mask |
| TestDilatedTemporalMask | 4 | make_dilated_temporal_mask |
| TestSinkAndReferenceFrameMasks | 5 | make_sink_window_mask, make_reference_frame_mask |
| TestCrossStreamMask | 5 | make_cross_stream_mask |
| TestSoftcap | 4 | Softcap tanh-capping correctness |
| TestALiBi | 4 | ALiBi linear position biases |
| TestRoPENonInterleaved | 3 | interleaved=False (GPT-NeoX split-halves RoPE) |
| TestPerBatchCacheSeqlens | 3 | List/array per-batch cache offsets |
| TestHeadDimVMismatch | 4 | D_v != D_qk graceful fallback |
| TestKVCacheAppendUnified | 9 | flash_attention_kvcache k_new/v_new append mode |
| TestAttentionDropout | 4 | Training dropout (dropout_p) |
| TestReturnAttnWeights | 4 | return_attn_weights=True -> (out, weights) |
| TestPagedKVCache | 7 | PagedKVCache allocator operations |
| TestPagedKVCacheGA | 14 | PagedKVCache dual-pool, gather, block table |
| TestPackedFormats | 10 | QKV/KV packed tensor formats |
| TestSteelBackwardGQA | 1 | STEEL backward with GQA |
| TestSteelBackwardD256 | 3 | STEEL backward at D=256 |
| TestVarlenBackward | 3 | Varlen autograd via custom_function |
| TestPagedBackward | 9 | Paged attention dQ + dK_pages/dV_pages scatter |
| TestVarlenPacked | 4 | Varlen packed formats |
| TestSlidingWindow | 4 | Sliding window tile-skip forward |
| TestUnifiedKVCache | 15 | flash_attention_kvcache: dense/paged/flash-decode dispatch |
| TestReturnLSE | 4 | return_lse=True -> (out, logsumexp) |
| TestCacheBatchIdx | 2 | cache_batch_idx per-batch cache indexing |
| TestRotaryDim | 2 | rotary_dim partial RoPE |
| TestKVCacheRopeAppend | 3 | flash_attention_kvcache_rope_append |
| TestPagedSteelForward | 11 | Kernel-level paged STEEL forward |
| TestPagedFlashDecode | 4 | Flash Decode with paged KV |
| TestD512Forward | 6 | D=512 forward: d-split correctness |
| TestD512Backward | 4 | D=512 backward via STEEL bwd |

### Test classes — `test_mlx_lm_integration.py` (7 classes, 33 methods)

| Class | Methods | What it tests |
|-------|---------|---------------|
| TestPatchUnpatch | 5 | patch_mlx_lm / unpatch_mlx_lm lifecycle |
| TestSignatureCompatibility | 3 | mlx_lm 0.30+ signature compat |
| TestNumericalCorrectness | 3 | Numerical output match vs reference |
| TestQuantizedKVCache | 5 | Quantized KV cache fallback |
| TestPatchMLXLMVerbose | 4 | verbose=True logging |
| TestGetPatchStats | 6 | get_patch_stats() counters |
| TestCheckModelCompatibility | 7 | check_model_compatibility() |

---

## Benchmarks (`benchmarks/` — 12 files)

| File | Lines | What it benchmarks |
|------|-------|-------------------|
| `bench_all.py` | 216 | Orchestrator: runs all benchmarks |
| `bench_attention.py` | 142 | MFA vs SDPA: D=64/128/256/512, causal |
| `bench_backward.py` | 145 | Backward pass: MFA vs SDPA vjp |
| `bench_compile.py` | 196 | Metal JIT compilation time |
| `bench_kvcache.py` | 138 | KV cache decode step throughput |
| `bench_mlx_lm.py` | 204 | mlx_lm integration: tokens/sec |
| `bench_paged_kv.py` | 159 | Paged attention vs dense |
| `bench_rope_3d.py` | 112 | 3D RoPE attention throughput |
| `bench_segment.py` | 101 | Segment mask attention |
| `bench_softcap_alibi.py` | 133 | Softcap and ALiBi overhead |
| `bench_spatial_masks.py` | 325 | Spatial mask benchmarks |
| `bench_varlen.py` | 113 | Varlen vs padded attention |

---

## Examples (`examples/` — 5 files)

| File | Lines | Description |
|------|-------|-------------|
| `basic_attention.py` | 83 | Drop-in flash_attention quickstart |
| `kvcache_decode.py` | 80 | Single-token decode with KV cache |
| `paged_kv_inference.py` | 115 | Paged KV cache multi-sequence |
| `sliding_window.py` | 87 | Sliding window attention |
| `varlen_training.py` | 143 | Variable-length training loop |

---

## Key constraints

| Constraint | Value |
|-----------|-------|
| Supported head_dims | {64, 128, 256, 512} |
| Supported dtypes | {float16, bfloat16, float32} |
| Layout | BHND [B, H, N, D] row-major |
| TGP budget | <= 32 KB threadgroup memory |
| D=512 varlen/paged STEEL | Falls back to SDPA (no d-split in those generators) |
| STEEL backward D limit | D<=512 (f16/bf16 only) |
| Platform | macOS arm64, Python 3.10+, mlx >= 0.18.0 |

# mlx-mfa Code Inventory — v2.6.0

All numbers verified by running shell commands against the source tree.
Regenerated: 2026-03-11.

---

## Version

| Key | Value |
|-----|-------|
| `pyproject.toml` | `2.6.0` |
| `mlx_mfa/__init__.py` | `2.6.0` |
| Latest git tag | `v2.6.0` |

---

## Source files

### Python (`mlx_mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `mlx_mfa/__init__.py` | 192 | Public API re-exports, ABI check, `__version__` |
| `mlx_mfa/attention.py` | 5009 | All attention functions + helpers + `DispatchPolicy.SAGE` |
| `mlx_mfa/inference.py` | 628 | `InferenceContext`, `DenseKVCache`, `PagedKVCache`, `SageInferenceContext`, `QuantizedKVCache` |
| `mlx_mfa/masks.py` | 1129 | 15 mask builders |
| `mlx_mfa/quantize.py` | 280 | `quantize_per_block`, `dequantize`, `smooth_k`, `sage_block_sizes` |
| `mlx_mfa/dispatch_policy.py` | 368 | Shape-aware MFA/SDPA routing + calibration |
| `mlx_mfa/compile_metallib.py` | 358 | AOT kernel compilation to AIR metallibs |
| `mlx_mfa/integrations/mlx_lm.py` | 431 | `patch_mlx_lm` / `unpatch_mlx_lm` + enrichment |
| `mlx_mfa/integrations/__init__.py` | 0 | Package marker |
| **Python total** | **8395** | |

### C++ / Objective-C++ / Metal (`csrc/`)

| File | Lines | Purpose |
|------|-------|---------|
| `csrc/bindings.cpp` | 553 | nanobind module + Python bindings |
| `csrc/mfa_attention.cpp` | 2214 | `MFAttention` Primitive: `eval_gpu`, `vjp` |
| `csrc/mfa_attention.hpp` | 513 | Primitive header |
| `csrc/mfa_steel_fwd.cpp` | 3265 | STEEL V1/V2 forward: JIT source gen, dispatch |
| `csrc/mfa_steel_fwd.hpp` | 259 | STEEL forward header |
| `csrc/mfa_steel_bwd.cpp` | 1295 | STEEL backward dQ + dKV kernels |
| `csrc/mfa_steel_bwd.hpp` | 68 | STEEL backward header |
| `csrc/mfa_paged_gather.cpp` | 242 | Paged KV gather Metal kernel |
| `csrc/mfa_paged_gather.hpp` | 84 | Paged gather header |
| `csrc/mfa_sage_fwd.cpp` | 524 | SageAttention Primitive + Metal JIT; window support |
| `csrc/mfa_sage_fwd.hpp` | 68 | `MFASageParams`, `mfa_sage_forward` declaration |
| `csrc/mfa_shader_gen.cpp` | 305 | ccv-based shader generator (f32 path) |
| `csrc/mfa_shader_gen.hpp` | 59 | Shader gen header |
| `csrc/shader_cache.hpp` | 105 | `KernelType` enum, cache interface |
| `csrc/shader_cache.mm` | 422 | Objective-C++ Metal pipeline compilation |
| `csrc/async_v2_kernel.metal` | 1088 | Hardware DMA V2 kernel (simdgroup_async_copy; macOS ≤15) |
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
| **C++ total** | **16036** | |

**Total source (Python + C++)**: 24431 lines across 37 files.

---

## Public API (`mlx_mfa.__all__`)

55 symbols total (54 exports + `__version__`).

### Core attention (14 functions + 2 classes)

| Symbol | Brief description |
|--------|-----------------|
| `flash_attention` | Standard BHND attention; AUTO/MFA/SDPA/SAGE backends |
| `flash_attention_rope` | RoPE-fused attention (3D RoPE, interleaved/split, rotary_dim) |
| `flash_attention_rope_unified` | Unified RoPE entry point |
| `flash_attention_sparse` | Block-sparse attention with `block_mask` |
| `flash_attention_varlen` | Variable-length (jagged) sequences, cu_seqlens |
| `flash_attention_kvcache` | Unified KV-cache: dense read/append, paged read/append |
| `flash_attention_kvcache_rope_append` | Fused RoPE + KV-cache append |
| `flash_attention_paged` | Paged KV pool attention (block_table) |
| `flash_attention_qkv_packed` | Fused QKV tensor input |
| `flash_attention_kv_packed` | Fused KV tensor input |
| `flash_attention_varlen_qkv_packed` | Varlen + packed QKV |
| `flash_attention_varlen_kv_packed` | Varlen + packed KV |
| `flash_attention_speculative_verify` | Speculative decoding target log-probs |
| `flash_attention_splitfuse` | Combined prefill + decode in one call |
| `KVCacheProtocol` | Abstract protocol for all KV cache types |
| `DenseKVCache` | Dense KV cache (append + read) |

### Dispatch (1 class + 1 function)

| Symbol | Brief description |
|--------|-----------------|
| `DispatchPolicy` | Backend constants: `.AUTO`, `.MFA`, `.SDPA`, `.SAGE` |
| `calibrate_dispatch` | Benchmark device and save optimal routing thresholds |

### LLM helpers (1 function)

| Symbol | Brief description |
|--------|-----------------|
| `make_shared_prefix_cache` | Build shared prefix KV cache for multi-request reuse |

### Mask builders (15)

| Symbol | Description |
|--------|------------|
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
|--------|------------|
| `make_rope_3d_tables` | Build 3D rotary frequency tables for video |

### SageAttention (3 functions + 1 class)

| Symbol | Brief description |
|--------|-----------------|
| `sage_attention` | int8 Q/K attention; `window_size=` supported |
| `sage_attention_prequantized` | Uses pre-stored int8 from `QuantizedKVCache` |
| `sage_attention_kvcache` | Decode variant: N_q ≠ N_kv native |
| `QuantizedKVCache` | Pre-stores K as int8; O(1) quantize per decode step |

### KV cache / inference contexts (3 classes)

| Symbol | Brief description |
|--------|-----------------|
| `InferenceContext` | Stateful KV-cache lifecycle: `prefill()`, `step()`, `reset()` |
| `PagedInferenceContext` | Stateful paged KV-cache lifecycle |
| `SageInferenceContext` | Stateful sage decode with `QuantizedKVCache` |
| `PagedKVCache` | Python KV block allocator (dual-pool design) |

### Quantization utilities (5)

| Symbol | Returns | Description |
|--------|---------|------------|
| `quantize_per_block` | `(int8, float32)` | Per-block int8 quantize `[B,H,N,D]` tensor |
| `dequantize` | `float32` | Reconstruct fp32 from int8 + per-block scale |
| `smooth_k` | `(fp16/bf16, float32)` | Per-channel mean subtraction for K |
| `sage_output_correction` | `float32` | Legacy no-op (not called) |
| `sage_block_sizes` | `(int, int)` | Returns `(BQ, BK)` for given head_dim |

### AOT compilation (1 function)

| Symbol | Returns | Description |
|--------|---------|------------|
| `compile_metallib` | `dict` | Pre-compile STEEL V2 kernel configs to AIR metallibs (`~/.mlx_mfa/metallib/`) |

### Utilities (4 + `__version__`)

| Symbol | Returns | Description |
|--------|---------|------------|
| `is_mfa_available` | `bool` | True when C++ ext + Metal GPU present |
| `get_device_info` | `dict` | device_name, gpu_family_gen, is_m3_plus, is_m5_plus, chip_name |
| `get_supported_configs` | `dict` | head_dims, dtypes, extension_available, features (22 flags), kernel_types |
| `warmup_kernels` | `None` | Pre-compile JIT kernels for given head_dims/dtypes |
| `__version__` | `str` | Package version string |

---

## Metal kernel types (`csrc/shader_cache.hpp` — 12 active)

| Value | Name | Description |
|-------|------|------------|
| 0 | `AttentionForward` | ccv MFA forward (f32) |
| 1 | `AttentionBackwardDQ` | ccv MFA backward dQ |
| 2 | `AttentionBackwardDKV` | ccv MFA backward dKV |
| 3 | `SteelForward` | STEEL V1/V2 forward (all D; d-split for D=256/512) |
| 4 | `FlashDecodePartial` | Flash Decode Phase 1: partial attn per KV split |
| 5 | `FlashDecodeReduce` | Flash Decode Phase 2: LSE reduce over splits |
| 6 | `SteelBackwardDQ` | STEEL native backward dQ (f16/bf16, D≤512) |
| 7 | `SteelBackwardDKV` | STEEL native backward dKV (f16/bf16, D≤512) |
| 8 | `SteelVarlenForward` | STEEL varlen forward (D≤256) |
| 9 | `PagedKVGather` | Paged KV gather: pool to contiguous BHND |
| 10 | `PagedSteelForward` | STEEL forward with kernel-level paged KV (D≤256) |
| 11 | `SageForward` | int8 Q/K quantized attention; window support |
| — | `TensorOpsForward` | Reserved: Metal 4 cooperative tensors (M5+/A19+ only) |

---

## Tests

**Total: 562 pytest-collected tests**

| File | Classes | Methods |
|------|---------|---------|
| `tests/test_attention.py` | 75 | 405 (473 collected with parametrize) |
| `tests/test_mlx_lm_integration.py` | 8 | 38 |
| `tests/test_sage_attention.py` | 5 | 30 |
| `tests/test_inference_context.py` | 6 | 21 |

---

## Benchmarks (`benchmarks/` — 18 files)

| File | Lines | What it benchmarks |
|------|-------|-------------------|
| `bench_all.py` | 433 | fwd + bwd + window + sage (comprehensive) |
| `bench_attention.py` | 142 | MFA vs SDPA: D=64/128/256/512, causal |
| `bench_auto_dispatch_validation.py` | 107 | Auto dispatch correctness validation |
| `bench_backward.py` | 145 | Backward pass: MFA vs SDPA vjp |
| `bench_backward_matrix.py` | 145 | Backward pass across all shapes |
| `bench_compile.py` | 196 | Metal JIT compilation time |
| `bench_dispatch_matrix.py` | 168 | MFA vs SDPA across all shapes |
| `bench_kvcache.py` | 138 | KV cache decode step throughput |
| `bench_mlx_lm.py` | 204 | mlx_lm integration: tokens/sec |
| `bench_paged_kv.py` | 159 | Paged attention vs dense |
| `bench_rope_3d.py` | 112 | 3D RoPE attention throughput |
| `bench_sage.py` | 93 | sage_attention vs flash_attention |
| `bench_segment.py` | 101 | Segment mask attention |
| `bench_softcap_alibi.py` | 133 | Softcap and ALiBi overhead |
| `bench_spatial_masks.py` | 325 | Spatial mask benchmarks |
| `bench_v2.py` | 98 | STEEL V2 vs V1 vs SDPA: D=64/128 |
| `bench_v2_final.py` | 267 | Comprehensive: dense+window+split-K (primary) |
| `bench_varlen.py` | 113 | Varlen vs padded attention |

---

## Key constraints

| Constraint | Value |
|-----------|-------|
| Supported head_dims | {64, 128, 256, 512} |
| Supported dtypes | {float16, bfloat16, float32} |
| Layout | BHND [B, H, N, D] row-major |
| TGP budget | ≤ 32 KB threadgroup memory |
| STEEL V2 dense dispatch | D=64/128 only; D=256/512 dense → SDPA (v2.6.0+) |
| D=256/512 window/sparse | Always routes to MFA (tile-skip regardless of D) |
| D=512 varlen/paged STEEL | Falls back to SDPA |
| STEEL backward D limit | D≤512 (f16/bf16 only) |
| Sage autograd | Not supported — inference-only |
| `QuantizedKVCache` contiguity | Slices need `mx.contiguous()` before C++ dispatch |
| Platform | macOS arm64, Python 3.10+, mlx ≥ 0.18.0 |
| AOT metallib cache | `~/.mlx_mfa/metallib/` (sync V2), `mlx_mfa/precompiled/` (async DMA) |

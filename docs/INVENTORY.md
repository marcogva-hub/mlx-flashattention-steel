# mlx-mfa Code Inventory — v1.4.0

All numbers verified by running shell commands against the source tree.
Regenerated: 2026-03-10.

---

## Version

| Key | Value |
|-----|-------|
| `pyproject.toml` | `1.3.0` → bump to `1.4.0` in CP11 |
| `mlx_mfa/__init__.py` | `1.3.0` → bump to `1.4.0` in CP11 |
| Latest git tag | `v1.3.0` |

---

## Source files

### Python (`mlx_mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `mlx_mfa/__init__.py` | 190 | Public API re-exports, ABI check, `__version__` |
| `mlx_mfa/attention.py` | 4951 | All attention functions + helpers + `DispatchPolicy.SAGE` (CP8) |
| `mlx_mfa/inference.py` | 628 | `InferenceContext`, `DenseKVCache`, `PagedKVCache`, `SageInferenceContext`, **`QuantizedKVCache`** (CP6) |
| `mlx_mfa/masks.py` | 1129 | 15 mask builders |
| `mlx_mfa/quantize.py` | 280 | `quantize_per_block`, `dequantize`, `smooth_k`, `sage_block_sizes` |
| `mlx_mfa/dispatch_policy.py` | 359 | Shape-aware MFA/SDPA routing + calibration |
| `mlx_mfa/integrations/mlx_lm.py` | 431 | `patch_mlx_lm` / `unpatch_mlx_lm` + enrichment |
| `mlx_mfa/integrations/__init__.py` | 0 | Package marker |
| **Python total** | **7968** | |

### C++ / Objective-C++ (`csrc/`)

| File | Lines | Purpose |
|------|-------|---------|
| `csrc/bindings.cpp` | 553 | nanobind module + Python bindings |
| `csrc/mfa_attention.cpp` | 2087 | `MFAttention` Primitive: `eval_gpu`, `vjp` |
| `csrc/mfa_attention.hpp` | 513 | Primitive header; `MFASageForward::Params` gains `window_left/right` (CP7) |
| `csrc/mfa_steel_fwd.cpp` | 3259 | STEEL forward: JIT source gen, dispatch |
| `csrc/mfa_steel_fwd.hpp` | 259 | STEEL forward header |
| `csrc/mfa_steel_bwd.cpp` | 1295 | STEEL backward dQ + dKV kernels |
| `csrc/mfa_steel_bwd.hpp` | 68 | STEEL backward header |
| `csrc/mfa_paged_gather.cpp` | 242 | Paged KV gather Metal kernel |
| `csrc/mfa_paged_gather.hpp` | 84 | Paged gather header |
| `csrc/mfa_sage_fwd.cpp` | 524 | SageAttention Primitive + Metal JIT; **window support** (CP7) |
| `csrc/mfa_sage_fwd.hpp` | 68 | `MFASageParams` (gains `window_right`), `mfa_sage_forward` declaration |
| `csrc/mfa_shader_gen.cpp` | 305 | ccv-based shader generator (legacy) |
| `csrc/mfa_shader_gen.hpp` | 59 | Shader gen header |
| `csrc/shader_cache.hpp` | 99 | `KernelType` enum, cache interface |
| `csrc/shader_cache.mm` | 263 | Objective-C++ Metal pipeline compilation |
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
| **C++ total** | **14650** | |

**Total source (Python + C++)**: 22618 lines across 36 files.

---

## Public API (`mlx_mfa.__all__`)

### Core attention (14 functions + 1 class)

| Symbol | Brief description |
|--------|------------------|
| `flash_attention` | Standard BHND attention; MFA/SDPA/**Sage** backend; `backend="sage"` new (CP8) |
| `flash_attention_rope` | RoPE-fused attention (3D RoPE, interleaved/split, rotary_dim) |
| `flash_attention_rope_unified` | Unified RoPE entry point (v1.1.0) |
| `flash_attention_sparse` | Block-sparse attention with `block_mask` |
| `flash_attention_varlen` | Variable-length (jagged) sequences, cu_seqlens |
| `flash_attention_kvcache` | Unified KV-cache: dense, paged, append |
| `flash_attention_kvcache_rope_append` | Fused RoPE + KV-cache append |
| `flash_attention_paged` | Paged KV pool attention (block_table) |
| `flash_attention_qkv_packed` | Fused QKV tensor input |
| `flash_attention_kv_packed` | Fused KV tensor input |
| `flash_attention_varlen_qkv_packed` | Varlen + packed QKV |
| `flash_attention_varlen_kv_packed` | Varlen + packed KV |
| `flash_attention_speculative_verify` | Speculative decoding target log-probs (v1.1.0) |
| `flash_attention_splitfuse` | Combined prefill + decode in one call (v1.1.0) |
| `PagedKVCache` | Python KV block allocator (dual-pool design) |

### Dispatch (1 class)

| Symbol | Brief description |
|--------|------------------|
| `DispatchPolicy` | Backend constants: `.AUTO`, `.MFA`, `.SDPA`, `.SAGE` (CP8) |

### LLM helpers (1 function)

| Symbol | Brief description |
|--------|------------------|
| `make_shared_prefix_cache` | Build shared prefix KV cache for multi-request reuse (v1.1.0) |

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

### SageAttention *(v1.2.0–v1.4.0)*

| Symbol | Brief description |
|--------|------------------|
| `sage_attention` | int8 Q/K attention; **`window_size=`** added (CP7) |
| `sage_attention_prequantized` | Uses pre-stored int8 from `QuantizedKVCache`; bypass re-quantize |
| `sage_attention_kvcache` | Decode variant: N_q ≠ N_kv native |

### KV cache classes

| Symbol | Brief description |
|--------|------------------|
| `InferenceContext` | Stateful KV-cache lifecycle: `prefill()`, `step()`, `reset()` (v1.2.1) |
| `DenseKVCache` | Base dense KV cache (v1.3.0) |
| `QuantizedKVCache` | **New v1.4.0 (CP6)** — pre-stores K as int8; O(1) quantize per decode step |
| `SageInferenceContext` | Stateful sage decode with `QuantizedKVCache` (v1.3.0) |

### Quantization utilities *(v1.2.0)*

| Symbol | Returns | Description |
|--------|---------|-------------|
| `quantize_per_block` | `(int8, float32)` | Per-block int8 quantize `[B,H,N,D]` tensor |
| `dequantize` | `float32` | Reconstruct fp32 from int8 + per-block scale |
| `smooth_k` | `(fp16/bf16, float32)` | Per-channel mean subtraction for K |
| `sage_output_correction` | `float32` | Legacy: smooth_k bias compensation (no-op; not called) |
| `sage_block_sizes` | `(int, int)` | Returns `(BQ, BK)` for given head_dim |

### Utilities (3 + `__version__`)

| Symbol | Returns | Description |
|--------|---------|-------------|
| `is_mfa_available` | `bool` | True when C++ ext + Metal GPU present |
| `get_device_info` | `dict` | device_name, gpu_family_gen, is_m3_plus, is_m5_plus, chip_name |
| `get_supported_configs` | `dict` | head_dims, dtypes, extension_available, features (16 flags), kernel_types |
| `__version__` | `str` | Package version string |

---

## Metal kernel types (`csrc/shader_cache.hpp` — 12 active)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `AttentionForward` | ccv MFA forward (legacy) |
| 1 | `AttentionBackwardDQ` | ccv MFA backward dQ |
| 2 | `AttentionBackwardDKV` | ccv MFA backward dKV |
| 3 | `SteelForward` | STEEL cooperative forward (all D; d-split for D=512) |
| 4 | `FlashDecodePartial` | Flash Decode Phase 1: partial attn per KV split |
| 5 | `FlashDecodeReduce` | Flash Decode Phase 2: LSE reduce over splits |
| 6 | `SteelBackwardDQ` | STEEL native backward dQ (f16/bf16, D≤512) |
| 7 | `SteelBackwardDKV` | STEEL native backward dKV (f16/bf16, D≤512) |
| 8 | `SteelVarlenForward` | STEEL varlen forward (D≤256; D=512 → SDPA fallback) |
| 9 | `PagedKVGather` | Paged KV gather: pool to contiguous BHND |
| 10 | `PagedSteelForward` | STEEL forward with kernel-level paged KV (D≤256) |
| 11 | `SageForward` | int8 Q/K quantized attention; **window support** (CP7) |
| — | `TensorOpsForward` | Reserved: Metal 4 cooperative tensors (M5+/A19+ only) |

---

## Tests

**Total: 553 pytest-collected tests**

| File | Classes | Methods |
|------|---------|---------|
| `tests/test_attention.py` | 73 | 396 |
| `tests/test_mlx_lm_integration.py` | 8 | 38 |
| `tests/test_sage_attention.py` | 5 | 30 |
| `tests/test_inference_context.py` | 6 | 21 |

### New test classes — v1.4.0 (CP6–CP7)

| Class | Methods | Track | What it tests |
|-------|---------|-------|---------------|
| `TestQuantizedKVCache` | 3 | CP6 | append shapes, O(1) decode step correctness, prequantized vs sage |
| `TestSageWindow` | 4 | CP7 | left window, both-sides window, shape preserved, no-window unchanged |

---

## Benchmarks (`benchmarks/` — 17 files)

| File | Lines | What it benchmarks |
|------|-------|-------------------|
| `bench_all.py` | 433 | **v1.4.0** — fwd + bwd + window + **sage** (CP9) |
| `bench_attention.py` | 142 | MFA vs SDPA: D=64/128/256/512, causal |
| `bench_auto_dispatch_validation.py` | 107 | Auto dispatch correctness validation |
| `bench_backward.py` | 145 | Backward pass: MFA vs SDPA vjp |
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
| `bench_v2_final.py` | 267 | Comprehensive: dense+window+split-K |
| `bench_varlen.py` | 113 | Varlen vs padded attention |

---

## Key constraints

| Constraint | Value |
|-----------|-------|
| Supported head_dims | {64, 128, 256, 512} |
| Supported dtypes | {float16, bfloat16, float32} |
| Layout | BHND [B, H, N, D] row-major |
| TGP budget | ≤ 32 KB threadgroup memory |
| D=512 varlen/paged STEEL | Falls back to SDPA (no d-split in those generators) |
| STEEL backward D limit | D≤512 (f16/bf16 only) |
| Sage autograd | **Not supported** — inference-only |
| `QuantizedKVCache` contiguity | Slices need `mx.contiguous()` / `.flatten().reshape()` before C++ dispatch |
| Platform | macOS arm64, Python 3.10+, mlx ≥ 0.18.0 |

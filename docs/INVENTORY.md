# mlx-mfa Code Inventory — v1.2.1

All numbers verified by running shell commands against the source tree.
Regenerated: 2026-03-09.

---

## Version

| Key | Value |
|-----|-------|
| `pyproject.toml` | `1.2.1` |
| `mlx_mfa/__init__.py` | `1.2.1` |
| Latest git tag | `v1.2.1` |

---

## Source files

### Python (`mlx_mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `mlx_mfa/__init__.py` | 166 | Public API re-exports, ABI check, `__version__` |
| `mlx_mfa/attention.py` | 4082 | All attention functions + helpers |
| `mlx_mfa/inference.py` | 264 | **New v1.2.1** — `InferenceContext` stateful KV-cache lifecycle |
| `mlx_mfa/masks.py` | 1129 | 15 mask builders |
| `mlx_mfa/quantize.py` | 260 | **New v1.2.0** — `quantize_per_block`, `dequantize`, `smooth_k`, `sage_block_sizes` |
| `mlx_mfa/integrations/mlx_lm.py` | 428 | `patch_mlx_lm` / `unpatch_mlx_lm` + enrichment |
| `mlx_mfa/integrations/__init__.py` | 0 | Package marker |
| **Python total** | **6329** | |

### C++ / Objective-C++ (`csrc/`)

| File | Lines | Purpose |
|------|-------|---------|
| `csrc/bindings.cpp` | 447 | nanobind module + Python bindings |
| `csrc/mfa_attention.cpp` | 1799 | `MFAttention` Primitive: `eval_gpu`, `vjp` |
| `csrc/mfa_attention.hpp` | 437 | Primitive header |
| `csrc/mfa_steel_fwd.cpp` | 3259 | STEEL forward: JIT source gen, dispatch |
| `csrc/mfa_steel_fwd.hpp` | 259 | STEEL forward header (+`mask_batch/head_stride`, `window_right`) |
| `csrc/mfa_steel_bwd.cpp` | 1295 | STEEL backward dQ + dKV kernels |
| `csrc/mfa_steel_bwd.hpp` | 68 | STEEL backward header |
| `csrc/mfa_paged_gather.cpp` | 242 | Paged KV gather Metal kernel |
| `csrc/mfa_paged_gather.hpp` | 84 | Paged gather header |
| `csrc/mfa_sage_fwd.cpp` | 446 | **New v1.2.0** — SageAttention Primitive + Metal JIT source gen |
| `csrc/mfa_sage_fwd.hpp` | 67 | **New v1.2.0** — `MFASageParams`, `mfa_sage_forward` declaration |
| `csrc/mfa_shader_gen.cpp` | 305 | ccv-based shader generator (legacy) |
| `csrc/mfa_shader_gen.hpp` | 59 | Shader gen header |
| `csrc/shader_cache.hpp` | 91 | `KernelType` enum, cache interface |
| `csrc/shader_cache.mm` | 235 | Objective-C++ Metal pipeline compilation |
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
| **C++ total** | **14241** | |

**Total source (Python + C++)**: 20570 lines across 34 files.

---

## Public API (`mlx_mfa.__all__` — 43 symbols)

### Core attention (14 functions + 1 class)

| Symbol | Brief description |
|--------|------------------|
| `flash_attention` | Standard BHND attention; MFA or SDPA backend |
| `flash_attention_rope` | RoPE-fused attention (3D RoPE, interleaved/split, rotary_dim) |
| `flash_attention_rope_unified` | **New v1.1.0** — unified RoPE entry point; replaces rope + rope_append |
| `flash_attention_sparse` | Block-sparse attention with `block_mask` |
| `flash_attention_varlen` | Variable-length (jagged) sequences, cu_seqlens |
| `flash_attention_kvcache` | Unified KV-cache: dense, paged, append (k_new/v_new) |
| `flash_attention_kvcache_rope_append` | Fused RoPE + KV-cache append (thin wrapper of rope_unified) |
| `flash_attention_paged` | Paged KV pool attention (block_table) |
| `flash_attention_qkv_packed` | Fused QKV tensor input |
| `flash_attention_kv_packed` | Fused KV tensor input |
| `flash_attention_varlen_qkv_packed` | Varlen + packed QKV |
| `flash_attention_varlen_kv_packed` | Varlen + packed KV |
| `flash_attention_speculative_verify` | **New v1.1.0** — speculative decoding target log-probs |
| `flash_attention_splitfuse` | **New v1.1.0** — combined prefill + decode in one call |
| `PagedKVCache` | Python KV block allocator (dual-pool design) |

### LLM helpers (1 function)

| Symbol | Brief description |
|--------|------------------|
| `make_shared_prefix_cache` | **New v1.1.0** — build shared prefix KV cache for multi-request reuse |

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

### SageAttention *(v1.2.0, 1 function)*

| Symbol | Brief description |
|--------|------------------|
| `sage_attention` | **New v1.2.0** — int8 quantized Q/K attention; fp16 GEMM + smooth_k |

### Quantization utilities *(v1.2.0, 5 symbols)*

| Symbol | Returns | Description |
|--------|---------|-------------|
| `quantize_per_block` | `(int8, float32)` | **New v1.2.0** — per-block int8 quantize `[B,H,N,D]` tensor |
| `dequantize` | `float32` | **New v1.2.0** — reconstruct fp32 from int8 + per-block scale |
| `smooth_k` | `(fp16/bf16, float32)` | **New v1.2.0** — per-channel mean subtraction for K; returns `(k_smooth, k_mean)` |
| `sage_output_correction` | `float32` | Legacy: smooth_k bias compensation (not used by `sage_attention`) |
| `sage_block_sizes` | `(int, int)` | **New v1.2.0** — returns `(BQ, BK)` for given head_dim |

### InferenceContext *(v1.2.1, 1 class)*

| Symbol | Brief description |
|--------|------------------|
| `InferenceContext` | **New v1.2.1** — stateful KV-cache wrapper: `prefill()`, `step()`, `reset()`, context-manager |

### Utilities (3 + `__version__`)

| Symbol | Returns | Description |
|--------|---------|-------------|
| `is_mfa_available` | `bool` | True when C++ ext + Metal GPU present |
| `get_device_info` | `dict` | device_name, gpu_family_gen, is_m3_plus, is_m5_plus, chip_name |
| `get_supported_configs` | `dict` | head_dims, dtypes, extension_available, features (23 flags), kernel_types |
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
| 6 | `SteelBackwardDQ` | STEEL native backward dQ (f16/bf16, D<=512) |
| 7 | `SteelBackwardDKV` | STEEL native backward dKV (f16/bf16, D<=512) |
| 8 | `SteelVarlenForward` | STEEL varlen forward (D<=256; D=512 -> SDPA fallback) |
| 9 | `PagedKVGather` | Paged KV gather: pool to contiguous BHND |
| 10 | `PagedSteelForward` | STEEL forward with kernel-level paged KV (D<=256) |
| 11 | `SageForward` | **New v1.2.0** — int8 Q/K quantized attention forward |
| — | `TensorOpsForward` | Reserved: Metal 4 cooperative tensors (M5+/A19+ only) |

---

## Tests

**Total: 486 pytest-collected tests** (423 test methods + parametrized expansion)

| File | Classes | Methods |
|------|---------|---------|
| `tests/test_attention.py` | 61 | 341 |
| `tests/test_mlx_lm_integration.py` | 8 | 38 |
| `tests/test_sage_attention.py` | 3 | 23 |
| `tests/test_inference_context.py` | 6 | 21 |

### New test classes — v1.2.1

| Class | Methods | Track | What it tests |
|-------|---------|-------|---------------|
| `TestWindowRight` | 8 | LA | `window_size=(left, right)` with right bound: correctness vs dense, fallback, errors |
| `TestBlockMask4D` | 14 | LB | 3-D/4-D block masks: shape validation, broadcast equivalence, per-head differences |
| `TestInferenceContextConstruct` | 3 | LC | `InferenceContext` init, repr, defaults |
| `TestInferenceContextPrefill` | 4 | LC | prefill output shape, cache state, overflow guard |
| `TestInferenceContextStep` | 5 | LC | step output, cache growth, multi-step, GQA |
| `TestInferenceContextReset` | 3 | LC | reset, prefill-then-reset, chaining |
| `TestInferenceContextManager` | 3 | LC | context manager: auto-reset, nested |
| `TestInferenceContextGQA` | 3 | LC | GQA shapes in prefill + step |

### New test classes — v1.2.0

| Class | Methods | Track | What it tests |
|-------|---------|-------|---------------|
| `TestQuantizeUtils` | 7 | KA | `quantize_per_block`, `dequantize`, `smooth_k`, `sage_block_sizes` |
| `TestSageAPI` | 7 | KC | `sage_attention()` interface: shapes, dtypes, NaN, smooth_k toggle, supported configs |
| `TestSageKernel` | 9 | KC | Numerical correctness (requires C++ ext): D=64/128/256, causal, GQA 2:1, batch>1 |

### New test classes — v1.1.0

| Class | Methods | Track | What it tests |
|-------|---------|-------|---------------|
| `TestRoPEUnified` | 7 | JB | `flash_attention_rope_unified`: standalone, cache-append, first-step, rope_3d |
| `TestPagedAppend` | 2 | JC | `k_new + block_table` combined: output correctness, pool shape |
| `TestSpeculativeVerify` | 4 | JD | `flash_attention_speculative_verify`: shape, lse, logprobs |
| `TestSharedPrefixCache` | 3 | JD | `make_shared_prefix_cache`: shapes, reuse |
| `TestSplitFuse` | 3 | JD | `flash_attention_splitfuse`: shapes, finite |
| `TestCrossAttentionKVCache` | 3 | JF | Cross-attention: shape, N_q=1, autograd |
| `TestTrackJEEnrichment` | 5 | JE | `patch_mlx_lm` stats, verbose_dispatch, KNOWN_MODEL_CONFIGS |

---

## Benchmarks (`benchmarks/` — 13 files)

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
| `bench_sage.py` | 93 | **New v1.2.0** — sage_attention vs flash_attention: N=512–4096, with/without smooth_k |
| `bench_segment.py` | 101 | Segment mask attention |
| `bench_softcap_alibi.py` | 133 | Softcap and ALiBi overhead |
| `bench_spatial_masks.py` | 325 | Spatial mask benchmarks |
| `bench_varlen.py` | 113 | Varlen vs padded attention |

---

## Examples (`examples/` — 6 files)

| File | Lines | Description |
|------|-------|-------------|
| `basic_attention.py` | 83 | Drop-in flash_attention quickstart |
| `cross_attention.py` | 104 | **New v1.1.0** — encoder-decoder cross-attention with GQA + autograd |
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
| Paged append + cache_batch_idx | `NotImplementedError` (cannot express without in-place updates) |
| Platform | macOS arm64, Python 3.10+, mlx >= 0.18.0 |

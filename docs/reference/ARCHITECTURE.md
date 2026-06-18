# mlx-mfa Architecture

Version: **2.27.0**

## 1) System Overview

`mlx-mfa` is a hybrid Python + C++/Metal package that integrates with MLX:

```text
Python API/runtime (mlx_mfa.*)
  -> nanobind extension (csrc/bindings.cpp)
    -> MFA Primitive dispatch (csrc/mfa_attention.cpp)
      -> JIT shader generation + pipeline cache
        -> Metal kernels (STEEL V2/V3/V4/V5, Sage, paged helpers, TurboQuant)
```

Key principle: keep dense production routing conservative and benchmark-backed,
while exposing advanced serving functionality through explicit runtime APIs.

## 2) Production vs Research Paths

- Production default: **V2 dense** where policy shows wins.
- Conservative fallback: **MLX SDPA** where wins are not established.
- Narrow promotion: D=256 causal long-N regimes only.
- Non-promotion outcomes retained:
  - D=512 remains SDPA-default
  - native dense backward remains non-default
- Experimental families remain opt-in: V3/V4/V5.

## 3) Core Runtime Surface

Main runtime entry point:
- `create_decode_runtime(...) -> DecodeRuntime`

`DecodeRuntime` unifies:
- dense decode/prefill
- paged decode/prefill
- packed query layout handling for paged varlen
- chunked prefill
- prefix reuse integration
- speculative draft/verify integration
- splitfuse helper integration
- runtime metadata for selected backend/cache/query-layout state

Lower-level context factory remains available:
- `create_inference_context(...)`

## 4) Attention API Families

Primary callable families:
- Dense/general: `flash_attention(...)`
- KV-cache/decode: `flash_attention_kvcache(...)`
- Paged: `flash_attention_paged(...)`
- Paged + packed varlen query: `flash_attention_paged_varlen(...)`
- Varlen training packed tensors: `flash_attention_varlen(...)`
- Sparse/window masks: `flash_attention_sparse(...)`
- Splitfuse helper path: `flash_attention_splitfuse(...)`
- Speculative verify helpers:
  - `flash_attention_speculative_verify(...)`
  - `flash_attention_speculative_verify_paged(...)`
- TurboQuant fused: `flash_attention_paged_varlen_turboquant(...)`

## 5) Serving-Oriented Flow Architecture

### 5.1 Paged + packed varlen

- Supports packed queries with per-sequence boundaries (`cu_seqlens_q`) over
  paged KV pools.
- Equal query lengths can use batched paged fast path.
- Heterogeneous query lengths use fused PagedVarlenForward kernel (single dispatch).

### 5.2 Paged continuous batching/remap

- Explicit remap via `cache_batch_idx` for scheduler-controlled active order.
- Runtime helpers support batched prefill/step and remapped packed-varlen calls.

### 5.3 Chunked prefill

- `DecodeRuntime.chunked_prefill(...)` provides explicit chunk boundaries for
  interleaving/scheduling.
- Designed as serving control capability; not assumed to improve total
  throughput on M1 Max.

### 5.4 Runtime-managed prefix reuse

- Prefix state can be registered/seeded/reused via runtime methods.
- Intended to reduce orchestration fragmentation vs helper-only wiring.

### 5.5 Runtime speculative decode

- `DecodeRuntime.speculative_step(...)` wraps verify output into explicit
  accept/reject bookkeeping (mask + accepted prefix length + accepted/rejected
  ids).
- Dense runtime path is primary; paged support is narrower and explicit.

### 5.6 Splitfuse runtime deepening

- Splitfuse is available via runtime helpers, including decode-step focused
  path (`splitfuse_step(...)`).
- Includes a narrow page-native paged decode-only route to reduce bridge glue.

## 6) Cache Architecture

## 6.1 Concrete cache classes

- `DenseKVCache`
- `PagedKVCache`
- `QuantizedKVCache`

## 6.2 Adapter/capability layer

`mlx_mfa.kv_cache` introduces cache abstraction components:
- `KVCacheCapabilities`
- `KVCacheAdapter` + concrete adapters
- `adapt_kv_cache(...)`
- `resolve_context_cache(...)`
- `resolve_context_cache_adapter(...)`

Goal: runtime code relies on capabilities rather than concrete cache internals.

## 6.3 Hybrid cache behavior

`HybridKVCache` now has real behavior:
- hot/cold/offloaded residency state
- promotion on access
- demotion/eviction under pressure
- reload/promotion back into hot tier
- prefetch intent hooks and runtime-visible metadata

This is a **minimal local offload milestone**, not distributed offload.

## 6.4 External cache adapter groundwork

`mlx_mfa.external_cache` provides extension points:
- `ExternalKVCacheAdapter`
- `ExternalKVCacheCapabilities`
- `LocalHostKVStoreAdapter` (first local backend)

This defines a future LMCache-like integration surface without claiming full
remote backend support in the freeze state.

## 7) TurboQuant KV Compression Architecture (v2.21.0–v2.23.0)

Three-phase integration of training-free KV cache compression:

### 7.1 Compression pipeline

```text
Input: fp16 [B, H, S, D]
  -> WHT rotation (Walsh-Hadamard Transform, orthogonal)
  -> Per-channel scalar quantization (PolarQuant centroids)
  -> 2-bit/3-bit/4-bit index packing (2 indices per byte)
  -> uint8 packed output [B, H, S, D/2]
```

### 7.2 Phase progression

| Phase | K path | V path | Output correction |
|-------|--------|--------|-------------------|
| 1 (v2.21.0) | Decompress before attention | Decompress before attention | None |
| 2 (v2.22.0) | Fused: Metal kernel reads packed K | fp16 V pool | None |
| 3 (v2.23.0) | Fused: Metal kernel reads packed K | Fused: Metal kernel reads packed V | WHT inverse on output |

### 7.3 V rotation asymmetry

K rotation cancels in Q@K^T: `R(Q) @ R(K)^T = Q @ K^T` (WHT is orthogonal).
V rotation does NOT cancel: `O_tq = P @ R(V) = R(P @ V) = R(O)`.
Phase 3 applies inverse WHT to the output after the kernel returns.
WHT is self-inverse: `R^{-1} = R`.

### 7.4 Metal kernel architecture

`MFAPagedVarlenTQForward` primitive (`csrc/mfa_steel_paged_varlen_tq_fwd.cpp`):
- K gather: centroid lookup from TGP-cached centroids + per-token scale
- V gather: same centroid+scale pattern, gated by `tq_v_enabled` uniform branch
- Centroids loaded once into threadgroup memory (16-element array)
- Buffer layout: Q(0), k_pool_tq(1), v_pool(2), O(3), L(4), params(5),
  cu_seqlens_q(6), tile_offsets(7), block_table(8), seq_lens_kv(9),
  centroids(10), k_scales(11), v_pool_tq(12), v_centroids(13), v_scales(14)

### 7.5 Runtime integration

`TurboQuantPagedInferenceContext` manages:
- Dual pools: uint8 K pool + uint8 V pool (Phase 3) or fp16 V pool (Phase 2)
- Auto-compression on `append(k, v)` via `pack_k_for_metal` / `pack_v_for_metal`
- Auto Q rotation with WHT before calling fused kernel
- `create_decode_runtime(turboquant=True)` instantiates this context

## 7.6 TurboQuant Phase 4 — Optimal Packing + WHT Fusion (v2.24.0)

- Optimal 3-bit bit-planar packing achieves **5.33× compression** (vs 4× with
  2 indices/byte): 3 quantization indices packed into 1 byte using bit-plane layout.
- WHT (Walsh-Hadamard Transform) rotation fused directly into the Metal kernel,
  eliminating the separate Python-side WHT pre/post-processing step.
- Packing/unpacking implemented in Metal via compile-time `PACK_BITS` template
  parameter; centroids adapted to the bit-planar index encoding.

## 7.7 SVDQuant — W4A16 + SVD Low-Rank Correction (v2.25.0)

- `SVDQuantLinear`: drop-in replacement for `nn.Linear` that stores the weight
  matrix in 4-bit quantized form plus an optional rank-r FP16 SVD residual
  correction (`U @ V^T` additive term after dequantization).
- `quantize_model()`: tree walker that replaces `nn.Linear` layers with
  `SVDQuantLinear` in-place, with configurable rank and group size.
- Located in `mlx_mfa/svdquant/linear.py` and `mlx_mfa/svdquant/quantize.py`.
- Activation remains in FP16 throughout; only weights are quantized.

## 7.8 GNA Native Kernel (v2.26.0)

- Native Metal kernel with inline 3D window check, replacing the sparse-path
  `make_gna_mask()` + `flash_attention_sparse()` fallback for forward pass.
- Two-level masking: `gna_tile_active()` for tile-level skip (avoids loading
  entire K/V tiles outside the GNA window) + per-element window mask applied
  inside the tile.
- Forward-only (no VJP); D=128, f16/bf16.
- Falls back to sparse path for backward and for configs outside D=128/f16/bf16.
- Located in `csrc/mfa_gna_fwd.cpp`, dispatched as `GNAForward = 24`.

## 7.9 Native `attn_bias` Metal Kernel (v2.27.0)

- Additive bias on attention logits computed inside the V2 STEEL tiling loop,
  applied after Q@K^T GEMM, before softmax. Bias is multiplied by `log2e`
  (scores are in log2 domain).
- **Mode 1** `[1,1,1,Nkv]`: single scalar per K position, broadcast to all
  Q rows/heads/batches. One `device half` read per K tile position.
- **Mode 2** `[1,H,1,Nkv]`: per-head per-KV bias. Indexed as
  `bias[head_idx * Nkv + k_pos]`.
- **Modes 0/3** (full bias): fall back to SDPA (would require BQ×BK tile loads).
- Compile-time gated: `#define HAS_ATTN_BIAS 0/1` and `ATTN_BIAS_MODE 1/2`.
  Zero overhead when `attn_bias=None`.
- `KernelKey.has_attn_bias` (bool) + `attn_bias_mode` (uint8_t) in
  `shader_cache.hpp`. Buffer index 10 for bias tensor.
- Split-K dispatch excluded when `has_attn_bias=true` (split-K partial kernel
  doesn't implement bias). Falls back to single-pass V2.
- Metallib pipelines (async + precompiled) bypassed when `has_attn_bias=true`
  (metallibs are pre-built without bias code).

## 8) Native Extension Architecture

Core native files:
- `csrc/mfa_attention.cpp`: primitive dispatch and routing
- `csrc/mfa_env.hpp`: MFAEnvConfig singleton (env var caching)
- `csrc/mfa_steel_fwd.cpp` + `csrc/mfa_steel_fwd_v2.cpp`: dense forward family
- `csrc/mfa_steel_fwd_v3.hpp/cpp`: V3 separate K/V smem kernel
- `csrc/mfa_steel_fwd_v5.hpp/cpp`: V5 D-blocked kernel (experimental)
- `csrc/mfa_steel_bwd.cpp`: native backward kernels (gated non-default)
- `csrc/mfa_sage_fwd.cpp`: Sage path
- `csrc/mfa_steel_paged_varlen_tq_fwd.hpp/cpp`: TurboQuant paged varlen kernel
- `csrc/mfa_gna_fwd.hpp/.cpp`: GNA native forward kernel JIT generator
- `csrc/mfa_paged_gather.cpp` / `csrc/mfa_scatter.cpp`: paged helpers
- `csrc/shader_cache.mm`: pipeline compilation/cache
- `mlx_mfa/svdquant/`: SVDQuantLinear (W4A16 + SVD low-rank correction)

### 8.0 KernelType enum (shader_cache.hpp)

Selected entries relevant to production dispatch:

| Value | Name | Description |
|-------|------|-------------|
| 0 | AttentionForward | ccv legacy path (f32) |
| 3 | SteelForward | STEEL V1 (D>128) |
| 4 | FlashDecodePartial | Flash Decode Phase 1 |
| 5 | FlashDecodeReduce | Flash Decode Phase 2 |
| 16 | SteelForwardV2 | STEEL V2 production default |
| 17 | SteelV2SplitKPartial | V2 split-K Phase 1 |
| 18 | SteelV2DSplit256 | D=256 two-pass D-split |
| 19 | SteelV2DSplit512 | D=512 four-pass D-split |
| 20 | SteelForwardV3 | V3 separate K/V smem (opt-in) |
| 21 | SteelForwardV4 | V4 direct device K reads (opt-in) |
| 23 | SteelForwardV5 | V5 D-blocked BK=128 (opt-in) |
| 24 | GNAForward | GNA inline 3D window (no block_mask) |
| — | (V2 with has_attn_bias) | Additive bias modes 1/2 via SteelForwardV2 |
| 27 | PagedVarlenForward | Fused packed varlen + paged KV |
| 28 | PagedVarlenTQForward | TurboQuant packed uint8 K/V + centroids |

### 8.1 MFAEnvConfig (v2.20.0)

Static singleton (`csrc/mfa_env.hpp`) that caches all `MFA_*` env vars at
first access. Eliminates per-dispatch `std::getenv()` syscall overhead.

**Cached fields** (read once): `force_gen`, `v2_force_bk`, `v3_force_bk_d64`,
`v3_force_bk_d128`, `v5_force_bk`, `v5_force_bd_tile`, etc.

**Live-read static methods** (uncached): `enable_v3()`, `enable_v4()`,
`enable_v5()`, `disable_v2()`, `force_v2()`, `force_splitk()`. These must
remain live-read because Python tests use `os.environ` patching at runtime.

`invalidate()` forces re-read of cached fields (test/bench use only).

### 8.2 Forward Dispatch Cascade

```
f32 → ccv legacy path
f16/bf16:
  Flash Decode (N≤4, S≥256) → split-KV two-phase
  V2 split-K (under-occupied grid) → V2 with parallel reduction
  V4 (M3+, MFA_ENABLE_V4=1) → direct device K reads
  V5 (MFA_ENABLE_V5=1) → D-blocked, Q in registers
  V3 (B*H≥4, causal, D=64 N≥4096 or D=128 N≥2048) → separate K/V smem
  V2 single-pass → production default
  V1 (D>128) → original STEEL kernel
```

## 9) Documentation and Historical Separation

Active references:
- `README.md`
- `docs/API_MANUAL.md`
- `docs/benchmarks/RESULTS.md`
- `RESULTS.md`

Historical branch/track artifacts:
- `devnotes/` (organized by pass/track)

This separation is intentional for freeze-readability.

## 10) Deferred Work

Deferred until future continuation (likely newer hardware generation):
- remote/distributed offload backends via external adapter contract
- broader speculative scheduler integration
- new hardware-family kernel redesign work

## 11) LLM Serving Layer Status (v2.27.0)

The serving layer is considered production-ready for local inference.
See `docs/SERVING_GUIDE.md` for usage guide.

| Component | Status |
|---|---|
| Dense decode runtime | Production |
| Paged KV (batched/packed) | Production (fused PagedVarlenForward kernel) |
| HybridKVCache (hot/cold/offloaded) | Production (local offload only) |
| Prefix caching | Production |
| Speculative decode | Production (narrow) |
| Chunked prefill (batched) | Production |
| Chunked prefill (packed) | Not supported |
| Splitfuse | Narrow/conditional |
| mlx-lm patch | Production |
| TurboQuant Phase 1 (non-fused) | Production (v2.21.0) |
| TurboQuant Phase 2 (K fused) | Production (v2.22.0) |
| TurboQuant Phase 3 (K+V fused) | Production (v2.23.0) |
| TurboQuant Phase 4 (packing + WHT) | Production (v2.24.0) |
| SVDQuantLinear | Production (v2.25.0) |
| GNA native kernel | Production (v2.26.0) |
| Native `attn_bias` | Production (v2.27.0) |
| Remote/distributed offload | Deferred (M5+) |

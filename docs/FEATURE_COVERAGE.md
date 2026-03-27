# mlx-mfa Feature Coverage

Version: **2.23.0**

## Attention Kernels

| Feature | Status | Notes |
|---------|--------|-------|
| Dense causal D=64/128 (V2) | Production | Benchmark-backed, auto-routed |
| Dense non-causal D=64/128 | Production (M1/M2) | M3+ stays SDPA |
| D=256 causal | Narrow | f16 both chips, bf16 M3+ only |
| D=512 | SDPA-default | No broad wins found |
| Block-sparse / window | Production | Tile-skip, up to 21x speedup |
| Softcap | Production | V2 tanh in log2 domain |
| ALiBi | Production | V2 bias addition |
| Sliding window | Production | V2 O(1) kb_start + kb_lim clip |
| RoPE (fused) | Production | Q+K rotation in kernel |
| GNA (neighborhood) | Production | Via sparse path |
| Flash Decoding (split-KV) | Production | N_q<=4, S>=256 |
| Native backward | Non-default | Benchmarked, not promoted |

## Serving / Runtime

| Feature | Status | Notes |
|---------|--------|-------|
| Dense decode runtime | Production | `create_decode_runtime()` |
| Paged KV (batched/packed) | Production | Fused PagedVarlenForward kernel |
| Paged continuous batching | Production | `cache_batch_idx` remap |
| Chunked prefill | Production | Batched + packed layouts |
| Prefix caching | Production | Register/seed/reuse |
| Speculative decode | Production (narrow) | Draft/verify integration |
| Splitfuse | Narrow | Shape-sensitive |
| mlx-lm patch | Production | mlx-lm 0.30+ |

## Cache Types

| Cache | Status | Notes |
|-------|--------|-------|
| DenseKVCache | Production | Pre-allocated [B,H,S,D] |
| PagedKVCache | Production | Block-allocated on demand |
| QuantizedKVCache | Production | SageAttention pre-quantized |
| HybridKVCache | Production (local) | Hot/cold/offloaded tiers |
| TurboQuantKVCache | Production | Phase 1: non-fused decompress |
| ExternalKVCacheAdapter | Groundwork | Local backend only |

## KV Cache Compression (TurboQuant)

| Phase | Feature | Status | Notes |
|-------|---------|--------|-------|
| 1 | Non-fused compress/decompress | Production (v2.21.0) | `turboquant_compress()` / `turboquant_decompress()` |
| 1 | TurboQuantKVCache | Production (v2.21.0) | Drop-in cache with transparent decompress |
| 1 | QJL 1-bit residual correction | Production (v2.21.0) | Optional, default on |
| 2 | K fused in paged varlen kernel | Production (v2.22.0) | Metal kernel reads packed K directly |
| 2 | Metal packing helpers | Production (v2.22.0) | `pack_k_for_metal()`, `build_tq_paged_k_pool()` |
| 3 | V fused in kernel | Production (v2.23.0) | Metal kernel reads packed V directly |
| 3 | V output un-rotation | Production (v2.23.0) | WHT inverse applied to P@V output |
| 3 | TGP centroid cache | Production (v2.23.0) | Centroids loaded once into threadgroup memory |
| 3 | TurboQuantPagedInferenceContext | Production (v2.23.0) | Stateful runtime with auto Q rotation |

### Compression quality (3-bit, cosine similarity vs fp16)

| Phase | K cos | V cos | Memory savings |
|-------|------:|------:|---------------:|
| Phase 2 (K-only) | 0.98 | — | ~1.6x |
| Phase 3 (K+V) | 0.98 | 0.97 | ~3.8x |

## SageAttention

| Feature | Status | Notes |
|---------|--------|-------|
| sage_attention() | Production | Per-block int8 quantized |
| sage_attention_kvcache() | Production | Decode backend |
| sage_attention_prequantized() | Production | Pre-quantized KV path |
| smooth_k() | Production | Per-channel mean subtraction |

## Experimental Kernels

| Kernel | Gate | Status | Notes |
|--------|------|--------|-------|
| V3 (separate K/V smem) | `MFA_ENABLE_V3=1` | Experimental | Occupancy regression vs V2 |
| V4 (direct device K reads) | `MFA_ENABLE_V4=1` | Experimental | Needs M3+ L2 cache |
| V5 (D-blocked, Q in registers) | `MFA_ENABLE_V5=1` | Experimental | Barrier-dominated on M1 |

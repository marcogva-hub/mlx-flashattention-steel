# mlx-mfa Feature Coverage

Version: **2.39.1** (PyPI) + master `82acc55` (post-Sprint A/B/C internal accumulation)
Last reviewed: 2026-05-13 (v50-nax-coverage audit)

> **For canonical M5+ NAX path classification per function**, see
> `docs/HARDWARE_SUPPORT.md` (the authoritative matrix derived from the
> v50-nax-coverage audit empirical bench).

## Updates since v2.27.0

| Version | Date | Major additions |
|---|---|---|
| v2.28-v2.36 | 2026-04 | Apple SDPA NAX auto-routing (`_M5_NAX_THRESHOLDS = 999_999`), LCSA NAX dispatcher, Conv3D NAX, auto-hooks (`_HOOKS_INSTALLED`) |
| v2.37.0 | 2026-05-04 | V6NAX NAX-direct backward kernel (forward + dQ + split dV/dK) |
| v2.37.1 | 2026-05-06 | V6NAX backward perf claim 1.4-1.85× (later retracted as overstated) |
| v2.37.2 | 2026-05-08 | V6NAX backward carve-out (`_v6nax_backward_carveout`) — D=64 qL≥4096 |
| v2.37.3 | 2026-05-09 | Institutional rules §Z (public API testing) + §AA (mandatory skill checkpoints); perf claim audit + retraction |
| v2.38.0 | 2026-05-13 | Refactor + cleanup: `_v6nax_eligible()` + `_v6nax_backward_vjp()` helper extraction; deletion of dormant placeholders |
| v2.38.1 | 2026-05-13 | D_vec precompute device buffer (M2-HIGH-01): D=64 V6NAX backward 1.91× / 1.87× / 1.80× vs SDPA-vjp at qL∈{4096, 8192, 16384} |
| v2.39.0 | 2026-05-13 | Option γ fused dK+dV (D=64) ships opt-in (outcome δ documented): BK=32 default caused -25 to -33% regression vs split |
| v2.39.1 | 2026-05-13 | H1 register-pressure root-caused + fixed: BK default 32→16; auto-default flips back to fused-D=64; speedups 2.00× / 1.95× / 1.72× vs SDPA-vjp |
| v2.39.2-internal (Sprint A) | 2026-05-13 | Carve-out broadened qL≥4096 → qL≥2048 (parity at qL=2048, 3-session variance 1.004) |
| v2.40.0-internal (Sprint B) | 2026-05-13 | D=128 fused architectural enablement (gate lift); auto-default UNCHANGED (regression on direct binding 3-7% vs split; outcome γ) |
| v2.40.x-internal (Sprint C) | 2026-05-13 | V6NAX backward Primitive pipeline-compile boilerplate consolidation (P3-HIGH-01) |
| **next: v2.50** | TBD | Tier 1+2 sprints per `docs/audits/v50-nax-coverage/03-sprint-sequence.md` |



## Attention Kernels

| Feature | M1/M2/M3 status | M5+ status | Notes |
|---------|---|---|---|
| Dense causal D=64/128 (V2) | Production STEEL | **Apple SDPA NAX (auto-routed)** | dispatch_policy `_M5_NAX_THRESHOLDS = 999_999` for D=64/128 |
| Dense non-causal D=64/128 | Production STEEL (M1/M2); SDPA on M3+ | **Apple SDPA NAX (auto-routed)** | same dispatch logic |
| D=256 causal | Narrow STEEL | STEEL (no NAX path) | f16 both chips, bf16 M3+ only |
| D=512 | SDPA-default | SDPA-default | No broad wins found |
| **V6NAX NAX-direct backward (D=64, opt-in v2.37.2+)** | N/A | **Production (env-gated `MFA_ENABLE_V6_BACKWARD=1`)** | qL≥2048 post-Sprint A; 1.91-2.00× SDPA-vjp |
| Block-sparse / window | Production | Tile-skip, up to 21x speedup |
| Softcap | Production | V2 tanh in log2 domain |
| ALiBi | Production | V2 bias addition |
| Sliding window | Production | V2 O(1) kb_start + kb_lim clip |
| RoPE (fused) | Production | Q+K rotation in kernel |
| GNA (neighborhood) | Production | Native Metal kernel (D=128) + sparse fallback |
| `attn_bias` (additive) | Production | Native Metal modes 1/2; SDPA fallback modes 0/3 |
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
| 4 | Optimal 3-bit packing + WHT fusion | Production (v2.24.0) | 5.33× memory savings; WHT fused into kernel |

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

## SVDQuant

| Feature | Status | Notes |
|---------|--------|-------|
| SVDQuantLinear | Production (v2.25.0) | W4A16 + SVD low-rank FP16 correction |
| quantize_model() | Production (v2.25.0) | Tree walker; replaces `nn.Linear` in-place |

## GNA Native Kernel

| Feature | Status | Notes |
|---------|--------|-------|
| GNA native Metal kernel | Production (v2.26.0) | Inline 3D window, forward-only, D=128 |

## Experimental Kernels

| Kernel | Gate | Status | Notes |
|--------|------|--------|-------|
| V3 (separate K/V smem) | `MFA_ENABLE_V3=1` | Experimental | Occupancy regression vs V2 |
| V4 (direct device K reads) | `MFA_ENABLE_V4=1` | Experimental | Needs M3+ L2 cache |
| V5 (D-blocked, Q in registers) | `MFA_ENABLE_V5=1` | Experimental | Barrier-dominated on M1 |

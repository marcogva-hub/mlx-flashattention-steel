# mlx-mfa Feature Coverage

Version: **2.61.0**
Last reviewed: 2026-06-19 (2.61.0 doc accuracy audit)

> **For canonical M5+ NAX path classification per function**, see
> `docs/reference/HARDWARE_SUPPORT.md` (the authoritative matrix derived from the
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
| v2.50 (shipped) | done | Tier 1+2 sprints — see CHANGELOG [2.50.0]; planning notes in the internal archive (not shipped) |



## Attention Kernels

| Feature | M1/M2/M3 status | M5+ status | Notes |
|---------|---|---|---|
| Dense D=128 (auto) | Production STEEL V2 | **V6_NAX matmul2d forward** (N≥`MFA_V6_DENSE_MIN_N`, default 2048) | `_select_dense_backend()` in attention.py; opt-out `MFA_DISABLE_V6_DENSE=1` → SDPA |
| Dense D=64 (auto) | Production STEEL V2 | SDPA (dense); D=64 *causal* large-N (B·H≥4, N≥4096) → MFA primitive via V3 cond-auto | per-(D,causal) `_M5_NAX_THRESHOLDS` dict in dispatch_policy.py (not a flat constant) |
| D=256 causal | Narrow STEEL | STEEL (no NAX path) | f16 both chips, bf16 M3+ only |
| D=512 | SDPA-default | SDPA-default | No broad wins found |
| **Backward D=64 (≥2048, fp16/bf16) — split-V6 NAX-direct, DEFAULT-ON** | `mx.vjp(SDPA)` | **split-V6 (default)** | **VERIFIED 2.16–3.05× vs SDPA-vjp** (nc 2.16×/2.21× @qL4096/8192; causal 2.77×/3.05×) — full-backward, gold which-binary trace + fp32 oracle, M5/MLX-0.31.2/2026-06-19. Opt out: `MFA_DISABLE_V6_BACKWARD=1`. Locked by `test_backward_routing_snapshot.py`. |
| **Backward D=128 / D=64<2048 / other** | `mx.vjp(SDPA)` | **SDPA-vjp (default)** | D=128 split-V6 is opt-in only + SLOWER (0.54× nc / 0.57× causal) → correctly not default. Prior 1.91–2.00× / parity / 2.55–5.75× withdrawn as artifacts (superseded by the verified D=64 number above). |
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
| V3 (separate K/V smem) | `MFA_ENABLE_V3=1` | Experimental (opt-in) | Conditionally auto-routed on M1–M4 (causal, B·H≥4, large N); occupancy regression vs V2 elsewhere |
| V4 (direct device K reads) | — | Retired (Lot-2) | Removed from build; gate `MFA_ENABLE_V4` no longer exists |
| V5 (D-blocked, Q in registers) | — | Retired (Lot-2) | Removed from build; gate `MFA_ENABLE_V5` no longer exists |

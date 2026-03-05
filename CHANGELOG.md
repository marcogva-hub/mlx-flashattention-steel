# Changelog

All notable changes to mlx-mfa are documented here.

## [0.7.0] — 2026-03-05

### Added
- **Track O: Spatial 2D/3D block masks** — `make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask`
- **Track P: Segment / document masks** — `make_segment_mask`, `make_causal_segment_mask`
- **Track Q: Adaptive window mask** — `make_adaptive_window_mask` (SeedVSR2-style resolution-scaled windows)
- **Track R: 3D RoPE table construction** — `make_rope_3d_tables` + `flash_attention_rope(rope_3d=...)` dict API
- **Track S: Variable-length batching** — `flash_attention_varlen` (split-concat implementation)
- **Track T: 4 benchmark scripts** — spatial masks, segment, varlen, 3D RoPE
- Pure Python release — no Metal kernel changes
- Tests: ~150 total


## [0.6.0] — 2026-03-05

### Added
- **Track K: Quantized KV cache** — Q4/Q8 dequantized before STEEL kernel
- **Track L: RoPE 1D fusion** — `flash_attention_rope()` with in-kernel rotary embeddings
- **Track M: Paged Attention design doc** — `docs/PAGED_ATTENTION_DESIGN.md`


## [0.5.0] — 2026-03-05

### Added
- **Flash Decoding (Track H)** — Two-phase split-KV attention for decode mode
  (N_q ≤ 4, S ≥ 256, f16/bf16). Phase 1 dispatches KV-sequence splits in
  parallel; Phase 2 reduces partial outputs via log-sum-exp. Activated
  automatically for eligible shapes.
  - New KernelType variants: `FlashDecodePartial`, `FlashDecodeReduce`
  - New params structs: `FlashDecodePartialParams`, `FlashDecodeReduceParams`
  - `compute_num_splits(kL, BK)` — targets ≥2 K-tiles per split, capped at 32
  - 11 new tests: non-causal/causal across D=64/128/256, GQA, bf16, boundary cases

- **M5+ detection stub (Track I)** — Forward-compatibility for Apple M5 (gen≥17,
  A19 SoC with Metal 4 tensor API)
  - `get_device_info()` now returns `is_m5_plus` (bool)
  - Gen 17 → `"M5"` chip name in `_GEN_TO_CHIP` mapping
  - `TensorOpsForward` KernelType reserved as commented stub in `shader_cache.hpp`
  - 3 new tests covering flag correctness, chip name, and M5 ⊇ M3+ logic

### Fixed
- `enc.barrier()` replaces `enc.maybeInsertBarrier()` between Flash Decode
  Phase 1 and Phase 2 — `maybeInsertBarrier()` is a no-op for raw
  `MTL::Buffer*` bindings (only `set_output_array()` sets `needs_barrier_`)
- `qL_off = S - N` for causal decode so query token at position `i` correctly
  sees keys `0..(S - N + i)` instead of starting from key 0

### Tests
- 107 tests total (was 93)

---

## [0.4.0] — 2026-02-xx

### Added
- **Track F** — M3+ architecture routing: BK=32 for D=128 on M3/M4 (gen≥15),
  `MFA_FORCE_GEN` env var override, `ARCHITECTURE_GEN` #define in Metal shader
- **Track G** — Sparse backward pass: tiled FA-2 dQ/dK/dV that skips inactive
  blocks; `flash_attention_sparse(backward='sdpa_sparse')` public API
- **Track C** — Native GQA: removed `mx.repeat` expansion, STEEL kernel handles
  `gqa_factor` natively in the Metal shader

### Tests
- 93 tests total (was 63)

---

## [0.3.0] — 2026-01-xx

### Added
- **Track D** — mlx-lm integration: `patch_mlx_lm()` / `unpatch_mlx_lm()`
- Native GQA support in STEEL kernel (gqa_factor parameter)
- `make_causal_block_mask()`, `make_sliding_window_mask()` public helpers
- mlx-lm integration tests (11 tests)

---

## [0.2.0] — 2025-12-xx

### Added
- **Track B** — Block-sparse attention: `flash_attention_sparse(q, k, v, mask)`
- Sparse STEEL kernel variant (K-loop skip, zero warp divergence)
- Sliding-window mask giving 3–6× speedup at long contexts

### Performance (M1 Max, B=1 H=8 f16, causal)
| D | N | Speedup |
|---|---|---------|
| 64 | 8192 | 2.11× SDPA |
| 128 | 8192 | 1.72× SDPA |
| 128 N=8192 sliding-window=512 | | 5.7× SDPA |

---

## [0.1.0] — 2025-11-xx

### Added
- Initial release: STEEL forward kernel replacing ccv-based MFA
- Full forward pass (D=64/128/256, f16/bf16/f32, causal/non-causal)
- Backward via `mx.vjp(scaled_dot_product_attention)`
- GQA via `mx.repeat` expand (later replaced by native GQA in v0.3)
- Public API: `flash_attention()`, `is_mfa_available()`, `get_device_info()`
- 41 tests

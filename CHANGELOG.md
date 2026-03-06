# Changelog

All notable changes to mlx-mfa are documented here.

## [1.0.0] — 2026-03-06

### Highlights

First stable public release. All features from v1.0.0-rc1 and v1.0.0-rc2.

| Track | Description | Tests added |
|-------|-------------|-------------|
| FA | Unified KV-cache API (`flash_attention_kvcache`) | 17 |
| FB | Native sliding-window in STEEL kernel | 4 |
| FC | Fused RoPE cache append (`flash_attention_kvcache_rope_append`) | 3 |
| FD | Kernel-level paged KV STEEL forward + Flash Decode | 15 |
| FX | `return_lse`, `cache_batch_idx`, `rotary_dim` | 8 |

**307 tests pass.** Full Python API with 33 public exports.

### Package
- First PyPI release: `pip install mlx-mfa`
- `pyproject.toml`: `Development Status :: 5 - Production/Stable`, `numpy` added to dependencies
- `MANIFEST.in`: adds `examples/`, `CHANGELOG.md`, `csrc/mfa/`
- `examples/`: 5 practical scripts covering all major API paths

See `[1.0.0-rc1]` and `[1.0.0-rc2]` below for the complete feature details.

---

## [1.0.0-rc2] — 2026-03-06

### Added
- **Track FD: Kernel-level paged KV streaming in STEEL forward kernel** — Metal kernel
  `mlx_mfa_paged_attention` reads K/V tiles directly from the `[num_blocks, block_size,
  H_kv, D]` pool via cooperative `block_table` lookup, eliminating a separate gather
  Metal dispatch. New `KernelType::PagedSteelForward`, `MFAPagedSteelParams`,
  `generate_paged_steel_forward_source()`, `MFAPagedSteelForward` Primitive, and
  `mfa_paged_steel_forward` nanobind binding. GQA, causal, sliding window all supported.
  `flash_attention_paged()` routes to the kernel for f16/bf16 D∈{64,128,256}.
  Benchmark (M1 Max, f16, B=1 H=8 D=128): **1.26–1.58x** faster than gather+attend.
- **Track FD-decode: Paged Flash Decode path** — For decode steps (N_q ≤ 4, S ≥ 256),
  `flash_attention_paged()` routes through Metal gather + `flash_attention()`, which
  activates the existing split-KV Flash Decode two-phase kernel for better SM parallelism.
- **Track FD-bench: `benchmarks/bench_paged_kv.py`** — Three-way comparison:
  gather+attend vs kernel-level paged STEEL vs pre-gathered Flash Decode.
- **307 tests pass** (up from 292 in rc1): 11 `TestPagedSteelForward` + 4
  `TestPagedFlashDecode`.

### Changed
- (infra) `has_window` added to `KernelKey` hash/equality; `window_left` wired into
  `MFASteelParams` — prerequisite for Track FD kernel dispatch.

---

## [1.0.0-rc1] — 2026-03-06

### Added
- **Track FB: Native sliding window in STEEL kernel** — `window_left` param in
  `MFASteelParams`; `has_window` KernelKey flag; K-tile `kb_start` computed per
  Q-block inside the persistent loop; boundary tiles apply element-wise mask.
  Fixed multi-tile boundary bug (only first boundary tile was masked), NaN-safe
  online softmax (all-masked-tile guard), and test reference `qL_off` alignment.
  `flash_attention(..., window_size=(left, right))` public API. 4 tests.
- **Track FA: Unified KV cache API** — `flash_attention_kvcache(q, k_cache, v_cache, ...)`
  replaces fragmented `with_kv_cache` / `paged` / `rope` paths. Dense + paged modes,
  RoPE, softcap, ALiBi, sliding window, `cache_seqlens`, `cache_batch_idx`. 17 tests.
- **Track FX-1: `return_lse` in `flash_attention`** — Expose logsumexp `L [B,H,N]`
  (log2 domain) alongside output when requested. MFA path uses `mfa_forward_with_lse`
  (free); fallback materialises log2-domain LSE via pure-MLX ops. 4 tests.
- **Track FX-2: `cache_batch_idx` in `flash_attention_kvcache`** — Non-contiguous
  batch→cache-slot mapping for continuous batching; `k_cache[cache_batch_idx]` gather
  before attention dispatch. 2 tests.
- **Track FX-3: `rotary_dim` partial RoPE** — Rotate only first `rotary_dim` dims;
  remainder passes through unchanged. STEEL kernel forces MLX fallback when
  `rotary_dim < head_dim`. 2 tests.
- **Track FC: Fused RoPE in cache append** — `flash_attention_kvcache_rope_append`
  rotates `k_new` BEFORE concat, storing pre-rotated keys in cache. O(1) rotation
  cost per decode step vs O(past_len) for naive re-rotation. `benchmarks/bench_kvcache.py`
  added for A/B comparison. 3 tests.

### Tests
Total collected: **292**

---

## [0.9.3] — 2026-03-06

### Added
- **Track EA: Differentiable `flash_attention_varlen`** — `mx.custom_function`
  wrapper adds full autograd. Forward: STEEL varlen kernel (f16/bf16, D=64/128/256);
  backward: splits per sequence through `flash_attention`. `TestVarlenBackward` (6 tests).
- **Track EB: Metal paged KV gather kernel** — `MFAPagedKVGather` Primitive
  gathers pool pages to `[B, H, max_kv_len, D]` in a single Metal dispatch.
  `flash_attention_paged` rewritten with `mx.custom_function`: `dQ` correct via
  `vjp(flash_attention)`; pool gradients are zeros (cache buffers).
  `TestPagedBackward` (6 tests).
- **Track EC: Varlen packed formats** — `flash_attention_varlen_qkv_packed` and
  `flash_attention_varlen_kv_packed` accept head-first or flat fused tensors and
  route to `flash_attention_varlen`. `TestVarlenPacked` (4 tests).
- **Track ED: Documentation refresh** — `docs/ARCHITECTURE.md` rewritten to 476 lines:
  updated backward routing tree (STEEL bwd / SDPA vjp / compiled vjp), new §8 (STEEL
  native backward — FA-2 log2 domain, GQA `gqa_factor`, D=256 three-phase D-split),
  new §9 (varlen backward via `mx.custom_function`), new §10 (paged KV gather — Metal
  kernel pseudocode, forward/backward flow, per-seq slicing rationale), expanded Public
  API table to all 31 exports. `docs/INVENTORY.md` regenerated from scratch: all line
  counts verified with `wc -l`, 31 `__all__` exports, 10 KernelType entries, 7 C++
  Primitive classes, 257 pytest runs / 212 test functions, 40 test classes, 10
  benchmarks. `README.md`: API Reference expanded from 7 to all 31 exports (param
  tables for core attention functions; compact reference table for 13 mask builders);
  Features section updated with v0.9.2–v0.9.3 additions.

### Tests
Total collected: **257 pytest runs / 212 test functions** (EA adds 6, EB adds 6, EC adds 4).

---

## [0.9.2] — 2026-03-06

### Added
- **Track DA: GQA backward guard fix** — Removed incorrect Python guard that blocked
  STEEL backward dispatch for grouped-query attention (H_q ≠ H_kv). The STEEL kernels
  have supported GQA since v0.9.0 via the `gqa_factor` Metal define; the Python
  `use_steel_bwd` predicate now correctly allows GQA shapes through.
- **Track DC: `mx.compile` for `_apply_rope_mlx`** — Shape-keyed compile cache
  (`_rope_compile_cache`) with separate `_impl` closures for interleaved and
  non-interleaved layouts. Scalars `offset` and `interleaved` are frozen in the
  closure to avoid dynamic control flow in the compiled graph. Median speedup ≈1.4×
  over the raw Python fallback (measured in `bench_compile.py`).
- **Track DC: `benchmarks/bench_compile.py`** — New benchmark (50-iteration median)
  comparing compiled vs raw latency for `_softcap_sdpa_ref`, `_alibi_sdpa_ref`, and
  `_apply_rope_mlx` (interleaved + non-interleaved) at N=2048 D=128 f16.
- **Track CE: D=256 D-split STEEL backward** — `generate_steel_backward_dq_source()`
  and `generate_steel_backward_dkv_source()` now emit D-split Metal code when
  `head_dim=256` (`BD_HALF=128`). Q/dO/K/V tiles are loaded in lo (0..127) and
  hi (128..255) passes sharing one threadgroup buffer; dQ/dK/dV accumulators become
  lo/hi register-tile pairs. TGP budget ≈ 23 KB (well below 32 KB limit). The
  `use_steel_bwd` guard is widened from `D ≤ 128` to `D ≤ 256`.
- **Track DD: Documentation refresh** — `docs/INVENTORY.md` updated to v0.9.2:
  test count 241, benchmark count 9, backward strategy table, DA–DE additions table.
  CE row in v0.9.1 table updated from "deferred" to "completed in v0.9.2".

### Fixed
- **Track DB: CHANGELOG inaccuracies** — v0.9.1 entry for Track CB now correctly states
  `_apply_rope_mlx` was NOT compiled in v0.9.1 (completed in Track DC / v0.9.2).
  Test count corrected to 232.

---

## [0.9.1] — 2026-03-06

### Added
- **Track CA: Vec4 block loads** — `MFABlockLoaderT` uses `float4`/`half4` aligned
  vector reads for all tile loads in the STEEL forward kernel, reducing instruction
  count per tile by 4× on cache-line-aligned data.
- **Track CB: `mx.compile` for fallback paths** — The Python fallback routes
  (`_softcap_sdpa_ref`, `_alibi_sdpa_ref`) are wrapped with `mx.compile`.
  `_apply_rope_mlx` and the sparse/varlen fallbacks are NOT yet compiled
  (completed in Track DC / v0.9.2).
- **Track CC: Persistent multi-Q-block kernel** — The STEEL forward kernel now iterates
  over an outer `qb` loop (`[0, NQ)`) within a single threadgroup dispatch, processing
  up to 4 Q-blocks per launch. Amortizes Metal command buffer overhead at N ≥ 4096.
- **Track CD: GQA in STEEL backward** — The STEEL dQ and dKV backward kernels now
  handle grouped-query attention.  The `gqa_factor` (H_q / H_kv) is baked into the
  Metal shader as `#define MFA_GQA_FACTOR <N>` at compile time, avoiding Metal
  `constant`-address-space struct-field read ambiguity.  `KernelKey` extended with
  `gqa_factor` so each GQA ratio compiles to a distinct cached pipeline.
- **Track CF: Double-buffer ping-pong** — Separate `K_smem` / `V_smem` threadgroup
  arrays when D ≤ 128 (TGP ≈ 19.2 KB < 32 KB limit).  Reduces barriers per K-tile
  from 4 → 2: V-tile stores overlap K-GEMM; K[n+1]-tile stores overlap P@V.
  Phase-0 preloads K[0] before the loop; `loader_k/v.next()` called inline.
  Disabled for D=256 (budget), RoPE (extra TGP), and sparse.
- **Track CG: `benchmarks/bench_all.py`** — Consolidated forward + backward benchmark
  suite (`--fwd-only`, `--bwd-only`, `--no-save` flags).  Appends markdown results
  table to `docs/benchmarks/RESULTS.md`.
- **Track CH: Documentation refresh** — `docs/INVENTORY.md` updated to v0.9.1
  (test count 232, benchmark count 8, kernel table, CA–CI additions table).
  `docs/ARCHITECTURE.md` adds notes on CF double-buffer and CC persistent kernel.
  `README.md` roadmap updated: N1 marked Done (v0.9.0); CA/CB/CC/CD/CF rows added.

### Deferred
- **Track CE: D=256 backward multi-pass** — 3D blocking for the STEEL dQ/dKV
  backward kernels (analogous to the forward D=256 path) is deferred to v1.0.
  D=256 backward continues to route to `mx.vjp(SDPA)` (same as v0.9.0).

---

## [0.9.0] — 2026-03-06

### Added
- **Track BA/BB/BC: STEEL native backward** — `mx.grad(flash_attention)` now dispatches
  native Metal STEEL backward kernels (`MFASteelBwdDQ`, `MFASteelBwdDKV`) for f16/bf16
  instead of `mx.vjp(SDPA)`. 2-3× backward speedup on D=64/128. f32 stays on ccv path.
  Key fixes: `Ktile[1,MFA_TK]` tile declaration (was 1×1, causing UB for ik>0) and
  `_sever_lazy_graph(cotangent)` before gradient checkpointing re-run of forward
  (prevents Metal buffer aliasing via lazy graph ancestry). 209 tests pass.
- **Track BD: STEEL varlen forward kernel** — `flash_attention_varlen` dispatches a
  dedicated Metal STEEL kernel instead of Python split-cat. Packed Q/K/V layout
  `[1, H, N_total, D]` with `cu_seqlens` offsets; per-threadgroup batch-item decode.
  Critical race-condition fix: `threadgroup_barrier` at START of K-loop prevents
  P@V reads (V from KV_smem) from racing against next iteration's K write.
  K-boundary `-INF` mask prevents softmax denominator inflation for partial K-tiles.
  215 tests pass.
- **Track BE: Paged KV Cache Phase 1** — `PagedKVCache` block allocator with pool
  `[num_blocks, block_size, H_kv, D]`; per-seq block table; `append`/`free_seq` helpers.
  `flash_attention_paged(q, k_pool, v_pool, block_table, seq_lens, ...)` reconstructs
  contiguous K/V per batch item via block-table gather, routes to `flash_attention`.
- **Track BF: QKV/KV packed tensor formats** — `flash_attention_qkv_packed` handles
  flat `[B, N, 3·H·D]` and head-first `[B, H, N, 3, D]` packed layouts.
  `flash_attention_kv_packed` handles `[B, S, 2·H·D]` and `[B, H, S, 2, D]`.
  Both raise `ValueError` for unsupported shapes.
- **Track BG: Backward benchmark** — `benchmarks/bench_backward.py` measures
  flash_attention VJP vs SDPA VJP across D=64/128, f16/bf16, causal/non-causal.
- **Track BH: Varlen benchmark update** — `benchmarks/bench_varlen.py` updated to
  note STEEL varlen kernel; section header updated to v0.9.0.
- **Tests: 232 pytest runs** (180+16 test functions; 232 with parametrize expansion)

## [0.8.0] — 2026-03-05

### Added
- **Track AA: Softcap** — `flash_attention(..., softcap=50.0)` applies `tanh(S/cap)*cap`
  before softmax; fused into Metal STEEL kernel for f16/bf16, Python fallback for f32.
- **Track AB: ALiBi** — `flash_attention_alibi(q, k, v, alibi_slopes, ...)` adds
  per-head linear position biases (slope_h × (k_pos − q_pos)). Metal kernel fuses
  bias into the QK tile accumulation; Python reference fallback included.
- **Track AC: RoPE non-interleaved (GPT-NeoX)** — `flash_attention_rope(..., interleaved=False)`
  supports split-halves RoPE layout `(d, d+D/2)` in addition to LLaMA adjacent pairs.
  Metal kernel and Python `_apply_rope_mlx` both branch on `interleaved`.
- **Track AD: Per-batch `cache_seqlens`** — `flash_attention_rope` now accepts
  `cache_seqlens` as a `list[int]`, `mx.array`, or `int`. Per-element dispatch via
  Python split-cat; MLX lazy eval fuses concurrent GPU dispatches.
- **Track AE: Graceful D_v ≠ D_qk fallback** — When `v.shape[-1] != q.shape[-1]`,
  routes to `mx.fast.scaled_dot_product_attention` instead of raising. K dimension
  must still equal Q (raises `ValueError` otherwise).
- **Track AF: `flash_attention_with_kv_cache`** — Fused KV cache append:
  `(output, k_updated, v_updated) = flash_attention_with_kv_cache(q, k_new, v_new, k_cache, v_cache)`.
  Concatenates along the sequence axis, dispatches one attention call.
- **Track AG: Attention dropout** — `flash_attention(..., dropout_p=0.2)` drops
  softmax weights during training. Uses `mx.where` causal masking to avoid
  `0.0 × −inf = NaN` in the masked region.
- **Track AH: Return attention weights** — `flash_attention(..., return_attn_weights=True)`
  returns `(output, attn_weights)` where weights are the full softmax probability matrix
  `[B, H, N, S]`. Compatible with softcap and dropout.
- **Track Z: Benchmark scripts** — `benchmarks/bench_softcap_alibi.py` measures
  softcap and ALiBi overhead vs SDPA baseline across four variants.
- **Tests: 209 total** (up from 93 in v0.4.0)

### Changed
- `flash_attention_rope` now accepts `Union[int, mx.array, Sequence[int]]` for `cache_seqlens`

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

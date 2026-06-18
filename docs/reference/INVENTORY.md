# mlx-mfa Inventory

Version: **2.58.1** (PyPI) — header updated 2026-06-12; module/LOC tables below are a 2026-05-13 snapshot (regeneration pending)

> **For canonical NAX path coverage**, see `docs/reference/HARDWARE_SUPPORT.md`.

## Public-API surface (v2.39.1)

22 `flash_attention*` functions + 3 `sage_attention*` functions exposed
via `mlx_mfa.__all__`.  Full per-function classification is in
`.doc-archive/docs/audits/v50-nax-coverage/02-consolidated-bench-results.md`.

## Major modules + current LOC (2026-05-13 snapshot)

| File | Lines (approx) | Notes |
|---|---|---|
| `mlx_mfa/attention.py` | 6058 | +472 LOC since v2.27.0 — V6NAX backward integration (v2.37.x-v2.39.x), helper extraction (v2.38.0), Sprint A carve-out broadening |
| `mlx_mfa/dispatch_policy.py` | 1037 | +199 LOC — V6NAX backward carve-out, M5+ NAX thresholds, custom dispatch table support |
| `mlx_mfa/__init__.py` | 467 | +173 LOC — auto-hooks installation log, diagnostics() function, 22 attention export update |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | 5800+ | V6NAX forward + 4 backward source generators (post Sprint A/B Phase B helper extraction) |
| `csrc/mfa_v6_nax_primitive.cpp` | 1800+ | 4 V6NAX backward Primitives (post Sprint C boilerplate consolidation) |
| `csrc/v6_nax_compile.mm` | 600+ | 4 dispatchers (BwdQ, BwdKV legacy, BwdDV split, BwdDK split, BwdFusedDKDV v2.39.0+) |

## Native extension (csrc/) recent additions

| Component | Version | LOC | Purpose |
|---|---|---|---|
| V6NAX forward NAX-direct kernel (`createV6NAXSource`) | v2.31.0 | ~1100 | Apple-style NAX with cooperative tensor MMA primitives |
| V6NAX backward dQ kernel (`createV6NAXBackwardQuerySource`) | v2.37.0 | ~470 | Per-Q-block dQ accumulation |
| V6NAX backward split dV kernel (`createV6NAXBackwardDVSource`) | v2.37.0 | ~300 | WM=4 Q-row partition |
| V6NAX backward split dK kernel (`createV6NAXBackwardDKSource`) | v2.37.0 | ~350 | WM=4 sister kernel to dV |
| V6NAX backward legacy fused dKdV (`createV6NAXBackwardKeyValueSource`) | v2.37.0 | ~440 | WM=1 single-SG, gated by `MFA_V6BWD_USE_FUSED=1` |
| V6NAX backward fused dK+dV (`createV6NAXBackwardFusedDKDVSource`) | v2.39.0+ | ~440 | Option γ — D=64 (auto-default) + D=128 (opt-in via Sprint B) |
| D_vec precompute device buffer | v2.38.1 | host + 3 kernels | Eliminates 2 in-kernel rowsums per V6NAX backward call |
| `naxHelpersBlock()` shared helper | v2.38.x Phase B | -1541 LOC dedup | Extracted from 5 source generators |
| V6NAX backward Primitive consolidation | v2.40.0-internal Sprint C | -68 LOC | `v6nax_get_or_compile_pipeline<Key, Hash>` template helper |

## Test suites

| Suite | Tests | Notes |
|---|---|---|
| `tests/test_v39_fused_dkdv.py` | 26 | Option γ fused tests (Sprint A/B/C accumulating) |
| `tests/test_v6nax_helpers.py` | 16 | `_v6nax_eligible` + `_v6nax_backward_vjp` direct |
| `tests/test_v32_sdpa_routing.py` | 19 | Dispatch policy + carve-out (incl. Sprint A qL=2048 threshold) |
| `tests/test_flash_attention_v6nax_backward.py` | 11 | V6NAX backward end-to-end via PUBLIC API |
| `tests/test_release_notes_perf_claims.py` | 12 | §Z regression: every documented perf claim reachable via PUBLIC API |
| `tests/test_attention.py` | ~700 | Core attention suite (unchanged scope from v2.27.0) |
| `tests/test_v6nax_bwd_multisg.py` | pre-existing failures unrelated to internal sprints (see Sprint B audit) | — |

## Internal-mode accumulation contract

Post-v2.39.1, master accumulates internal sprints (Sprint A v2.39.2-internal,
Sprint B v2.40.0-internal, Sprint C v2.40.x-internal) **without bumping the
PyPI version**.  pyproject.toml + `mlx_mfa/__init__.py:__version__` + README
banner all remain at `2.39.1` until the v2.50 bundle release.

CHANGELOG `[Unreleased — for v2.50]` section accumulates entries.  At v2.50
ship time, this section is renamed to `[2.50.0] — <date>`.

See `.doc-archive/docs/audits/v50-nax-coverage/03-sprint-sequence.md` for the 5-sprint
plan to reach v2.50.

## Scope

This inventory reflects the retained codebase at freeze-prep time, including:
- dense V2 production path and dispatch policy;
- serving/runtime expansion (paged/packed/chunked/prefix/speculative/splitfuse);
- cache abstraction and hybrid local offload-capable behavior;
- TurboQuant KV cache compression (Phase 1–4);
- SVDQuant linear compression (W4A16 + low-rank correction);
- GNA native Metal kernel (inline 3D window, D=128);
- historical development artifacts moved under `.doc-archive/devnotes/`.

## Top-Level Layout

| Path | Purpose |
|---|---|
| `mlx_mfa/` | Public Python API, runtime layer, cache abstractions, dispatch policy |
| `csrc/` | C++/Objective-C++ Metal extension and kernel generation |
| `benchmarks/` | Benchmark/profiling scripts (now defaulting outputs to `.doc-archive/devnotes/`) |
| `tests/` | Unit/integration correctness coverage |
| `.doc-archive/docs/` | API manual, architecture guide, inventory, benchmark interpretation |
| `examples/` | Current usage examples (dense, paged, varlen, runtime flows) |
| `.doc-archive/devnotes/` | Historical R&D artifacts by branch/track |

## Python Modules (`mlx_mfa/`)

Current line counts (2026-03-27 snapshot):

| File | Lines | Notes |
|---|---:|---|
| `mlx_mfa/attention.py` | 5586 | Core attention APIs: dense/paged/varlen/sparse/speculative/splitfuse/turboquant |
| `mlx_mfa/runtime.py` | 1748 | `DecodeRuntime`, runtime integration, serving flow helpers |
| `mlx_mfa/inference.py` | 1245 | Inference contexts incl. TurboQuantPagedInferenceContext |
| `mlx_mfa/turboquant.py` | 938 | TurboQuant compress/decompress, Metal packing helpers |
| `mlx_mfa/kv_cache.py` | 822 | Cache adapter layer + hybrid cache behavior |
| `mlx_mfa/dispatch_policy.py` | 838 | Benchmark-backed dispatch policy and calibration |
| `mlx_mfa/masks.py` | 1129 | Mask builders |
| `mlx_mfa/quantize.py` | 280 | Quantization helpers |
| `mlx_mfa/external_cache.py` | 181 | External cache adapter contract + local host backend |
| `mlx_mfa/compile_metallib.py` | 364 | AOT metallib tooling |
| `mlx_mfa/integrations/mlx_lm.py` | 431 | mlx-lm integration hooks |
| `mlx_mfa/__init__.py` | 294 | Public exports and version |
| `mlx_mfa/svdquant/__init__.py` | — | SVDQuant public API |
| `mlx_mfa/svdquant/linear.py` | — | SVDQuantLinear nn.Module |
| `mlx_mfa/svdquant/quantize.py` | — | quantize_model() tree walker |

## Native Extension (`csrc/`)

Current line counts (major files, 2026-03-27 snapshot):

| File | Lines | Notes |
|---|---:|---|
| `csrc/mfa_attention.cpp` | 2967 | Primitive dispatch and routing hooks |
| `csrc/mfa_steel_fwd.cpp` | 3349 | Shared forward template generation |
| `csrc/mfa_steel_fwd_v2.cpp` | 2138 | V2 production kernel family |
| `csrc/mfa_steel_paged_varlen_tq_fwd.cpp` | 511 | TurboQuant paged varlen kernel generator |
| `csrc/mfa_steel_fwd_v3.cpp` | 642 | V3 separate K/V smem kernel |
| `csrc/mfa_steel_fwd_v5.cpp` | 683 | V5 D-blocked kernel (experimental) |
| `csrc/mfa_steel_bwd.cpp` | 1295 | Native backward (kept non-default) |
| `csrc/mfa_env.hpp` | 108 | MFAEnvConfig env var singleton |
| `csrc/mfa_gna_fwd.hpp` | — | GNA forward kernel declarations |
| `csrc/mfa_gna_fwd.cpp` | — | GNA forward kernel JIT generator |
| `csrc/mfa_sage_fwd.cpp` | 520 | Sage forward path |
| `csrc/shader_cache.mm` | 480 | Metal pipeline compilation/cache |
| `csrc/bindings.cpp` | 638 | nanobind module bindings |
| `csrc/async_v2_kernel.metal` | 1088 | Async metallib kernel source |

## Public API Snapshot

`mlx_mfa.__all__` exports: **90** symbols (`+ __version__`).

Major groups:
- Core attention: `flash_attention*` dense/paged/varlen/packed/splitfuse/speculative/turboquant.
- Runtime/context: `create_decode_runtime`, `create_inference_context`,
  `InferenceContext`, `PagedInferenceContext`, `SageInferenceContext`,
  `TurboQuantPagedInferenceContext`, `DecodeRuntime`.
- Cache abstraction: `KVCacheAdapter`, `KVCacheCapabilities`,
  `DenseKVCacheAdapter`, `PagedKVCacheAdapter`, `QuantizedKVCacheAdapter`,
  `HybridKVCacheAdapter`, `HybridKVCache`, `adapt_kv_cache`,
  `resolve_context_cache*`.
- TurboQuant: `turboquant_compress`, `turboquant_decompress`,
  `TurboQuantKVCache`, `pack_k_for_metal`, `pack_v_for_metal`,
  `build_tq_paged_k_pool`, `build_tq_paged_v_pool`.
- External cache groundwork: `ExternalKVCacheAdapter`,
  `ExternalKVCacheCapabilities`, `LocalHostKVStoreAdapter`.

## Benchmarks + Artifacts

- Benchmark scripts live in `benchmarks/`.
- `benchmarks/bench_utils.py` (61 lines): shared `med()`, `geomean()`, `env_override()`.
- `benchmarks/bench_turboquant_full.py`: TurboQuant Phase 1-3 benchmark matrix.
- `benchmarks/bench_v3_autoresearch.py`, `bench_v5_autoresearch.py`,
  `bench_v3_promotion.py`, `bench_d512_autoresearch.py`,
  `bench_dispatch_d256_kernel.py`, `bench_dispatch_d512_vae.py`: autoresearch scripts.
- Runtime/decision artifacts are archived under `.doc-archive/devnotes/<track>/`.

## Configuration Files (new in v2.20.0)

- `ENV_VARS.md` (63 lines): documents all 18+ `MFA_*` env vars.
- `AUTORESEARCH.md` (137 lines): D=512 VAE autoresearch protocol.
- `AUTORESEARCH_KERNEL.md` (227 lines): D=256 kernel autoresearch protocol.

## Tests

Primary suites used in recent passes:
- `tests/test_attention.py`
- `tests/test_inference_context.py`
- `tests/test_kv_cache_abstraction.py`
- `tests/test_external_cache.py`
- `tests/test_turboquant.py`
- `tests/test_gna_native.py` (11 tests — GNA native kernel)
- `tests/test_svdquant.py` (21 tests — SVDQuant)

## Historical Notes

Development history is intentionally separated from active docs:
- active docs: `README.md`, `.doc-archive/docs/*`, `RESULTS.md`;
- historical R&D traces: `.doc-archive/devnotes/`.

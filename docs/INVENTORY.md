# mlx-mfa Inventory

Version: **2.26.0**
Regenerated: 2026-03-31

## Scope

This inventory reflects the retained codebase at freeze-prep time, including:
- dense V2 production path and dispatch policy;
- serving/runtime expansion (paged/packed/chunked/prefix/speculative/splitfuse);
- cache abstraction and hybrid local offload-capable behavior;
- TurboQuant KV cache compression (Phase 1–4);
- SVDQuant linear compression (W4A16 + low-rank correction);
- GNA native Metal kernel (inline 3D window, D=128);
- historical development artifacts moved under `devnotes/`.

## Top-Level Layout

| Path | Purpose |
|---|---|
| `mlx_mfa/` | Public Python API, runtime layer, cache abstractions, dispatch policy |
| `csrc/` | C++/Objective-C++ Metal extension and kernel generation |
| `benchmarks/` | Benchmark/profiling scripts (now defaulting outputs to `devnotes/`) |
| `tests/` | Unit/integration correctness coverage |
| `docs/` | API manual, architecture guide, inventory, benchmark interpretation |
| `examples/` | Current usage examples (dense, paged, varlen, runtime flows) |
| `devnotes/` | Historical R&D artifacts by branch/track |

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
- Runtime/decision artifacts are archived under `devnotes/<track>/`.

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
- active docs: `README.md`, `docs/*`, `RESULTS.md`;
- historical R&D traces: `devnotes/`.

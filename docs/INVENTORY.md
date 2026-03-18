# mlx-mfa Inventory

Version: **2.14.1**
Regenerated: 2026-03-18

## Scope

This inventory reflects the retained codebase at freeze-prep time, including:
- dense V2 production path and dispatch policy;
- serving/runtime expansion (paged/packed/chunked/prefix/speculative/splitfuse);
- cache abstraction and hybrid local offload-capable behavior;
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

Current line counts (2026-03-13 snapshot):

| File | Lines | Notes |
|---|---:|---|
| `mlx_mfa/attention.py` | 5321 | Core attention APIs: dense/paged/varlen/sparse/speculative/splitfuse |
| `mlx_mfa/runtime.py` | 1724 | `DecodeRuntime`, runtime integration, serving flow helpers |
| `mlx_mfa/inference.py` | 957 | Inference contexts and cache implementations |
| `mlx_mfa/kv_cache.py` | 822 | Cache adapter layer + hybrid cache behavior |
| `mlx_mfa/dispatch_policy.py` | 809 | Benchmark-backed dispatch policy and calibration |
| `mlx_mfa/masks.py` | 1129 | Mask builders |
| `mlx_mfa/quantize.py` | 280 | Quantization helpers |
| `mlx_mfa/external_cache.py` | 181 | External cache adapter contract + local host backend |
| `mlx_mfa/compile_metallib.py` | 364 | AOT metallib tooling |
| `mlx_mfa/integrations/mlx_lm.py` | 431 | mlx-lm integration hooks |
| `mlx_mfa/__init__.py` | 243 | Public exports and version |

## Native Extension (`csrc/`)

Current line counts (major files):

| File | Lines | Notes |
|---|---:|---|
| `csrc/mfa_attention.cpp` | 2589 | Primitive dispatch and routing hooks |
| `csrc/mfa_steel_fwd.cpp` | 3343 | Shared forward template generation |
| `csrc/mfa_steel_fwd_v2.cpp` | 2064 | V2 production kernel family |
| `csrc/mfa_steel_bwd.cpp` | 1295 | Native backward (kept non-default) |
| `csrc/mfa_sage_fwd.cpp` | 520 | Sage forward path |
| `csrc/shader_cache.mm` | 437 | Metal pipeline compilation/cache |
| `csrc/async_v2_kernel.metal` | 1088 | Async metallib kernel source |

## Public API Snapshot

`mlx_mfa.__all__` exports: **81** symbols (`+ __version__`).

Major groups:
- Core attention: `flash_attention*` dense/paged/varlen/packed/splitfuse/speculative.
- Runtime/context: `create_decode_runtime`, `create_inference_context`,
  `InferenceContext`, `PagedInferenceContext`, `SageInferenceContext`,
  `DecodeRuntime`.
- Cache abstraction: `KVCacheAdapter`, `KVCacheCapabilities`,
  `DenseKVCacheAdapter`, `PagedKVCacheAdapter`, `QuantizedKVCacheAdapter`,
  `HybridKVCacheAdapter`, `HybridKVCache`, `adapt_kv_cache`,
  `resolve_context_cache*`.
- External cache groundwork: `ExternalKVCacheAdapter`,
  `ExternalKVCacheCapabilities`, `LocalHostKVStoreAdapter`.

## Benchmarks + Artifacts

- Benchmark scripts live in `benchmarks/`.
- Runtime/decision artifacts are archived under `devnotes/<track>/`.
- `docs/benchmarks/archive/benchmarks_v2.0.0/` contains historical JSON moved
  out of active benchmark docs during freeze cleanup.

## Tests

Primary suites used in recent passes:
- `tests/test_attention.py`
- `tests/test_inference_context.py`
- `tests/test_kv_cache_abstraction.py`
- `tests/test_external_cache.py`

## Historical Notes

Development history is intentionally separated from active docs:
- active docs: `README.md`, `docs/*`, `RESULTS.md`;
- historical R&D traces: `devnotes/`.

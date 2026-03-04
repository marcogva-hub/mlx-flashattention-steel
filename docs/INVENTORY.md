# mlx-mfa Repository Inventory

## Project structure

```
mlx-mfa-v2/
├── .github/workflows/ci.yml       # GitHub Actions CI
├── benchmarks/
│   └── bench_attention.py         # MFA vs MLX SDPA timing harness
├── csrc/                          # C++/Objective-C++ extension
│   ├── bindings.cpp               # nanobind Python bindings
│   ├── kernels/
│   │   └── attention_forward.metal # Placeholder (real kernels are JIT)
│   ├── mfa/                       # ccv-derived shader generator
│   │   ├── AttentionKernel.cpp/.hpp
│   │   ├── AttentionKernelDescriptor.cpp/.hpp
│   │   ├── AttentionKernelType.hpp
│   │   ├── AttentionOperand.hpp
│   │   ├── CodeWriter.cpp/.hpp
│   │   ├── DeviceProperties.hpp
│   │   ├── GEMMHeaders.cpp/.hpp
│   │   ├── GEMMOperandPrecision.hpp
│   │   └── mfa_compat.h
│   ├── mfa_attention.cpp/.hpp     # MFAttention MLX Primitive
│   ├── mfa_shader_gen.cpp/.hpp    # KernelKey → ccv shader source bridge
│   ├── mfa_steel_fwd.cpp/.hpp     # STEEL cooperative forward kernel generator
│   ├── shader_cache.hpp           # C++ cache interface
│   └── shader_cache.mm            # Obj-C++ Metal compile/cache
├── docs/
│   ├── ARCHITECTURE.md            # End-to-end technical architecture
│   ├── INVENTORY.md               # This file
│   └── benchmarks/RESULTS.md      # v0.1.0 benchmark results
├── mlx_mfa/
│   ├── __init__.py                # Public re-exports
│   └── attention.py               # flash_attention() + helpers
├── scripts/
│   └── check_env.py               # Pre-build environment validation
├── tests/
│   ├── __init__.py
│   └── test_attention.py          # 41-test suite
├── CMakeLists.txt                 # C++ build (scikit-build-core + nanobind)
├── LICENSE                        # MIT
├── MANIFEST.in                    # sdist file inclusion rules
├── THIRD_PARTY_LICENSES           # Attribution for ccv, MFA, FlashAttention
├── pyproject.toml                 # Package metadata + build config
└── README.md                      # User-facing documentation
```

## Source files

### Python (`mlx_mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `mlx_mfa/__init__.py` | 40 | Public re-exports: `flash_attention`, `is_mfa_available`, `get_device_info`, `get_supported_configs` |
| `mlx_mfa/attention.py` | 370 | Core Python layer: input validation, dtype/GQA routing, `mx.custom_function` backward, fallback SDPA |

### C++/Objective-C++ (`csrc/`)

| File | Lines | Purpose |
|------|-------|---------|
| `csrc/bindings.cpp` | 134 | nanobind module `_ext`: exposes `mfa_attention_forward`, `mfa_forward_with_lse`, `get_device_info` to Python |
| `csrc/mfa_attention.cpp` | 512 | `MFAttention` MLX Primitive — `eval_gpu()` dispatches either STEEL or ccv forward kernel based on dtype/D |
| `csrc/mfa_attention.hpp` | 158 | `MFAttention` class declaration; `MFAParams` struct (shape, dtype, causal, blocking params) |
| `csrc/mfa_steel_fwd.cpp` | 751 | STEEL forward kernel source generator — produces the Metal shader string for cooperative tiled attention; contains blocking config table and `SteelBlockConfig` selection |
| `csrc/mfa_steel_fwd.hpp` | 65 | `SteelBlockConfig` struct; `generate_steel_forward_source()` declaration |
| `csrc/mfa_shader_gen.cpp` | 305 | ccv-path shader generation: maps `KernelKey` → `AttentionKernelDescriptor` → Metal source (used for f32 and legacy configs) |
| `csrc/mfa_shader_gen.hpp` | 59 | `KernelKey` struct; `generate_attention_source()` declaration |
| `csrc/shader_cache.hpp` | 64 | `ShaderCache` singleton interface — `get_or_compile(key, device)` returns `void*` pipeline |
| `csrc/shader_cache.mm` | 175 | Obj-C++ implementation: Metal `newLibraryWithSource`, `newComputePipelineState`, cache keyed on `KernelKey` hash; env-gated shader dump |

### ccv-derived shader generator (`csrc/mfa/`)

| File | Lines | Purpose |
|------|-------|---------|
| `AttentionKernel.cpp` | 3324 | Core shader source emitter — generates Metal attention kernel string from `AttentionKernelDescriptor`; includes QK-GEMM, softmax, PV-GEMM, causal mask, backward loops |
| `AttentionKernel.hpp` | 124 | `AttentionKernel` class; constructor from descriptor, `source` string output |
| `AttentionKernelDescriptor.cpp` | 42 | `AttentionKernelDescriptor` construction from `KernelKey` |
| `AttentionKernelDescriptor.hpp` | 70 | Descriptor fields: `headDimension`, `transposeState[3]`, `blockDimensions`, `GEMM` precision flags |
| `AttentionKernelType.hpp` | 41 | Enum: `loopForward`, `loopBackwardQuery`, `loopBackwardKeyValue` |
| `AttentionOperand.hpp` | 296 | Address-space-aware operand helper — computes byte offsets for Q/K/V/O given batch/head/sequence strides |
| `CodeWriter.cpp/.hpp` | 51/59 | Simple string builder with indentation support for generated Metal source |
| `DeviceProperties.hpp` | 8 | `DeviceProperties` struct (empty placeholder — device caps are passed via `KernelKey`) |
| `GEMMHeaders.cpp` | 786 | Generates Metal simdgroup matrix multiply headers (HGEMM/SGEMM) ported from philipturner/metal-flash-attention |
| `GEMMHeaders.hpp` | 27 | `GEMMHeaders::generate()` declaration |
| `GEMMOperandPrecision.hpp` | 89 | Enum and helpers for `regP[Q/K/V]` precision selection (FP16 vs FP32 GEMM accumulators) |
| `mfa_compat.h` | 81 | Shim replacing ccv-specific headers: `CCV_NNC_MFA_PRECONDITION`, hash utilities, `ccv::nnc::mfa::hash` namespace |

### Metal shaders (`csrc/kernels/`)

| File | Lines | Purpose |
|------|-------|---------|
| `attention_forward.metal` | 30 | Placeholder only — real kernels are JIT-generated strings (not pre-compiled `.metal`) |

### Tests (`tests/`)

| File | Lines | Purpose |
|------|-------|---------|
| `tests/test_attention.py` | 593 | 41-test suite: fallback SDPA, public API, forward correctness (all D/dtype/causal), backward gradients, edge cases (GQA, N=1, cross-attention), backward edge cases |

### Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/check_env.py` | 89 | Pre-build validation: Python version, MLX version, macOS, nanobind, Apple Silicon |
| `benchmarks/bench_attention.py` | 73 | Timing harness: median-of-20-iterations comparison, configurable B/H/D/N/dtype/causal |

### Build & Config

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | scikit-build-core + nanobind build; detects MLX via Python; compiles `csrc/` with Metal framework; produces `mlx_mfa/_ext.*.so` |
| `pyproject.toml` | Package metadata (v0.1.0, MIT, Apple Silicon only), build backend (scikit-build-core), test config |
| `MANIFEST.in` | Ensures `csrc/`, `LICENSE`, `THIRD_PARTY_LICENSES` are included in sdist |
| `.github/workflows/ci.yml` | Two-job CI: fallback tests (no C++ build) + full MFA tests (arm64 runner, Metal GPU) |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | User-facing: performance table, quick start, API reference, supported configs, roadmap |
| `THIRD_PARTY_LICENSES` | Attribution for metal-flash-attention (MIT), ccv (Apache 2.0), FlashAttention (BSD-3-Clause) |
| `docs/benchmarks/RESULTS.md` | Full v0.1.0 benchmark results: B=1/B=2, f16/bf16/f32, causal/non-causal, crossover analysis |
| `docs/ARCHITECTURE.md` | End-to-end architecture reference (see that file) |
| `docs/INVENTORY.md` | This file |

## Dependencies

### Build-time
- `scikit-build-core >= 0.9` — PEP 517 build backend for C++ extensions
- `nanobind >= 2.0` — Python/C++ bindings (NB_DOMAIN "mlx" required for array ABI sharing)
- `cmake >= 3.24` — build system
- `MLX C++ headers` — detected via `mlx.__path__[0]`
- `Metal.framework`, `Foundation.framework` — macOS system frameworks (linked automatically)

### Runtime
- `mlx >= 0.18.0` — MLX array operations and Primitive base class

### Test
- `pytest >= 7.0`
- `numpy` — used in tests for numerical comparison (f32 reference)
- `pytest-benchmark` (optional)

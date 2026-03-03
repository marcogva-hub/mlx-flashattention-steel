# mlx-mfa Development Roadmap

## Phase 1 --- Foundations (2-3 weeks)

### Phase 1.1 --- Scaffolding [DONE]
- [x] Project structure (CMake + nanobind + scikit-build-core)
- [x] Python package with public API + fallback path
- [x] C++ primitive skeleton (MFAttention class)
- [x] Shader cache skeleton (JIT compilation framework)
- [x] Test suite (fallback + extension-gated)
- [x] Benchmark skeleton
- [x] Environment check script
- [x] shader_cache.mm uses Obj-C Metal API (no metal-cpp dep)

### Phase 1.2 --- Extract ccv MFA kernels
- [ ] Clone liuliu/ccv, isolate `lib/nnc/mfa/` subtree
- [ ] Extract `ccv_nnc_mfa_attention.cpp/.hpp` and dependencies
- [ ] Extract `ccv_nnc_mfa_hash.hpp` (kernel key hashing)
- [ ] Extract `v2/` JIT shader generation code
- [ ] Identify all ccv-specific types to replace

### Phase 1.3 --- Decouple from ccv
- [ ] Replace ccv tensor types with MLX array types
- [ ] Replace ccv Metal device access with MLX metal device API
- [ ] Replace ccv allocator with `mlx::core::allocator::malloc_or_wait()`
- [ ] Wire JIT generation into ShaderCache
- [ ] Wire kernel dispatch into MFAttention::eval_gpu()
- [ ] Forward pass functional (single head_dim first, e.g. D=128)

### Phase 1.4 --- Full forward integration
- [ ] All head dims: D=64, 128, 256
- [ ] All dtypes: float16, bfloat16, float32
- [ ] BF16 emulation for M1/M2
- [ ] Causal masking
- [ ] Correctness tests pass vs MLX SDPA reference

## Phase 2 --- Performance & Validation (1-2 weeks)

- [ ] Benchmark sweep D={64,128,256}, N={512..16384}
- [ ] Profiling with Metal GPU profiler (Instruments)
- [ ] Tuning blocking parameters per generation (M1-M4)
- [ ] Test BF16 emulation vs hardware
- [ ] Multi-head batch dispatching optimization

## Phase 3 --- Backward pass (2-3 weeks)

- [ ] Kernel 1: dQ (3D + 5 ops/elem, parallel rows)
- [ ] Kernel 2: dK/dV (4D + 5 ops/elem, parallel cols)
- [ ] Integrate via Primitive::vjp() in MLX autograd
- [ ] Gradient checking (finite differences)
- [ ] End-to-end training test

## Phase 4 --- Production-ready (1-2 weeks)

- [ ] CI (GitHub Actions macOS)
- [ ] Full API documentation
- [ ] Hardware auto-detection (M1 vs M3 vs M4)
- [ ] Wheel distribution
- [ ] Integration guide for mlx-lm, mlx-vlm

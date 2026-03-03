# Claude Code Prompt — mlx-mfa Phase 1.2 to 1.4

## Context

You are working on `mlx-mfa`, a C++/Python library that integrates Metal Flash Attention (MFA) kernels into Apple's MLX framework. Phase 1.1 (project scaffold) is complete. The scaffold includes a working Python API with fallback to `mx.fast.scaled_dot_product_attention`, a C++ Primitive skeleton (`MFAttention`), a shader cache skeleton, nanobind bindings, tests, and benchmarks.

Read `CLAUDE.md` at the repo root first. It contains the full architecture, technical decisions, type mappings, and references.

## Your mission

Complete Phases 1.2, 1.3, and 1.4 to make `flash_attention(q, k, v)` execute a real MFA Metal kernel for the forward pass, producing correct results for head_dim in {64, 128, 256}, dtypes float16/bfloat16/float32, with optional causal masking.

## Phase 1.2 — Extract MFA kernels from ccv

1. Clone `https://github.com/liuliu/ccv` (branch `unstable`) into a temporary directory.

2. Study the MFA subtree at `lib/nnc/mfa/`. The key files are:
   - `ccv_nnc_mfa_attention.cpp` and `.hpp` — main attention dispatch, parameter resolution, blocking tables
   - `ccv_nnc_mfa_hash.hpp` — kernel key hashing (FNV-1a)
   - `ccv_nnc_mfa_gemm.cpp` and `.hpp` — GEMM primitive used internally by attention
   - `v2/` directory — the actual JIT shader generation code (this is the core)

3. Also study the Swift reference at `https://github.com/philipturner/metal-flash-attention` for:
   - `Sources/FlashAttention/Attention/AttentionKernel/` — shader generation logic
   - `Documentation/` — blocking tables, performance data, algorithm pseudocode

4. Copy the relevant ccv files into `csrc/ccv_extracted/` (temporary staging area). Do NOT modify them yet. Document what each file does and what ccv dependencies it has.

5. Create `csrc/ccv_extracted/EXTRACTION_NOTES.md` documenting:
   - Which files were extracted and why
   - All ccv-specific types/functions that need replacement
   - The dependency graph between extracted files
   - Which parts of the JIT generation are essential vs optional

## Phase 1.3 — Decouple from ccv and integrate

This is the hardest phase. Work incrementally: get D=128 forward working first, then expand.

1. Create `csrc/mfa_kernels.hpp` and `csrc/mfa_kernels.cpp` (or `.mm` if Metal API calls needed) containing the decoupled MFA kernel generation and dispatch logic. Key changes:
   - Replace all `ccv_nnc_tensor_t` usage with buffer pointer + shape/stride info passed as parameters
   - Replace ccv device access with the Metal device obtained from MLX (see below)
   - Replace ccv command queue/buffer with MLX's compute encoder pattern
   - Keep the JIT shader generation logic as intact as possible — it is heavily optimized and subtle
   - Keep the blocking parameter tables verbatim — they encode months of tuning

2. Getting the Metal device and command encoder from MLX:
   - MLX exposes its Metal device and command encoder through internal C++ APIs
   - Study how MLX's own `scaled_dot_product_attention` dispatches kernels. Look at:
     - `mlx/backend/metal/primitives.cpp` or similar in the MLX source
     - `mlx/backend/metal/kernels/` for how MLX structures its own Metal kernels
   - The pattern is typically:
     ```cpp
     auto& s = stream();
     auto& d = mlx::core::metal::device(s.device);
     auto command_buffer = d.get_command_buffer(s.index);
     auto encoder = command_buffer->computeCommandEncoder();
     // ... set pipeline, buffers, dispatch ...
     encoder->endEncoding();
     ```
   - NOTE: MLX's internal Metal API may differ between versions. Check the installed MLX source headers.

3. Wire the JIT generation into `ShaderCache::generate_shader_source()`. The generated source should be a complete Metal shader string that compiles via `newLibraryWithSource`.

4. Wire the kernel dispatch into `MFAttention::eval_gpu()`:
   - Resolve block params for the given head_dim
   - Get/compile pipeline from ShaderCache
   - Get Metal command encoder from MLX
   - Set buffer arguments: Q, K, V, O, L (logsumexp), and a params struct
   - Dispatch threadgroups: grid = (ceil(N/block_q), B*H, 1), threadgroup = (n_warps*32, 1, 1)

5. Validate: `pip install -e . && pytest tests/ -v -k "MFAKernel and test_forward_correctness and D128"` should pass.

## Phase 1.4 — Full forward integration

1. Extend to all head dims: D=64, D=128, D=256
2. Extend to all dtypes: float16, bfloat16, float32
3. Implement BF16 emulation path for M1/M2 (check if device supports native bfloat: Metal feature set)
4. Implement causal masking in the generated shader
5. All tests in `TestMFAKernel` should pass
6. Run benchmarks and record results in `docs/benchmarks.md`

## Important technical notes

### MFA algorithm (forward)
```
for each K/V tile (traversal dimension):
    load K_tile into threadgroup memory (simdgroup_async_copy if available)
    S = Q_tile @ K_tile^T            (SIMD FMA in registers/threadgroup)
    if causal: apply mask to S
    online softmax update: m_new = max(m_old, rowmax(S))
                          l_new = l_old * exp(m_old - m_new) + rowsum(exp(S - m_new))
                          O = O * (l_old/l_new) * exp(m_old - m_new) + exp(S - m_new) @ V_tile / l_new
    load V_tile
    O_acc += softmax(S) @ V_tile
write O and L = m + log(l)
```

### Register spilling (D=256)
At D=256, nothing fits in registers. MFA intentionally spills to threadgroup memory with optimized access patterns. The 3D blocking adds a loop over head_dim sub-tiles. This is all encoded in the blocking tables and JIT generation — preserve it.

### BF16 emulation
On M1/M2 (no hardware bfloat16), MFA uses: `as_type<float>(uint(bf16_val) << 16)` for conversion. On M3+ with native BF16, this is unnecessary. The JIT generator should check device capabilities.

### Thread dispatch geometry
- Each threadgroup processes one Q tile (block_q rows)
- Grid X = ceil(N / block_q) — one threadgroup per Q tile
- Grid Y = B * H — one "row" per batch*head
- Threadgroup size = n_warps * 32 threads (SIMD groups)

### Buffer layout
Q, K, V are [B, H, N, D] contiguous. The kernel indexes via:
```
offset = ((b * H + h) * N + row) * D + d
```
Make sure inputs are contiguous before dispatch (MLX may give non-contiguous views). Use `mlx::core::copy(arr, ...)` or check strides.

## Constraints

- Do NOT break the existing fallback path — it must continue working for unsupported configs
- Do NOT modify the Python API signature
- Do NOT add heavy dependencies (no Boost, no Torch, etc.)
- Keep shader_cache.mm as Objective-C++ (not metal-cpp)
- All new C++ code should compile with `-std=c++17`
- Test with `pytest tests/ -v` after every significant change
- If you hit a wall with MLX internal APIs, check the MLX source directly: `python -c "import mlx; print(mlx.__file__)"` then inspect the include directory

## Success criteria

```bash
pip install -e .
pytest tests/ -v                 # ALL pass (fallback + extension)
python benchmarks/bench_attention.py --head-dim 128 --seq-len 1024 4096
# MFA should produce correct results (matching SDPA within tolerance)
# Performance comparison will show actual vs SDPA timing
```

## How to work

1. Start by reading CLAUDE.md
2. Run `python scripts/check_env.py` to validate the environment
3. Run `pytest tests/ -v` to confirm fallback tests pass
4. Work on Phase 1.2 (extraction) first — understand the ccv code thoroughly before modifying
5. Phase 1.3: start with D=128 only, get one config working end-to-end
6. Phase 1.4: expand to all configs
7. Commit frequently with descriptive messages
8. If MLX internal API access is problematic, document the issue and try alternative approaches (e.g., mx.fast.metal_kernel for a simpler initial version, then migrate to full Primitive later)

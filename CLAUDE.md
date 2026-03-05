# CLAUDE.md

## MANDATORY — Git safety
NEVER delete, move, or `rm -rf` the repository directory.
ALWAYS `git push origin master` before starting any destructive operation.
NEVER run `rm -rf` on any path containing the repo root.

## What is this project?

mlx-mfa is a Python/C++ library that brings Metal Flash Attention (MFA) kernels to Apple's MLX framework. It exposes `flash_attention(q, k, v)` as a drop-in replacement for `mx.fast.scaled_dot_product_attention()`.

## Architecture

```
Python API (mlx_mfa.flash_attention)
    |
    +-- fallback --> mx.fast.scaled_dot_product_attention
    |
    +-- MFA path --> C++ Extension (nanobind, scikit-build-core)
                         |
                         +-- MFAttention : mlx::core::Primitive
                         |     eval_gpu()  -> forward Metal dispatch
                         |     vjp()       -> backward (Phase 3)
                         |
                         +-- ShaderCache (singleton)
                         |     JIT Metal shader source strings
                         |     Compile via MTLDevice newLibraryWithSource
                         |     Cache MTLComputePipelineState by param hash
                         |
                         +-- Metal GPU Kernels (JIT-generated, NOT static .metal)
```

## Key technical decisions

1. **C++ extension (nanobind)** over `mx.fast.metal_kernel()`: backward pass needs 2 separate kernel dispatches (dQ and dK/dV), JIT generation too complex for inline API, need autograd via `Primitive::vjp()`.

2. **shader_cache.mm is Objective-C++**: uses native Metal API directly instead of metal-cpp wrapper. Interface uses `void*` with `__bridge_retained` for ARC-safe pipeline management.

3. **Kernel source**: from C++ port in [liuliu/ccv](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa), NOT Swift reference (philipturner/metal-flash-attention). The ccv code is proven in production (Draw Things).

4. **JIT shader generation**: Metal shaders are source strings generated at runtime, parameterized by head_dim, dtype, block sizes, causal flag, device caps. NOT pre-compiled .metal files.

5. **Backward uses 7 GEMMs** (vs standard 5) to avoid FP32 atomics not natively supported on Apple Silicon. Two kernels: dQ (parallel rows), dK/dV (parallel cols).

## Build system

- scikit-build-core + CMake + nanobind
- MLX detected via Python (no find_package)
- Requires: macOS arm64, Python 3.10+, mlx >= 0.18.0, nanobind >= 2.0
- `pip install -e .` builds everything
- `python scripts/check_env.py` validates env

## Source layout

```
csrc/
  mfa_attention.hpp/.cpp  -- MFAttention Primitive
  shader_cache.hpp        -- Cache interface (pure C++)
  shader_cache.mm         -- Obj-C++ Metal compilation
  bindings.cpp            -- nanobind module
  kernels/                -- Placeholder .metal (real kernels are JIT)
mlx_mfa/
  __init__.py             -- Public API
  attention.py            -- flash_attention() + fallback
tests/test_attention.py   -- Fallback + extension-gated tests
benchmarks/               -- MFA vs MLX SDPA comparison
scripts/check_env.py      -- Pre-build validation
```

## MFA kernel source (ccv)

Extract from:
```
liuliu/ccv  (branch: unstable)
  lib/nnc/mfa/
    ccv_nnc_mfa_attention.cpp/.hpp  -- Param resolution, dispatch
    ccv_nnc_mfa_hash.hpp            -- Kernel key hashing
    ccv_nnc_mfa_gemm.cpp/.hpp       -- GEMM primitive
    v2/                             -- JIT shader generation (current)
    3rdparty/                       -- metal-cpp headers (skip)
```

ccv type mapping:

| ccv | MLX |
|-----|-----|
| `ccv_nnc_tensor_t` | `mlx::core::array` |
| ccv Metal device | `mlx::core::metal::device()` |
| ccv allocator | `mlx::core::allocator::malloc_or_wait()` |
| ccv command buffer | MLX compute encoder |
| ccv stream | `mlx::core::Stream` |

## MFA blocking parameters

Deformed tile aspect ratios:
- Parallelization dim: 16-32 (small, many parallel tiles)
- Traversal dim: 80-128 (large, amortize register spilling)
- D=256: 3D blocking splits head_dim into sub-tiles of 128

Block params vary by generation (M1/M2 vs M3/M4). Lookup tables in ccv source.

## MLX Primitive pattern

```cpp
class MFAttention : public mlx::core::Primitive {
  void eval_gpu(inputs, outputs) override;
  std::vector<array> vjp(...) override;
};

auto outputs = array::make_arrays(
    {out_shape, lse_shape}, {q.dtype(), float32},
    std::make_shared<MFAttention>(stream, params), {q, k, v});
```

## Current status

v0.5.0 in progress — 107 tests pass.

| Track | Description | Status |
|-------|-------------|--------|
| 1.1–1.4 | Forward pass, ccv kernels | Done |
| 1.5 | Backward via mx.vjp(SDPA) | Done |
| 4 | GQA, public API, CI | Done |
| A/B | STEEL kernel, D=256 block config | Done (v0.1.0) |
| B | Block-sparse attention | Done (v0.2.0) |
| C/D | Native GQA, mlx-lm integration | Done (v0.3.0) |
| F/G | M3+ routing, sparse backward | Done (v0.4.0) |
| H | Flash Decoding (split-KV) | Done (v0.5.0) |
| I | M5+ detection stub | Done (v0.5.0) |

## Post-Phase 1 Technical Notes

### transposeState Fix (Critical)

The original ccv code sets `transposeState = true` for all operands. This was intended to
compute head offsets as `head * D * seqLen`, but it also switched the inner GEMM to
column-major addressing (`K[d, s]` instead of `K[s, d]`) with `seqLen` as leading
dimension instead of `D`.

Fix applied in two places:

1. `csrc/mfa/AttentionKernel.cpp` `operandLocationWithHeadOffsetValue` — Both transposed
   and non-transposed branches now unconditionally emit `* {{SEQUENCE_LENGTH}}` for head
   offset. This decouples head-offset calculation from GEMM behavior.

2. `csrc/mfa_shader_gen.cpp` — `transposeState[all] = false` so inner GEMMs use
   `leadingDimension = headDimension` (D) and row-major `apply_offset`, correctly reading
   `Q[n,d]` and `K[s,d]`.

**DO NOT REVERT THIS FIX.** If backward kernels need transposed operands, handle via
explicit tensor transpose in MLX before passing to the kernel, not via `transposeState`.

### bfloat16 numpy conversion

`numpy` PEP 3118 does not support `bfloat16`. When converting MLX bfloat16 arrays to
numpy for testing, cast to float32 first within MLX:

```python
np.array(mlx_bf16_array.astype(mx.float32))
```

### Memory layout

MLX arrays passed to `eval_gpu()` are expected to be contiguous in BHND layout.
The kernel assumes:

- `Q`: `[B, H, N, D]` row-major, leading dim = `D`
- `K`: `[B, H, S, D]` row-major, leading dim = `D`
- `V`: `[B, H, S, D]` row-major, leading dim = `D`

If MLX passes non-contiguous arrays, they must be made contiguous before dispatch.

## Performance targets

- Forward D=128 N=4096: >= 20% faster than MLX SDPA
- Forward D=256 N=8192: >= 30% faster
- ALU utilization >= 70%
- Max abs error < 1e-5 (f32), < 1e-2 (f16)

## Testing

```bash
pytest tests/ -v                     # Fallback (no build needed)
pytest tests/ -v -k "MFAKernel"     # Extension tests (needs build)
python benchmarks/bench_attention.py
```

## References

- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention) -- Algorithm, blocking tables, pseudocode
- [liuliu/ccv mfa subtree](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa) -- C++ source
- [Draw Things blog](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18) -- Production validation
- [MLX custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX C++ extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)

## v0.5.0 Technical Notes

### Flash Decoding (Track H)

Two-phase split-KV decode for N_q ≤ 4 queries with S ≥ 256 KV (f16/bf16 only).
Activated automatically in `eval_gpu()` when conditions are met.

**Phase 1** (`mlx_mfa_flash_decode_partial`): Splits KV into `num_splits` chunks.
Grid = `(NQ * num_splits, H, B)`. Outputs partial O/L to scratch buffers.

**Phase 2** (`mlx_mfa_flash_decode_reduce`): Combines splits via exp2 LSE.
Grid = `(N, H, B)`. Writes final O and logsumexp to MLX output arrays.

**Critical fixes during development:**
- `enc.barrier()` not `enc.maybeInsertBarrier()` between phases. The
  `maybeInsertBarrier()` method is a no-op if `needs_barrier_` is false,
  which is only set by `set_output_array()`. Raw `set_buffer()` calls for
  scratch buffers never set this flag.
- `qL_off = S - N` for causal decode (not 0). Query at position `i` must
  see keys `0..(S - N + i)` — the K-loop start must be offset accordingly.
- P@V loop: `for iq → for ik → for id` with V indexed as
  `Vs[Vs_off + ik*8*LDV + id*8]` and `Stile.frag_at(iq, ik)`.

**`compute_num_splits(kL, BK)`**: targets ≥2 K-tiles per split, capped at 32.

### M5+ Detection Stub (Track I)

`get_device_info()` now returns `is_m5_plus = (gen >= 17)`. Gen 17 = M5 family
(A19 SoC with Metal 4 tensor API — `MTLTensor`, cooperative tensors). Not
available on M1–M4. `TensorOpsForward` KernelType is reserved as a commented
stub in `shader_cache.hpp` for when M5 hardware is available for implementation.

`is_m5_plus` implies `is_m3_plus` (gen ≥ 17 ⊃ gen ≥ 15).

## Post-Phase 2 Technical Notes

### simdgroup_async_copy — Definitive Status (macOS 26)

`simdgroup_async_copy` is a private AIR intrinsic (`__asm("air.simdgroup_async_copy_2d...")`)
that Apple removed from runtime Metal shader compilation in macOS 26. Confirmed by liuliu
(ccv maintainer) on Apple Developer Forums. Not a regression — intentional removal.

- **Metal 4 tensor API** (`MTLTensor`, cooperative tensors): M5+/A19+ only. Not available on M1–M4.
- **MLX SDPA** runs at full speed on macOS 26 because it uses standard per-thread
  threadgroup loads + `threadgroup_barrier`, not async DMA.

### MFA tile load paths

The kernel has three compile-time paths controlled by `preferAsyncCache` and `preferAsyncLoad`:

| Path | preferAsyncCache | preferAsyncLoad | K/V load | Q load | macOS 26 status |
|------|:---:|:---:|---|---|:---:|
| M1/M2 original | false | true | simdgroup_async_copy DMA | simdgroup_async_copy | BROKEN |
| M3+ (AsyncCache) | true | false | per-lane device reads | simdgroup_async_copy | WORKS |
| software fallback | any | any | disableAsyncCopy=true loop | ditto | SLOW (~12×) |

**Fix applied**: `mfa_shader_gen.cpp` forces `preferAsyncCache=true, preferAsyncLoad=false`
for ALL GPU generations. K/V are read directly from device memory per lane — no async DMA.
Q still goes through the software async_copy fallback, but Q is loaded only once per
head-dim slice (8× for D=128, block_d=16), amortized over N/block_k ≈ 51 K-tile iterations.
Total async_copy work reduced by ~86%.

### Fix A/B results (pre-quick-win baseline)

Fix B (f16 MAD GEMM via `regP[Q/K/V]=FP16` when `low_prec_inputs=true`) and Fix A
(compile-time div/mod in the software fallback) had **no measurable performance impact**:
127ms → 127ms. The tile-load bottleneck completely dominated.

Key insight: `low_prec_inter` and `low_prec_inputs` are **decoupled** in `mfa_shader_gen.cpp`.
`low_prec_inputs` controls `regP[Q/K/V]=FP16` (GEMM register precision, independent of tile size).
`low_prec_inter` controls the blocking table (forward vs forwardMixed). Setting
`low_prec_inter=true` for f16 caused a 53% regression because forwardMixed tiles
(bk=128, bd=32 = 4096 elems/tile) are 2× larger than forward tiles (bk=80, bd=16 = 1280
elems/tile), making the software async_copy fallback 2× slower. Fix: keep `low_prec_inter=false`
(small tiles) while `low_prec_inputs=true` still gives f16 MAD GEMM.

### is_m3_plus threshold

Architecture gen from `get_architecture_gen()` returns the numeric suffix of the architecture
string (e.g. "applegpu_g13s" → 13). **NOT** the `MTLGPUFamilyApple` enum value.

| Gen | Chip |
|-----|------|
| 13 | M1 |
| 14 | M2 |
| 15 | M3 |
| 16 | M4 |

Correct threshold for M3+: `>= 15`. The old `>= 9` threshold was always true on all
modern Apple Silicon (M1 has gen 13). Fixed in forward pass eval_gpu; backward passes
also patched.

### STEEL kernel — replacement for ccv MFA

STEEL is a rewritten attention kernel using standard threadgroup memory loads instead of
`simdgroup_async_copy`. It completely replaces the ccv-based approach for the forward pass.

**Performance (M1 Max, f16, B=2 H=8)**

| D | N=4096 | vs ccv |
|---|--------|--------|
| 64 | 1.01x SDPA | was 0.16x |
| 128 | 0.95x SDPA | was 0.17x |
| 256 | 0.28x SDPA | was 0.06x |

**Key optimizations applied**

- Standard threadgroup loads (no async copy dependency)
- Removed 16,384 unnecessary `simdgroup_barrier` calls per kernel for BD=128
- PV loop reorder: `for(ik)` outer, `for(id)` inner — keeps `Stile[iq][ik]` in registers
  across all TD iterations; V_smem access stride=8 (16 bytes, cache-line friendly) vs
  the previous 2176-byte jumps between ik steps
- ccv routing removed for D>128 (STEEL handles all D values; ccv 3D-blocking +
  async_copy fallback is slower than STEEL register spill for D=256)

**D=256 register spill**

D=256 requires 256-wide head dimension in registers → spill on M1/M2 (32K register file).
Needs 3D blocking (block_d tiling along head dim) to fit in registers. Current target:
0.5–0.7x SDPA (realistic ceiling without hardware async copy).

**Remaining performance ceiling**

The 5% gap at D=128 is likely the cost of software tile loads vs what hardware DMA
(`simdgroup_async_copy`) provided. Cannot be closed without hardware support
(M5+ tensor API or restored async copy).

## Output constraint — MANDATORY
NEVER produce a monolithic response exceeding 20000 tokens.
### Reading large files
NEVER open an entire file without checking its size first. Before reading any source file:
1. Run `wc -l <file>` to check line count
2. If > 500 lines, NEVER read the whole file. Instead:
   - Use `grep -n` to locate relevant sections
   - Use `head -n` / `tail -n` to read specific portions
   - Use `sed -n 'START,ENDp'` to extract targeted line ranges
   - Read the file in chunks using view with line ranges
3. If you need to understand a file's structure, use `grep -n "function\|class\|struct\|def \|void \|enum" <file>` first
### Writing output
For long tasks, systematically break down the work:
1. Make ONE change (one fix, one file)
2. Commit
3. Test
4. Briefly summarize what was done (~200 words max)
5. Move to the next change
NEVER write long recap reports at the end of a session.
Summarize in 500 words maximum, using a table format when relevant.

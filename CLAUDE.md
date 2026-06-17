# CLAUDE.md

## MANDATORY — Git safety
NEVER delete, move, or `rm -rf` the repository directory.
ALWAYS `git push origin master` before starting any destructive operation.
NEVER run `rm -rf` on any path containing the repo root.

## Auto-default principle (Sprint U, 2026-05-12)

Every PyPI release of `mlx-mfa` must be **fully functional transparently**
for users. When implementing a new optimization:

1. **Default path**: integrate auto-routing through existing mlx-mfa
   public surfaces (`flash_attention*`, `sparse_attention*`,
   `conv3d_nax_forward`, etc.) OR register a hook in
   `mlx_mfa/_auto_hooks.py::install_hooks()` if external `mx.*` surface
   needs patching.
2. **Validation pending**: if the optimization can't ship as default
   yet (e.g., methodology issues, perf claim unconfirmed), introduce
   an env var (`MFA_*`) as transitional opt-in. Document the path to
   default in CHANGELOG.
3. **Granular control**: if the optimization needs per-module decisions,
   ship a named patcher (`patch_<feature>`) as expert API. Document but
   don't make it primary.

Pre-tag audit checklist in `CLAUDE_V6_NAX.md` §5.X enforces this. See
`docs/RELEASE_PHILOSOPHY.md` for the canonical statement and three
usage levels (default / explicit / expert).

## Public API path + skill checkpoints (2026-05-13)

Two institutional rules added to `CLAUDE_V6_NAX.md` §Z + §AA after the
v2.37.0/v2.37.1 silent integration bug:

1. **Every perf claim in release notes / CHANGELOG / public docs must
   be reproducible via the documented public API path** (e.g.,
   `mx.grad(flash_attention(...))` with default `backend="auto"` and
   documented env vars), NOT just internal kernel benchmarks or
   forced-backend (`backend="mfa"`) measurements.  See §Z.  This is
   enforced by `tests/test_release_notes_perf_claims.py` — every
   parameterized claim must reach its kernel via the default API or
   the test fails.

2. **Mandatory skill invocations at critical decision points**:
   `/mlx-code-review` pre-merge, pre-release, at perf discoveries, at
   FALSIFIED outcomes, and post-doc-creation when perf claims are
   involved.  `/metal-kernel-dev` pre-kernel-design and at register-
   pressure decisions.  `/repo-release-prep` pre-release tag.  See
   §AA for the full mandatory/recommended matrix.

Both rules trace to the v2.37.0/v2.37.1 silent integration bug:
documented "D=64 1.4-1.85× faster" was unreachable via the public API
because `should_use_mfa()` returns False for non-causal D ∈ {64, 128}
and short-circuits to SDPA fallback before the V34 env-var check
runs.  100% of tests passed (every test used `backend="mfa"` forced
path).  Reference audit: `docs/v6-nax/v2.37.x-perf-claim-audit.md`.

## §AA mandatory blocking (added 2026-05-13, Sprint 4)

`CLAUDE_V6_NAX.md` §AA skill invocation checkpoints are **MANDATORY
BLOCKING gates**, not advisory.  Missing invocations halt the
workflow.  Sprint 3 created mlx-mfa-* specialized skills that
automate the mandatory checkpoints:

| Checkpoint | Automation skill |
|---|---|
| Perf discovery ("X× speedup" / "Y% speedup") | `/mlx-mfa-perf-audit` |
| Pre-version-bump (canonical pre-tag gate) | `/mlx-mfa-release-audit` |
| Sub-ms bench / cross-session variance | `/mlx-mfa-bench-methodology` |
| New kernel write | `/mlx-mfa-kernel-design` (deferred post-Sprint-6) |
| Before audit-prescribed kernel sprint (§AA.5 premise validation) | `/mlx-mfa-apple-primitives-coverage` (added 2026-05-14 post-Sprint-3+4 retrospective) |

Every sprint deliverable doc MUST include a "Skill invocations"
table per §AA.2 (templates in `docs/templates/`).  Empty/missing
table = audit fails.  `/mlx-mfa-release-audit` Check 5 enforces this
before any version bump.

See `docs/skills/README.md` for the full mlx-mfa-* skill set +
invocation patterns.  See `CLAUDE_V6_NAX.md` §AA.1-§AA.5 for the
hardened rule + halt protocol + premise validation discipline.

## §AA.5 premise validation (added 2026-05-14)

Before committing to any audit-prescribed kernel sprint, run a
**premise validation check** (~30 min) per `CLAUDE_V6_NAX.md` §AA.5:

1. Inventory available Apple/MLX primitives (`dir(mx)`, `dir(mx.fast)`)
2. Component-decompose the audit's measured regression
3. Bench candidate primitive-based dispatch paths
4. Issue verdict: **FULL_INVERSION** / **PARTIAL_INVERSION** / **CONFIRMATION**

Three sprints in v2.50 found the audit's framing inverted (Sprints 1, 2)
or partially inverted (Sprint 3).  See `docs/v50/audit-framing-inversions.md`
for the empirically-validated catalogue.

## §AA.5.x multi-gate audit requirement (added 2026-05-14)

When an investigation surfaces a kernel-input compatibility issue
(LSE convention, scale convention, dtype packing, buffer layout), the
fix MUST enumerate ALL dispatch sites that produce that input — not
just the one the failing test touches.  See `CLAUDE_V6_NAX.md` §AA.5.x
for the full audit checklist.

Companion methodology doc: `docs/methodology/kernel-debugging.md` —
codifies sentinel-write debugging (~20 min to isolate dispatch-routing
bugs vs ~6h of gradient bisection).  Use sentinel writes BEFORE any
deep kernel-disassembly work when a multi-gate dispatch chain is
suspected.

## Canonical Python environment (2026-05-13)

Always use `.venv/bin/python` for all mlx-mfa work.  **`.venv/` is the
single source of truth.**  Any other venv directories (e.g., legacy
`venv/`, archived as `venv.deprecated.YYYY-MM-DD/`) are deprecated and
must NOT be used.  See Sprint 1 venv-consolidation
(`chore/venv-consolidation`, 2026-05-13) for rationale.

**All tools required for the workflow live in `.venv/`:**

| Tool | Form | Verify with |
|---|---|---|
| `twine` | binary `.venv/bin/twine` | `test -f .venv/bin/twine` |
| `build` | Python module (not a binary) | `.venv/bin/python -c "import build"` |
| `pytest` | binary `.venv/bin/pytest` | `test -f .venv/bin/pytest` |
| `mlx` | importable package | `.venv/bin/python -c "import mlx.core"` |

Note: `build` ships as a Python module only — it does NOT install a
`bin/build` executable.  Use `.venv/bin/python -m build` to invoke it.
Checking for it via `test -f .venv/bin/build` will always fail; use
the import check instead.  This subtlety caused a false "twine not
found" diagnosis in the v2.37.3 release session — the actual problem
was using `which twine` (which searches `$PATH`, not the venv) instead
of `.venv/bin/twine` (which works fine).

**If any tool is missing**, install in-place: `.venv/bin/pip install
<tool>`.  Do NOT create a fresh venv to "start clean" — that's how the
deprecated `venv/` came to exist alongside `.venv/`.

**Before any release flow**, run the sanity check:

```bash
bash scripts/check_venv.sh
```

This is enforced as a pre-tag gate per `CLAUDE_V6_NAX.md` §X.5.

### Build extension

```bash
cd ~/code/mlx-mfa-v2
CMAKE_ARGS="-DPython_EXECUTABLE=.venv/bin/python" \
  .venv/bin/python -m pip install --no-build-isolation -e .
```

### Run tests

```bash
.venv/bin/python -m pytest tests/ -q
```

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
  mfa_attention.hpp/.cpp  -- MFAttention + MFAGNAForward Primitives
  mfa_gna_fwd.hpp/.cpp    -- GNA native kernel JIT generator
  mfa_steel_fwd.cpp       -- STEEL V1 forward kernel
  mfa_steel_fwd_v2.cpp    -- STEEL V2 forward kernel (sequential K/V)
  mfa_sage_fwd.cpp        -- SageAttention forward kernel
  shader_cache.hpp        -- Cache interface (pure C++)
  shader_cache.mm         -- Obj-C++ Metal compilation
  bindings.cpp            -- nanobind module
  kernels/                -- Placeholder .metal (real kernels are JIT)
mlx_mfa/
  __init__.py             -- Public API
  attention.py            -- flash_attention() + variants
  masks.py                -- Block masks (causal, sliding, GNA, diagonal, etc.)
  turboquant.py           -- TurboQuant KV compression
  svdquant/               -- SVDQuantLinear (W4A16 + SVD low-rank)
    linear.py             -- SVDQuantLinear nn.Module
    quantize.py           -- quantize_model() tree walker
tests/
  test_attention.py       -- Core attention tests (800+)
  test_gna_native.py      -- GNA native kernel tests (11)
  test_attn_bias_native.py -- Native attn_bias kernel tests (17)
  test_svdquant.py        -- SVDQuant tests (21)
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

v2.56.0 — 1827 tests pass (see CHANGELOG.md for the current feature matrix; the track table below is a historical v0–v2.27 record). Phase IV complete: TQ-decode eval-collapse gains (IV-D1/D2), whole-repo correctness review (no CRITICAL; A3-1 latent int64-overflow fix), incremental optimization closed-at-floor. `MFA_FORCE_NATIVE_BWD` removed (kernel retained); V3 conditionally-auto-routed (M5-validated).

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
| V2-1 | STEEL V2 (sequential K/V phases, 2× BK) | Done (v1.4.0) |
| V2-2 | V2 split-K for under-occupied grids | Done (v1.4.0) |
| V2-3 | V2 D=256 support (implemented; reverted: regression) | Done (v1.4.0) |
| V2-4 | V2 softcap + sliding window | Done (v1.4.0) |
| GNA | Generalized Neighborhood Attention | Done (v2.12.0) — sparse path production |
| Phase B | Sparse mask utilities (diagonal, strided, temporal group, bias) | Done (v2.13.0) |
| Phase C | Top-k dynamic sparse attention (Python ref) | Done (v2.13.0) — Metal kernel deferred |
| LLM Serving | HybridKVCache, runtime, external cache finalization | Done (v2.14.0) |
| PagedVarlenFwd | Fused packed varlen + paged KV kernel | Done (v2.14.1) |
| Paged causal fix | Per-tile causal zone accounting for qL_off | Done (v2.14.1) |
| TurboQuant P1 | Non-fused KV compression (compress/decompress/cache) | Done (v2.21.0) |
| TurboQuant P2 | K fused in Metal paged varlen kernel | Done (v2.22.0) |
| TurboQuant P3 | V fused in kernel, TGP centroids, runtime integration | Done (v2.23.0) |
| TurboQuant P4 | Optimal 3-bit packing (5.33×) + WHT fusion in kernel | Done (v2.24.0) |
| SVDQuant P1 | SVDQuantLinear (W4A16 + rank-r FP16 correction) | Done (v2.25.0) |
| GNA Native | Inline 3D window Metal kernel (forward-only, D=128) | Done (v2.26.0) |
| attn_bias | Native Metal bias kernel (modes 1/2), DiT dispatch audit | Done (v2.27.0) |

## Post-Phase 1 Technical Notes

### GNA Kernel — BlockLoaderT Template Parameters

CRITICAL: When creating a new kernel that uses BlockLoaderT for K/V loading,
copy the EXACT typedef from a working kernel (V1 forward in mfa_steel_fwd.cpp).

The six template parameters control memory layout:
  `<T, BROWS, BCOLS, kDstStrRow, kDstStrCol, reduction_dim, tgp_size>`

For K loader in Q@K^T GEMM (K needs transposed layout [D, K_seq] in smem):
  `kDstStrRow=1, kDstStrCol=LDK, reduction_dim=0`

Getting kDstStrRow and kDstStrCol swapped produces silent data corruption:
the loader writes K in [K_seq, D] but the GEMM reads it as [D, K_seq].
Q_smem and K_smem values will look correct in isolation — the bug only
manifests as NaN/garbage in the GEMM output.

### GNA Kernel — Architecture Decision (v2.12.0 → v2.26.0)

v2.12.0: Early native GNA kernels (v1: inline ND test, v2: 3D strided window loader)
benchmarked at 0.24-0.89x vs sparse path. Production used sparse fallback only.

v2.26.0: Reimplemented native GNA kernel using V2 STEEL infrastructure (BlockLoaderT,
MMAFrag/MMATile). Two-level masking: `gna_tile_active()` for O(1) tile skip +
per-element window mask after Q@K^T. The native kernel applies the **exact** GNA
window formula per (query, key) pair, which is more precise than the sparse path's
tile-level block mask (conservative over-approximation).

Production path: `flash_attention_gna()` tries native kernel first (D=128, 3D, f16/bf16),
falls back to `make_gna_mask()` + `flash_attention_sparse()` for other configs or backward.
Native kernel is forward-only (no VJP). Backward uses sparse path.

### Paged Causal Zone Fix (v2.14.1)

The paged kernel's causal masking zone check `kb >= (kb_lim - (BQ+BK-1)/BK)`
only applies causal masking to the LAST few K-tiles. This is correct when
qL_off=0 (N_q == S_kv) but wrong when qL_off > 0 (N_q < S_kv, decode/prefill).

Fix: `first_causal_kb = (qb * BQ + qL_off) / BK` — start causal masking from
the K-tile where the first query's causal boundary falls.

The bug was invisible for N_q=1 decode (K-boundary mask coincides with causal)
and for kv_len aligned to block_size. Exposed by the PagedVarlenForward fused
kernel which was a clean implementation that didn't inherit the bug.

Lesson: always test against SDPA (ground truth), not against other internal paths.

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

## v1.4.0 Technical Notes — STEEL V2

### V2 kernel overview
STEEL V2 shares K_smem and V_smem in a single `KV_smem` buffer (sequential K phase
then V phase). This doubles BK vs V1 within the same TGP, halving K-tile iterations:

| D | BQ | BK | WM | TGP bytes | V1 BK | V1 TGP |
|---|----|----|----|-----------:|------:|-------:|
| 64  | 32 | 64 | 4 | 13,824 | 32 | 14,336 |
| 128 | 32 | 32 | 4 | 18,944 | 16 | 19,200 |

D=256 is **not dispatched** to V2. The V2 config (BQ=16, BK=32, WM=2) requires halving
BQ to stay under 32KB TGP, which also halves WM. Fewer warps/TG causes a net regression:
0.62–0.84× causal, 0.58–0.62× non-causal vs V1. D=256 always routes to V1.

### V2 dispatch routing
Two routing checks in `eval_gpu()`:
1. **V2 split-K** (Phase 3): `v2sk_eligible` — fires for under-occupied grids
   (`total_tgs < 0.8 * gpu_cores`). Softcap: OK. Sliding window: excluded (interacts with
   split ranges). D=64/128 only.
2. **V2 single-pass**: `v2_eligible` — fires for all other well-occupied grids.
   Softcap + sliding window: both OK. D=64/128 only.

Both blocks gated by `if (!std::getenv("MFA_DISABLE_V2"))` for benchmarking.

### V2 benchmark results (M1 Max, B=2 H=8, f16, causal)

| D | N | V2/SDPA | V2/V1 |
|---|---|--------:|------:|
| 64  | 4096 | 1.95× | 1.66× |
| 64  | 8192 | 2.07× | 1.21× |
| 128 | 4096 | 1.67× | 1.51× |
| 128 | 8192 | 1.74× | 1.26× |

Non-causal: V2 1.04–1.32× vs V1 (smaller benefit; fewer K-tiles to amortize).

### V2 Phase 5 — softcap + sliding window
Softcap (`has_softcap`): tanh applied in log2 domain via `log2e`/`ln2` conversion,
after QK scale, before masking. Uses `precise::tanh` for Metal accuracy.

Sliding window (`has_window`): `kb_start` computed before MFABlockLoaderT construction;
K and V base pointers advanced O(1) before loader creation. Right bound clips `kb_lim`
and masks per-element in the K-tile loop.

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

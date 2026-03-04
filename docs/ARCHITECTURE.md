# mlx-mfa Architecture

## Overview

mlx-mfa provides `flash_attention(q, k, v, scale, causal)` as a drop-in
replacement for `mx.fast.scaled_dot_product_attention`.  The implementation
uses hand-tuned Metal GPU kernels compiled JIT at runtime, dispatched through
an MLX C++ Primitive.

---

## Data flow

```
Python: flash_attention(q, k, v, causal=True)
         │
         ▼
mlx_mfa/attention.py
  1. Validate shapes (4-D, matching head_dim)
  2. GQA: if H_kv < H_q, tile k/v via mx.repeat(k, ratio, axis=1)
  3. Check _can_use_mfa: head_dim ∈ {64,128,256}, dtype ∈ {f16,bf16,f32},
     extension importable
     ├── no  → _fallback_sdpa: mx.fast.scaled_dot_product_attention
     └── yes → _mfa_forward
                 │
                 ▼
         mx.custom_function (_make_mfa_custom)
           Forward: mfa_forward_with_lse (C++ binding)
           Backward: mx.vjp(_fallback_sdpa)
                 │
                 ▼
         C++ binding: csrc/bindings.cpp
           mfa_attention_forward(q, k, v, scale, causal)
                 │
                 ▼
         MFAttention::eval_gpu (csrc/mfa_attention.cpp)
           ├── dtype == f32 → ccv path (generate_attention_source)
           └── dtype == f16/bf16 → STEEL path (generate_steel_forward_source)
                 │
                 ▼
         ShaderCache::get_or_compile (csrc/shader_cache.mm)
           ├── key in cache → return cached id<MTLComputePipelineState>
           └── key missing  → newLibraryWithSource + newComputePipelineState
                 │
                 ▼
         Metal GPU kernel execution
           Result: O [B, H, N, D], L [B, H, N] (logsumexp)
```

---

## STEEL kernel architecture (`csrc/mfa_steel_fwd.cpp`)

The STEEL (Structured Tiled Execution Engine Layer) kernel is the primary
forward path for f16 and bf16.  It is JIT-generated as a Metal source string
from `generate_steel_forward_source()` — there is no pre-compiled `.metal` file.

### Tiling strategy

Each threadgroup processes one Q-block (`BQ` rows) against all K/V blocks:

```
for kb in 0..num_k_blocks:
    if causal and entire tile is masked: continue (tile skip)

    // Cooperative load: all threads load K tile [BK × D] to threadgroup SRAM
    load K[kb] → K_smem
    load V[kb] → V_smem
    barrier

    // QK GEMM: S[BQ × BK] = Q_reg[BQ × D] × K_smem[D × BK]
    // (Q is pre-loaded into registers before this loop)
    S = simdgroup_multiply(Q_reg, K_smem)

    // Online softmax (running max + sum across K tiles)
    apply causal element mask (diagonal tiles only)
    update running_max, running_sum; rescale accumulator O_reg

    // PV GEMM: O_reg[BQ × D] += P[BQ × BK] × V_smem[BK × D]
    O_reg += simdgroup_multiply(P, V_smem)

// Write O_reg to device memory; write logsumexp L
```

**Key design decision**: Q is loaded into registers *once* before the K-loop
(Q hoisting).  K and V are loaded into threadgroup SRAM for each tile.
This avoids re-loading Q every iteration (O(N²) savings for long sequences).

### Causal tile skipping

For a Q-block at row offset `q_start` and a K-block at column offset `k_start`:
- If `k_start >= q_start + BQ` (entire K tile is after all Q rows): skip entirely.
- If `q_start <= k_start < q_start + BQ` (diagonal tile): partial mask applied per-element.
- Otherwise (K tile is fully before Q rows): no masking needed.

This halves the effective K-tile iterations for causal attention → **1.5–2.9× speedup**
vs SDPA at large sequence lengths (SDPA applies the causal mask without skipping tiles).

### Blocking tables

| head_dim | dtype | BQ | BK | WM | WN | TGP threads | TGP memory |
|:--------:|-------|:--:|:--:|:--:|:--:|:-----------:|:----------:|
| 64 | f16/bf16 | 32 | 32 | 4 | 1 | 128 | ~16 KB |
| 128 | f16/bf16 | 32 | 16 | 4 | 1 | 128 | ~20 KB |
| 256 | f16/bf16 | 32 | 16 | 4 | 1 | 128 | ~29 KB |
| any | f32 | 16 | 16 | 2 | 1 | 64 | ~12 KB |

- `BQ` = Q rows per threadgroup
- `BK` = K/V rows per tile iteration
- `WM × WN × 32` = number of SIMD groups × 32 threads = total threadgroup size
- TGP memory < 32 KB (Metal hard limit per threadgroup)

**D=256 note**: f16/bf16 uses `BQ=32/WM=4` (128 threads) for better occupancy.
Larger configs were tested but regressed due to TGP memory exceeding 32 KB.
The f32 path is routed to the ccv kernel (not STEEL) due to register pressure.

---

## ccv-derived kernel (`csrc/mfa/`)

Used for f32 inputs (and as a fallback for configurations STEEL doesn't support).
Significantly slower than STEEL on macOS 26 due to the `simdgroup_async_copy` removal:

| Path | macOS 26 | Notes |
|------|:--------:|-------|
| STEEL (no async DMA) | ✓ | `preferAsyncCache=true` — per-lane device reads |
| ccv original | ✗ | `simdgroup_async_copy` AIR intrinsic removed |
| ccv + disableAsyncCopy | ✓ (slow) | Software fallback, ~10–15× slower than STEEL |

The ccv path generates Metal source via `AttentionKernel` (3324 lines, `csrc/mfa/`)
using the same JIT cache.  Its blocking tables (`mfa_shader_gen.cpp`) mirror
the ccv `AttentionDescriptor::forward()` lookup from liuliu/ccv.

---

## Backward pass

```
mx.grad(flash_attention)(q, k, v)
         │
         ▼
mx.custom_function vjp  (attention.py:_make_mfa_custom)
  dO = _sever_lazy_graph(cotangent)   ← buffer-aliasing fix (see below)
  _, (dQ, dK, dV) = mx.vjp(
      _fallback_sdpa(q, k, v, scale, causal),
      [q, k, v],
      [cotangent]
  )
  return dQ, dK, dV
```

### Why not the C++ Primitive vjp?

`mfa_attention_forward` returns only `O` to Python.  MLX prunes `L` (logsumexp)
from the graph.  When `MFAttention::vjp` runs in C++, `outputs[1]` (L) is gone
— accessing it is undefined behaviour and corrupts gradients.

The Python `custom_function` sidesteps this by *recomputing* gradients from scratch
via `mx.vjp(_fallback_sdpa)`.  This is slightly slower (SDPA backward instead of
STEEL backward) but correct, simple, and MLX-maintained.

### Buffer aliasing fix (`_sever_lazy_graph`)

Inside the vjp, `cotangent` is `ones_like(O_fwd)` — a lazy node inheriting
`O_fwd`'s buffer ancestry.  If the backward re-runs `mfa_forward_with_lse`
in the same Metal encoder, the allocator can alias `O_r`'s output buffer with
the freed `O_fwd` buffer, corrupting `L_r`.

Fix: `cotangent + mx.zeros_like(cotangent)` writes to a fresh buffer with no
shared ancestry, preventing the alias.  See `attention.py:_sever_lazy_graph`
for the full analysis and alternatives tested.

---

## Build system

```
pyproject.toml
  build-backend: scikit-build-core
  ↓
CMakeLists.txt
  Languages: CXX + OBJCXX (for shader_cache.mm)
  Finds: Python.Development.Module, MLX (via python -c "import mlx")
  Frameworks: Metal, Foundation
  Produces: mlx_mfa/_ext.cpython-3XX-darwin.so
```

**nanobind domain**: `NB_DOMAIN "mlx"` is required so that `mlx.core.array`
objects are recognised across the `mlx` and `mlx_mfa._ext` extension boundaries.
Without this, passing MLX arrays to the extension raises a type error.

---

## Device detection and configuration

```cpp
// csrc/mfa_attention.cpp
int gen = d.get_architecture_gen();  // e.g. "applegpu_g13s" → 13
bool is_m3_plus = (gen >= 15);       // 13=M1, 14=M2, 15=M3, 16=M4
```

| Gen | Chip | preferAsyncCache | Notes |
|:---:|------|:----------------:|-------|
| 13 | M1 | false | STEEL: per-lane loads from device |
| 14 | M2 | false | Same as M1 for STEEL |
| 15 | M3 | true | ccv-path uses preferAsyncCache (K/V from device, no DMA) |
| 16 | M4 | true | Same as M3 |

For the **STEEL** path, `is_m3_plus` is not used (the kernel always uses
per-lane loads with no async DMA dependency).  It is used only for the ccv
(f32) path's blocking table selection.

---

## Public API reference

| Function | Signature | Returns |
|----------|-----------|---------|
| `flash_attention` | `(q, k, v, scale=None, causal=False, stream=None)` | `mx.array [B,H,N,D]` |
| `is_mfa_available` | `()` | `bool` |
| `get_device_info` | `()` | `dict` with `device_name`, `gpu_family_gen`, `is_m3_plus`, `chip_name`, `extension_available` |
| `get_supported_configs` | `()` | `dict` with `head_dims`, `dtypes`, `extension_available` |

All functions are exported from `mlx_mfa/__init__.py` and re-exported from
`mlx_mfa/attention.py`.

---

## Key design decisions and non-obvious constraints

1. **JIT Metal compilation** (not pre-compiled `.metal`): kernels are parameterized
   by head_dim, dtype, block dims, and causal flag.  Pre-compiling all combinations
   would require 3 × 2 × N_configs `.air` files shipped in the wheel.

2. **`NB_DOMAIN "mlx"`**: mandatory for sharing `mlx.core.array` ABI between
   MLX and the extension.  Without it, Python raises `RuntimeError: Unable to
   cast Python instance to C++ type`.

3. **`transposeState = false`**: the original ccv code set `transposeState = true`
   which coupled head-offset computation to GEMM inner loop addressing.  mlx-mfa
   unconditionally uses `transposeState = false` and forces `SEQUENCE_LENGTH` in
   the head-offset expression.  See `CLAUDE.md § transposeState Fix`.

4. **`disableAsyncCopy = true`**: `simdgroup_async_copy` (a private AIR intrinsic)
   was removed from Metal shader compilation on macOS 26.  All ccv-path kernels
   use the software-loop fallback.  The STEEL path was designed to avoid this
   intrinsic entirely.

5. **`MTLLanguageVersion3_1`**: required for `bfloat4` vectors used in bf16 kernels.
   Metal 3.0 (macOS 14) only has scalar bfloat; 3.1 (macOS 14.2+) adds vector types.

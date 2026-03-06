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
  2. GQA: if H_kv < H_q, native GQA via gqa_factor in kernel
  3. Check _can_use_mfa: head_dim ∈ {64,128,256}, dtype ∈ {f16,bf16,f32},
     extension importable
     ├── no  → _fallback_sdpa: mx.fast.scaled_dot_product_attention
     └── yes → _mfa_forward
                 │
                 ▼
         mx.custom_function (_make_mfa_custom)
           Forward: mfa_forward_with_lse (C++ binding)
           Backward: see "Backward pass routing" below
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

## Backward pass routing

```
mx.grad(flash_attention)(q, k, v)
         │
         ▼
mx.custom_function vjp  (attention.py:_make_mfa_custom._backward)
         │
         ├── f16/bf16, D≤256, softcap==0, alibi==False
         │       │
         │       ▼
         │   mfa_steel_backward (C++ binding → MFASteelBwdDQ + MFASteelBwdDKV)
         │   • dQ kernel grid: (N_q/BQ, H_q, B)
         │   • dK/dV kernel grid: (N_k/BK, H_kv, B)
         │   returns (dQ, dK, dV)
         │
         ├── f32
         │       ▼
         │   mx.vjp(_fallback_sdpa(q,k,v))   ← MLX SDPA backward
         │
         ├── softcap > 0
         │       ▼
         │   mx.vjp(_softcap_sdpa_ref)        ← compiled via mx.compile
         │
         └── alibi == True
                 ▼
             mx.vjp(_alibi_sdpa_ref)           ← compiled via mx.compile
```

### Why not the C++ Primitive vjp?

`mfa_attention_forward` returns only `O` to Python.  MLX prunes `L` (logsumexp)
from the graph.  When `MFAttention::vjp` runs in C++, `outputs[1]` (L) is gone
— accessing it is undefined behaviour and corrupts gradients.

The Python `custom_function` sidesteps this by recomputing gradients either via
native STEEL backward kernels (fast path) or `mx.vjp(_fallback_sdpa)` (safe
fallback for unsupported dtypes/configs).

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

31 symbols exported in `mlx_mfa.__all__` (version 0.9.3):

### Core attention (11)

| Function | Key arguments | Returns |
|----------|--------------|---------|
| `flash_attention` | `(q,k,v, scale, causal, softcap, dropout_p, return_attn_weights)` | `[B,H,N,D]` |
| `flash_attention_rope` | `(q,k,v, cos,sin, scale, causal, cache_seqlens, rope_3d, interleaved)` | `[B,H,N,D]` |
| `flash_attention_sparse` | `(q,k,v, block_mask, scale, causal, backward)` | `[B,H,N,D]` |
| `flash_attention_varlen` | `(q,k,v, cu_q,cu_k, max_q,max_k, scale, causal)` | `[1,H,total,D]` |
| `flash_attention_with_kv_cache` | `(q, k_new,v_new, k_cache,v_cache, scale, causal, softcap)` | `[B,H,N,D]` |
| `flash_attention_paged` | `(q, k_pages,v_pages, block_table,seq_lens, scale, causal)` | `[B,H,N_q,D]` |
| `flash_attention_qkv_packed` | `(qkv, scale, causal, num_heads, num_kv_heads)` | `[B,H,N,D]` |
| `flash_attention_kv_packed` | `(q, kv, scale, causal, num_kv_heads)` | `[B,H,N,D]` |
| `flash_attention_varlen_qkv_packed` | `(qkv, cu_q,cu_k, max_q,max_k, scale, causal, num_heads, num_kv_heads)` | `[1,H,total,D]` |
| `flash_attention_varlen_kv_packed` | `(q, kv, cu_q,cu_k, max_q,max_k, scale, causal, num_kv_heads)` | `[1,H,total_q,D]` |
| `PagedKVCache` | `(num_blocks, block_size, H, D, dtype)` | allocator class |

### Mask builders (15)

| Group | Functions |
|-------|-----------|
| Dense-sparse | `make_causal_block_mask`, `make_sliding_window_mask` |
| Spatial | `make_spatial_2d_mask`, `make_spatial_3d_mask`, `make_topk_spatial_mask` |
| Segment | `make_segment_mask`, `make_causal_segment_mask` |
| Adaptive | `make_adaptive_window_mask` |
| Video/VSR | `make_lcsa_mask`, `make_axial_spatial_mask`, `make_axial_temporal_mask` |
| Temporal | `make_dilated_temporal_mask` |
| Special | `make_sink_window_mask`, `make_reference_frame_mask`, `make_cross_stream_mask` |

### RoPE helpers (1)

- `make_rope_3d_tables(freqs_h, freqs_w, freqs_t, ...)` — build 3D rotary tables for video

### Utilities (3 + `__version__`)

- `is_mfa_available()` — `True` when C++ extension loaded and Metal available
- `get_device_info()` — `dict`: `device_name`, `gpu_family_gen`, `is_m3_plus`, `is_m5_plus`, `chip_name`, `extension_available`
- `get_supported_configs()` — `dict`: `head_dims`, `dtypes`, `extension_available`

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

6. **Double-buffer ping-pong (Track CF, v0.9.1)**: For D≤128 without RoPE/sparse,
   the STEEL forward kernel uses *separate* `K_smem` and `V_smem` threadgroup arrays
   instead of a shared `KV_smem`.  This lets V-tile stores complete concurrently
   with K-GEMM (different TGP regions, no hazard) and K[n+1]-tile stores complete
   during P@V, reducing threadgroup barriers per K-tile from 4 to 2.

   ```
   Shared KV_smem (4 barriers):
     barrier → K-load → barrier → K-GEMM → barrier → V-load+softmax → barrier → P@V

   Separate K_smem/V_smem (2 barriers):
     [Phase-0: K[0]-load, barrier]
     K-GEMM + V[kb]-load → barrier-1 → softmax → P@V + K[kb+1]-load → barrier-2
   ```

   TGP budget check: D=128 requires `(BK+padK)*BD + BK*(BD+padV)` ≈ 19.2 KB < 32 KB.
   D=256 would exceed 32 KB, so `double_buf` is disabled for D=256.

7. **Persistent multi-Q-block kernel (Track CC, v0.9.1)**: The kernel iterates
   over an outer `qb` (Q-block) loop (`[0, NQ)`) within a single threadgroup
   dispatch, processing 4 Q-blocks per launch. This amortizes Metal command
   buffer overhead at large N (N ≥ 4096) where grid occupancy would otherwise
   leave the GPU partly idle between sequential dispatches.

---

## 8. STEEL native backward kernels (Tracks BA/BB/BC, v0.9.0; CE, v0.9.2)

For f16/bf16 D≤256 with `softcap==0` and `alibi==False`, `_make_mfa_custom`
routes the backward through `mfa_steel_backward` (C++ binding) which dispatches
two Metal kernels: `MFASteelBwdDQ` and `MFASteelBwdDKV`.

**Algorithm**: FlashAttention-2 backward, log2 domain throughout.

```
Given: O, L (logsumexp from forward), dO (cotangent)

P = exp2(S × scale × log2(e) − L_log2)   // recompute attention probabilities
delta = scale × rowsum(O ⊙ dO)            // D_i = sum_j(O_ij * dO_ij)
dS = P × (dO @ V^T − delta)               // dSoftmax

dQ += scale × dS @ K    [dQ kernel,   grid: (N_q/BQ, H_q, B)]
dK += scale × dS^T @ Q  [dKV kernel,  grid: (N_k/BK, H_kv, B)]
dV += P^T @ dO
```

**GQA support**: `gqa_factor = H_q / H_kv` is baked into the Metal shader as
`#define MFA_GQA_FACTOR`.
- dQ kernel: maps `kv_head = tid.y / gqa_factor` to find the K/V row.
- dKV kernel: iterates `for q_head in [kv_head*f .. (kv_head+1)*f)`, accumulating
  dK/dV across all Q-heads in the GQA group.

**D=256 D-split (Track CE, v0.9.2)**: D=256 exceeds the 32 KB TGP budget
(`Q_smem + dO_smem + KV_smem ≈ 46,080 bytes` with standard tiling).
Solution: split D into two halves (`BD_HALF = 128`), reducing TGP to ≈23,552 bytes.

Three-phase algorithm per K-tile (dQ kernel):

```
// Phase 1: accumulate S from both D-halves
load Q_lo  [BQ × 128], K^T_lo  [128 × BK] → S  += Q_lo  @ K^T_lo
load Q_hi  [BQ × 128], K^T_hi  [128 × BK] → S  += Q_hi  @ K^T_hi
// S is now complete → apply softmax/online-max

// Phase 2: accumulate dP from both D-halves
load dO_lo [BQ × 128], V^T_lo  [128 × BK] → dP += dO_lo @ V^T_lo
load dO_hi [BQ × 128], V^T_hi  [128 × BK] → dP += dO_hi @ V^T_hi
// dS = P × (dP − delta) — computed once, shared across phases

// Phase 3: accumulate dQ in both D-halves
load K_lo  [BK × 128] → dQ_lo += dS @ K_lo   (stored at offset 0)
load K_hi  [BK × 128] → dQ_hi += dS @ K_hi   (stored at offset 128)
```

Register tiles `Qtile_lo/hi` and `dOtile_lo/hi` are declared as explicit named
variables *outside* all loops so the compiler keeps them in registers across phases.
`KV_smem` is shared between D-halves (single buffer reused per phase).

---

## 9. Varlen backward via `mx.custom_function` (Track EA, v0.9.3)

`flash_attention_varlen` is wrapped in `mx.custom_function` to provide full
autograd support without a dedicated varlen backward Metal kernel.

```
flash_attention_varlen(q, k, v, cu_q, cu_k, max_q, max_k, ...)
         │
         ▼
@mx.custom_function _varlen_impl(q_, k_, v_)
  Forward:
    ├── f16/bf16, D∈{64,128,256} → STEEL varlen kernel (single Metal dispatch)
    └── else → _varlen_split_concat (Python per-sequence)

@_varlen_impl.vjp _varlen_bwd(primals, cotangent, output)
  Backward: split per sequence → mx.vjp(flash_attention) for each seq
    for i in 0..num_seqs:
        q_i = q_[:, :, cu_q[i]:cu_q[i+1], :]
        k_i = k_[:, :, cu_k[i]:cu_k[i+1], :]
        v_i = v_[:, :, cu_k[i]:cu_k[i+1], :]
        _, (dq_i, dk_i, dv_i) = mx.vjp(flash_attention, [q_i, k_i, v_i], [dO_i])
    dQ = concat(dq_i, axis=2)
    dK = concat(dk_i, axis=2)
    dV = concat(dv_i, axis=2)
```

**Closure design**: `cu_seqlens_q` and `cu_seqlens_k` are materialised to
Python `list[int]` via `.tolist()` *once* in the outer function (before the
`@mx.custom_function` definition), then captured by closure.  Since Python
scalars/lists are transparent to the MLX autograd tape, no `mx.array` slicing
occurs inside the autograd graph — only MLX arrays `q_`, `k_`, `v_` are tracked.

**Per-sequence vjp efficiency**: each `mx.vjp(flash_attention)` call dispatches
to the STEEL backward kernels (for f16/bf16 D≤256), so the split-concat backward
is nearly as fast as a hypothetical single-kernel varlen backward.

---

## 10. Metal paged KV gather kernel (Track EB, v0.9.3)

`MFAPagedKVGather` (`csrc/mfa_paged_gather.cpp`) is a Metal Primitive that
materialises contiguous K/V tensors from a paged block pool in a single GPU
dispatch, replacing the Python for-loop gather from v0.9.0.

**Pool and output layouts**:

```
Pool input:  [num_blocks, block_size, H_kv, D]   (token-major within block)
Output:      [B, H_kv, max_kv_len, D]             (BHND — STEEL-ready)
```

The kernel simultaneously transposes `[block_size, H_kv, D] → [H_kv, block_size, D]`
during the copy, eliminating a separate transpose operation.

**Grid structure**: 1-D grid, one thread per output element.
Total elements = `B × H_kv × max_kv_len × D`.

```metal
kernel void paged_kv_gather(..., uint gid [[thread_position_in_grid]])
{
    // Decode (b, h, kv_t, d) from flat gid
    int d    = gid % D;
    int kv_t = (gid / D) % max_kv_len;
    int h    = (gid / D / max_kv_len) % H;
    int b    = gid / D / max_kv_len / H;

    if (kv_t >= seq_lens[b]) { out[gid] = 0; return; }  // padding

    int log_blk  = kv_t / block_size;
    int tok_off  = kv_t % block_size;
    int phys_blk = block_table[b * max_blocks + log_blk];
    if (phys_blk < 0) { out[gid] = 0; return; }         // sentinel

    // Pool: [phys_blk][tok_off][h][d]
    out[gid] = pool[phys_blk * (block_size*H*D) + tok_off * (H*D) + h*D + d];
}
```

**`flash_attention_paged` with `mx.custom_function`** (v0.9.3):

```
Forward:
  1. K_contig = mfa_paged_kv_gather(k_pages, block_table, seq_lens, max_kv_len)
  2. V_contig = mfa_paged_kv_gather(v_pages, block_table, seq_lens, max_kv_len)
  3. for b in 0..B:
       out[b] = flash_attention(q[b], K_contig[b, :, :kv_len[b], :],
                                       V_contig[b, :, :kv_len[b], :])
  4. return concat(out, axis=0)

Backward:
  1. Re-gather K_contig, V_contig (same as forward)
  2. for b in 0..B:
       _, (dQ[b], _, _) = mx.vjp(flash_attention, [q[b], K_b, V_b], [dO[b]])
  3. dK_pages = zeros_like(k_pages)   ← pools are cache buffers, not parameters
  4. dV_pages = zeros_like(v_pages)
```

**Why per-sequence slicing after gather**: `K_contig[b]` has zeros for positions
`seq_lens[b]:max_kv_len`.  Passing the full `[H, max_kv_len, D]` slice to
`flash_attention` would include those zeros in the softmax denominator
(`exp(Q·0) = 1` per padded position), corrupting the output for shorter sequences
in a batch.  Slicing to `[:kv_len]` ensures only real tokens participate in attention.


## 11. PagedKVCache — Python-level allocator (Track GA, v1.0.1)

`PagedKVCache` (`mlx_mfa/attention.py`) is a pure-Python paged KV cache allocator
that provides the block management layer above the Metal gather kernel.

### Design rationale — numpy backing store

MLX arrays are **immutable** (functional style): every `.at[].set()` call creates
a new array.  For a decode loop appending one token at a time, this means
`O(T × H)` full-array allocations for `T` steps with `H` heads — untenable.

The fix is a **numpy float32 backing store** (`_k_np`, `_v_np`):

```python
# In __init__:
self._k_np = np.zeros((num_blocks, block_size, H, D), dtype=np.float32)
self._v_np = np.zeros((num_blocks, block_size, H, D), dtype=np.float32)
```

`numpy` supports true in-place slice writes:

```python
self._k_np[block_id, ptr:ptr+chunk] = k_tokens   # O(chunk*H*D), no allocation
```

**bfloat16 note**: numpy has no `bfloat16` dtype (PEP 3118 gap).  All backing
stores use `float32`; the `k_pool` / `v_pool` properties convert at access time:

```python
@property
def k_pool(self):
    if self._k_pool_cached is None:
        self._k_pool_cached = mx.array(self._k_np).astype(self.dtype)
    return self._k_pool_cached
```

### Block allocator lifecycle

```
__init__(num_blocks=128, block_size=16, H=8, D=128)
  _free = [0..num_blocks-1]
  _block_table = {}    # seq_id -> [block_ids]
  _write_ptr   = {}    # seq_id -> offset within last block

append(k, v, seq_id=0)
  1. Force MLX graph materialisation + numpy copy
  2. Transpose [B,H,T,D] -> [T,H,D]
  3. while written < T:
       if write_ptr == block_size: allocate new block
       slice-write chunk to numpy backing store
       advance write_ptr
  4. invalidate cached mx.array views

k_pool / v_pool (property)
  mx.array(np_store).astype(dtype)
  cached between appends; invalidated on any mutation

gather(seq_id)
  np.concatenate([k_np[b] for b in block_table[seq_id]])[:seqlen]
  -> mx.array [1, H, seqlen, D]

free_seq(seq_id)
  return blocks to _free list
  invalidate cache
```

### `get_block_table()` and `get_seq_lens()`

These return `mx.int32` arrays suitable as inputs to `flash_attention_kvcache`
(paged mode) and `flash_attention_paged`:

```python
block_table = cache.get_block_table([0, 1])  # mx.int32 [2, max_blocks] (-1 = padding)
seq_lens    = cache.get_seq_lens([0, 1])     # mx.int32 [2]

out = flash_attention_kvcache(
    q, cache.k_pool, cache.v_pool,
    block_table=block_table,
    seq_lens=seq_lens,
    block_size=cache.block_size,
    scale=scale,
    causal=True,
)
```

### `seq_lengths` formula

```python
@property
def seq_lengths(self):
    return {
        sid: (len(blks) - 1) * self.block_size + self._write_ptr[sid]
        for sid, blks in self._block_table.items()
    }
```

This is O(active_sequences) — constant per sequence regardless of total tokens
stored — because it uses `_write_ptr` (the offset within the *last* block) rather
than summing individual block fills.

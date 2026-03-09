# mlx-mfa Tech Debt Remediation Plan -- V2

**Date**: 2026-03-09
**Version**: v1.2.2 (486 tests pass)
**Scope**: NEW or REMAINING friction points after Phases 1-4 remediation
**Baseline**: Previous 25 items mostly fixed; 2 deferred (C.2 batched RoPE, E.3 array seq_lens)
**Constraint**: No precision regressions; no API signature changes

---

## Executive Summary

Post-remediation analysis of the mlx-mfa-v2 codebase identifies **18 new friction points** introduced or revealed by the Phase 3-4 changes (fused quantize primitive, scatter KV, saved LSE backward, etc.). The findings cluster into five categories:

| Category | Count | Worst-case impact |
|----------|-------|-------------------|
| **F. Scatter/Pool write patterns** (full-pool copy, Python token loops) | 4 | O(pool_size) copy per decode token |
| **G. Backward path overhead** (6-tensor fence, zeros_like waste) | 3 | ~0.5ms per backward pass |
| **H. Per-batch Python serialization** (paged gather, varlen backward, RoPE) | 4 | O(B) serial dispatches where 1 suffices |
| **I. Cache/KV append patterns** (concat growth, tolist sync) | 4 | O(seqlen) memory copy per decode step |
| **J. Code-level dead weight** (dead branches, duplicate code blocks) | 3 | Code clarity, maintenance cost |

**Deferred items carried forward** (from V1, require Metal shader changes):
- C.2: Batched RoPE offsets in Metal (requires new C++/Metal shader)
- E.3: Pass seq_lens/block_table as MLX arrays to C++ (requires Metal refactor)

**Top-3 highest-ROI fixes:**
1. **F.1**: Scatter kernel copies entire pool on every 1-token decode write (~256 blocks x 16 x H x D elements). Make it block-scoped so only modified blocks are processed.
2. **H.1**: `_attn_per_seq()` runs B serial `flash_attention` calls in paged decode. Replace with one batched SDPA call using a padding mask (pattern already exists in `_paged_batched_bwd`).
3. **I.1**: `InferenceContext.step()` grows cache via `mx.concatenate` per token. Pre-allocate + write-pointer pattern eliminates the O(seqlen) copy.

---

## Category F: Scatter/Pool Write Patterns

### F.1 -- `mfa_scatter_kv` copies entire pool per write

**Location**: `csrc/mfa_scatter.cpp:49-80` (Metal kernel), called from `mlx_mfa/attention.py:1459-1468`
**Impact**: HIGH -- the scatter kernel reads and writes *every element* of the pool, even when only 1 token out of 256 blocks changes. For a pool of 256 blocks x 16 slots x 8 heads x 128 dims = 4M elements at 2 bytes each = 8 MB read + 8 MB written per decode token.
**Frequency**: Every decode step in paged mode (via `PagedKVCache.append()` or `flash_attention_kvcache` paged-append)

**Current kernel** (Metal):
```metal
// Each thread copies one element from pool_in -> pool_out,
// overriding if (blk, off) matches a scatter target.
// This processes ALL pool elements even when N_write = 1.
for (int n = 0; n < p.N_write; n++) {
    if (blk_ids[n] == blk && blk_offs[n] == off) {
        val = tokens[n * p.H * p.D + h * p.D + d];
        break;
    }
}
pool_out[elem] = val;
```

**Fix**: Two options (increasing effectiveness):
1. **Block-scoped scatter**: Only dispatch threads for blocks that contain at least one write target. Requires a host-side pre-pass to build a "dirty block list" (trivial: unique values of `blk_ids`). Unmodified blocks get a pointer alias or zero-copy.
2. **In-place write with COW**: Use MLX's `array::at().set()` pattern or a C++ primitive that does `pool_out = pool_in` (pointer copy), then writes only the N_write slots. Requires MLX in-place mutation support or a custom `copy_on_write + scatter` primitive.

**Effort**: MEDIUM (option 1), HIGH (option 2)
**Expected savings**: From O(pool_size) to O(N_write x H x D) per decode step. For 256-block pool with 1-token decode: ~4000x fewer elements processed.

---

### F.2 -- Paged-append Python loop builds scatter targets one-at-a-time

**Location**: `mlx_mfa/attention.py:1444-1457`
**Impact**: MEDIUM -- O(B x N_new) Python loop with per-iteration `k_new[b, :, t, :]` array slicing
**Frequency**: Every paged-append step

**Current code**:
```python
for b in range(B_p):
    kv_len = seq_lens_list_p[b]
    tb = block_table_list_p[b]
    for t in range(N_new_p):
        pos = kv_len + t
        blk_idx = pos // blk_sz
        blk_off = pos % blk_sz
        phys = int(tb[blk_idx])
        sc_blk_ids.append(phys)
        sc_blk_offs.append(blk_off)
        sc_k_rows.append(k_new[b, :, t, :])   # MLX slice per token
        sc_v_rows.append(v_new[b, :, t, :])
```

Each `k_new[b, :, t, :]` creates an MLX array view node. For B=4, N_new=1: 8 slices + 2 stacks + 2 scatter calls.

**Fix**: Compute `blk_ids` and `blk_offs` arrays via vectorized arithmetic:
```python
positions = seq_lens_per_batch[:, None] + mx.arange(N_new)[None, :]  # [B, N_new]
blk_indices = positions // blk_sz
blk_offsets = positions % blk_sz
phys_ids = block_table[mx.arange(B)[:, None], blk_indices]
```
Then reshape `k_new` from `[B, H, N_new, D]` to `[B*N_new, H, D]` and pass directly.

**Effort**: MEDIUM
**Expected savings**: Eliminate B x N_new Python iterations + B x N_new array slices

---

### F.3 -- `PagedKVCache.append()` scatter path still has Python token loop

**Location**: `mlx_mfa/attention.py:3378-3393` (scatter path inside `PagedKVCache.append`)
**Impact**: MEDIUM -- same O(T) Python loop for building scatter target lists
**Frequency**: Every `PagedKVCache.append()` call

**Current code**:
```python
for i in range(chunk):
    all_blk_ids.append(blk_id)
    all_blk_offs.append(ptr + i)
```

**Fix**: Replace inner `for i in range(chunk)` with:
```python
all_blk_ids.extend([blk_id] * chunk)
all_blk_offs.extend(range(ptr, ptr + chunk))
```

**Effort**: EASY
**Expected savings**: Minor for decode (chunk=1), meaningful for prefill (chunk=16)

---

### F.4 -- `_scatter_to_pool()` backward is O(num_blocks) Python accumulation

**Location**: `mlx_mfa/attention.py:3629-3669`
**Impact**: HIGH -- in the paged attention backward, each block's gradient tile is sliced, transposed, and accumulated in a Python dict loop. For 64 blocks: 64 transpose + 64 pad + 1 stack.
**Frequency**: Every backward pass through `flash_attention_paged`

**Fix**: Write a C++ `mfa_scatter_grad_kv` primitive that takes the dense dK/dV gradients, block_table, and seq_lens, and produces per-block accumulated gradients in one Metal dispatch. This is the inverse of `mfa_paged_kv_gather`.

**Effort**: HIGH (new C++ primitive)
**Expected savings**: From O(B x num_blocks_per_seq) Python iterations to 1 Metal dispatch

---

## Category G: Backward Path Overhead

### G.1 -- 6-tensor GPU synchronization fence in STEEL backward

**Location**: `mlx_mfa/attention.py:2714`
**Impact**: MEDIUM -- forces a full GPU synchronization barrier on 6 tensors before the backward kernel. This is necessary for correctness (buffer aliasing prevention), but it serializes all prior GPU work.
**Frequency**: Every backward pass through `_make_mfa_custom` for f16/bf16 D<=512

**Current code**:
```python
mx.eval(q, k, v, O, L, dO)
q  = mx.contiguous(q)
k  = mx.contiguous(k)
# ... 4 more mx.contiguous calls ...
dQ, dK, dV = mfa_steel_backward(q, k, v, O, L, dO, scale, causal)
```

**Fix**: Move the synchronization + contiguity check into the C++ `mfa_steel_backward` binding itself. The binding can call `mlx::core::eval()` on its inputs and `contiguous()` internally, avoiding 6 Python-to-C++ round-trips for contiguous calls.

**Effort**: MEDIUM (C++ binding change)
**Expected savings**: ~100-500us per backward pass (GPU sync + Python overhead)

---

### G.2 -- `mx.zeros_like(mask_uint8)` in sparse backward return

**Location**: `mlx_mfa/attention.py:1833, 1855, 1874`
**Impact**: LOW-MEDIUM -- allocates a full-sized zero tensor matching `mask_uint8` shape just to return a dummy gradient.
**Frequency**: Every backward pass through `flash_attention_sparse`

**Fix**: Use `mx.zeros((1,), dtype=mask_uint8.dtype)` as a scalar zero.

**Effort**: TRIVIAL
**Expected savings**: Eliminates one NQ x NK uint8 tensor allocation per sparse backward

---

### G.3 -- Windowed backward rebuilds mask tensors from scratch

**Location**: `mlx_mfa/attention.py:2681-2697`
**Impact**: MEDIUM -- builds the full `[N, S]` window mask with `mx.arange`, `mx.where`, `mx.zeros`, `mx.full` on every backward call, even though window parameters are static.
**Frequency**: Every backward pass with `window_size != None`

**Fix**: Cache the mask inside the closure keyed by `(N, S)` using a small dict:
```python
_mask_cache = {}
def _windowed_sdpa(q, k, v):
    N, S = q.shape[2], k.shape[2]
    if (N, S) not in _mask_cache:
        _mask_cache[(N, S)] = _build_window_mask(N, S)
    mask = _mask_cache[(N, S)]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

**Effort**: EASY
**Expected savings**: ~4-6 eliminated graph nodes per windowed backward call

---

## Category H: Per-Batch Python Serialization

### H.1 -- `_attn_per_seq()` runs B serial `flash_attention` calls

**Location**: `mlx_mfa/attention.py:3707-3718`
**Impact**: HIGH for B > 1 -- each sequence dispatches a separate Metal kernel. For B=8 with paged decode, this is 8 serial flash_attention calls.
**Frequency**: Every paged attention forward when `_USE_PAGED_FLASH_DECODE` is True

**Current code**:
```python
def _attn_per_seq(q_, K_contig, V_contig):
    outputs = []
    for b in range(B):
        kv_len = seq_lens_list[b]
        out_b = flash_attention(q_[b:b+1], K_contig[b:b+1, :, :kv_len, :], ...)
        outputs.append(out_b)
    return mx.concatenate(outputs, axis=0)
```

**Fix**: Use the batched pattern from `_paged_batched_bwd` (already in the same file):
```python
pad_mask = build_kv_padding_mask(seq_lens_list, max_kv_len, q.dtype)
return mx.fast.scaled_dot_product_attention(q, K_contig, V_contig, scale=scale, mask=pad_mask)
```

**Effort**: EASY (pattern already exists in `_paged_batched_bwd`)
**Expected savings**: From B Metal dispatches to 1

---

### H.2 -- Varlen backward runs num_seqs serial `mx.vjp` calls

**Location**: `mlx_mfa/attention.py:3205-3221`
**Impact**: HIGH for many sequences -- each sequence creates a separate backward graph.
**Frequency**: Every backward pass through `flash_attention_varlen`

**Fix**: Implement a STEEL varlen backward kernel in C++/Metal.

**Effort**: HIGH (new C++ primitive + Metal shader)
**Expected savings**: From num_seqs backward dispatches to 1

---

### H.3 -- Paged-mode `_gather_contig()` fallback has O(B x num_blocks) Python loop

**Location**: `mlx_mfa/attention.py:3679-3705`
**Impact**: MEDIUM -- Python fallback gather iterates per-batch per-block. Only used when `_ext_available()` is False or dtype is f32.
**Frequency**: f32 paged attention or missing extension

**Fix**: Use vectorized gather: `k_flat = k_p[block_table_flat].reshape(B, -1, H_kv, D)`.

**Effort**: EASY
**Expected savings**: From O(B x blocks_per_seq) array slices to 1 advanced-index op

---

### H.4 -- `flash_attention_rope_unified` per-batch recursive dispatch

**Location**: `mlx_mfa/attention.py:583-609`
**Impact**: HIGH for multi-sequence serving
**Frequency**: Every call with per-batch `cache_seqlens`

This is the same as **deferred item C.2** from V1 (batched RoPE offsets).

**Fix (Python-side partial)**: Detect uniform cache_seqlens and skip the loop:
```python
if len(set(cs_list)) == 1:
    return flash_attention_rope_unified(q, k, v, ..., cache_seqlens=cs_list[0], ...)
```

**Effort**: EASY (partial fix), HIGH (full fix requires C.2 Metal changes)

---

## Category I: Cache/KV Append Patterns

### I.1 -- `InferenceContext.step()` grows cache via `mx.concatenate`

**Location**: `mlx_mfa/inference.py:232-241`
**Impact**: HIGH -- at step N, concatenate copies all N-1 previous tokens. Over 4096 steps: sum(1..4096) = ~8.4M token copies. For H=8, D=128, f16: ~17 GB cumulative memory traffic.
**Frequency**: Every decode step through `InferenceContext`

**Fix**: Pre-allocate cache buffer to `max_seq_len` and use a write pointer. See V1 item E.1 for details.

**Effort**: HIGH
**Expected savings**: From O(seqlen^2) total to O(seqlen) total memory work

---

### I.2 -- `flash_attention_kvcache` dense-append path copies entire cache

**Location**: `mlx_mfa/attention.py:1538-1540`
**Impact**: HIGH -- same O(seqlen) concatenation as I.1
**Frequency**: Every dense-append call

**Fix**: Provide a `DenseKVCache` helper with pre-allocation, or document that callers should use `PagedKVCache` for long sequences.

**Effort**: MEDIUM

---

### I.3 -- `seq_lens.tolist()` and `block_table.tolist()` force GPU sync

**Location**: `mlx_mfa/attention.py:3600-3601, 1420-1421`
**Impact**: MEDIUM -- `.tolist()` triggers GPU synchronization on the decode hot path.
**Frequency**: Every `flash_attention_paged` and paged-append call

Same as **deferred item E.3**. Becomes unnecessary when H.1, F.2, and F.4 are fixed (C++ primitives take MLX arrays directly).

**Effort**: LOW (once dependencies are fixed)

---

### I.4 -- `cache_seqlens` resolution has complex branching

**Location**: `mlx_mfa/attention.py:1512-1516`
**Impact**: LOW -- 6 lines of type-checking logic. Error-prone, not performance-critical.
**Frequency**: Every paged-append call

**Fix**: Extract to a utility function:
```python
def _resolve_scalar_seqlens(cache_seqlens) -> int:
    if isinstance(cache_seqlens, int):
        return cache_seqlens
    if isinstance(cache_seqlens, mx.array):
        return int(cache_seqlens.item())
    return int(next(iter(cache_seqlens)))
```

**Effort**: TRIVIAL

---

## Category J: Code-Level Dead Weight

### J.1 -- Duplicate paged-mode branch in `flash_attention_rope_unified`

**Location**: `mlx_mfa/attention.py:618-630`
**Impact**: NONE (runtime) -- both `if` and `else` branches execute identical code
**Frequency**: N/A -- code clarity issue

**Current code** (both branches identical):
```python
if not _can_use_mfa(q, head_dim) or q.dtype == mx.float32 or _partial:
    q_rot, _ = _apply_rope_to_qk(q, k, rotary_cos, rotary_sin,
        q_offset=cs, k_offset=cs, interleaved=interleaved, rotary_dim=rotary_dim)
else:
    q_rot, _ = _apply_rope_to_qk(q, k, rotary_cos, rotary_sin,
        q_offset=cs, k_offset=cs, interleaved=interleaved, rotary_dim=rotary_dim)
```

**Fix**: Remove the conditional; keep one call.

**Effort**: TRIVIAL

---

### J.2 -- `_mfa_rope_forward` has Python-side `mx.contiguous` calls despite D.5 fix

**Location**: `mlx_mfa/attention.py:3056-3058`
**Impact**: LOW -- three redundant `mx.contiguous()` calls; the D.5 fix already handles this in C++.
**Frequency**: Every in-kernel RoPE forward

**Fix**: Remove the three `mx.contiguous()` calls.

**Effort**: TRIVIAL
**Expected savings**: ~300ns per call

---

### J.3 -- `_make_mfa_sparse_custom` imports numpy inside closure

**Location**: `mlx_mfa/attention.py:1791`
**Impact**: LOW -- `import numpy` at top of factory function runs on every `flash_attention_sparse` call. Only the deprecated `sdpa_sparse` backward actually uses numpy.
**Frequency**: Every sparse forward call

**Fix**: Move the import into the `sdpa_sparse` branch (line 1851) where it is used.

**Effort**: TRIVIAL

---

## Remediation Priority Matrix

| Priority | Fix ID | Description | Impact | Effort | Phase |
|----------|--------|-------------|--------|--------|-------|
| **P0** | J.1 | Remove duplicate paged-mode RoPE branch | None (clarity) | Trivial | 1 |
| **P0** | J.2 | Remove redundant mx.contiguous in _mfa_rope_forward | Low | Trivial | 1 |
| **P0** | J.3 | Move numpy import out of sparse closure | Low | Trivial | 1 |
| **P0** | G.2 | Replace zeros_like(mask) with scalar zero | Low-Med | Trivial | 1 |
| **P0** | I.4 | Extract cache_seqlens resolver to utility | Low | Trivial | 1 |
| **P0** | F.3 | Vectorize PagedKVCache scatter target loop | Medium | Easy | 1 |
| **P1** | H.1 | Batched paged forward (eliminate _attn_per_seq loop) | High | Easy | 2 |
| **P1** | H.4 | Skip per-batch RoPE loop when offsets are uniform | High | Easy | 2 |
| **P1** | H.3 | Vectorize Python fallback gather with advanced indexing | Medium | Easy | 2 |
| **P1** | G.3 | Cache windowed backward mask in closure dict | Medium | Easy | 2 |
| **P2** | G.1 | Move fence into C++ backward binding | Medium | Medium | 3 |
| **P2** | F.2 | Vectorize paged-append scatter target computation | Medium | Medium | 3 |
| **P2** | I.2 | Provide DenseKVCache helper with pre-allocation | High | Medium | 3 |
| **P3** | F.1 | Block-scoped scatter (avoid full-pool copy) | HIGH | Medium | 4 |
| **P3** | F.4 | Fused scatter-grad C++ primitive for paged backward | High | High | 4 |
| **P3** | I.1 | Pre-allocated InferenceContext cache | High | High | 4 |
| **P3** | H.2 | STEEL varlen backward kernel | High | High | 4 |
| **P3** | C.2 | Batched RoPE offsets in Metal shader | High | High | 4 |

---

## Implementation Phases

### Phase 1: Quick Wins (all trivial/easy, <2 hours total)

**Goal**: Clean up dead code, eliminate unnecessary allocations.

| Fix | Lines changed | Test impact |
|-----|---------------|-------------|
| J.1 Remove duplicate RoPE branch | ~10 lines removed in attention.py | Run RoPE tests |
| J.2 Remove redundant mx.contiguous | 3 lines removed in attention.py | Run RoPE tests |
| J.3 Move numpy import | 1 line moved in attention.py | Run sparse tests |
| G.2 Scalar zero for mask gradient | 3 lines in attention.py | Run sparse backward |
| I.4 Extract cache_seqlens utility | ~10 lines in attention.py | Run kvcache tests |
| F.3 Vectorize scatter target loop | ~8 lines in attention.py | Run paged tests |

**Expected total savings**: ~500ns per call + code clarity improvement

### Phase 2: Easy-Medium Fixes (~1 day)

**Goal**: Eliminate the worst per-batch serialization patterns.

| Fix | Description |
|-----|-------------|
| H.1 | Batched paged forward via padding mask (pattern from _paged_batched_bwd) |
| H.4 | Detect uniform cache_seqlens, skip per-batch loop |
| H.3 | Advanced-indexing gather for Python fallback path |
| G.3 | Per-(N,S) cached window mask in backward closure |

**Expected savings**: Up to Bx reduction in paged forward dispatches; cleaner backward

### Phase 3: Structural Improvements (~2-3 days)

**Goal**: Move Python overhead into C++; provide pre-allocated cache alternatives.

| Fix | Description |
|-----|-------------|
| G.1 | Encapsulate fence + contiguous in C++ mfa_steel_backward binding |
| F.2 | Vectorize paged-append scatter target computation |
| I.2 | DenseKVCache helper class with write-pointer pattern |

**Expected savings**: ~100-500us per backward; O(seqlen) cache writes for new helper

### Phase 4: New C++ Primitives (~1-2 weeks)

**Goal**: Fuse scatter patterns, implement missing backward kernels.

| Fix | Description | New files |
|-----|-------------|-----------|
| F.1 | Block-scoped scatter (skip unchanged blocks) | `csrc/mfa_scatter.cpp` modification |
| F.4 | `mfa_scatter_grad_kv` (inverse of paged gather) | `csrc/mfa_scatter_grad.hpp/.cpp` |
| I.1 | Pre-allocated InferenceContext (write-pointer) | `inference.py` rewrite |
| H.2 | STEEL varlen backward kernel | `csrc/mfa_steel_bwd_varlen.hpp/.cpp` |
| C.2 | Batched RoPE offset in Metal shader | `csrc/mfa_attention.cpp` + shader |

**Expected savings**: F.1: ~4000x fewer elements per scatter; I.1: O(seqlen) total cache work

---

## Estimated Aggregate Impact

### Decode Hot Path (per-token, 32-layer LLM, paged mode, B=1)

| Component | Current | After Phase 1+2 | After Phase 3+4 |
|-----------|---------|------------------|------------------|
| Scatter KV pool copy (F.1) | ~80us | ~80us | ~0.1us |
| Python scatter target loop (F.2/F.3) | ~15us | ~5us | ~0us |
| tolist GPU sync (I.3) | ~20us | ~20us | ~0us |
| Per-batch attention loop (H.1) | 0 (B=1) | 0 | 0 |
| **Total per-token paged overhead** | **~115us** | **~105us** | **~0.1us** |

### Decode Hot Path (per-token, B=8 batched paged)

| Component | Current | After Phase 1+2 | After Phase 3+4 |
|-----------|---------|------------------|------------------|
| Per-batch attention loop (H.1) | ~8x flash_attention | ~1x | ~1x |
| Per-batch scatter (F.2) | ~8x Python loops | ~1x vectorized | ~0 (C++) |
| tolist GPU sync (I.3) | ~40us | ~40us | ~0us |
| **Total per-token overhead** | **~960us** | **~180us** | **~20us** |

### Backward Pass (D=128 N=4096, standard dense)

| Component | Current | After G.1 |
|-----------|---------|-----------|
| GPU fence + contiguous (G.1) | ~300us | ~50us (in C++) |
| STEEL backward kernels | ~16ms | ~16ms |
| **Total backward** | **~16.3ms** | **~16.05ms** |

### InferenceContext step() (N=4096, H=8, D=128, f16)

| Component | Current | After I.1 |
|-----------|---------|-----------|
| Concat per step | ~32MB copy at step 4096 | ~2KB write |
| Total over 4096 steps | ~67 GB cumulative | ~8 MB cumulative |

---

## Testing Strategy

Each phase must pass all 486 existing tests before proceeding:
```bash
.venv/bin/python -m pytest tests/ -q       # all 486 pass
.venv/bin/python benchmarks/bench_all.py   # no regressions
```

For Phase 4 (new C++ primitives), add:
- `TestScatterBlockScoped` -- correctness vs full-pool scatter
- `TestScatterGradKV` -- gradient correctness vs Python `_scatter_to_pool`
- `TestVarlenBackward` -- per-sequence gradient vs split-concat reference
- `TestDenseKVCache` -- cache equivalence with concat approach

---

## Files Requiring Changes

| Phase | File | Type of change |
|-------|------|----------------|
| 1 | `mlx_mfa/attention.py` | Remove dead branches, scalar zero, extract utility |
| 2 | `mlx_mfa/attention.py` | Batched paged forward, uniform-seqlens shortcut, vectorize gather |
| 3 | `mlx_mfa/attention.py` | Vectorize scatter targets |
| 3 | `csrc/bindings.cpp` | Encapsulate fence + contiguous in backward binding |
| 3 | `mlx_mfa/inference.py` | DenseKVCache helper (new class, same file) |
| 4 | `csrc/mfa_scatter.cpp` | Block-scoped scatter optimization |
| 4 | `csrc/mfa_scatter_grad.hpp/.cpp` | NEW: inverse-gather for paged backward |
| 4 | `csrc/mfa_steel_bwd_varlen.hpp/.cpp` | NEW: STEEL varlen backward kernel |
| 4 | `csrc/mfa_attention.cpp` | Batched RoPE offset support (C.2) |

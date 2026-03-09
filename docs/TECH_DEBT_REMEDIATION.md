# mlx-mfa Tech Debt Remediation Plan

**Date**: 2026-03-09
**Version**: v1.2.1 (486 tests pass)
**Scope**: Every Python-side friction point that slows GPU kernel execution
**Constraint**: No precision regressions; no API signature changes

---

## Executive Summary

Four-agent deep analysis of the mlx-mfa-v2 codebase identified **25 friction points** where Python code adds overhead to GPU kernel dispatch. The findings cluster into five categories:

| Category | Count | Worst-case impact |
|----------|-------|-------------------|
| **A. Fused kernel gaps** (Python MLX ops that should be C++/Metal) | 3 | SageAttention 0.52x slower than flash_attention |
| **B. Unnecessary recomputation** (forward re-run, double masks) | 3 | Backward 2x slower than needed |
| **C. O(N) Python loops** on the hot path | 5 | O(num_blocks) pool rebuild per token |
| **D. Per-call dispatch overhead** (import probes, validation, no-ops) | 9 | ~2.4us wasted per flash_attention call |
| **E. Architectural bottlenecks** (concat cache, pool rebuild) | 5 | O(seqlen^2) total memory work |

**Top-3 highest-ROI fixes:**
1. Fuse `quantize_per_block` + `smooth_k` into a single C++ primitive (eliminates 30+ graph nodes to 1)
2. Save logsumexp L from forward pass to eliminate backward recomputation (halves backward cost)
3. Cache `_ext_available()` result (eliminates ~1us per call x 32 layers x 100 tokens/s)

---

## Category A: Fused Kernel Gaps

These are Python-side MLX op chains that should be single C++/Metal kernels.

### A.1 — `quantize_per_block()`: 12+ MLX kernel launches per call

**Location**: `mlx_mfa/quantize.py:70-92`
**Impact**: CRITICAL — root cause of SageAttention being 0.52-0.89x slower than flash_attention
**Frequency**: Called 2x per `sage_attention()` invocation (once for Q, once for K)

**Current flow** (each line = separate Metal kernel dispatch):
```
mx.pad -> reshape -> astype(f32) -> mx.abs -> mx.max -> div -> mx.maximum ->
astype(f32) [again] -> div -> mx.round -> mx.clip -> astype(int8) -> reshape -> slice
```

**Problem**: 12-16 Metal kernel dispatches with ~5-20us overhead each = 60-320us of dispatch overhead alone. Plus each intermediate creates a full-sized temporary tensor.

**Fix**: Write a fused `MFAQuantizePerBlock` C++ primitive in `csrc/`:
- Single Metal kernel: reads fp16/bf16 input, computes per-block absmax, scales, rounds, clips, outputs int8 + f32 scale
- Register the primitive in `bindings.cpp`
- Python side: `from mlx_mfa._ext import mfa_quantize_per_block`
- Expected reduction: 12 kernel launches to 1

**Effort**: HIGH (new C++ primitive + Metal shader + tests)
**Expected speedup**: SageAttention from 0.52x to potentially >1.0x vs flash_attention

---

### A.2 — `smooth_k()`: 5 GPU passes over full KV tensor

**Location**: `mlx_mfa/quantize.py:165-166`
**Impact**: HIGH — adds ~4 unnecessary memory passes on [B,H,S,D]
**Frequency**: Once per `sage_attention()` call when `apply_smooth_k=True`

**Current flow**:
```python
k_mean = mx.mean(k.astype(mx.float32), axis=2, keepdims=True)  # cast + reduce
k_smooth = (k.astype(mx.float32) - k_mean).astype(k.dtype)     # cast + sub + cast
```
5 passes: read-write(f32), read-write(mean), read+read-write(sub), read-write(f16).

**Fix**: Fuse into the `MFAQuantizePerBlock` primitive from A.1:
- Accept an `apply_smooth=True` flag
- In the same Metal kernel: compute per-channel mean, subtract, then quantize
- Single pass: read fp16 -> compute mean -> subtract -> quantize -> write int8 + scale

**Effort**: MEDIUM (extend A.1 kernel)
**Expected savings**: 4 eliminated GPU passes on the full KV tensor

---

### A.3 — `quantize_per_block` double float32 cast

**Location**: `mlx_mfa/quantize.py:76` and `:82`
**Impact**: LOW-MEDIUM — same `x_blocked.astype(mx.float32)` computed twice

**Quick fix** (independent of A.1):
```python
x_f32 = x_blocked.astype(mx.float32)  # compute once
absmax = mx.max(mx.abs(x_f32), axis=(3, 4), keepdims=True)
...
x_quant = mx.clip(mx.round(x_f32 / scale), -128, 127).astype(mx.int8)
```

**Effort**: TRIVIAL (1-line change)
**Expected savings**: 1 eliminated full-tensor cast (MLX may or may not CSE this)

---

## Category B: Unnecessary Recomputation

### B.1 — Backward re-runs full forward (gradient checkpointing)

**Location**: `mlx_mfa/attention.py:2657-2666` (`_backward()` in `_make_mfa_custom`)
**Impact**: HIGH — backward cost is >=2x what it needs to be
**Frequency**: Every backward pass through flash_attention

**Current flow**:
```python
def _backward(res, cotangent):
    ...
    O_re, L = mfa_forward_with_lse(q, k, v, scale, causal)  # RE-RUNS FORWARD
    ...
    dQ, dK, dV = mfa_steel_backward(q, k, v, O_re, L, dO, ...)
```

The logsumexp L was already computed during the forward pass (line 2615: `O, _ = mfa_forward_with_lse(...)`) but discarded. The backward then re-runs the entire forward just to recover L.

**Fix**: Save L from the forward pass by having `_impl()` return `(O, L)` as a tuple:
```python
def _impl(q, k, v):
    O, L = mfa_forward_with_lse(q, k, v, scale, causal)
    return O, L  # save both

def _backward(res, cotangent):
    (O, L) = res  # L is already available, no recomputation needed
    dO = cotangent[0]
    dQ, dK, dV = mfa_steel_backward(q, k, v, O, L, dO, ...)
```

The public API still returns only O via `result = impl(q, k, v)[0]`.

**Effort**: MEDIUM (restructure mx.custom_function return, update vjp signature)
**Expected savings**: ~50% backward time reduction (eliminates 1x forward pass)

---

### B.2 — `_fallback_sdpa_with_lse` builds causal mask twice

**Location**: `mlx_mfa/attention.py:2751-2776`
**Impact**: MEDIUM — two [N,S] tensor allocations + triu operations
**Frequency**: Per-call on `return_lse=True` without extension

**Fix**: Compute the mask once, reuse for both the manual logit computation and the SDPA call.

**Effort**: TRIVIAL

---

### B.3 — `_sever_lazy_graph()` injects gratuitous elementwise add

**Location**: `mlx_mfa/attention.py:2494-2528`
**Impact**: LOW-MEDIUM — one unnecessary elementwise kernel on [B,H,N,D]
**Frequency**: Every backward pass for f16/bf16 D<=512

**Current**: `arr + mx.zeros_like(arr)` — creates a new array to break buffer aliasing.

**Fix**: Replace with `mx.contiguous(arr)` (which the docstring at line 2521 already acknowledges works). Or, if B.1 is implemented (saving L from forward), this workaround may become unnecessary entirely.

**Effort**: TRIVIAL

---

## Category C: O(N) Python Loops on Hot Paths

### C.1 — Paged-append pool rebuild: O(num_blocks x block_size) Python ops

**Location**: `mlx_mfa/attention.py:1428-1465`
**Impact**: HIGH — O(num_blocks) iterations creating MLX graph nodes per decode token
**Frequency**: Every decode step in paged-append mode

**Current**: A `for b in range(B): for t in range(N_new):` loop that indexes individual tokens into a dict, then a `for i in range(num_blks):` loop that reconstructs the pool via `mx.concatenate` per block.

For pool with 256 blocks x 16 block_size: 4096 array slices + 256 concatenates + 1 stack.

**Fix options** (ordered by effort):
1. **In-place scatter C++ primitive**: Write an `mfa_scatter_kv` binding that writes tokens directly into the pool buffer at target offsets. O(N_new) instead of O(pool_size).
2. **mx.scatter** (if MLX supports it): Use `pool.at[block_idx, :, offset, :].set(token)` pattern.
3. **Batch the updates**: Collect all (block_idx, offset) pairs in Python, then do a single vectorized scatter.

**Effort**: MEDIUM-HIGH
**Expected savings**: From O(num_blocks) to O(N_new) per step

---

### C.2 — Per-batch Python loop for RoPE offsets

**Location**: `mlx_mfa/attention.py:563-600`
**Impact**: HIGH for multi-sequence decode (the dominant LLM serving case)
**Frequency**: Per-call when `cache_seqlens` is a per-batch array

**Current**: `.tolist()` forces GPU-to-CPU sync, then `for b, cs in enumerate(cs_list):` calls `flash_attention_rope_unified()` recursively B times.

**Fix**: Implement batched RoPE offset support in the C++ `mfa_attention_rope_forward` binding:
- Pass `cache_seqlens` as an int32 array buffer argument
- Metal shader indexes per-batch offset via `batch_idx`
- Eliminate the Python loop entirely

**Effort**: HIGH (C++ + Metal shader changes)
**Expected savings**: From O(B) dispatch calls to 1

---

### C.3 — `_sparse_backward_tiled()`: O(NQ x NK) Python loop

**Location**: `mlx_mfa/attention.py:2152-2268`
**Impact**: HIGH for large sequences with sparse patterns
**Frequency**: Per backward pass with `backward='sdpa_sparse'`

For N=4096 (NQ=128, NK=256): up to 32,768 loop iterations, each creating ~8 graph nodes.

**Fix**: Deprecate this path in favor of `backward='steel_sparse'` which uses the native `mfa_steel_backward_sparse` Metal kernel. Fix C.4 first (the numpy round-trip blocking steel_sparse).

**Effort**: LOW (deprecation) once C.4 is fixed

---

### C.4 — Sparse backward steel: 7-tensor numpy round-trip

**Location**: `mlx_mfa/attention.py:1793-1803`
**Impact**: VERY HIGH — ~112MB of GPU-to-CPU-to-GPU transfer for D=128 N=4096
**Frequency**: Per backward on `backward='steel_sparse'` path

**Current**: Forces `mx.synchronize()` on all tensors, then `_to_fresh(a)` does `mx.array(np.array(a.astype(f32)))` for each of q, k, v, O, L, dO, mask.

**Fix**: Replace numpy round-trip with `mx.contiguous()`:
```python
mx.synchronize()
q2 = mx.contiguous(q); k2 = mx.contiguous(k)  # breaks aliasing without CPU trip
```
If `mx.contiguous` alone doesn't break aliasing, use `arr * 1.0` or `arr + mx.zeros(1)` (a scalar zero, not `zeros_like` which allocates a full tensor).

**Effort**: LOW (replace 7 lines)
**Expected savings**: Eliminates ~10-50ms of CPU-to-GPU transfer per backward

---

### C.5 — `speculative_verify`: Python double-loop for log-prob extraction

**Location**: `mlx_mfa/attention.py:1917-1919`
**Impact**: MEDIUM-HIGH — O(B x N_draft) Python scalar extractions with `float(log_probs[b, t, int(ids[b,t])])`
**Frequency**: Per speculative decoding verify step

**Fix**: Replace with vectorized MLX indexing:
```python
target_logprobs = mx.take_along_axis(log_probs, ids[..., None], axis=-1).squeeze(-1)
```
Also remove the premature `mx.synchronize(out, lse)` on line 1905.

**Effort**: EASY

---

## Category D: Per-Call Dispatch Overhead

These individually add 0.1-1.5us per `flash_attention()` call. Compounded across 32 LLM layers at 100 tokens/sec = significant aggregate waste.

### D.1 — `_ext_available()` uncached import probe

**Location**: `mlx_mfa/attention.py:2485-2491`
**Impact**: MEDIUM (~1us per call)
**Frequency**: Every `flash_attention()`, `flash_attention_sparse()`, `flash_attention_varlen()`, `flash_attention_paged()` invocation

**Fix**:
```python
_ext_avail_cached: Optional[bool] = None

def _ext_available() -> bool:
    global _ext_avail_cached
    if _ext_avail_cached is not None:
        return _ext_avail_cached
    try:
        from mlx_mfa._ext import mfa_attention_forward
        _ext_avail_cached = True
    except ImportError:
        _ext_avail_cached = False
    return _ext_avail_cached
```

**Effort**: TRIVIAL (5 lines)

---

### D.2 — `sage_attention()` uncached try/except import

**Location**: `mlx_mfa/attention.py:981-985`
**Impact**: MEDIUM (~1us per call)
**Fix**: Same pattern as D.1. Cache after first probe.
**Effort**: TRIVIAL

---

### D.3 — `_VALID_BACKENDS` set literal created per call

**Location**: `mlx_mfa/attention.py:163`
**Impact**: LOW (~200ns per call — Python rebuilds the set object on each function entry)
**Fix**: Move to module scope: `_VALID_BACKENDS = frozenset({"auto", "mfa", "sdpa"})`
**Effort**: TRIVIAL

---

### D.4 — `_steel_sdpa()` double validation through `flash_attention()`

**Location**: `mlx_mfa/integrations/mlx_lm.py:191-192`
**Impact**: MEDIUM-HIGH — ~1.5us redundant validation per layer per token
**Frequency**: Every LLM decode step (32x per token for 32-layer model = ~48us/token)

**Current**: `_steel_sdpa()` validates head_dim, dtype, extension availability, then calls `flash_attention()` which re-does ALL the same checks.

**Fix**: Have `_steel_sdpa()` call `_mfa_forward()` directly (bypassing `flash_attention()` validation):
```python
from mlx_mfa.attention import _mfa_forward
return _mfa_forward(queries, keys, values, scale, causal=True, ...)
```

**Effort**: EASY
**Expected savings**: ~50% reduction in per-call Python overhead for mlx-lm integration

---

### D.5 — Three `mx.contiguous()` no-op calls per MFA dispatch

**Location**: `mlx_mfa/attention.py:2706-2708` (also 2573-2575, 3007-3009)
**Impact**: LOW-MEDIUM (~300-600ns per call — 3 Python-to-C++ round-trips)
**Frequency**: Every MFA forward dispatch

**Fix**: Move contiguity check into C++ `mfa_attention_forward` binding:
```cpp
auto q = ensure_row_contiguous(inputs[0]);  // in C++, zero overhead if contiguous
```

**Effort**: MEDIUM (C++ changes to all binding entry points)

---

### D.6 — `_make_mfa_sparse_custom()` not cached

**Location**: `mlx_mfa/attention.py:1727+`
**Impact**: MEDIUM-HIGH — `mx.custom_function` closure rebuilt on every `flash_attention_sparse()` call
**Frequency**: Every sparse attention call

**Fix**: Cache with `lru_cache` using hashable parameters only (scale, causal, head_dim). The `block_mask` array changes per call, so pass it as an argument to the cached callable, not as a closure capture:
```python
@functools.lru_cache(maxsize=32)
def _make_mfa_sparse_custom(scale, causal, head_dim, backward):
    def _impl(q, k, v, mask):  # mask as argument, not closure
        ...
    return _impl
```

**Effort**: MEDIUM

---

### D.7 — `scale` recomputed per call

**Location**: `mlx_mfa/attention.py:206-207` and ~10 other functions
**Impact**: LOW (~50ns per call)
**Fix**: Not worth caching independently. If D.4 is implemented (direct _mfa_forward), the mlx-lm path won't recompute.

---

### D.8 — Stats dict mutations in `_steel_sdpa()`

**Location**: `mlx_mfa/integrations/mlx_lm.py:127-132`
**Impact**: LOW (~100-200ns per call, 32x per token)
**Fix**: Use a list with integer indices instead of dict string keys, or make stats tracking optional via a flag.
**Effort**: TRIVIAL

---

### D.9 — `hasattr()` checks on cache object in `_steel_sdpa()`

**Location**: `mlx_mfa/integrations/mlx_lm.py:138, 172`
**Impact**: LOW (~200ns per call)
**Fix**: Use `getattr(cache, "bits", None)` (faster, no exception frame) or determine cache capabilities once at `patch_mlx_lm()` time.
**Effort**: EASY

---

## Category E: Architectural Bottlenecks

### E.1 — `InferenceContext.step()` grows cache by concatenation

**Location**: `mlx_mfa/inference.py:233-234`
**Impact**: HIGH — O(seqlen) memory copy per decode step leads to O(seqlen^2) total
**Frequency**: Every decode step

**Current**:
```python
self._k_cache = mx.concatenate([self._k_cache, k_new], axis=2)
```
At step 4096 with H=32 D=128 f16: copies 32MB per token.

**Fix**: Pre-allocate cache to `max_seq_len` and use a write pointer:
```python
def __init__(self, ..., max_seq_len=8192):
    self._k_buf = mx.zeros([B, H_kv, max_seq_len, D], dtype=dtype)
    self._write_pos = 0
```

**Challenge**: MLX is functional (no in-place mutation). Options:
1. Use `mx.scatter` or `.at[].set()` (creates new array but smaller graph)
2. Use a C++ in-place write primitive
3. Use numpy backing store (like `PagedKVCache` already does)

**Effort**: HIGH

---

### E.2 — `PagedKVCache.append()` forces full pool rebuild + `mx.synchronize`

**Location**: `mlx_mfa/attention.py:3355-3366`
**Impact**: HIGH — GPU sync + O(pool_size) copy per token
**Frequency**: Every decode step in paged mode

**Current**: Slices pool into 3 parts (prefix, new block, suffix), concatenates, then `mx.synchronize()`.

**Fix**: Same approach as C.1 — implement a C++ scatter primitive for block-table-indexed writes.

**Effort**: HIGH (same C++ work as C.1)

---

### E.3 — `seq_lens.tolist()` + `block_table.tolist()` GPU-to-CPU sync

**Location**: `mlx_mfa/attention.py:1406-1407, 3528-3529`
**Impact**: MEDIUM — forces GPU synchronization on the decode hot path
**Frequency**: Every paged attention call

**Fix**: Keep `seq_lens` and `block_table` as MLX arrays and pass them to C++ bindings directly. The Metal shader can index them via buffer arguments.

**Effort**: MEDIUM-HIGH (requires C++ binding changes)

---

### E.4 — Per-sequence Python loops in paged/varlen backward

**Location**: `mlx_mfa/attention.py:3694-3706, 3724-3734, 3156-3172`
**Impact**: HIGH for B > 1 — serializes what should be parallel batch processing
**Frequency**: Every backward pass with batched paged/varlen attention

**Fix**: Batch the gathered K/V into a single `[B, H, max_kv, D]` tensor with padding and run one batched `mx.vjp(flash_attention(...))`.

**Effort**: MEDIUM

---

### E.5 — Identity transpose no-op

**Location**: `mlx_mfa/attention.py:2078`
**Impact**: NEGLIGIBLE — `mx.transpose(float_block, (0,1,2,3))` is an identity permutation
**Fix**: Remove the line.
**Effort**: TRIVIAL

---

## Remediation Priority Matrix

| Priority | Fix ID | Description | Impact | Effort | Phase |
|----------|--------|-------------|--------|--------|-------|
| **P0** | D.1 | Cache `_ext_available()` result | Medium | Trivial | 1 |
| **P0** | D.3 | Move `_VALID_BACKENDS` to module scope | Low | Trivial | 1 |
| **P0** | A.3 | Deduplicate float32 cast in quantize | Low-Med | Trivial | 1 |
| **P0** | B.3 | Replace `_sever_lazy_graph` with `mx.contiguous` | Low-Med | Trivial | 1 |
| **P0** | E.5 | Remove identity transpose | Negligible | Trivial | 1 |
| **P0** | D.2 | Cache sage_attention import probe | Medium | Trivial | 1 |
| **P1** | D.4 | `_steel_sdpa` direct dispatch (bypass validation) | Med-High | Easy | 2 |
| **P1** | C.5 | Vectorize speculative_verify log-prob extraction | Med-High | Easy | 2 |
| **P1** | C.4 | Replace numpy round-trip in sparse backward | Very High | Low | 2 |
| **P1** | C.3 | Deprecate `_sparse_backward_tiled` Python loops | High | Low | 2 |
| **P1** | B.2 | Deduplicate causal mask in `_fallback_sdpa_with_lse` | Medium | Trivial | 2 |
| **P1** | D.6 | Cache `_make_mfa_sparse_custom` | Med-High | Medium | 2 |
| **P1** | D.8 | Optimize stats tracking | Low | Trivial | 2 |
| **P1** | D.9 | Replace `hasattr` with `getattr` | Low | Easy | 2 |
| **P2** | B.1 | Save L from forward; eliminate backward recompute | HIGH | Medium | 3 |
| **P2** | D.5 | Move contiguity check to C++ bindings | Low-Med | Medium | 3 |
| **P2** | E.4 | Batch paged/varlen backward | High | Medium | 3 |
| **P3** | A.1+A.2 | Fused quantize C++ primitive (+ smooth_k) | CRITICAL | High | 4 |
| **P3** | C.1+E.2 | In-place scatter for paged pool writes | High | Med-High | 4 |
| **P3** | E.1 | Pre-allocated cache with write pointer | High | High | 4 |
| **P3** | C.2 | Batched RoPE offset in C++/Metal | High | High | 4 |
| **P3** | E.3 | Pass seq_lens/block_table as MLX arrays to C++ | Medium | Med-High | 4 |

---

## Implementation Phases

### Phase 1: Quick Wins (all trivial, <1 hour total)

**Goal**: Eliminate per-call overhead on the hot path. No C++ changes.

| Fix | Lines changed | Test impact |
|-----|---------------|-------------|
| D.1 Cache `_ext_available()` | ~8 lines in attention.py | None |
| D.2 Cache sage import | ~6 lines in attention.py | None |
| D.3 Module-scope `_VALID_BACKENDS` | 1 line move | None |
| A.3 Deduplicate f32 cast | 3 lines in quantize.py | Run sage tests |
| B.3 Replace `_sever_lazy_graph` body | 1 line in attention.py | Run backward tests |
| E.5 Remove identity transpose | 1 line in attention.py | Run sparse tests |

**Expected total savings**: ~2-3us per `flash_attention()` call

### Phase 2: Easy-Medium Fixes (~1 day)

**Goal**: Fix the worst Python-loop bottlenecks and dispatch redundancies.

| Fix | Description |
|-----|-------------|
| D.4 | `_steel_sdpa` calls `_mfa_forward` directly |
| C.4 | Replace numpy round-trip with mx.contiguous in sparse backward |
| C.3 | Deprecate `_sparse_backward_tiled` in favor of steel_sparse |
| C.5 | Vectorize speculative_verify with `mx.take_along_axis` |
| B.2 | Deduplicate causal mask build |
| D.6 | Cache `_make_mfa_sparse_custom` |

**Expected savings**: ~48us/token for mlx-lm; ~10-50ms per sparse backward

### Phase 3: Structural Improvements (~2-3 days)

**Goal**: Halve backward cost by saving logsumexp; batch paged/varlen ops.

| Fix | Description |
|-----|-------------|
| B.1 | Restructure `_make_mfa_custom` to return (O, L), eliminate backward recompute |
| D.5 | Move `mx.contiguous` into C++ binding entry points |
| E.4 | Batch paged/varlen backward loops over B dimension |

**Expected savings**: ~50% backward time reduction; eliminated per-batch serialization

### Phase 4: New C++ Primitives (~1-2 weeks)

**Goal**: Fuse Python op chains into single Metal kernels; implement in-place cache writes.

| Fix | Description | New files |
|-----|-------------|-----------|
| A.1+A.2 | `MFAQuantizePerBlock` C++ primitive + Metal shader | `csrc/mfa_quantize.hpp/.cpp`, Metal JIT |
| C.1+E.2 | `mfa_scatter_kv` C++ primitive for paged writes | `csrc/mfa_scatter.hpp/.cpp` |
| E.1 | Pre-allocated InferenceContext cache | `inference.py` rewrite |
| C.2 | Batched RoPE offsets in Metal shader | `csrc/mfa_attention.cpp` + shader changes |

**Expected savings**: SageAttention from 0.52x to >1.0x vs flash_attention; O(1) paged writes

---

## Estimated Aggregate Impact

### Decode Hot Path (per-token, 32-layer LLM)

| Component | Current overhead | After Phase 1+2 | After Phase 3+4 |
|-----------|-----------------|------------------|------------------|
| Python validation (x32) | ~77us | ~26us | ~16us |
| mx.contiguous no-ops (x32) | ~19us | ~19us | ~0us |
| Import probes (x32) | ~32us | ~0us | ~0us |
| Stats/hasattr (x32) | ~10us | ~6us | ~6us |
| **Total per-token Python overhead** | **~138us** | **~51us** | **~22us** |

### SageAttention (per-call, N=4096)

| Component | Current | After A.1+A.2 |
|-----------|---------|---------------|
| quantize Q + K (Python ops) | ~11ms | ~0.5ms (fused kernel) |
| smooth_k (Python ops) | ~3ms | ~0ms (fused into quantize) |
| sage_forward Metal kernel | ~9ms | ~9ms (unchanged) |
| **Total** | **~23ms** | **~9.5ms** |
| **vs flash_attention 12ms** | **0.52x** | **~1.26x** |

### Backward Pass (D=128 N=4096)

| Component | Current | After B.1 |
|-----------|---------|-----------|
| Forward recompute (L recovery) | ~12ms | ~0ms (saved from fwd) |
| STEEL dQ kernel | ~8ms | ~8ms |
| STEEL dKV kernel | ~8ms | ~8ms |
| _sever_lazy_graph overhead | ~1ms | ~0ms |
| **Total backward** | **~29ms** | **~16ms** |

---

## Testing Strategy

Each phase must pass all 486 existing tests before proceeding:
```bash
.venv/bin/python -m pytest tests/ -q       # all 486 pass
.venv/bin/python benchmarks/bench_all.py   # no regressions
```

For Phase 4 (new C++ primitives), add:
- `TestQuantizeFused` — correctness vs Python `quantize_per_block`
- `TestScatterKV` — correctness vs Python pool rebuild
- `TestInferenceContextPrealloc` — cache equivalence with old concat approach

---

## Files Requiring Changes

| Phase | File | Type of change |
|-------|------|----------------|
| 1 | `mlx_mfa/attention.py` | Cache `_ext_available`, move constants, fix `_sever_lazy_graph` |
| 1 | `mlx_mfa/quantize.py` | Deduplicate float32 cast |
| 2 | `mlx_mfa/attention.py` | Vectorize speculative_verify, fix sparse backward, cache sparse custom |
| 2 | `mlx_mfa/integrations/mlx_lm.py` | Direct dispatch bypass |
| 3 | `mlx_mfa/attention.py` | Restructure custom_function for (O,L) return |
| 3 | `csrc/bindings.cpp` | Add contiguity check to binding entry points |
| 4 | `csrc/mfa_quantize.hpp/.cpp` | NEW: fused quantize primitive |
| 4 | `csrc/mfa_scatter.hpp/.cpp` | NEW: in-place KV scatter primitive |
| 4 | `mlx_mfa/inference.py` | Pre-allocated cache rewrite |
| 4 | `csrc/mfa_attention.cpp` | Batched RoPE offset support |

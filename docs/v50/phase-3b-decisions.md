# v2.50 Phase 3b — Top-K native Metal kernel — DECISIONS

**Sprint date**: 2026-05-14 (Prompt 3)
**Branch**: `feat/v50-phase3b-topk-native`
**Master tip pre-Phase-3b**: `23edb40` (post-Prompt 2 Section D merge)

## §AA.5 Premise validation

**Audit prescription** (from `02-consolidated-bench-results.md` G5 +
`03-sprint-sequence.md` Sprint 3):
> L effort (~3-6h) — new top-K-fused source generator (sparse-attention
> variant with score-based block selection) + Primitive + binding + tests
> + routing in `flash_attention_topk`.

### Available Apple primitives checked

Re-inventory since Prompt 2 §AA.5 check (Sprint 3 Phase 3a):

| Primitive | Signature | Applicability to top-K attention |
|---|---|---|
| `mx.topk(a, k, axis)` | returns top-k values, unsorted within partition | Used in Prompt 2 dispatch fix |
| `mx.partition(a, kth, axis)` | partition-based selection | Equivalent perf to mx.topk in MLX 0.31 |
| `mx.argpartition(a, kth, axis)` | indices version | Used in Prompt 2 candidate path bench |
| `mx.take_along_axis(a, indices, axis)` | per-element axis gather | Used in compact-K gather variant |
| **`mx.gather_mm` (NEW check)** | batch-level gather + matmul fused | NOT applicable — gathers entire matrices per batch, not row-level |
| **`mx.block_masked_mm` (NEW check)** | matmul with block-size masking | NOT applicable for exact top-K (block-approximation only) |
| `mx.fast.scaled_dot_product_attention(q,k,v, mask=array)` | Apple SDPA NAX with arbitrary float mask | Used in Prompt 2 dispatch fix as pass-2 |
| `mx.compile` (graph fusion) | compile-time op fusion | Marginal effect (1.03× speedup confirmed in this check) |

### Component decomposition (re-confirmed Prompt 3)

Audit shape B=1 H=16 qL=4096 D=128 fp16 k_count=64:

| Path | Median | Notes |
|---|---|---|
| Prompt 2 dispatch (mx.topk → bias → mx.fast.sdpa) | 44.39 ms | Production baseline |
| Same chain wrapped in `mx.compile` | 43.08 ms | 1.03× — marginal |
| argpartition → take_along_axis V (compact gather) | 193.46 ms | 4.4× SLOWER — broadcast tensor blows up |
| Dense SDPA reference (no top-K) | 3.15 ms | Unreachable upper bound |
| **Audit baseline (Python reference, sort-based)** | **55.6 ms** | Pre-Prompt-2 starting point |

Sort/partition/topk cost on the materialised [1,16,4096,4096]=256M-element
score tensor remains ~33 ms — the unavoidable floor when scores hit
global memory.  Bandwidth-bound at ~30 GB/s effective (well below
M5 Max's ~700 GB/s peak), indicating MLX's sort is compute-bound or
has cache-unfriendly access patterns.

### Premise verdict: **CONFIRMATION**

The audit's L (~3-6h) prescription for a streaming top-K Metal kernel
is **empirically justified**.  No Apple primitive composition closes
the remaining 14× gap vs dense SDPA.  The Prompt 2 dispatch fix
(1.25× recovery) is the architectural ceiling for non-kernel paths.

### Downstream actions

- Proceed with kernel implementation (Phase A.3).
- Scope-refine: PASS-1 kernel outputs top-K INDICES (small buffer)
  rather than full bias (1 GB tensor).  PASS-2 reuses Apple SDPA NAX
  via a sparse bias built from those indices (Python-side).
- This is the "hybrid Python-extraction + NAX-attention" pivot
  explicitly authorised by the user's Phase A.0 halt criterion, but
  with the EXTRACTION done in a native kernel (not Python) so the
  ~33 ms sort cost is avoided.

## DC1 — Design: PASS-1 kernel + PASS-2 SDPA dispatch

### Why two-pass with indices buffer (not bias buffer)

A full streaming-attention kernel would compute scores + maintain
top-K heap + use the heap to do attention all in one pass.  That's
~600-800 LOC of MSL and contains a non-contiguous K-gather pattern
in pass 2 (loading K rows by index, not by tile-row).  The non-
contiguous gather doesn't play well with Apple NAX cooperative-tensor
MMA primitives (which assume contiguous BK x BD K-tiles).

**Cleaner architecture**: separate the top-K extraction from the
attention computation.

**PASS-1 (new Metal kernel)**:
- Compute Q @ K^T tile-by-tile, scores live in registers only
- Maintain a per-Q-row top-K heap in threadgroup memory
- Output: `topk_idx [B, H, N, k_count]` int32 buffer (256 KB at audit shape)
- Score tensor NEVER materialised to global memory

**PASS-2 (Python-side)**:
- Convert `topk_idx` to a float bias `[B, H, N, S]` via scatter:
  - bias[b,h,n,s] = 0 if s ∈ topk_idx[b,h,n,:] else -1e4
  - Implementable as: bias = mx.full(-1e4); bias[scatter(topk_idx)] = 0
- Call `mx.fast.scaled_dot_product_attention(q, k, v, mask=bias)`
- Apple SDPA NAX kernel handles the masked attention in fused fashion

### Why this scope is realistic in one session

The PASS-1 kernel is structurally similar to a forward-attention
kernel WITHOUT the V@P matmul:
- Same Q-tile loading
- Same Q@K^T MMA computation
- Replaces P@V matmul with heap maintenance
- Replaces lse-write with topk_idx-write

Estimated: ~300-400 LOC MSL (vs ~600-800 for full streaming attention),
no non-contiguous gather (PASS-2 uses standard SDPA), reuses
`naxHelpersBlock()` + `createV34Source()` structural template.

### Heap structure choice

Per-row heap stored in threadgroup memory:
- For k_count ≤ 64: 32 rows × 64 slots × 8 bytes (4-byte score + 4-byte idx) = 16 KB
- For k_count ≤ 128: 32 rows × 128 slots × 8 bytes = 32 KB

M5 NAX threadgroup memory budget: 32 KB per kernel typical.  Cap
k_count ≤ 64 for v1 implementation; document k_count ≤ 128 as future
work.

**Heap operations**:
- Initialisation: all slots = (-inf, -1)
- Insert: linear scan find min → conditional replace (O(k) per insert)
- Final extraction: indices only (scores discarded)

Linear-scan heap is simpler than min-heap on GPU + good cache locality
when heap fits in TGM.  For k=64, each insert is 64 sequential lane-
local comparisons — ~2 cycles each = 128 cycles per insert.

**Per-tile insert cost**: BQ × BK inserts per K-tile = 32 × 64 = 2048
inserts; 2048 × 128 cycles = 262K cycles per tile.  At M5 ~1 TCycles/sec
GPU clock, that's ~262 ns per K-tile.  For S=4096 BK=64: 64 tiles × 262 ns
= ~17 µs per (head, batch).  Across H=16 B=1: 16 × NQ × 17 µs (NQ=128) =
~35 ms.  Hmm, that's not great.  But this is back-of-envelope and
ignores SIMD parallelism.

**Optimised heap insert**: use SIMD operations to parallelise the
linear-scan find-min across 32 lanes.  Each lane checks one slot, simd-reduce
to find min lane, conditional replace.  Reduces insert cost ~32×.

### Algorithm pseudocode

```cpp
// PASS-1: top-K index extraction
[[kernel]] void topk_extract(
    const device T* Q,  // [B, H, N, D]
    const device T* K,  // [B, H, S, D]
    constant TopKParams& params,
    device int32_t* topk_idx,  // [B, H, N, k_count]
    /* thread + grid IDs */
) {
  // Per-Q-block top-K heap in threadgroup memory
  threadgroup float heap_scores[BQ][K_COUNT];  // init -inf
  threadgroup int32_t heap_idx[BQ][K_COUNT];   // init -1
  
  // K-loop
  for (int kb = 0; kb < NK; kb++) {
    // Compute S_tile = Q @ K_tile^T in registers (NAX MMA)
    // S_tile shape: [BQ, BK]
    NAXTile<float, TQ, TK> Stile;
    compute_qk_mma(Q, K + kb*BK, Stile);  // standard V34 forward pattern
    
    // For each (q_row, k_col), update heap
    for q_row in 0..BQ:
      for k_col in 0..BK:
        score = Stile.frag_at(q_row/16, k_col/16)[q_row%16, k_col%16];
        global_k_idx = kb * BK + k_col;
        update_heap(heap_scores[q_row], heap_idx[q_row], score, global_k_idx);
  }
  
  // Write heap_idx to topk_idx buffer
  if (thread_index_in_threadgroup < BQ * K_COUNT) {
    int r = thread_index_in_threadgroup / K_COUNT;
    int c = thread_index_in_threadgroup % K_COUNT;
    topk_idx[batch, head, tid.x * BQ + r, c] = heap_idx[r][c];
  }
}
```

### Python integration

```python
def flash_attention_topk(q, k, v, topk_ratio, ...):
    ...
    k_count = max(1, math.ceil(topk_ratio * S))
    if (_get_has_nax_cached() and D in (64, 128)
            and q.dtype in (mx.float16, mx.bfloat16)
            and k_count <= 64
            and mask is None
            and not _disable_topk_native):
        # PHASE 3B PATH: PASS-1 native kernel + PASS-2 Apple SDPA
        topk_idx = _ext.topk_extract(q, k, k_count, scale)  # [B,H,N,k]
        # Build sparse bias from indices
        # bias[b,h,n,s] = 0 if s ∈ topk_idx[b,h,n,:] else -1e4
        bias = _topk_idx_to_bias(topk_idx, S, q.dtype)
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)
    # Else: Prompt 2 dispatch (Apple SDPA NAX via sort-based bias) — 1.25× over reference
    # OR: reference path (M1-M4, mask, fp32, k_count > 64)
    ...
```

## DC2 — Scope discipline (§AA.1 budget management)

Phase 3b is part of a 3-section Prompt 3 (A + B + C).  Section A
budget should not consume Sections B and C.  Mitigation:

1. Start with simplest possible PASS-1 kernel (single Q-block per
   threadgroup, BQ=32, k_count ≤ 64)
2. Correctness-first: verify pass-1 output against `mx.topk`/`mx.argpartition`
   reference BEFORE doing pass-2 integration
3. If pass-1 kernel correctness validation fails by end of ~2h work:
   §AA.1 halt + STATUS doc + move to Section B (which is independent)
4. If pass-1 works but pass-2 integration shows no speedup:
   ship pass-1 as infrastructure (gated, opt-in) + STATUS doc for
   future pass-2 optimization

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| 3b.0 §AA.5 premise validation | `/mlx-mfa-apple-primitives-coverage` | done — CONFIRMATION verdict |
| 3b.1 Design + decisions doc | (no skill — direct reads + bench check) | done |
| 3b.2 register budget pre-impl | `/metal-kernel-dev` | (pending) |
| 3b.3 implementation | new MSL kernel | (pending) |
| 3b.4 three-axis validation | (test suite) | (pending) |
| 3b.5 corruption audit | `/mlx-debug-forensics` | (pending) |
| 3b.6 perf bench | `/mlx-mfa-bench-methodology` | (pending) |
| 3b.7 pre-merge | `/mlx-code-review` | (pending) |

## DC3 — `/metal-kernel-dev` pre-impl design review verdict

Skill invoked Phase 3b.2 (pre-implementation gate per §AA + Phase 3b.0
GO).  Synthesised review:

| Question | Verdict | Notes |
|---|---|---|
| Register budget | GREEN | Pass-1 state smaller than V34 forward (no Otile/softmax); fits 128-reg budget |
| Heap insert efficiency | **YELLOW — DESIGN RISK** | See below |
| TGM layout (16 KB) | GREEN | Half of 32 KB M5 budget |
| Last-K-block edge | GREEN | Mirror V34's `is_last_k` + per-element check pattern |
| WM Q-row partition | GREEN | SG s owns rows [s*8, s*8+8); independent heaps per SG, no cross-SG sync |

### Heap-insert design risk (Question 2)

**Naïve approach (serial in lane 0)**:
- Per K-tile: lane 0 processes BK=64 candidate scores sequentially
- Per candidate: linear scan of 64-slot heap to find min, conditional replace
- Cost per Q-row: NK × BK × k_count = 64 × 64 × 64 = 262K cycles
- Per Q-block (32 rows): 8M cycles ≈ 8 ms
- × NQ blocks ÷ WM concurrent = ~256 ms total
- **Catastrophic** — slower than Prompt 2 baseline (44 ms)

**SIMD-parallel lane-per-slot approach**:
- 32 lanes, 64 slots → 2 slots/lane
- After Q@K^T MMA, each lane has its slice of the score row in registers
- Cooperative reduction to find heap_min across SG (~5 cycles)
- Each lane independently checks if any of its 2 slots can be replaced
- Conflict resolution when multiple lanes want to replace: needs serialisation or atomic CAS
- Cost per insert: ~10-15 cycles SIMD-parallel
- Per Q-row K-loop: NK × BK × 15 = 64 × 64 × 15 = 60K cycles ≈ 60 µs per Q-row
- × BQ × NQ ÷ WM = ~30 ms total
- **Better but still risky** — conflict resolution algorithm subtle, easy to introduce silent correctness bugs

**Radix-select alternative**:
- Multi-pass bit-level selection: ~log2(k_count) = 6 passes
- Each pass examines one bit of the floating-point representation
- More complex MSL but more parallelism-friendly
- Estimated implementation: 800+ LOC of careful MSL, multi-day work
- **Scope exceeds single session**

### Verdict: **NO_GO for full implementation in this session**

The heap-maintenance algorithm is the central design problem.  Both
SIMD-parallel insertion (correctness risk) and radix-select (scope
risk) require careful Apple-GPU-specific design work that exceeds
the safe envelope for an autonomous session also tasked with shipping
Sections B and C (which deliver causal-training and sparse-training
features — arguably higher user value than top-K which serves niche
attention patterns).

## DC4 — §AA.1 halt protocol invoked

Section A halted per §AA.1.  Reallocating session budget to Sections
B (Phase 4b-complete K-parallel kernels) and C (Sprint 5 sparse
backward) where:
- Scope is well-defined (extend existing kernel pattern with causal
  mask block × 3 kernels for Section B; add sparse iteration to same
  kernels for Section C)
- Success probability high (V34 dQ kernel causal extension already
  shipped in Prompt 2 as exact template for Section B's K-parallel
  kernels; Section C extends Section B's pattern)
- User value clear (causal LLM training + VSR-style sparse training)

Phase 3b deferred with:
- §AA.5 CONFIRMATION verdict preserved (kernel work empirically justified)
- Design analysis preserved (DC1 + DC3 above)
- Scope estimate refined: L (~6h CC dedicated session) is correct;
  the heap-maintenance algorithm is the critical design problem
  requiring focused work, not a peripheral concern
- Production impact: zero — Prompt 2 dispatch fix (1.25× speedup) remains
  the production path; users unaffected

## Recommended next steps for Phase 3b (dedicated future session)

1. **Prototype the SIMD-parallel heap insertion in isolation** (~1h):
   write a standalone Metal kernel that takes a [BQ, BK] score tile +
   current heap state, runs the SIMD-parallel update, validates
   against a Python reference.  Iterate on correctness BEFORE
   integrating into the full top-K pipeline.

2. **Alternative: explore radix-select implementation** (~2h):
   if SIMD-parallel heap proves too brittle, multi-pass radix-select
   on the floating-point bit pattern may be more robust at the cost
   of higher kernel LOC.

3. **Once heap algorithm validated, full kernel integration** (~2h):
   wrap heap algorithm in standard V34-forward-style outer loop +
   threadgroup-memory setup + topk_idx output write.

4. **Python integration + tests + bench** (~1h):
   PASS-2 (scatter indices → bias → mx.fast.sdpa) is straightforward
   Python.  Tests reuse Sprint 3 Phase 3a test patterns.

Total dedicated session: ~6h CC.  Matches audit's L estimate.

## Skill invocations log (per §AA.2) — UPDATED

| Phase | Skill | Status |
|---|---|---|
| 3b.0 §AA.5 premise validation | `/mlx-mfa-apple-primitives-coverage` | done — CONFIRMATION |
| 3b.1 Design + decisions doc | (no skill — direct reads + bench check) | done |
| 3b.2 register budget pre-impl | `/metal-kernel-dev` | done — NO_GO heap-maintenance algorithm risk |
| 3b.3-3b.7 | not initiated due to NO_GO at 3b.2 | N/A — halt |

# v2.50 Sprint 3 Phase 3b — Native streaming top-K Metal kernel — DEFERRED

**Sprint date**: 2026-05-13
**Status**: **DEFERRED to dedicated future session**
**Phase 3a status**: SHIPPED — see `docs/v50/sprint3-decisions.md`

## TL;DR

Phase 3a (dispatch fix) ships 1.25× speedup on `flash_attention_topk`
(55.6 ms → 44.4 ms at canonical audit shape).  Phase 3b — a native
streaming top-K Metal kernel — would recover the remaining ~14× gap
vs dense SDPA but requires **L effort (~5h CC)**: dedicated kernel
design, register-budget pre-flight, MLX Primitive scaffolding, binding,
three-axis validation.

Per §AA.1 failure-mode handling, halting Phase 3b in this session +
documenting the design + deferring to a dedicated future session.
Phase 3a deliverable preserved on master.

## Why Phase 3a alone is not enough

The 1.25× speedup brings `flash_attention_topk` from 17.95× regression
to 14.32× regression vs dense SDPA.  This is meaningfully better but
still leaves top-K attention "essentially unusable at scale" per the
audit's verdict.

The fundamental ceiling: any path that materialises the [B,H,N,S]
score tensor pays ~33 ms in MLX 0.31 just for the score-tensor I/O
and global threshold-finding (sort/partition/topk — empirically all
the same cost on M5 Max).  At B=1 H=16 N=S=4096 D=128 that tensor is
1 GB of f16 data; the bandwidth alone exceeds the entire dense SDPA
runtime.

A native streaming kernel never materialises the full score tensor.
Scores live tile-by-tile in registers, exactly like flash attention's
forward pass — but with a per-query online top-K heap maintained
alongside the running softmax statistics.

## Phase 3b design sketch

### Algorithm variant 1 — Exact streaming top-K (preferred)

Two-pass kernel:

**Pass 1 — top-K index selection (no V access)**:

```
For each query tile Q_tile [BQ=32 rows, in registers]:
  Initialise per-row heap: heap[r] = [k_count slots, value=-inf, idx=-1]
  For each key tile K_tile [BK=64 cols] in 0..S step BK:
    Load K_tile into threadgroup memory (use Apple NAX cooperative tensor)
    Compute S_tile = Q_tile @ K_tile^T [BQ x BK] in registers
    For each row r in 0..BQ:
      // Update heap with the BK scores in this row
      For each col c in 0..BK:
        global_idx = key_tile_start + c
        if S_tile[r,c] > heap[r].min_value():
          heap[r].replace_min(S_tile[r,c], global_idx)
  Write heap indices to global memory: topk_idx [B,H,N,k] int32
```

Heap structure choice: BQ=32 rows × k_count slots × 8 bytes each.
For k_count=64: 16 KB per threadgroup.  For k_count=256: 64 KB — too
much.  Cap k_count ≤ 128 for v1; larger k uses the Phase 3a fallback.

**Pass 2 — Streaming attention over top-K positions**:

```
For each query tile Q_tile:
  Read heap indices topk_idx[b,h,n,:] for each row
  // Build a per-row sorted index list to enable coalesced K/V reads
  Sort indices in-register (k_count ≤ 128 → 1-pass radix or bitonic)
  // Standard flash-attention online softmax over the gathered tiles
  For each chunk_start in 0..k_count step BK:
    For each row r:
      Load K[topk_idx[r, chunk_start:chunk_start+BK], :] into registers
      Compute partial scores, update running L, M, O
  Write final O
```

**Register budget pre-flight (`/metal-kernel-dev` required before impl)**:
- BQ=32, BK=64, D=128, k_count=64
- Q_tile: 32×128 f16 = 4 KB threadgroup or 128 regs/lane
- Heap: k_count=64 × (f16 value + i32 idx) = 384 B/row × 32 rows = 12 KB tgm
- Total tgm: Q + heap + K_tile = ~20 KB — fits in 32 KB M5 NAX budget
- Per-warp registers: ~96 — within M5 NAX 128-reg budget

### Algorithm variant 2 — Block-level approximate top-K (simpler, semantics change)

One-pass kernel:

```
For each query tile Q_tile:
  per_block_max[NK] = -inf
  For each key tile K_tile in 0..NK:
    Compute S_tile = Q_tile @ K_tile^T
    per_block_max[K_tile] = max(S_tile across BQ x BK)
  Sort per_block_max to find top-N_kept blocks
  // Now do flash attention only over the top-N_kept blocks
  For each kept block:
    ...standard flash attention online softmax...
```

This changes semantics from exact top-K (keeps the k_count highest-
scoring keys globally) to block-approximate top-K (keeps all keys in
the N_kept highest-block-max blocks).  Simpler to implement, faster
in steady state, but a different operation.

Recommend variant 1 (exact) for v2.50 to preserve API contract; offer
variant 2 as a separate `flash_attention_topk_blockwise` if there's
user demand.

## Estimated Phase 3b effort (dedicated session)

| Phase | Component | CC time |
|---|---|---|
| 3b.1 | Read existing V6NAX/STEEL kernel templates + Apple NAX cooperative-tensor primitives | 30 min |
| 3b.2 | `/metal-kernel-dev` pre-impl: register budget + heap structure validation | 30 min |
| 3b.3 | Pass-1 kernel: streaming top-K index selection generator (~400 LOC MSL) | 1.5h |
| 3b.4 | Pass-2 kernel: gather-attention generator (~300 LOC MSL, mostly reused from V6NAX) | 1h |
| 3b.5 | `csrc/mfa_topk_fwd.cpp`: MFATopKForward Primitive + eval_gpu | 45 min |
| 3b.6 | `csrc/bindings.cpp`: `mfa_topk_forward` binding | 15 min |
| 3b.7 | Routing in `flash_attention_topk` (Phase 3a fallback preserved) | 15 min |
| 3b.8 | Three-axis tests (correctness, public API, edges) | 45 min |
| 3b.9 | Three-axis bench + cross-session §AA.4 | 30 min |
| 3b.10 | `/mlx-debug-forensics` corruption audit | 30 min |
| 3b.11 | `/mlx-code-review` + decisions doc + CHANGELOG | 30 min |
| **Total** | | **~6h CC** |

The estimate is on the upper end of L (~3-6h) per the audit, because
the streaming top-K kernel is genuinely novel (no existing Apple NAX
template for online heap maintenance) and the three-axis test scaffold
needs careful design to cover (a) heap correctness vs exact sort,
(b) attention output vs the Phase 3a reference, (c) gradient via
`mx.vjp` against the Phase 3a path.

## What's still preserved

- **Phase 3a SHIPPED at Sprint 3 branch** (will merge to master with
  this STATUS doc): `flash_attention_topk` M5+ NAX dispatch fix,
  1.25× speedup, 17/17 tests pass, zero regressions in the 1112 pre-
  existing baseline tests.
- **Test count post-merge**: 1129 passing (1112 baseline + 17 Sprint 3).
- **50 pre-existing failures** (test_v6nax_backward_kv.py +
  test_v6nax_bwd_multisg.py: D_vec API mismatch from v2.38.1) are
  unrelated to Sprint 3 and tracked separately.

## Recommended next steps for Phase 3b (dedicated future session)

1. Read Apple NAX cooperative-tensor MMA primitive docs + existing
   V6NAX forward kernel structure (`createV6NAXSource()` in
   `csrc/mfa/v6_nax/NAAttentionKernel.cpp`)
2. `/metal-kernel-dev` pre-impl for streaming heap data structure +
   register budget at D=64 and D=128
3. Implement Pass-1 kernel (`createTopKIndicesSource()`)
4. Verify Pass-1 produces correct top-K indices via bit-identical
   comparison to `mx.argpartition` on the same input
5. Implement Pass-2 kernel (gather attention, mostly reused from V6NAX)
6. Three-axis validation + cross-session §AA.4 perf bench
7. Update `flash_attention_topk` routing to call native kernel; keep
   Phase 3a path as fallback for unsupported edges (large k_count,
   block mask, etc.)

## Files in this Phase 3b deferral

Only this STATUS doc + entries in CHANGELOG/sprint3-decisions.md.  No
code changes for Phase 3b.  The Sprint 3 branch
(`feat/v50-sprint3-topk-dispatch`) ships ONLY Phase 3a.

## Master state post-Sprint-3 (Phase 3a only)

- `master`: ~`<Sprint 3 merge commit>` (Phase 3a merged)
- Sprint 3 Phase 3a complete: dispatch fix, 1.25× speedup.
- Sprint 3 Phase 3b deferred with corrected scope estimate (~6h CC).
- CHANGELOG `[Unreleased — for v2.50]`: 3 entries (Sprint 1, Sprint 2,
  Sprint 3 Phase 3a).
- Tests: 1129/1129 of relevant tests pass (50 pre-existing failures
  unrelated to Sprint 3).
- No version bump, no tag, no PyPI publication per internal-mode
  contract.

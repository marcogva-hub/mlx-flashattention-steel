# V6NAX backward Option β — design hints from V6NAX forward mechanistic findings

**Generated**: 2026-05-12 (post V6NAX forward investigation)
**Source**: `docs/v6-nax/v6nax-forward-mechanisms.md`
**Purpose**: feed-forward design constraints for the next prompt
(V6NAX backward NAX-direct rearchitect Option β).

## TL;DR

V6NAX forward's +18% baseline gain comes from a **bundled** mechanism trio
(cross-SG sync elim + simd_shuffle_xor + M5-tuned defaults). All three
transfer cleanly to backward. One **bonus anti-pattern finding** worth
addressing in the backward implementation: V6NAX's default `EXEC_SG=4` is
sub-optimal for mid-range shapes; SG=8 wins +32% on mid_d128.

V6NAX backward Option β should:
1. Apply the same B+C+E bundle (≥18% expected on similar regime)
2. Default `EXEC_SG=8` for D=128 mid shapes (or shape-aware heuristic)
3. NOT inherit Apple MPP autotune defaults (E confirmed structurally)
4. Use `NAXFrag::row_reduce` for ALL row-reductions (softmax recompute
   + dS row-sum + D accumulator)

## Backward pattern overview

V6NAX forward computes `O = softmax(Q @ K^T * scale) @ V`. V6NAX backward
must compute three gradients:
- **dQ**: requires re-computing P (softmax probabilities) and matmul
  with dO + correction
- **dK**: requires P^T @ dO accumulation across all Q tiles for this K tile
- **dV**: requires P^T @ dO accumulation across all Q tiles for this V tile

Standard FA-2 backward uses 7 GEMMs (2 to recompute attention, 5 for
gradient terms) and 2 separate kernel dispatches (dQ-loop and dK/dV-loop).

## Per-hypothesis transfer evaluation (with findings)

### Hypothesis A — TGP occupancy (forward verdict: FALSIFIED at baseline, REVERSE at SG=8)

**Forward finding**: V6NAX's default `EXEC_SG=4` is sub-optimal for mid_d128.
SG=8 wins +32% on mid_d128, 16% median across usable shapes. Large_d128 is
already saturated at SG=4 (no benefit from SG=8). Small_d128 also benefits
marginally from higher SG.

**Backward transfer**:
- **dQ kernel**: per-Q-tile parallelism naturally aligns with forward's
  per-Q-tile dispatch. The same SG sub-optimality likely applies — use
  `EXEC_SG=8` as default for D=128 mid shapes; consider shape-aware
  heuristic with `EXEC_SG=4` only for the largest shapes (already
  saturated regime).
- **dK/dV kernel**: per-K-tile parallelism with Q-tile inner loop.
  Different scheduling pattern than forward. **Recommend autoresearch
  sweep on dK/dV SG defaults** as part of Option β implementation;
  don't assume the forward result transfers without verification.
- **ACTION**: V6NAX backward should ship with `EXEC_SG=8` as default for
  D=128, NOT inherit V6NAX forward's `EXEC_SG=4` choice.

### Hypothesis B — cross-SG sync elimination (forward verdict: CONFIRMED structurally + bundled)

**Forward finding**: V6NAX K-loop uses only `simdgroup_barrier(mem_none)`
(line 2906; intra-SG, lightweight). Predecessors use
`threadgroup_barrier(mem_threadgroup)` (lines 1059, 1290; cross-SG,
heavyweight). The mechanism contributes to the bundled 18% gain.

**Backward transfer**:
- **Directly transferable to dQ kernel**: same per-Q-tile structure,
  no cross-SG accumulation needed (dQ for a given Q tile only depends
  on that tile's row context).
- **Adaptable for dK/dV kernel**: dK[k_tile] = ∑_{q_tile} P[q,k]^T @ dO[q].
  Naive backward accumulates across SGs. **Goal**: per-SG accumulation
  in private registers, followed by ≤1 `threadgroup_barrier(mem_threadgroup)`
  per K-tile (after all Q-tiles processed). DO NOT add intermediate
  threadgroup_barriers inside the Q-tile loop.

### Hypothesis C — simd_shuffle_xor vs MPP reduce (forward verdict: CONFIRMED structurally + bundled)

**Forward finding**: V6NAX uses `Stile.template row_reduce<MaxOp>(...)`
(line 2889) → internally `simd_shuffle_xor` (line 2546). Predecessors use
`mpp::reduce_rows(...)` cooperative-tensor reduction.

**Backward transfer**:
- **dQ kernel softmax recompute**: needs `row_reduce<MaxOp>` +
  `row_reduce<SumOp>` patterns — **directly transferable**, use
  `NAXFrag::row_reduce`.
- **dK/dV kernel `dS = P * (dP - rowsum(P * dP))`**: the rowsum is a
  row reduction. **Use simd_shuffle_xor via NAXFrag::row_reduce here too**.
- **D accumulator (`rowsum(dO ⊙ O)` for gradient correction)**: also a
  row reduction → simd_shuffle_xor.
- **ACTION**: every row-reduction in V6NAX backward must use
  `NAXFrag::row_reduce` (the simd_shuffle_xor path). Do NOT use
  `mpp::reduce_rows` (cooperative-tensor MPP) anywhere in the backward
  kernel.

### Hypothesis D — register pressure (forward verdict: NULL at baseline, REVERSE on small shapes when larger)

**Forward finding**: V6NAX default tile (BQ=64 for D=128, BQ=32 for D=64)
is NOT register-pressure-bottlenecked. Larger tiles (BLOCK_R=64,
BLOCK_C=64) don't slow down V6NAX baseline; small shapes actually benefit
from LARGER tile (less iteration overhead).

**Backward transfer**:
- **dQ kernel**: similar tile shape constraints as forward. BQ=BK=32 to 64
  workable.
- **dK/dV kernel**: NEW constraint — dK and dV accumulators are
  `BK × D` FP32 tiles per K-tile. With BK=32 D=128 = 16 KB FP32
  accumulator per SG. With WM=4: 4 × 16 KB = 64 KB across the TGP
  for dK alone, plus the same for dV.
- M5 Max register file: ~32 KB per SG. dK accumulator at 16 KB ≤ 32 KB
  fits, but the COMBINED dK + dV + intermediate state may push toward
  spill.
- **Recommended**: start with BK=32 for D=128 (smaller than forward's
  BK=32, same value); BQ=32 for the dK/dV inner loop. If perf testing
  reveals spill is hurting, the M5-tuned route is to reduce TQ (tile
  rows per thread) before reducing BK.

### Hypothesis E — Apple defaults mis-tuned for M5 (forward verdict: CONFIRMED structurally + bundled)

**Forward finding**: V6NAX uses explicit M5-tuned BQ/BK/WM defaults bypassing
Apple's MPP autotune. The mechanism contributes to the bundled 18% gain.

**Backward transfer**:
- **DO NOT inherit Apple's MPP autotune defaults for V6NAX backward**.
- Apple's autotune optimizes for generic shapes; M5 Max has specific
  characteristics (cluster-shared L2 sizing, register file per SG,
  SG scheduling priority) that V6NAX forward explicitly tuned for.
- **Recommended starting defaults for V6NAX backward**:
  - dQ kernel: BQ=32, BK=32, WM=4 (D=128); BQ=32, BK=32, WM=2 (D=64)
  - dK/dV kernel: BK=32, BQ=32, WM=4 (D=128); BK=32, BQ=32, WM=2 (D=64)
  - Per Hypothesis A finding: bump WM=8 for mid_d128 shape regime
- Run Option β implementation with EXEC_SG autoresearch sweep to
  validate.

## Open questions for V6NAX backward Option β

1. **dQ vs dK/dV kernel split**: standard FA-2 uses 2 separate dispatches.
   V6NAX backward Option β should also use 2 kernels (cleaner per-SG
   partitioning per gradient term).

2. **NAXFrag accumulator types**: dK/dV accumulators are FP32 (loss of
   precision tolerance is low). NAXFrag supports FP32 cooperative tensors.
   Verify scope `<1>` works for the dK/dV accumulator pattern (V6NAX forward
   uses `<1>` for the Otile accumulator, so this should transfer).

3. **Block mask + causal interaction**: V6NAX forward Option α may handle
   causal differently than V6NAX backward will need to. Causal backward
   needs the same triangular mask within the diagonal block but on the
   gradient terms.

4. **K-tile loop direction**: dK/dV accumulates across Q tiles. Loop
   order Q outer vs K outer affects cache behavior. V6NAX forward iterates
   K-tiles outer (per-Q-tile in dispatch); backward may need K outer
   (per-K-tile dispatch for dK/dV).

5. **Shared softmax recompute or per-kernel recompute?**: dQ and dK/dV
   both need P (softmax). If kernels are split, P must be recomputed in
   each. Option β should benchmark recompute-vs-store-in-tgp tradeoff.

## Recommended next-sprint scope (V6NAX backward Option β)

1. **dQ kernel**: clone V6NAX forward structure with backward-specific
   inner loop (recompute P; D = rowsum(dO ⊙ O); dS = P * (dP - D);
   dQ += dS @ K).
2. **dK/dV kernel**: per-K-tile dispatch, per-SG partition of Q-tiles;
   dV[k] += P^T @ dO; dK[k] += dS^T @ Q.
3. **NAXFrag::mma** for all GEMMs (Q@K^T recompute, S@dO, P^T@dO, etc.).
4. **NAXFrag::row_reduce** (simd_shuffle_xor) for all row-reductions
   (softmax recompute + dS row-sum + D accumulator).
5. **`simdgroup_barrier(mem_none)`** intra-SG only; minimize
   `threadgroup_barrier(mem_threadgroup)` to ≤1 per K-tile in dK/dV kernel
   (after all Q-tile contributions accumulated).
6. **M5-tuned defaults** matching V6NAX forward (BQ=32-64, BK=32, WM=2-4
   depending on D), with **EXEC_SG=8 default for D=128 mid shapes**
   per the V6NAX forward Hypothesis A anti-pattern finding.
7. **Autoresearch sweep on dK/dV EXEC_SG** as final tuning step before ship.

## Next prompt scope estimate

V6NAX backward NAX-direct monolithic rearchitect (Option β) using this doc
as canonical input. Per memory #30 roadmap note: estimate ~1 week CC work.

The bonus finding (anti-pattern A — EXEC_SG=4 sub-optimal for mid shapes)
is independently actionable as a smaller follow-up patch to V6NAX FORWARD
itself, separate from the Option β backward sprint.

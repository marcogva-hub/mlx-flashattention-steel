# Sprint III-8d — Standalone MMA reproduction: the faithful-read blocker

**Date:** 2026-06-15 (continuation of the III-8 marathon)
**Executor:** Claude Opus 4.8 High
**Outcome:** mechanism **NOT isolated**, fix **NOT landed**. But the sprint
**rigorously closed the register-read approach via the mandated `col≤row`
self-check** (not by inference, as before) and pinpointed the exact blocker:
**faithfully serializing the Q@K^T cooperative-MMA accumulator fragment.**
No functional kernel edit (no guess). Reports honestly per the prompt's
explicit fallback.

## MMA machinery understood (R.1 prep)
- `MFAMMAFrag` wraps `simdgroup_matrix<T,8,8>` (Apple cooperative matrix, 2
  elements per lane). `MFAMMATile<T,M,N>` is a grid of 8×8 frags.
- Q@K^T: `MFAMMAFrag::mma(Stile.frag_at(iq,ik), Qtile.frag_at(iq,dd),
  Ktile.frag_at(0,ik), …)` accumulated over the D-chunk `dd`. `Stile` is a
  `MFAMMATile<AccT, MFA_TQ, MFA_TK>` **accumulator**.
- Store (`MFAMMATile::store<U,1,1>`): per-lane, `base = i*8*ld + j*8`,
  `dst[base+0],[+1] = frag_at(i,j)[0],[1]`, with `dst` pre-offset by the
  lane's `simd_coord` `(sm,sn)`. This same store writes `Otile` correctly
  (causal output is correct).

## The decisive self-check (R.2) — register read is UNFAITHFUL
Re-added the `Stile.store<T,1,1>` dump (allowed for causal) and applied the
**mandatory `col≤row` self-check**: causal output is correct ⇒ the causal
`col≤row` scores ARE correct ⇒ a faithful dump must match numpy there.

| dump | `col≤row` MAE vs numpy | `col>row` MAE |
|---|---|---|
| causal | **1.18** (should be ~0!) | 1.21 |
| non-causal | 1.16 | 1.17 |
| causal-dump vs non-causal-dump (pre-mask, should be identical) | **MAE 0.24** | |

The causal dump **fails** the self-check (1.18 on a region known to be
correct), and the causal/non-causal dumps differ pre-mask (they must be
identical — same Q@K^T). ⇒ **`MFAMMATile::store` does not faithfully
serialize the `Stile` Q@K^T accumulator.** This rigorously closes the
register-dump approach (previously only inferred unreliable; now proven by
the prompt's mandated self-check).

## The precise blocker
The kernel **consumes** `Stile` correctly via `frag_at(i,j)[jj]` (the mask +
P@V use it; causal output is correct) — but **reading `Stile` out to memory
via `store` is unfaithful**, even though the identical `store` works for
`Otile`. So the Q@K^T **accumulator** tile's lane→(row,col) layout differs
from what `store`/`simd_coord` assume for `Otile` (an accumulator-vs-operand
/ WM-row-mapping subtlety of cooperative simdgroup matrices). Every read
method tried — `store_contiguous`, element-wise `frag_at`→memory, `store` —
fails the `col≤row` self-check or gives inconsistent results. **A standalone
harness uses the same store and faces the identical blocker.**

## Why this is genuinely hard
To read the `col>row` scores (the only place the bug lives, since causal
zeros and never exercises them), one must serialize the cooperative-MMA
accumulator fragment — and that serialization is itself the unsolved
sub-problem (3 read methods defeated, each failing the self-check). The
kernel never needs to serialize Stile to memory (it consumes it in-register
via `frag_at` for P@V), so there is no proven-faithful Stile→memory path to
borrow.

## Honest status + next fork (per the prompt's fallback)
Mechanism localized to: **wrong `col>row` contributions on a single Q@K^T
tile, non-causal single-pass** (reliable: III-8c). Reading those scores
requires a faithful cooperative-MMA-accumulator serialization, which has
defeated every method tried, validated rigorously by the self-check.

Next fork (scoped, not executed — needs a focused dedicated effort):
1. **Derive the exact accumulator lane→(row,col) layout** via a trivial
   numpy-parity single-MMA test: feed `MFAMMAFrag::mma` a KNOWN tiny input
   (e.g. Q=I, K=I) where every (i,j) score is analytically known, and
   reverse-engineer which `(simd_lane, i, j, [0|1])` maps to which (row,col)
   — until a read matches numpy on `col≤row`. THEN read `col>row`.
2. **Escalate to the MLX MMA-primitive layer**: reproduce with `mx`'s own
   simdgroup-matrix path / a minimal `mx.fast.metal_kernel` doing one
   `simdgroup_multiply_accumulate`, comparing every element to numpy.

This has consumed III-7 + III-8 + III-8c + III-8d. The bug is non-default-
reachable (dispatch routes non-causal dense → SDPA). **No guessed edit.**
Marco's call: commit a fresh focused session to the layout-derivation fork,
or take the route-around (forced-mfa non-causal → V1/SDPA, correct API
output now) as interim while the layout fork is scheduled separately.

## Institutional note (extends lesson #12)
Cooperative-MMA **accumulator** fragments cannot be assumed serializable by
the tile's own `store` even when that `store` works for a sibling tile
(`Otile`): operand-vs-accumulator and multi-simdgroup (WM) row mappings
differ. To read an MMA tile mid-pipeline, first derive+validate the
lane→(row,col) layout on a known-input self-check; an unvalidated read is an
unreliable probe (this cost III-8/8d three defeated read methods).

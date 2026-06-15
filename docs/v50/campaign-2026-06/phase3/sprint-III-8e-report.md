# Sprint III-8e — known-answer probe cracks the mechanism (behavioral); exact line still open

**Date:** 2026-06-15 (continuation of the III-8 marathon)
**Executor:** Claude Opus 4.8 High
**Outcome:** the **known-answer uniform-P probe RELIABLY cracked the
behavioral mechanism** (where 4 sprints of register dumps / one-hot recovery
failed) — but the exact codegen line is **not yet pinned** (the signature
contradicts every obvious source), and the fix is **not landed**. No guessed
edit. The probe technique is the institutional win.

## The technique that worked (codify — companion to lesson #12)
Register dumps of the cooperative-MMA accumulator are unfaithful (lesson #12,
re-proven in III-8d via the `col≤row` self-check). Instead, read the
**effective attention through the correct full pipeline** with **known-answer
inputs**, isolating each stage:
- **Q=0** ⇒ all scores 0 ⇒ **uniform P** regardless of any Q@K^T bug ⇒
  isolates softmax-sum + P@V (scores irrelevant).
- **V[j,0]=j** (ramp) ⇒ `O[i,0] = Σ_j P[i,j]·j` = mean of the attended key
  set ⇒ reveals WHICH keys are attended (a scalar summary of P).
- **V = 1[j∈S]** ⇒ `O[i,0]` = P-mass on key-set S ⇒ confirms the attended set.
All validated against fp32 (and self-consistent: V=ones ⇒ O=1.0 ⇒ P
normalized). No dump, no fragment-serialization, no one-hot-O=P confound.

## The mechanism (reliable, reproducible, no confound)
With **Q=0 (uniform P)** and **V[j,0]=j**:

| Q-tile qb | rows | O[row,0] | attended keys |
|---|---|---|---|
| 0 | 0–31 | 15.5 | 0–31 (= **(0+1)·BQ**) |
| 1 | 32–63 | 31.5 | 0–63 (= (1+1)·BQ) |
| 2 (N=128) | 64–95 | 47.5 | 0–95 (= (2+1)·BQ) |
| 3 (N=128) | 96–127 | 63.5 | 0–127 (= (3+1)·BQ) |

**Non-causal single-pass attends exactly `(qb+1)·BQ` keys per Q-tile** —
tile-uniform (all rows in a tile attend the same set), sub-tile granular (32
keys of a 64-key BK tile for qb=0). That is precisely the **causal
`q_max = (qb+1)·BQ` key bound leaking into the non-causal path.** P is
normalized over that truncated set (V=ones ⇒ 1.0; V=1[j<32] ⇒ 1.0 for qb=0;
V=1[j≥32] ⇒ 0.0).

**This resolves the four-sprint paradox**: causal masks keys ≥ q_max anyway
(col>row), so the truncation is invisible to causal (correct), and qb=1+ tiles
attend enough keys to be correct — non-causal qb=0 is the only visibly-wrong
case, exactly matching III-8c's single-tile finding. The earlier
"col>row scores wrong" / "denominator under-accumulated" reads are subsumed:
the keys ≥ q_max are simply **not attended at all** (truncated), not
miscomputed.

## Why the exact line is still open (honest)
The signature is `q_max=(qb+1)·BQ` **keys** (32 for qb=0), tile-uniform — and
it matches NONE of the obvious sources:
- `kb_lim` non-causal `= p->NK` (q-independent; = 1 tile = 64 keys for N=64,
  not 32). Ruled out.
- causal `kb_lim` formula would give `ceil(q_max/BK)` = 1 tile = 64 keys (not
  32). Ruled out.
- diagonal mask + K-boundary mask: gated `if(causal)` / `if(kb==NK_aligned)`;
  inactive for the N=64 repro. Ruled out.
- dispatch params (`NK2=1`, `NK_aligned=1`, `kL_rem=64`): correct,
  q-independent. Ruled out.
Yet the limit is q-dependent (`qb`) — so it must be an in-kernel `qb`-using
expression I have not located by static reading (candidate: the loaders'
effective key extent, the MMA K-frag iteration, or a subtler interaction).

## Next step (now enabled by a reliable oracle)
The uniform-P attended-key probe is a **fast, reliable oracle**. Bisect the
kernel source with it: systematically disable/alter candidate `qb`-using
regions (loaders, MMA K-frag loop bounds, the kb_lim/mask emission even when
"inactive") and re-measure the attended-key count until it becomes `N` for all
qb. Then the fix is the line whose change restores full attention; validate by
O vs fp32 across the domain (R.4), then R.5 generalization (any path reading
above-`q_max` positions), R.6 re-bench/promote, R.7 lock. **No guessed edit
— the bisection identifies the line empirically via the oracle.**

## Status
Mechanism: **reliably pinned** (non-causal attends `(qb+1)·BQ` keys = causal
q_max; the keys ≥ q_max are truncated, not miscomputed). Exact line: **open**
(contradicts obvious sources; oracle-bisection is the tractable continuation).
Fix: **not landed.** Non-default-reachable. No guessed edit. Consumed III-7
through III-8e. The route-around remains the low-risk interim for API
correctness if the oracle-bisection is deferred.

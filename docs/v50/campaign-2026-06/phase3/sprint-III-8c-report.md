# Sprint III-8c — V2 single-pass non-causal: static-diff reorientation

**Date:** 2026-06-15
**Executor:** Claude Opus 4.8 High (continuation of the III-8 session)
**Outcome:** repair **NOT landed**. The static `kb_lim`/mask diff + git history
**narrowed the bug very tightly but did not isolate the exact line.** Per the
sprint's explicit fallback, reported honestly with the narrowed state + the
scoped next fork. **No functional kernel edit made** (no guess; race risk).

## Approach (replacing the retracted register-dump method)
Per Marco's reorientation: abandon mid-kernel register dumps (proven
unreliable twice — cooperative-MMA fragments don't serialize per-lane).
Diagnose only via (a) **static causal-vs-non-causal source diff**, (b) **git
archaeology**, (c) **end-to-end O vs independent fp32** as the sole oracle.

## R.1 — Static diff (exhaustive): only two differences, both read correct
Exhaustive enumeration of `causal` in the single-pass generator
(`generate_steel_v2_source`, lines 139–890): **exactly two** causal-conditionals —
- `kb_lim` (line 435): causal bounds the K-loop by query position; non-causal
  `= p->NK` (all tiles). **Reads correct.**
- diagonal-mask block (line 665): causal masks `col>row` to `-INF`; non-causal
  skips it. **Reads correct.**

Both are **original code (commit 81d801f7, 2026-03-10), unchanged.**

## R.2 — git archaeology: resolves the v1.4.0 contradiction
No `v1.4.0` tag exists (tags jump v1.3.0 → v2.5.0). The `kb_lim` + mask are
original/unchanged. Verdict: the III-7 "v1.4.0 benched V2 non-causal working"
was a **coverage illusion** — speed was measured, but **non-causal single-pass
correctness was never fp32-validated**. NOT a regression; the bug is
longstanding and original. (Another instance of III-4 lesson #10 / III-7
Class C: "working-but-unvalidated path.")

## Reliable empirical narrowing (O vs fp32 only)
- **Single FULL tile, identical kb_lim:** N=64 D=128 (BK=64) is exactly 1 K-tile
  with `kb_lim=NK=1` for **both** causal and non-causal. Causal MAE **0.0001**,
  non-causal MAE **0.19**. So the bug is exposed **purely by the diagonal
  mask's absence** — not by `kb_lim` (identical here), not cross-tile, not the
  partial-tile path.
- **It's the `col>row` (future-key) positions.** Causal (masking `col>row`) is
  correct → the `col<=row` contributions are correct. Non-causal additionally
  uses `col>row` and is wrong → the **`col>row` positions' contributions are
  wrong.** Error decreases monotonically with N (0.19→0.13 as N 64→256):
  the wrong fixed per-tile contribution dilutes as the softmax denominator
  grows.
- **BK-independent:** `MFA_V2_FORCE_BK=32` does not fix it (non-causal still
  wrong) — not a BK=64 tile-size bug.
- **Deterministic** (maxdiff 0.0) — not a race.

## Why the line is still not isolated
The bug lives at the `col>row` positions, which the causal path **masks to
zero and therefore never exercises** — so causal-correctness cannot tell us
whether those positions' **scores (Q@K^T)** or their **P@V contribution** are
the fault. And the two methods that could separate them are unavailable here:
- mid-kernel register dumps — **unreliable** (cooperative-MMA fragment layout;
  proven by a causal control yielding wrong dumped scores despite correct
  output);
- one-hot-V (O=P) recovery — **reliable only for causal**; for non-causal it
  assumes P@V is correct, which is one of the two suspects.

The two source diffs both read correct, so the fault is a **subtle MMA-fragment
issue** (coverage/layout of the `col>row` fragments) that source reading does
not reveal.

## Next fork (scoped, not executed — needs a focused pass)
1. **Layout-correct `simd_shuffle` fragment gather**: serialize the Stile MMA
   fragment into a known lane order in a scratch buffer so mid-kernel scores
   for `col>row` can be trusted — the reliable replacement for the dead-end
   per-lane dump. Distinguishes wrong-scores vs wrong-P@V definitively.
2. **Minimal standalone MMA reproduction**: replicate just the
   `MFAMMATile`/`MFAMMAFrag` Q@K^T for one 32×64 tile outside the full kernel,
   compare every (i,j) to numpy — isolates whether the MMA fragment math is
   wrong for `col>row`.

## Disposition
Not default-reachable (dispatch routes non-causal dense → SDPA for perf;
reachable only via forced `backend="mfa"`). Repair has now consumed III-7 +
III-8 + III-8c. The route-around option (forced-mfa non-causal → V1/SDPA)
remains available as a low-risk interim that yields correct API output.
Marco's call on the next fork vs route-around vs pause. **No release impact on
the default path. No guessed edit.**

## Methodological lesson (codified — see audit-framing-inversions.md)
For cooperative-MMA (simdgroup-matrix) kernels: **mid-kernel register dumps are
unreliable** (fragment layout doesn't serialize per-lane; the value you write
out is not the value the kernel computes with). **Diagnose via end-to-end O vs
an independent fp32 reference + static causal-vs-non-causal source differential**,
and when a probe (e.g. one-hot-V O=P) depends on an unproven component, treat
its result as confounded, not conclusive. This cost III-8 two retractions.

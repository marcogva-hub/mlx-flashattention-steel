# Sprint II-10 — V34 XL Top-K Filtered-SDPA Variant (2026-06-12)

**Status**: COMPLETE — **DECLINED with bench; Approach 5 closed
permanently** (second measured negative, after II-3's streaming build).

## R.0 — Blueprint recovered

Architecture B (production AUTO): materialized scores + 32-iter
bisection threshold + elementwise bias + SDPA NAX = **11.34 ms** at the
audit shape (B1 H16 N=S=4096 D=128 K=64).  Refined Approach 5: PASS-1
matmul-grade top-K INDICES -> scatter-to-bias -> unchanged SDPA NAX;
projected ~7 ms (1.6x); kill gate PASS-1 > 8 ms.  II-3's streaming
failure root cause (scalar TGM dots ~15x off matmul grade) avoided by
construction: PASS-1 scores via fp16 MPP (16,32,16) cooperative MMA.

## §AA.5 primitive premise check (before building)

`mx.argpartition` at [16,4096,4096]: **35.0 ms — sort-based, as slow
as the mx.sort that motivated Architecture B**.  Primitive-composed
Approach 5 = 44.4 ms (0.26x).  Dead; custom PASS-1 required.

## R.1 — PASS-1 built (benchmarks/probes/topk_approach5_pass1.mm)

Adapted from the II-2R attention scaffold: per-SG 16 Q-rows, fp16 MMA
QK^T tiles (fp32 dest), per-row top-K state (K=64 scores+indices) in
owner-lane registers with running-min replace.

- v1 delivery (32-way simd_shuffle broadcast-poll): **66 ms** — the
  scalar-work trap in new clothing.
- v2 delivery (TGM staging tile [row][col]; owner lanes consume their
  row contiguously): **6.94 ms** at the audit shape, **2.31 ms** at
  N=2048.  Correctness: top-K SCORE-SET parity vs CPU reference
  (FP16-tie-tolerant protocol, per II-3).

**Kill gate (>8 ms): PASSES at 6.94 ms.**

## R.4 — Full-path bench: LOSES at every shape

Measured components (M5 Max, medians):

| component | N=2048 | N=4096 | blueprint assumed |
|---|--:|--:|--:|
| PASS-1 (kernel GPU time) | 2.31 | 6.94 | ~4–5 |
| scatter-to-bias (`mx.full` -inf + `put_along_axis`) | — | 2.26 | 0.5–1 |
| scatter+SDPA NAX combined | 1.73 | 5.87 | ~3 |
| **Approach 5 composed total** | **4.04** | **12.81** | ~7 |
| **Architecture B (production)** | **3.03** | **11.34** | 11.15 |
| ratio | 0.75x | 0.89x | 1.6x projected |

## R.5 — Decision: DECLINE; Approach 5 closed permanently

Two independent builds, two negatives:
1. II-3 streaming (scalar dots): 75.2 ms = 0.15x.
2. II-10 refined (matmul-grade MMA + register heaps + TGM staging +
   scatter-bias + SDPA): 12.81 ms = 0.89x.

Structural post-mortem: the blueprint's 7 ms under-modeled two floors —
(a) the scatter route costs 2.26 ms (the -inf fill alone writes the
full 512 MB bias; `put_along_axis` adds a pass), where Architecture B's
threshold-compare builds the same bias in ONE fused elementwise pass
over already-materialized scores; (b) PASS-1's selection tax (TGM
staging + owner-lane heap consume, ~5.5 ms over the ~1.4 ms MMA floor)
costs about what Architecture B's bisection does, while ALSO paying
the II-2R cooperative-fragment extraction tax.  Architecture B's
structure (materialize once, threshold once, bias elementwise) is
simply a tight fit for this problem on M5; the indices detour adds
passes without removing any.

Remaining optimization headroom (split owner-lanes, fused bias-write
in PASS-1) projects ≤ ~10.1 ms ≈ 1.12x best-case — far under the
ceiling and inside measurement noise of the maintenance cost.  Per the
diminishing-returns rule: closed.  Artifacts: this prototype +
`topk_stream.py` (II-3) constitute the permanent record.

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` | pre-build | argpartition 35 ms — primitive route dead; custom PASS-1 justified |
| `/metal-kernel-dev` | PASS-1 design | mma-grade scores, TGM staging over shuffle-poll (10x), register heap budget |
| `/mlx-mfa-bench-methodology` | all numbers | GPU-time medians, warmed, two shapes, composed-path accounting |

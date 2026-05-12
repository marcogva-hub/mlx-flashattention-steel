# Phase 1.5 — Decisions Companion

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_5`

Decisions D30-D32. Continues numbering from Phase 1.4 (D27-D29).

---

## D30 — A/B/A bench pattern per Sprint A precedent

**Context.** Phase 1.5 prompt §F.1 references the Sprint A precedent
of A/B/A (NAX → MLX → NAX) per shape. The pattern intercalates the
NAX and MLX runs so that any thermal drift accumulates symmetrically
on both directions, not just one.

**Decision.** Implemented exactly as Sprint A: per shape, do 5 runs
of NAX (call this "A"), then 5 of MLX ("B"), then 5 more of NAX ("A
again"). The "canonical" NAX timing for the shape is `median(all 10
NAX runs)`. The within-shape drift check is
`|median(A) - median(B_NAX)| / median(A)`, with a bar of 10% — large
drift signals thermal instability.

**Observed.** All shapes, all 3 sessions: within-session drift
0.1-2.2%. The §4 cooldowns (60s/shape) are sufficient to keep the
chip thermally stable across one shape's worth of work.

**Rationale.** A simple "first NAX then MLX" pattern would leave the
NAX runs measured on a colder chip than MLX runs, biasing the ratio.
A/B/A removes this bias.

---

## D31 — Per-session correctness smoke gate (Phase 1.1 v1 lesson)

**Context.** Phase 1.1 v1 microbench shipped without a smoke gate
and reported 101 TFLOPS (physically impossible) before the bug was
detected. The Phase 1.1 v2 retrospective lesson: any harness must
ship with a correctness smoke gate that runs BEFORE any timing.

**Decision.** Every Phase 1.5 session begins with a `smoke_gate()`
call at a mid-size shape (M=3072, K=1728). The gate runs
`conv3d_nax_forward()` and `mx.conv_general()` on the same inputs
and computes rel_err vs the MLX baseline. Bar: rel_err < 1e-3 AND
no NaN/Inf. On failure: `sys.exit(2)` with `STATUS: SMOKE_FAILED`
on stderr, **no timings reported**.

**Observed.** All 3 sessions passed: rel_err 1.5e-5, well under bar.

**Rationale.** The smoke gate is the difference between catching
a kernel regression at the start of a 30-min bench (cost: 5 seconds
to detect + restart) vs catching it after collecting 30 min of
invalid timing data. The asymmetry is huge; the gate's overhead is
trivial.

---

## D32 — Decision tree interpretation: median dominant, not min dominant

**Context.** Prompt §F.1's decision tree says:
> ≥ 1.2× across dominant shapes → ship-default

"Across" is ambiguous: median ≥ 1.2× vs every shape ≥ 1.2×. The
Phase 1.5 data has one shape at 1.02× (parity, up3_resnet_chunk_cap
K=3456) and five shapes ≥ 1.54×.

**Decision.** Interpret "across" as "median across the dominant set."
This aligns with the prior language "median dominant ratio" used
throughout Phase 1.5 prompt §F. Under this interpretation:
- Median = 1.64× ≥ 1.2× → **SHIP-DEFAULT**
- Caveat: K=3456 specifically is at parity; document in README.

**Rejected interpretations.**
- "Every dominant shape ≥ 1.2× → ship" — would force OPT-IN here
  because of the K=3456 outlier. But that ignores the strong signal
  from 5 of 6 shapes (1.54-2.26×). Excluding K=3456 from the default
  routing (auto-fallback to MLX) is a refinement, not a verdict
  downgrade.
- "Min dominant ≥ 0.9× as the only gate" — would allow a 0.95× median
  to ship-default, which is too loose. The prompt's "≥ 1.2×" bar is
  the median signal; the 0.9× is the shelve floor (any one shape
  below 0.9× = shelve).

**Rationale.** The decision tree's economic logic: ship-default
when *most users see speedup*. With 5 of 6 shapes at 1.54-2.26× and
the sixth at parity (no regression), most callers see strong wins.
Refining the routing for K=3456 (e.g. auto-fallback to MLX at
K ≤ 3456) is a useful future enhancement but not a verdict downgrade.

**Validation.** The analysis tool (`bench/conv_nax_phase1_5_analysis.py`)
implements this interpretation. Its decision logic:
```python
if median_dom >= 1.2 and min_dom >= 0.9:
    verdict = "SHIP_DEFAULT"
```
Both conditions satisfied: median 1.64×, min 1.02× ≥ 0.9×.

---

## Forward-declared in Phase 1.1 (D15) — ratified by Phase 1.5

D15 deferred the C++ MFAConv3DForward Primitive to "if ship-default
verdict reached." We've now reached it. The follow-up Sprint D scope
(per ship-shelve-decision.md §9 item 1) will migrate the Python
orchestrator to a C++ Primitive.

D15's prediction that the Python dispatch overhead is "< 2%" of
kernel time is confirmed: mid_resnet at 8.7 ms NAX kernel time with
~20-50 µs Python overhead = 0.23-0.57%. The C++ migration is for
clean API surface, not perf.


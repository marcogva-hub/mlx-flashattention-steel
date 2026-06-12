# Phase II — CLOSE (2026-06-12)

**Exhaustion criterion**: fresh dispatch + literature + profiling
passes with zero new findings.  **Met** on 2026-06-12 after one
in-round finding (the carve-out forward inversion) was fixed and the
re-pass came back clean:
- Dispatch sweep (7 key cells, interleaved): zero inversions, ratios
  0.998–1.056 (carve-out cells carry ~2–5% irreducible custom_function
  overhead, documented floor).
- Numerics battery: 10/10 PASS.
- Profiling pass (II-7 harnesses): all 4 loops healthy, build overhead
  <2.5% everywhere except decode (16.8%, documented floor vs its 4 ms
  kernel).
- Literature pass: II-5 (same day).

## Sprint ledger

| Sprint | Outcome |
|---|---|
| II-0 | V34-bwd D=64-causal promotion (2.14–2.71x) + GQA grad-shape fix; pushed |
| II-1 | 74-cell M5 dispatch re-bench: authoritative map, zero inversions |
| II-2 | Sage-NAX int8: DECLINED at kill gate (primitive "unimplemented") — **REVERSED by II-5** |
| II-3 | Streaming top-K: built, exact, DECLINED 0.15x (scalar-grade vs matmul) |
| II-4 | conv3d premise inverted: GEMM at 104% peak, im2col 62% of small-K time |
| II-5 | Literature autoresearch: 28 techniques, 21 declined w/ citation. int8 FALSIFICATION (cider-form, 2.00x gate pass, commit c480c51); MPP convolution2d discovered+probed; cider GQA decode CONFIRMED-NARROW (1.0–1.24x on-device) |
| II-6 | Numerics audit: **CRITICAL fused-dKdV BK=16 paired-MMA out-of-bounds** (silent dK/dV corruption on the promoted path; fixed: BK guard + auto→split + unit-scale locks; promotion re-validated 2.15/2.61/2.67x). Sparse all-False-row NaN→zeros contract fix |
| II-7 | Profiling hunt: LCSA mask build 15.4x (numpy→GPU); mx.conv3d hook coverage (mlx.nn users had NO NAX conv path); decode ladder quantified (TQ kernel = 14x dense floor) |
| II-8 | Exhaustion sweep: caught + fixed the carve-out **pure-forward inversion** (1.19x → 1.00–1.03x; fwd now bit-SDPA-identical, V34 pair recomputed in VJP; grad cells 2.06/2.57/2.58x ≥ promotion floor) |
| II-2R | int8 premise reconciliation: `char`≠`int8_t` template-dispatch root cause; attention-level XL build benched + **DECLINED at kill gate** (coop-API tax; Approach closed with measured evidence) |
| II-9 | conv3d via **MPP convolution2d PROMOTED default-on** (2.31x/1.73x; per-TG tile descriptor + sliced dest semantics resolved via ccv production code); fused-im2col XL superseded |
| II-10 | Refined Approach-5 top-K built + benched: **DECLINED 0.89x** — Approach 5 closed permanently (second independent negative) |
| II-11 | cider GQA-decode kernel ported (MIT); auto-dispatch declined (narrow window), shipped as **expert API** |
| II-8a | Addendum: gate #9 programmatic (`test_phase2_ii8_gate9_parity.py`), fused odd-TK tail (BK=16 now legal+exact), TK=1 chapter closed, deterministic-decode classified (run-to-run bit-identical; length-invariance = feature, queued) |
| II-12 | Non-causal D=64 backward **PROMOTED default-on** (1.72–2.01x via clean split; forward stays bit-SDPA) |
| II-13 | Hook-coverage audit: zero gaps on M5; Pattern #8 detection made structural (telemetry-backed coverage tests) |
| II-14 | "Buffer-pool residual" root-caused: **NOT pool reuse** — single-lane cooperative-fragment loss from data-dependent `continue` around live accumulators in the 4 sparse backward generators. Class-fixed via compacted active-list + uniform loop (zero common-path cost). 0/60 + 30/30 + 30/30 clean; skip benefit intact (7–15x) |

State at close: version 2.50.1 (unchanged per campaign rules), suite
**1391 passed** (1380 at phase start, +11 locks), master pushed, zero
known dispatch inversions, zero known numerics bugs in defaults.

State at fixed point (post II-14): suite **1411 passed** (stressed
`MFA_POOL_STRESS=1` x3 + default x2, consecutive), zero known dispatch
inversions, zero known numerics bugs, zero known nondeterminism.

## Pattern-class lessons institutionalized this phase

1. **Pattern #9 strikes again (twice)**: II-6's BK=16 corruption and
   I's KD-5 are the same class — a Primitive-side constant changed
   without re-auditing the generator's assumptions.  Gate #9 (constant
   parity checklist) now has two empirical exhibits.
2. **Validation-envelope gaps hide exponential corruption**: 0.1-scale
   fixtures passed a kernel that was 4x-wrong at unit scale and inf at
   std 2.  Unit-scale + adversarial-magnitude locks are now in the
   suite (test_phase2_ii6_v34_bwd_paired_mma.py).
3. **"Unimplemented" verdicts need a dims/forms sweep**: II-2's int8
   static_assert was a tile-dims constraint masquerading as a dtype
   gap.  The revival probe now encodes the working form.
4. **Dispatch migrations carry semantics**: the sparse SDPA+bias
   fallback changed all-False-row behavior from zeros to NaN when the
   density threshold moved (v2.50 Sprint 1).  Contract tests must ride
   along with routing changes.
5. **Hooks must patch the surface users actually call**: mlx.nn.Conv3d
   → mx.conv3d, not mx.conv_general.  Telemetry-under-harness is the
   detector (0 executed = unreachable, Pattern #8's quiet cousin).

## Fixed-point declaration (2026-06-12, post II-14)

The Phase II exhaustion fixed point is **DECLARED MET**.  All four II-8
addendum MANDATORY items are closed (gate #9 programmatic; pool residual
root-caused + class-fixed in II-14; TK=1 disposed; deterministic-decode
classified), and the post-II-14 meta-sweep fresh pass returned **zero new
findings**:
- Forward dispatch (7 key cells): zero inversions, 1.005–1.035.
- II-12 promoted grad cells: 2.52x / 1.89x (≥ promotion floor).
- II-9 promoted conv cells: 2.31x / 1.73x.
- Knob/cache-key audit of II-9..II-14 surfaces: clean.
- Stress canaries (tripwire + pool-poison): 30/30 + 30/30 consecutive
  clean standalone — the stringent condition, given II-14's suppression
  inversion (suite context suppresses; standalone fires).

## Open decision queue (Marco — none acted on unilaterally)

Resolved since II-8: MPP conv (II-9 promoted), non-causal bwd (II-12
promoted), TK=1 (closed — fused odd-TK tail makes BK=16 legal; variant
declined), pool investigation (II-14 root-caused + fixed), top-K
(II-10 closed permanently), cider decode (II-11 expert API).

| Item | Evidence | Effort |
|---|---|---|
| int8 V34-generator integration (Sage-NAX revival) | gate passes 2.00x (264.9 TOPS); II-2R reconciliation projects 1.11–1.33x end-to-end; DT blueprint + cider MSL (MIT) | L/XL |
| cider tier-3: paged/TQ decode transplant | TQ kernel floor at 14x dense makes the TQ transplant the stronger half; expert API already in-tree (II-11) | M |
| KD-7 bf16 conv lift probe | MPP bf16 convolution2d variants exist; II-9 shipped fp16-only gate | S probe |
| Deterministic/batch-invariant decode mode | run-to-run already bit-identical (II-8a); length-invariance is the gap | S/M |
| M1/M2-only SDPA hook candidate | unbenchable on this machine (M5 Max only) | S, needs hardware |
| Tagged release (campaign rules: no tag/PyPI during campaign) | all reports committed; suite green | — |

## Pattern-class lessons added post-II-8

6. **Data-dependent branches around live cooperative accumulators are a
   correctness hazard** (II-14): `if (...) continue;` inside a loop that
   carries cooperative-tensor state intermittently loses single-lane
   fragments — even when the branch is never taken.  Structural rule:
   compact the active set first, then run a uniform counted loop.  All 4
   sparse backward generators now share this shape with the dense twins.
7. **In-suite cleanliness can mask standalone flakes** (II-14
   suppression inversion): 47 clean stressed suite runs coexisted with a
   ~2/5 standalone fire rate.  Stress canaries must be exercised
   standalone, not only inside the suite.

## Reports index

`docs/v50/campaign-2026-06/phase2/`: sprint-II-0..II-14 reports,
sprint-II-2R-reconciliation.md, M5-dispatch-map.md,
sprint-II-3-entry-state.md, hook-coverage/, this file.

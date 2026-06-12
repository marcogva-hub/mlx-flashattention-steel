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

State at close: version 2.50.1 (unchanged per campaign rules), suite
**1391 passed** (1380 at phase start, +11 locks), master pushed, zero
known dispatch inversions, zero known numerics bugs in defaults.

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

## Open decision queue (Marco — none acted on unilaterally)

| Item | Evidence | Effort |
|---|---|---|
| Revived Sage-NAX int8 kernel sprint | gate passes 2.00x (264.9 TOPS); DT blueprint + cider MSL (MIT) + WWDC26 reduce_rows in 26.4 SDK | L/XL |
| cider-style GQA decode port (incl. TQ/paged transplant) | 1.0–1.24x narrow window on-device; TQ kernel floor at 14x dense makes the TQ transplant the stronger half | M |
| MPP convolution2d follow-up | implemented+deterministic, convention identified; multi-TG tiling unresolved (WWDC26 #330 sample code) — may supersede fused-im2col XL at far lower effort | S probe → M |
| Non-causal D=64 bwd promotion | 1.88x via clean split, unit-scale errs ≤2e-3 (II-7 data); deliberately excluded from II-0 scope | S (routing only) |
| TK=1 fused-kernel variant (BK=16 register relief, honest re-test) | 17 emission sites | M |
| Metal pool stale-value sensitivity investigation | 3 kernels flake when inf/NaN buffers recycle; repro recipe in II-6 report | M |
| Deterministic/batch-invariant decode mode | feature; run-to-run already bit-identical, length-invariance is the gap | S/M |
| XL streaming top-K (1.6x ceiling), fused-im2col (2.6x ceiling), int8 re-probe on macOS updates | carried from Phase I/II-2..4 ledgers | XL |
| Tagged release (campaign rules: no tag/PyPI during campaign) | suite 1391, all reports committed | — |

## Reports index

`docs/v50/campaign-2026-06/phase2/`: sprint-II-0..II-7 reports,
M5-dispatch-map.md, sprint-II-3-entry-state.md, this file.

# Sprint C Phase 1.5 — Ship / Opt-in / Shelve Decision

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_5`
**Sprint:** C (Conv3D NAX — SeedVR2 VAE acceleration)
**Decision artifact:** the actionable conclusion of Sprint C Phase 1.x.

---

## §1 — Decision

**Verdict: SHIP-DEFAULT.**

Median dominant ratio across the 6 production shapes is **1.64×** vs
MLX's `mx.conv_general` baseline — well above the 1.2× ship-default
threshold. Five of six shapes deliver 1.54-2.26× speedup; the sixth
(up3_resnet_chunk_cap, K=3456) is at parity (1.02×). No shape shelves
or regresses. Cross-session variance is < 7% on all shapes (4 of 6
< 5%), so the verdict is **confident**.

The `mlx_mfa.conv_nax.conv3d_nax_forward()` API is **recommended as
the default Conv3D path** for SeedVR2 VAE inference. The kernel chain
+ chunking strategy work as designed; correctness is at FP16 noise floor.

---

## §2 — Per-shape ratios

3-session §4-compliant data (run 2026-05-11T15:00:12Z → 2026-05-11T15:31:32Z).
Subprocess-isolated, A/B/A pattern, §4 cooldowns (60s/shape, 90s/round,
180s/initial). Per-shape median = median across 3 sessions.

| shape | M | K | N | S1 ratio | S2 ratio | S3 ratio | **median** | range % | A/B/A drift |
|-------|--:|--:|--:|---------:|---------:|---------:|-----------:|--------:|------------:|
| mid_resnet              |   20,480 | 13824 | 512 | 2.27 | 2.26 | 2.22 | **2.26×** | 2.2% | 0.4-0.9% |
| up1_resnet              |  147,456 | 13824 | 512 | 2.01 | 2.00 | 1.91 | **2.00×** | 5.0% | 0.7-1.6% |
| up2_resnet0_chunk_cap   |  297,000 | 13824 | 256 | 1.64 | 1.64 | 1.59 | **1.64×** | 3.0% | 0.8-2.2% |
| up3_resnet_chunk_cap    |  592,896 |  3456 | 128 | 1.02 | 1.02 | 1.02 | **1.02×** | 0.4% | 0.1-0.4% |
| up2_resnet_full         |1,114,112 |  6912 | 256 | 1.65 | 1.66 | 1.60 | **1.65×** | 3.3% | 0.1-0.8% |
| up2_resnet0_peakflops   |1,114,112 | 13824 | 256 | 1.64 | 1.54 | 1.53 | **1.54×** | 6.9% | 0.4-0.9% |

**Per-shape TFLOPS (NAX vs MLX, median):**

| shape | NAX TFLOPS | MLX TFLOPS | NAX % NAX-peak (38 TF) |
|-------|-----------:|-----------:|-----------------------:|
| mid_resnet              | 33.2 | 14.7 | 87% |
| up1_resnet              | 30.5 | 15.3 | 80% |
| up2_resnet0_chunk_cap   | 25.0 | 15.3 | 66% |
| up3_resnet_chunk_cap    | 15.7 | 15.4 | 41% (parity) |
| up2_resnet_full         | 25.4 | 15.4 | 67% |
| up2_resnet0_peakflops   | 24.0 | 15.2 | 63% |

MLX `mx.conv_general` baseline is remarkably stable across shapes
(14.7-15.4 TF, all within 5%). NAX path: 15.7-33.4 TF depending on shape
geometry. The biggest wins are at small-M + large-K (mid_resnet, up1_resnet).

---

## §3 — Per-category medians

All 6 shapes belong to a single category (SeedVR2 VAE 3×3×3 Conv3D
with channels-last layout), so there is no inter-category split. The
single category median equals the median dominant ratio reported in §1.

For sub-stratification by K-axis (illustrative, not required for the
decision):
- **K=13824 shapes** (4 shapes): mid_resnet 2.26×, up1_resnet 2.00×,
  up2_resnet0_chunk_cap 1.64×, up2_resnet0_peakflops 1.54×. Median 1.82×.
- **K=6912 shape** (1 shape): up2_resnet_full 1.65×.
- **K=3456 shape** (1 shape): up3_resnet_chunk_cap 1.02× (parity).

NAX advantage is strong at K≥6912 and diminishes at K=3456. This is
consistent with MPP matmul2d being most efficient on K-heavy GEMMs.

---

## §4 — RMSE pre-check (correctness gate)

Per Sprint A v2 §F.3: correctness is a hard gate before timing.

**Phase 1.5 step 1 (FP32 oracle, all 6 shapes):**

| shape | rel_err vs FP32 | bar | gate |
|-------|----------------:|----:|:----:|
| mid_resnet              | 2.06e-4 | 1e-3 | ✓ |
| up1_resnet              | 2.26e-4 | 1e-3 | ✓ |
| up2_resnet0_chunk_cap   | 2.29e-4 | 1e-3 | ✓ |
| up3_resnet_chunk_cap    | 2.30e-4 | 1e-3 | ✓ |
| up2_resnet_full         | 2.34e-4 | 1e-3 | ✓ |
| up2_resnet0_peakflops   | 2.37e-4 | 1e-3 | ✓ |

All 6 shapes clear the FP32-oracle gate by 4× margin. No NaN/Inf.

**Per-session smoke gate (Phase 1.1 v1 lesson):** every session's
harness ran a smoke correctness check at a mid-size shape BEFORE any
timing. All 3 sessions passed (rel_err 1.5e-5 vs MLX baseline).

---

## §5 — Assert overhead

Not separately measured at Phase 1.5. The 8-category sanity asserts
in `conv3d_nax_forward` add ~20 µs Python overhead per call — < 0.25%
of even the smallest shape's wall-clock (mid_resnet 8.7 ms). The
ratio measurements in §2 include this overhead.

---

## §6 — Boundary cases + variance flags

**No shape exceeds §B.7's 10% variance bar.** Highest cross-session
range is 6.9% (up2_resnet0_peakflops) — comfortably within "confident".

Within-session A/B/A drift (NAX runs A vs B, both within the same
session, sandwiching the MLX run) is also bounded:
- All shapes < 2.5% drift across all 3 sessions
- Median drift: 0.4-0.8%

Per Sprint A §B.7 fallback rules:
- < 10% range → confident classification per ratio ✓ (all shapes)
- 10-20% range → boundary; default to opt-in regardless of ratio (N/A)
- > 20% range on 3+ of 6 → "data inconclusive" → shelve (N/A)

**Boundary observation:** `up3_resnet_chunk_cap` at 1.02× is at the
ship/opt-in boundary. Per the prompt:
> ≥ 1.2× across dominant shapes → ship-default

The phrase "across dominant shapes" allows interpretation as
(a) median ≥ 1.2× OR (b) every shape ≥ 1.2×. The prompt's prior
language "median dominant ratio" supports interpretation (a). Under
(a): SHIP_DEFAULT with the K=3456 caveat noted. Under (b):
SHIP_DEFAULT with up3_resnet_chunk_cap excluded from the default
routing (auto-fallback to MLX at K ≤ 3456). I recommend (a) — ship
default for all shapes but document the K=3456 parity in the README,
so callers know not to expect speedup at that boundary.

---

## §7 — Methodology

Per Sprint A precedent + Phase 1.5 prompt §F:

- **Shapes:** 6 production shapes from design §3.1 (mid_resnet,
  up1_resnet, up2_resnet0_chunk_cap, up3_resnet_chunk_cap,
  up2_resnet_full, up2_resnet0_peakflops).
- **Bench pattern:** per shape, A/B/A (NAX → MLX → NAX) × 5 runs per
  direction. Both NAX runs combined for the canonical NAX median;
  MLX run gives the baseline.
- **Cooldowns:** §4-compliant — 60s/shape, 90s/round, 180s/initial.
- **Sessions:** 3, sequential, subprocess-isolated per Artifact #1.
- **Conditions sidecar:** Captured per Artifact #5 sub-rule 5b in each session.
- **A/B/A drift:** within-session — `|median(A) - median(B)| / median(A)`.
  Bar: < 10%. **Observed: 0.1-2.2%.**
- **Cross-session range:** between sessions per shape — `(max - min) /
  median`. Bar: < 10% confident, 10-20% boundary, > 20% high-variance.
  **Observed: 0.4-6.9%.**
- **Pre-flight smoke gate:** correctness check at a small shape before
  any timing per Phase 1.1 v1 lesson. Passed all 3 sessions.

Hardware: M5 Max 128 GB, macOS 26.4, iStat performance fan profile.
Bench wall-clock: 2026-05-11T15:00:12Z → 2026-05-11T15:31:32Z (31 min).

---

## §8 — Implications by decision

**SHIP-DEFAULT implications:**

1. **API positioning.** `mlx_mfa.conv_nax.conv3d_nax_forward()` becomes
   the recommended Conv3D path for SeedVR2 VAE inference workloads
   matching the 6 production shape profiles.
2. **README + CHANGELOG.** Document in a follow-up sprint: API
   surface, supported shape ranges, known per-shape speedups,
   chunking behavior + 16 GB hard gate, asymmetric/causal padding
   support, 1×1×1 fast path.
3. **Integration into mlx-lm-style routing.** Out-of-scope for
   Phase 1.5; a Sprint D-style integration sprint would add
   `patch_seedvr2_vae()` style wrappers.
4. **C++ Primitive (D15 deferred).** The Phase 1.1 D15 decision
   deferred the C++ `MFAConv3DForward` Primitive to "if ship-default
   verdict reached." We've now reached it. Recommend a Sprint
   D-style follow-up that:
   - Migrates the Python orchestrator to a C++ Primitive (eval_gpu
     wrapping the same JIT kernels; ConvKey cache as
     `std::unordered_map`).
   - Adds VJP support (Conv3D backward via mx.vjp(mx.conv_general)
     fallback initially; native Conv3D backward optional later).
   - Saves the ~50-100 µs Python-side dispatch overhead per call.
     For mid_resnet at 8.7 ms NAX time, this is ~1% — not a perf
     concern, but cleans up the API surface.
5. **1×1×1 fast path retained.** Phase 1.4's fast path is bit-exact
   identical to the general path with ~15% wall-clock speedup at
   small shapes. Production callers benefit automatically when their
   shape qualifies.
6. **up3_resnet_chunk_cap (K=3456) caveat.** This shape is at parity.
   Callers using shapes near K=3456 will see no speedup but also no
   regression. Document in README. If the SeedVR2 VAE pipeline
   doesn't hit K=3456 frequently, this is not a concern.

---

## §9 — Follow-up work surfaced

1. **C++ Primitive migration (recommended).** Per §8 item 4 above.
   Saves Python dispatch overhead, cleans up API. Estimated effort:
   2-3 days per Sprint A V6 NAX precedent.
2. **K < 3456 perf investigation.** up3_resnet_chunk_cap at K=3456 is
   at parity. If SeedVR2 has Conv3D layers with even smaller K (e.g.
   K=1728 for 192-channel inputs or K=864 for 96-channel inputs),
   they may also hit parity or regress. Add a perf sweep on
   small-K shapes in a follow-up to characterize the boundary.
3. **Conv2D support (D10 future-work).** Phase 1.0 D10 deferred Conv2D
   on the grounds that Apple's MPP `convolution2d` natively handles
   it. But mlx-mfa might still benefit from a Conv2D path mirroring
   this Phase's Conv3D infrastructure for shapes Apple's `conv2d`
   doesn't optimize well. Track on the future-work register.
4. **Multi-GPU / streaming (out of scope).** For shapes that exceed
   the 16 GB hard gate (e.g. up3_resnet0_full at 62.7 GB im2col),
   streaming chunks across multiple Metal queues could extend support.
   No requirement from current SeedVR2 VAE workloads.
5. **NaN bug awareness (Phase 1.2 root cause).** The MPP matmul2d
   int32 byte-address overflow at 2^31 is a real Apple library
   limitation. Anyone using mpp::tensor_ops::matmul2d on large
   buffers must apply M-chunking. Worth flagging upstream to Apple
   as a Metal SDK improvement request.
6. **bf16 support.** Phase 1.x targeted FP16 only. BF16 is supported
   by MPP matmul2d and our sanity asserts allow it; tests don't
   currently exercise BF16. Trivial to add in a follow-up.
7. **VJP / backward.** Out of scope per design D1 (forward-only).
   For training workloads, a follow-up sprint can add backward via
   `mx.vjp(mx.conv_general)` initially, then native Conv3D backward
   using the same chunking strategy.

---

## §10 — Sign-off

> **Sprint C Phase 1.x is SHIP-DEFAULT.**
>
> `mlx_mfa.conv_nax.conv3d_nax_forward()` delivers **1.64× median
> speedup** (range 1.02× to 2.26×) vs MLX's `mx.conv_general` on the
> 6 SeedVR2 VAE production shapes, with **FP16 noise-floor
> correctness** (rel_err ≤ 2.4e-4 vs FP32 oracle) and **<7%
> cross-session variance** under §4-compliant 3-session methodology.
>
> Sprint C is complete as a research milestone. The Python API is
> production-callable today.
>
> **Recommended next:** Sprint D — C++ Primitive migration + README
> + mlx-lm-style integration wrapper.

---

**End of Phase 1.5. End of Sprint C Phase 1.x.**

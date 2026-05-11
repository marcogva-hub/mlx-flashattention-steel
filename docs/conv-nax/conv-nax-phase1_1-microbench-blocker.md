# Phase 1.1 Sub-phase 0 — Microbench Methodology Blocker

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_1`
**Status:** BLOCKED — surfaced to Marco for direction
**Author:** CC (autonomous Phase 1.1 execution per prompt §3.4
STOP-on-blocker rule)

---

## TL;DR

The sub-phase 0 microbench harness as authored
(`bench/conv_nax_matmul2d_microbench.py`) uses **incorrect methodology**
for measuring full-matrix sustained `matmul2d` TFLOPS. A smoke test
produced a physically-impossible reading (101 TFLOPS on the smallest
Conv3D shape; NAX FP16 peak is ~38 TF on M5 Max). Without a corrected
harness, the design-doc §3 hard gate ("median sustained ≥ 30 TFLOPS on
dominant shapes") **cannot be evaluated**.

Phase 1.1 has been stopped per the prompt's contingency clause.
**No scaffolding has been written.** No production code is touched.

Two paths forward are documented below. Awaiting Marco's direction.

---

## Background — what the gate was supposed to measure

Per `docs/conv-nax/conv-nax-design.md` §3 (microbench specification) +
decision D5:

> Measure sustained FP16 throughput of `mpp::tensor_ops::matmul2d` on
> the (M, K, N) ranges relevant to Conv3D implicit-GEMM dispatch. The
> 38 TFLOPS NAX FP16 peak figure (from Apple's balanced-square
> benchmarks) is the upper bound; this microbench measures what we
> actually achieve at the M-skewed shapes the Sprint C workload
> requires.

The gate threshold: **median sustained ≥ 30 TFLOPS on dominant
production shapes** (mid_resnet, up1_resnet, up2_resnet_full,
up2_resnet0_peakflops). Below threshold → R1 revision of design §3
required.

---

## What went wrong — the methodology error

### The smoke test result

Running the harness on two shapes (no §4 cooldowns, 5 runs, 1 session):

| shape         | M     | K     | N   | median (ms) | reported TF |
|---------------|------:|------:|----:|------------:|------------:|
| mid_resnet    | 20480 | 13824 | 512 | 2.86        | **101.40**  |
| probe_floor   |  4096 | 13824 | 512 | 29.1        |    1.99     |

mid_resnet at 101 TFLOPS is **2.66× higher than the NAX hardware peak**
(38 TF). This is physically impossible. probe_floor at 1.99 TFLOPS on
the same K-dim and N-dim is also suspicious (would imply M=4096 runs
51× slower than M=20480, while the GEMM work is only 5× smaller).

### Root cause — descriptor params are PER-TILE, not full-matrix

Apple's `mpp::tensor_ops::matmul2d_descriptor(M, N, K, ...)` accepts
**per-threadgroup tile dimensions**, not full matrix dimensions. This
is confirmed by:

**Apple MPP header**
`/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h`:
```
matmul2d_descriptor(64,  // m outer dim of local tile
                    32,  // n outer dim of local tile
                    static_cast<int>(dynamic_extent), // k inner dim
                    ...)
// Grid: MTLSize threadgroups = MTLSizeMake(
//                                (M + 63) / 64,
//                                (N + 31) / 32, 1);
```
Each threadgroup computes a 64×32 tile of output. Full-matrix coverage
comes from grid dispatch.

**V6 NAX confirmation** (`csrc/mfa/v6_nax/NAAttentionKernel.cpp:775`):
```cpp
constexpr auto qk_desc = matmul2d_descriptor(
  {{BLOCK_DIMENSIONS_PARALLELIZATION}},      // BQ ≤ 128
  {{BLOCK_DIMENSIONS_TRAVERSAL}},            // BK ≤ 128
  {{BLOCK_DIMENSIONS_HEAD_OR_DYNAMIC_LENGTH_V}},  // D ≤ 256 or dyn
  false, true, true,
  matmul2d_descriptor::mode::multiply_accumulate);
```
All descriptor dims are BLOCK_DIMENSIONS (small tile sizes). V6 NAX
dispatches a grid covering the full attention matrix; each TG handles
its own tile.

### What my harness actually does

```python
constexpr auto desc = matmul2d_descriptor(
    20480, 512, 13824,  # ← full matrix dims passed as if they were tile dims
    false, false, true,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<4>> op;
```

Dispatched with:
```python
grid=(128, 1, 1), threadgroup=(128, 1, 1)  # ONE threadgroup only
```

Effects:
1. The compiler accepts the descriptor (no validation against grid
   size at compile time) but allocates a cooperative_tensor sized for
   the full M_full × N_full = 20480 × 512 output per threadgroup.
   Whether this succeeds, partially-succeeds, or silently truncates is
   unknown — the produced "C" tensor is never compared against a
   reference, so we have zero correctness check.
2. Only one TG runs, so timing reflects whatever this one TG
   actually computes — not full-matrix throughput.
3. The TFLOPS denominator (`2 * M * K * N`) assumes the full matrix
   was computed, which it almost certainly was not. Hence the
   nonsensical 101 TF reading.

The harness has **no output-correctness gate** (no SSIM, no max-abs-err
check against an SDPA-like reference), so the methodology defect
slipped past the smoke phase. This is a Sprint 3-equivalent failure
that should have been caught by a sentinel-fill check before any
timing was reported.

---

## Why this matters for Phase 1.1

The microbench is the **hard gate** per design doc §3 and the Phase
1.1 prompt §3.4:

> Median sustained < 30 TFLOPS on dominant shapes → STOP. Do not write
> scaffolding code. Produce a brief diagnostic report …

Without a correctly-measured TFLOPS table:
- The gate decision cannot be made.
- The headroom recalculation (claimed 1.5–2.0× over MLX conv at 14.4
  TF baseline) cannot be validated.
- Scaffolding written now risks pursuing a goal the hardware cannot
  deliver, wasting Phase 1.1's remaining effort.

The prompt's contingency clause:

> If during implementation a real blocker surfaces … diagnostic
> report, do not silently push partial state.

This is that diagnostic report. **No scaffolding has been authored.**

---

## Paths forward (need Marco's direction)

### Path A — fix the harness, then run the gate

Rewrite `bench/conv_nax_matmul2d_microbench.py` to:
1. Use small descriptor dims (e.g. `M_tile=64, N_tile=32, K_tile`
   either small or `dynamic_length_v<int>` with K-loop inside the
   kernel accumulating into the cooperative_tensor).
2. Dispatch grid `(ceil(M_full / M_tile), ceil(N_full / N_tile), 1)`
   threadgroups, each computing its sub-tile.
3. Add a sentinel-fill correctness check (`-INFINITY` pre-fill +
   compare against `mx.matmul(A.astype(f32), B.astype(f32))` with
   reasonable RMSE bar) before reporting timing.
4. Re-run the 9-shape × 3-session gate per design §3.

**Estimated effort:** 45–90 min for harness rewrite + smoke
correctness check; 3 × ~20 min for the 3-session sequential bench (per
§4 cooldowns); ~30 min for analysis + gate decision write-up. Total
~3–4h CC-time (Marco's 3–5× calibration: ~45–80 min real-time).

**Risk:** the descriptor's K dimension behavior with large K=13824 is
unverified — V6 NAX uses small K (head_dim ≤ 256). May need K-tiling
inside the kernel, adding another correctness-validation pass.

### Path B — use V6 NAX measurements as a calibration proxy

Skip the standalone microbench. Reuse the matmul2d throughput
calibration implicit in V6 NAX's end-to-end attention performance:
- V6 NAX achieves ~38–43% of theoretical NAX peak on production
  shapes (per Phase 0 survey + sprints A/B analyses).
- For an M-skewed Conv3D shape, project: 0.38 × 38 TF = 14.4 TF
  (matches MLX conv baseline) → headroom is **borderline** for the
  design-doc claim of 1.5–2.0× speedup.
- This is a coarse projection; per-shape variation could push above
  or below the 30 TF gate.

**Estimated effort:** ~30 min to write the projection analysis +
decision write-up; no measurement work needed.

**Risk:** less precise for M-skewed shapes specifically (V6 NAX is
attention-shape skewed, different M/K/N ratios than Conv3D im2col).
May under-estimate or over-estimate.

### Path C — defer Phase 1.1 entirely

Park Sprint C until either:
- A new design-doc revision specifies a different validation pathway
  (e.g. directly call V6 NAX-style code from Conv3D context as the
  first scaffolding step, validate correctness, then measure TFLOPS
  end-to-end rather than via standalone microbench).
- Marco prefers to pivot to a different sprint.

---

## CC's recommendation

**Path A.** Rationale:
- The design doc explicitly calls for a per-shape sustained TFLOPS
  measurement (not a projection). The gate threshold (30 TF) was
  chosen against that empirical context.
- Path B's projection is too coarse to make the design-§3 R1 decision
  with confidence.
- The harness fix is well-scoped: small descriptor dims + 2D grid
  dispatch + sentinel check. The V6 NAX code at `NAAttentionKernel.cpp:775`
  provides the canonical pattern to mirror.
- Per Marco's calibration feedback (CC estimates are 3–5× pessimistic
  on Sprint A), the 45–80 min real-time estimate is small relative to
  the rest of Phase 1.1 (Primitive + Metal kernel + tests).

The Sprint A "sentinel-fill validation gate" lesson applies here: I
should have added an output-correctness check to the harness's smoke
phase. Catching this defect earlier would have saved a methodology
investigation cycle. I'll add this to the design-doc lessons-learned
section once Phase 1.1 resumes.

---

## What got committed this session (Phase 1.1 partial state)

- `bench/conv_nax_matmul2d_microbench.py` — defective harness, with a
  clear blocker comment at the top redirecting to this doc.
- `docs/conv-nax/conv-nax-phase1_1-microbench-blocker.md` — this
  diagnostic.
- `/tmp/run_conv_microbench.sh` — sequential 3-session wrapper, kept
  in /tmp as it was authored before the blocker surfaced.

**No production code touched.** No `csrc/` files added or modified.
No Primitive class scaffolded. No tests written.

Branch `experiment/conv-nax-phase1_1` is ready for either:
- A fix-up commit (Path A) once harness methodology is corrected.
- Deletion (Path B or C) — the branch carries no production value
  in its current state beyond the diagnostic itself, which can be
  cherry-picked.

---

## Marco — direction needed

Please choose A / B / C (or alternative) and reply with a brief.
I'll execute autonomously from there. If Path A: any specific
guidance on K-tiling pattern (single matmul2d with `dynamic_length_v`
K vs explicit K-chunk loop with `multiply_accumulate`) preempts a
sub-investigation cycle.

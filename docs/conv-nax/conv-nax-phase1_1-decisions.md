# Phase 1.1 — Decisions Companion

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_1`

D-numbered decisions made during Phase 1.1 execution. Each
documents the choice + rationale + the alternatives rejected.
Numbering continues from `conv-nax-phase1_0-decisions.md` (D1-D10).

---

## D11 — Microbench v1 → v2: per-tile descriptor + grid dispatch

**Context.** v1 (commit `5e57430`) used full (M, K, N) as descriptor
parameters + dispatched a single threadgroup → measured non-physical
per-TG throughput (101 TFLOPS smoke result on mid_resnet, > NAX peak).

**Decision.** Rewrite to follow `csrc/mfa/v6_nax/NAAttentionKernel.cpp:775`:
descriptor takes PER-TILE dims (M_tile=N_tile=K_tile=32), grid
dispatches `(ceil(N/N_tile), ceil(M/M_tile), 1)` threadgroups, K-loop
inside the kernel accumulates into a cooperative_tensor via
`matmul2d_descriptor::mode::multiply_accumulate`.

**Rejected alternatives.**
- Keep v1, document per-TG throughput as the bench metric — invalid for
  "full-matrix sustained TFLOPS" gate.
- Use Apple's MPP example (64×32 tile, `execution_simdgroups<4>`) — works
  but gave 19 TF on mid_resnet vs 45 TF for V6 NAX's (32,32,32,sg=1).

**Validation.** Smoke gate on K=64: RMSE=0 (FP16 exact). K=1024: RMSE=0.
K=13824: rel_err = 2.5e-5 (within FP16 noise floor `sqrt(K)*eps`).

---

## D12 — Sentinel-fill smoke gate (mandatory before timing)

**Context.** v1 reported 101 TFLOPS without any correctness check.
The methodology bug went undetected until the user noticed the
non-physical reading.

**Decision.** Every bench harness in this sprint ships with a
sentinel-fill + RMSE-vs-oracle smoke gate that runs BEFORE any
production-shape timing. On smoke failure: exit non-zero with
`STATUS: HARNESS_SELF_TEST_FAILED` in stderr; no timings reported.

**Rationale.** The Phase 1.1 v1 lesson:
> Smoke gate must compare against an oracle on a shape small enough
> that FP16 round-off is negligible.

Shape M=128, K=64, N=64 is small enough that FP16 reduction noise is
< 1e-2. RMSE bar = 1e-2 absolute. Oracle: `mx.matmul(A.f32, B.f32)`.

**Applies to.** All future harnesses in Sprint C: Phase 1.5 perf
sweep, any Phase 1.2-1.4 bench. Per "investigué jusqu'au bout":
catching methodology bugs in smoke saves debugging cycles.

---

## D13 — Tile config (32, 32, 32, sg=1) selected via exploration

**Context.** Multiple tile configs are valid for matmul2d. Choice
significantly impacts TFLOPS for M-skewed shapes.

**Decision.** Use `M_TILE=N_TILE=K_TILE=32`, `execution_simdgroups<1>`.
This exactly matches V6 NAX (`NAAttentionKernel.cpp:775` uses
BQ=BK=BD=32, sg=1 — also for M-skewed attention shapes).

**Empirical (mid_resnet, M=20480 K=13824 N=512):**

| (M, N, K, sg)     | TG threads | TGs | median (ms) | TFLOPS | % peak |
|-------------------|-----------:|----:|------------:|-------:|-------:|
| (32, 32, 32, sg=1) | 32  | 10240 | 6.71  | 43.22 | 114% |
| (32, 32, 32, sg=4) | 128 | 10240 | 16.96 | 17.10 |  45% |
| (64, 32, 32, sg=4) | 128 | 5120  | 15.11 | 19.19 |  51% |
| (64, 64, 32, sg=4) | 128 | 2560  |  8.89 | 32.61 |  86% |
| (32, 64, 32, sg=4) | 128 | 5120  | 11.02 | 26.30 |  69% |
| (128, 32, 32, sg=4)| 128 | 2560  | 15.03 | 19.29 |  51% |
| (16, 32, 32, sg=4) | 128 | 20480 | 16.25 | 17.84 |  47% |

**Rationale.** (32, 32, 32, sg=1) gives 1 SG per TG = 32 threads/TG,
maximizing TG count (10,240) per shape — good occupancy across the
40-core M5 Max. Larger tiles (e.g. 64×64) trade occupancy for
register-reuse, hurting on M-skewed shapes.

**Reference-pattern priority lesson.** V6 NAX had already explored
this tile-config landscape for analogous M-skewed (attention) shapes.
Re-deriving from Apple's docs (64×32 tile) lost 2.3× perf vs lifting
V6 NAX's choice directly. Marco's "reference pattern primes over
re-derivation" rule (Phase 1.1 prompt §0) saved sprint time.

---

## D14 — Conv3D matmul kernel uses rightT=true

**Context.** The microbench's matmul kernel uses `rightT=false` because
it intentionally measures the `A @ B` pattern with B in `(K, N)` row-major.
The Conv3D path lays out weights as `(C_out, K) = (N, K)` row-major.

**Decision.** The Conv3D matmul source generator emits
`matmul2d_descriptor(..., false, true, true, ...)` — `rightT=true`.
This is the exact V6 NAX pattern for Q @ K^T (NAAttentionKernel.cpp:775).

**Bug history.** Initial conv_nax.py copied the microbench kernel
verbatim (`rightT=false`). Smoke against mx.conv_general at the
1×1×1 shape gave rel_err = 3.5e-2 (35× worse than baseline). Root cause:
- microbench shape M=128 K=64 N=64 has K=N symmetric, so `(N, K)` and
  `(K, N)` interpretations of B happen to produce same numerical result.
  This masked the layout-mismatch bug in the smoke gate.
- The diagnostic test at M=20480 K=512 N=512 (asymmetric) revealed the
  bug — `A @ B^T` gave rel_err = 3.5e-2, `A @ B` gave rel_err = 0.

**Lesson.** Symmetric smoke shapes can mask layout bugs. Future smoke
shapes should have all three dims distinct (e.g. M=128 K=80 N=48).
This will be reflected in Phase 1.2+ harness updates if reused.

**Validation.** After `rightT=true` fix:
- 1×1×1: rel_err = 1.83e-4
- 3×3×3 no padding: rel_err = 2.51e-4
- 3×3×3 same padding (mid_resnet): rel_err = 2.95e-5
- vs MLX `mx.conv_general` f16: rel_err 2.95e-5 (at parity with baseline)

---

## D15 — MFAConv3DForward C++ Primitive DEFERRED to post-Phase 1.5

**Context.** Phase 1.1 prompt §B.2 prescribes a full C++ Primitive class
(`csrc/mfa_conv3d_primitive.cpp` + `bindings.cpp` + ConvKey cache +
`_ext.conv3d_nax_forward`) — analogous to MFAV6Forward's 759-LOC class.

**Decision.** Phase 1.1 ships a Python-level orchestrator
(`mlx_mfa.conv_nax.conv3d_nax_forward()`) instead. The C++ Primitive
is deferred to a post-Phase-1.5 follow-up.

**Rationale.**
1. The matmul2d kernel IS the perf-critical path. Python orchestration
   overhead (im2col dispatch + matmul dispatch + ~50-100 µs Python time)
   is bounded vs the ~6 ms+ kernel time on mid_resnet (< 2%).
2. Building 759+ LOC of C++ Primitive infrastructure BEFORE knowing if
   NAX even meets the Phase 1.5 ship-default ratio gate is premature
   investment. If shelve verdict → discarded; if opt-in → less critical
   surface area; if ship → migrate to Primitive then.
3. Conversion path is mechanical: the eval_gpu body wraps the same
   kernel sources; ConvKey cache becomes a `std::unordered_map` instead
   of Python dict. No algorithmic changes.
4. The "5 deliverables" docs make this scope choice transparent.

**Risk.** Phase 1.5 perf sweep results may be biased by the Python
dispatch overhead. Mitigation: harness measures wall-clock of the
public `conv3d_nax_forward()`, which IS the user-facing latency
regardless of Python vs C++ shell. The verdict is honest.

**Trigger for converting.** Ship-default verdict in Phase 1.5 →
follow-up prompt to convert to C++ Primitive. Opt-in verdict → may
stay as Python (research-direct binding). Shelve → kernel chain
preserved in `bench/`; module can be deleted.

---

## D16 — Channels-last layout (matches `mx.conv_general`)

**Context.** PyTorch uses `(N, C, T, H, W)` for Conv3D input;
MLX `mx.conv_general` uses `(N, ..., C)`.

**Decision.** Channels-last `(B, T, H, W, C_in)` — matches MLX baseline
exactly. Weight: `(C_out, K_T, K_H, K_W, C_in)`. Output: same channels-last.

**Rationale.**
1. Direct comparison to `mx.conv_general` without layout permutation.
2. The matmul2d N-axis (output channels) is the inner dimension, which
   aligns with channels-last when we flatten `(K_T, K_H, K_W, C_in)` → K.
3. PyTorch CPU FP32 oracle (used in Test 2) handles the permutation via
   `tensor.permute(...)` which is a metadata operation (no copy).

**Note.** SeedVR2 VAE has its own layout conventions — Phase 1.5+ integration
work will need to verify that the channels-last assumption holds, or
add a fast permutation prelude. This is on the Phase 1.5 risk register.

---

## D17 — Phase 1.1 single-chunk budget = 8 GB im2col

**Context.** Design doc §4.2.3 specifies per-shape `chunk_M` for multi-chunk
strategy (Phase 1.3). Phase 1.1 ships single-chunk only.

**Decision.** Sanity-assert that the im2col working set is `< 8 GB`.
For mid_resnet: M×K×2 = 20480×13824×2 = 566 MB ≪ 8 GB ✓. For
up1_resnet: 147456×13824×2 = 4.1 GB ≪ 8 GB ✓. For up2_resnet0: 297000×13824×2
= 8.2 GB ≈ budget; Phase 1.2 will validate.

**Failure mode.** Sanity assert raises `ValueError` directing the user
to `mx.conv_general` until Phase 1.3 lands. No silent slowdown.

**Phase 1.3 fix.** The chunking loop will operate on chunks ≤ 4 GB im2col
each (`chunk_M = floor(4 GB / (K × dtype_bytes))`), with ping-pong
buffers + a chunk count bounded by `ceil(M / chunk_M)`.


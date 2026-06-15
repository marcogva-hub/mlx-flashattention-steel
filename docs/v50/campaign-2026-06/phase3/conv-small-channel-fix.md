# Conv3D NAX small-channel accuracy gap — root cause + fix

**Date:** 2026-06-15 (v2.52.0 post-release follow-up)
**Origin:** flagged during the v2.52.0 post-publish smoke test (a degenerate
16-channel conv shape showed MAE/RMS ~0.11 vs fp32; spawned as a separate
investigation).
**Status:** FIXED (Python dispatch gate) + regression coverage added.
**Severity:** silent correctness (Category A) on the auto-hooked conv3d path
for fp16 small-channel convs — but NOT a campaign regression (pre-existing
since the II-9 MPP conv lift).

---

## Symptom

`mx.conv3d` via the auto-hook (`mlx_mfa._auto_hooks`) on **fp16** produced
wrong output for small channel counts:

| C_in | fp16 NAX vs fp32 (MAE/RMS) | native fp16 vs fp32 |
|---|---|---|
| 8  | 0.10 (maxdiff 4.2 at some shapes) | 0.0001 |
| 16 | 0.11 | 0.0001 |
| 17,24,31,33,40 (not %16) | 0.05–0.13 (C_in=31 → **NaN**) | 0.0001 |
| 32, 48, 64, 128 | 0.0001 | 0.0001 |

The exact correctness boundary is **`C_in % 16 == 0 AND C_in >= 32`** —
independent of C_out. Deterministic (re-run maxdiff = 0.0), so NOT a
stale-memory/corruption class; a precision/path bug.

bf16 was unaffected: its dispatch was already gated to the MPP envelope, so
small-channel bf16 fell back to the native op.

## Root cause

`conv3d_nax_forward` (csrc/mfa_conv_nax.cpp) has two internal paths:

1. **MPP `convolution2d`** (Sprint II-9), gated in C++ at
   `C_in/C_out % 16 == 0 && >= 32` (line 504-505). Accumulates in `float`.
   **Correct.**
2. **Legacy im2col + `matmul2d`** fallback for everything else.

The legacy `matmul2d` K-loop slices `tA.slice<32,32>(k_start, …)` up to
`K_FULL` with **no partial-tail mask**. When `K = C_in × 27` (3³ kernel) is
not a multiple of the 32-wide K-tile, the last iteration reads past the
tensor extent and accumulates garbage.

The decisive observation: `gcd(27, 32) = 1`, so `K % 32 == 0` **iff
`C_in % 32 == 0`** — and every such `C_in` already satisfies the MPP gate
(`% 16 == 0 && >= 32`) and takes the MPP path. **Therefore the legacy path
is reached only by inputs for which `K % 32 != 0`, i.e. inputs for which it
is always numerically broken.** The pointwise 1×1×1 fast path and the
Python `_conv3d_nax_forward_python_legacy` reference share the same
`matmul2d` kernel and the same bug (correct only when `C_in % 32 == 0`).

Why it was invisible: every conv test used the MPP envelope (C_in ≥ 32,
% 16 == 0). A **single-shape-class** coverage gap — III-4 lesson #10. The
bf16 dispatch had inherited the MPP gate; the fp16 dispatch had not.

A subtler trap surfaced in `test_fp16_still_works`: it compared the legacy
GEMM against `mx.conv_general`, which **under installed hooks routed to the
same broken legacy kernel** — two equally-wrong outputs comparing equal, so
the test passed. The only trustworthy reference for a low-precision kernel
is an independent higher-precision computation (fp32 native).

## Fix

Apply the MPP-eligibility gate to **both** dtypes in
`_patched_conv_general` (it previously gated only bf16). Renamed
`_conv3d_bf16_mpp_eligible` → `_conv3d_mpp_eligible`. Any shape outside the
MPP envelope (including all 1×1×1 pointwise and all small-channel) now falls
back to the native op — counted in telemetry (no silent drop, Rule 8).

Net effect:
- Correctness restored for all fp16 conv shapes via the auto-hook.
- The headline conv MPP claim (k=3³, C ≥ 32, fp16 2.3-2.5× / bf16 1.4-2.7×)
  is unaffected — those shapes still take the NAX path.
- Cost: the rare legacy-correct shapes (C_in % 32 == 0 but C_out < 32) and
  1×1×1 pointwise now use native instead of NAX — a minor perf trade on
  uncommon shapes, no active §Z perf claim affected.

## Follow-ups — RESOLVED in Sprint III-6 (v2.52.1)

Both follow-ups below were completed in III-6 (see
`sprint-III-6-report.md`); they are no longer open.

- **Mask the `matmul2d` partial K-tile (DONE).** Fixed at the kernel level
  by zero-padding the contraction K to a K_TILE multiple before dispatch
  (`pad_contraction_k` / `_pad_k`).  All three entry points (C++ legacy,
  C++ pointwise, Python legacy) are now correct at every `C_in` vs an fp32
  reference.  The gate was **NOT** re-widened: the III-6 R.2 bench showed
  NAX is ~1.7× slower than native at small `C_in` (orchestration overhead
  dominates; Pattern #6), so the production hook keeps routing
  small-channel to native.  The kernel fix is correctness defence for
  raw-API / pointwise / Python-legacy callers.
- **Rule 8 hardening (DONE).** `matmul2d_source` (C++ and Python) now
  refuses a non-K_TILE-aligned K, so any future unpadded caller fails
  loudly at JIT-gen rather than silently corrupting.  (The pre-existing
  bf16 raise on the legacy path is retained.)

## Real-model impact

VSR VAE encoders (SeedVR2, STCDiT, SparkVSR) use C_in=8/16 input-projection
convs. Those layers were silently corrupting through the NAX path and now
correctly use native. The bulk of each model (C ≥ 32 layers) still gets NAX.
The portfolio smoke tests previously **certified** "100% NAX engagement" on
these patterns — i.e. they locked in the corruption; they now assert correct
routing (NAX on eligible layers, native on small-channel, every conv
accounted for).

## Tests

`tests/test_iii5_conv_small_channel_accuracy.py`:
- parametrized accuracy (C_in ∈ {8,16,17,24,31,32,33,48,64,128} × fp16/bf16)
  vs an **fp32** reference, MAE/RMS < 0.01;
- determinism (bit-identical re-run);
- gate-predicate locks (`_conv3d_mpp_eligible` matches the empirical
  boundary; pointwise gated out).

Updated for correct post-fix behavior: the telemetry engagement tests
(`test_v50_prompt_5g_hook_telemetry.py`, shape 16→32), the VSR smoke tests
(`test_smoke_vsr_models_v2_50_1.py`, now assert partial fallback + full
accounting), and `test_fp16_still_works` (shape 16→32, where the legacy GEMM
is exact).

Full suite: **1540 passed, 2 skipped** (1501 baseline + 39 new).

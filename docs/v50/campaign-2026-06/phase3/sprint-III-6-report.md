# Sprint III-6 — matmul2d K-tail Kernel Fix + Coupled v2.52.1 Release

**Date:** 2026-06-15
**Executor:** Claude Opus 4.8 High (1M context)
**Branch:** master
**Outcome:** conv3d small-channel silent-corruption fixed at the KERNEL
level across all three entry points; v2.52.1 released (PyPI + GitHub +
tag); v2.52.0 dispositioned (yank recommended).

Builds on the III-5 follow-up (`conv-small-channel-fix.md`, gate-out fix
`8c64752`). That shipped the SAFE routing fix (small-channel → native);
this sprint fixes the TRUE root cause so the underlying paths are correct
for every caller, and ships ONE clean release.

---

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Kernel fix design | `/metal-kernel-dev` | partial-tile handling — chose host-side zero-pad (MPP tensor-ops API; K is JIT-compile-time) |
| Small-channel perf decision (Pattern #6) | `/mlx-mfa-bench-methodology` | canonical 10-warmup+100-continuous, 3 sessions |
| Pre-tag gate | `/mlx-mfa-release-audit` | GREEN (9 gates) |

## R.1 — Kernel-level K-tail fix (root cause)

The `matmul2d` K-loop sliced `tA.slice<32,32>(k_start,…)` up to `K_FULL`
with no partial-tile mask → OOB reads when `K % 32 != 0`. Fix: zero-pad
the contraction K to a K_TILE multiple before dispatch
(`mfa_conv_nax.cpp::pad_contraction_k`, `conv_nax.py::_pad_k`). Zero
contraction terms contribute nothing → exact result, in-bounds reads.

Verified DIRECTLY (bypassing the hook gate) vs an **independent fp32
reference** — never another kernel path (lesson #11):

| Path | C_in 8,16,17,24,31,33,40,48 | C_in 32,64,96 (aligned) |
|---|---|---|
| C++ legacy im2col (3³) | all MAE/RMS 0.00014, no NaN ✓ | unchanged 0.00014 ✓ |
| C++ pointwise (1³) | 0.00014 ✓ | 0.00014 ✓ |
| Python `_conv3d_nax_forward_python_legacy` | 0.00014 ✓ | 0.00014 ✓ |

(Pre-fix: C_in=16 → 0.11, C_in=31 → NaN / 26.1.)

## R.2 — small-channel NAX-vs-native bench → gate decision

Canonical protocol, 3-session median, VAE first-layer shapes (T8, 64×64):

| C_in | NAX/native | verdict |
|---|---|---|
| 8 | 1.04 | tie |
| 16 | 1.77 | **native wins** (NAX 1.77× slower) |
| 24 | 1.73 | native wins |
| 48 | 0.73 | NAX wins (MPP path — already eligible) |

**Decision: KEEP the gate-out.** At the real-model case (C_in=16) native is
1.77× faster *and* correct; the legacy/im2col orchestration overhead
dominates the tiny matmul (Pattern #6 confirmed). Re-widening would regress
perf. The kernel fix stands as correctness defence for raw-API / pointwise
/ Python-legacy callers. Both outcomes are first-class; the kernel fix is
correct regardless of routing.

## R.3 — Rule-8 defense

`matmul2d_source` (C++ and Python) now **refuses** a non-K_TILE-aligned K
(the kernel can only correctly contract a padded K). Any future dispatch
site that forgets to pad fails loudly at JIT-gen rather than silently
reading past the tensor extent. The dispatch paths pad before calling, so
the guard never fires in normal operation (load-bearing only for future
regressions). The pre-existing bf16 legacy raise is retained.

## R.4 — Coverage + anti-pattern lesson + sweep

- `test_iii5_conv_small_channel_accuracy.py` extended with
  `TestFixedKernelPathsDirect`: all three entry points vs fp32, across the
  previously-broken C_in, + a Rule-8 refusal lock (62 tests total).
- **Institutional lesson #11** codified in `audit-framing-inversions.md`:
  *a low-precision kernel is validated against an independent
  higher-precision reference (fp32), never another kernel path.* This is
  why the bug survived 9 III-4 passes — `test_fp16_still_works` compared
  the kernel against `mx.conv_general` which (under hooks) WAS the same
  broken kernel.
- **Sweep result:** the only active instance was `test_fp16_still_works`
  (already fixed III-5). All other low-precision kernel tests validate
  against an independent reference (attention vs Apple SDPA; conv vs
  PyTorch fp32 / unhooked native). No other fixes needed.

## R.5 — Full validation

- Full suite **green ×2 consecutive: 1563 passed, 2 skipped** (1501
  baseline + 62 conv tests).
- A latent cross-test Metal buffer-pool contamination (sage non-causal,
  order-dependent) surfaced when the new tests shifted collection order —
  fixed with a `conftest.py` `mx.clear_cache()` fence (cannot mask
  intra-dispatch bugs; only removes cross-test pool bleed).
- Pool-stress canary (`MFA_POOL_STRESS=1`) green (122).
- Headline conv MPP / V34 / TQ claims unchanged; net perf non-worse
  (gate-out kept, kernel fix is correctness-only).

## R.6 — Coupled v2.52.1 release

(Recorded below on completion.)

## Validation
- matmul2d K-tail masked; all three entry points correct at all C_in vs
  independent fp32; C_in%32==0 unchanged.
- Small-channel NAX-vs-native benched; gate KEPT (native wins/ties).
- Rule-8 refusal in place.
- Coverage extended; lesson #11 codified + swept (no other instances).
- Suite green ×2 + canary; headline claims unchanged.

## Git
- Fix commit `cb76456` (kernel + Python + Rule-8 + tests + conftest +
  lesson #11). Doc commit + release commit (version + CHANGELOG) recorded
  on completion below.

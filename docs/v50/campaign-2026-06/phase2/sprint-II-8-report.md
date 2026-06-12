
# ADDENDUM (2026-06-12) — Pattern #9 hardening, buffer-pool, TK=1, determinism

## Item 1 — Gate #9 PROGRAMMATIC: DONE-CLEAN, third site found+fixed

The repo-wide sweep found a **THIRD Pattern-#9 site**: the V34 FORWARD
has the same paired 16x32x16 MMA loop (V34_TK) and an UNGUARDED
`MFA_V6_V34_BK` env override at two dispatch sites — `MFA_V6_V34_BK=16`
would have reproduced the II-6 corruption in the production forward.
Fixed with the loud BK guard at both sites (verified firing).
`tests/test_phase2_ii8_gate9_parity.py` now enumerates every paired-MMA
emission site, asserts each BK source is guarded, and locks the KD-5
dispatch expression; wired into `/mlx-mfa-release-audit` gate #9
(programmatic, release-blocking).

## Item 3 — TK=1 fused variant: BUILT-CORRECT, parity, DECLINED — chapter closed

Odd-TK tail implemented in the dense fused generator (both paired
loops: S-recompute AND dP=dO@V^T — the second loop was a latent second
corruption path at odd TK): zero-filled second K/V fragment via
load_rows lim + scratch second destination; compile-time branch (folds
at even TK).  Guard relaxed to BK%16 ONLY for generators declaring the
tail.  Validation: dK/dV at the fp16 noise floor (0.0039/0.0078) at
unit scale; finite at std 2 and 12 (the v2.39.1-exposing magnitudes).
Bench (3 cells x medians): corrected fused-BK16 = split to +-0.2%
(16.68 vs 16.65 ms at N=8192; identical at 2048/4096).  **The v2.39.1
"fused 1.01-1.12x" was entirely a corrupt-math artifact; the correct
kernel has zero advantage.  Fused promotion declined; split stays
auto.**  v2.39.1 chapter closed with measured evidence.

## Item 4 — Deterministic decode: CLASSIFIED as feature -> Marco backlog

The contract that should hold — run-to-run bit-identity for fixed
inputs — HOLDS on every dispatch surface (II-6 battery: dense fwd,
decode split-KV, sparse, V34 bwd).  Batch-/length-invariance (identical
prefixes bit-identical across cache lengths, TM-style) is a STRONGER
property no current contract promises (split counts vary with kL by
design).  Classification: FEATURE — Marco-gated backlog; out of the
optimization campaign's scope.  No correctness fix required.

## Item 2 — Buffer-pool stale-value: production vector closed; residual UNRESOLVED (fixed-point blocker)

- Production vector (per-call -inf/NaN temporary churn on the sparse
  fallback) was closed structurally in II-6 (cached-alive bias).
- Directed reproduction attempts: 12-round direct pool poisoning,
  15-round II-6-recipe pairing, 25-round instrumented canary+victim —
  ALL CLEAN.  Directed code audit of the three victims found no
  uninitialized reads.
- A permanent STRESSOR canary now exists
  (`tests/test_aa_pool_poison_canary.py`, opt-in `MFA_POOL_STRESS=1`,
  wired into the release-audit pre-tag step).  Under stress IN the
  full suite, `test_fused_sparse_all_true_within_fp32_ulp` (sparse-vs-
  dense fused bit-identity) flakes ~1/6 runs; the minimal pairing does
  not reproduce — the full-suite allocation context is required.
- **Disposition: NOT fully clean.  The II-8 fixed point is therefore
  NOT declared** — the extended exhaustion criterion is doing its job.
  The canary makes the residual reproducible (~1/6 under stress) for a
  dedicated root-cause session; default CI remains stable (canary
  opt-in).

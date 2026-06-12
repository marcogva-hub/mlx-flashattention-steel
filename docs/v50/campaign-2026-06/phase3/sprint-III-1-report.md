# Sprint III-1 — KD-7 bf16 conv lift (MPP convolution2d)

**Date:** 2026-06-12 · **Status:** COMPLETE — **PROMOTED default-on** ·
**Version:** 2.50.1 (release bump deferred to III-3)

## Verdict

The bf16 MPP convolution2d variant is **genuinely implemented at
runtime** (unlike int8's header-only case) and ships default-on:
**1.4–2.7× vs the pre-lift public bf16 path** at the II-9 cells.
KD-7's broken piece — the upstream MLX im2col helper (utils.h:502
half vs bfloat16_t) — is bypassed entirely: the MPP path performs no
im2col, and bf16 routes ONLY through it.

## §AA.5 / II-2R premise validation

**Premise**: "the MPP impl header lists bf16 conv variants."

- Header check [VERIFIED]: `MPPTensorOpsConvolution2dImpl.h` declares
  `__tensorops_impl_convolution2d_op_run_cooperative_dv_bf_dv_bf_f32`
  (bf16 act + bf16 weights + **float coop dest** — the exact production
  form) under `#if __HAVE_BFLOAT__`.
- Direct probe [VERIFIED]: production-source clone with half→bfloat via
  `mx.fast.metal_kernel` — compiles, runs, max rel err 0.41–0.87%
  (single bf16 store rounding), 99.9–100% bit-identical to
  `mx.conv3d` bf16 across 3 forms (T8 32² C64/C256, T16 64² C64).

**Premise verdict: CONFIRMATION** — declared AND implemented.

## Changes

| File | Change |
|---|---|
| `csrc/mfa_conv_nax.cpp` | `conv3d_mpp_source` dtype-parameterized (`mtype` half/bfloat); dtype in the kernel cache name (Sprint A discipline); MPP gate widened to fp16∨bf16; loud call-time guard for bf16 reaching the legacy im2col path (Rule 8) |
| `mlx_mfa/_auto_hooks.py` | `_conv3d_nax_eligible` admits bf16; new `_conv3d_bf16_mpp_eligible()` mirrors the C++ MPP gate (incl. `MFA_DISABLE_CONV3D_MPP` -> ineligible) so bf16 NEVER reaches the broken legacy path; non-MPP bf16 falls back to the original op bit-identically |
| `tests/test_phase3_iii1_conv_bf16.py` | 6 locks: engagement (telemetry), correctness, fp16 unchanged, non-MPP fallback bit-identical, env opt-out safe, raw-C++ legacy bf16 raises loudly |
| `tests/test_release_notes_perf_claims.py` | conv claim kind (`expected="conv3d_mpp"`, telemetry-based reachability); `ii9_*` fp16 row registered (gap: II-9 had no §Z row) + `iii1_*` bf16 row |
| `docs/PERF_CLAIMS.md` + `tests/test_perf_claims_doc_sync.py` | 2 active rows; ID grammar widened `ii\d+` → `i{2,3}\d+` |

## Bench (public path: `install_hooks()` → `mx.conv3d`, bf16, 3 sessions, 30-iter medians)

Baseline = pre-lift public bf16 path (hook fell back to Apple
`mx.conv3d`; the legacy NAX path was never reachable for bf16).

| cell | pre-lift ms | MPP ms (median of 3) | speedup |
|---|--:|--:|--:|
| T8 64x64 C128 | 2.19–2.20 | 0.90 | **2.43×** |
| T16 64x64 C128 | 4.09 | 1.54 | **2.66×** |
| T8 32x32 C256 | 2.22–2.24 | 0.85 | **2.62×** |
| T4 64x64 C64 | 0.44–0.45 | 0.32 | **1.40×** |

Correctness through the public path: max rel ≤ 0.84%, 99.94–99.96%
bit-identical to the original op; engagement + fallback both
telemetry-verified.

## Fresh finding for III-4 (not acted on)

C_in/C_out=16 measured **correct** in BOTH dtypes through the isolated
probe (fp16 max rel 0.09–0.11%, bf16 0.39%) — contradicting II-9's
"err 0.17–0.31 at C=16" which was measured through the production
path.  The C≥32 gate is left untouched (conservative; II-9's
measurement context differs).  III-4 must re-derive the C=16 verdict
through the production path. [VERIFIED probe / II-9 context UNCERTAIN]

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Premise validation (§AA.5) | `/mlx-mfa-apple-primitives-coverage` | CONFIRMATION (declared + implemented); probe before any code change |
| Perf discovery (§Z reachability) | `/mlx-mfa-perf-audit` rubric, made executable | telemetry-based conv claim kind in test_release_notes_perf_claims.py; both conv rows REACHABLE |
| Bench methodology | §AA.4 | 3 sessions × 30 iters, medians, public surface |

## Validation

Suite: **1417 passed, 2 skipped** (was 1409+2; +6 locks, +2 claim
params).  fp16 conv path unchanged (69 conv tests pass; bitwise
determinism lock).

# Release v2.56.0 — SHIPPED

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**PyPI:** https://pypi.org/project/mlx-mfa/2.56.0/ · **GitHub:** https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.56.0
**Tag SHA:** `97fed31c582177f09c01936ca2d49da0d8056564` (annotated `v2.56.0` on commit `dee8957`).
macOS 26.6 (25G5028f), M5 Max, mlx 0.31.2.

## Both gates ran (in order)

1. **`repo-release-prep` skill** (Marco-mandatory, all docs): 7-phase doc inventory (955 .md;
   `docs/v50/*` preserved as historical) + per-file audit + version/orphan/`__all__`/benchmark-
   freshness. Doc fixes committed `dee8957`.
2. **`/mlx-mfa-release-audit`** (§AA 9-check): **verdict GREEN, all checks PASS, 0 advisories**
   (Check 1 version-bump cleared on the bump commit `9db5efd`).

## Scope (all on master since v2.55.0)

| Item | commits | kind |
|---|---|---|
| `MFA_FORCE_NATIVE_BWD` removal (kernel retained, keep-all-paths) | `e14ddd9`+`c304d5e` | **breaking → minor bump** |
| V3 auto-routing validated + reframed (M5/26.6) + correctness test | `c304d5e`/`75f3510`/`4c88f02` | validated + test |
| IV-D1/D2 TQ-decode eval-collapse (~1.63× tq_v=False, ~1.36–1.39× tq_v=True default) | `c93a03f`+`b1d81bf` | **perf headline** |
| A3-1 V6 NAX device-offset int64 widening (latent-overflow fix) | `011e34a` | latent-correctness |

Version bump 2.55.0 → 2.56.0 (own commit `9db5efd`) — minor bump for the breaking public-env-var
removal, per the established convention.

## Doc audit (what the skill updated)

- **CHANGELOG [2.56.0]**: header corrected ("no kernel *math* changed; one address-arithmetic fix");
  added **`### Fixed`** (A3-1 latent int64 overflow + A5-1 V3 coverage test) alongside the existing
  Removed/Changed/Validated/Performance sections + the breaking-change migration note.
- **ENV_VARS.md**: `MFA_FORCE_NATIVE_BWD` → **REMOVED v2.56.0** (was already updated in Wave M3) with
  the migration note + kernel-retained clause.
- **README**: version header → 2.56.0; perf footnote → v2.56.0 (IV-D1/D2 + A3-1; kernels otherwise
  unchanged from v2.52.1). V3 framing already corrected in RESULTS.md (queue-closure).
- **CLAUDE.md**: Current status → v2.56.0 / 1827 tests + Phase IV summary (was stale v2.52.1/1563).
- **API/`__all__`**: 101 exports all importable (the 34 lazy `__getattr__` entries resolve);
  no orphans; gitignore complete; 0 tracked-ignored.
- No historical record rewritten (Phase 2.8 preserved).

## Build + publish

`v2.56.0` tag → clean build (`mlx_mfa-2.56.0-cp311-cp311-macosx_26_0_arm64.whl` + sdist) →
`twine check` PASSED → `twine upload` (PyPI live) → `gh release` (draft=false) → master + tag pushed.

## Post-publish smoke (CLEAN env, PUBLISHED wheel — `pip install mlx-mfa==2.56.0`)

| Check | Result |
|---|---|
| (1) `MFA_FORCE_NATIVE_BWD` inert (no routing change, no warning, import OK) | ✅ |
| (2) V3 windowed-causal vs fp32 (D=64 N=4096; D=128 N=2048) | ✅ err ~7.5e-5 |
| (3) IV-D1/D2 decode deferred==eager bit-identity, **both** tq_v (30 steps) | ✅ max_diff 0.00e+00 |
| (4) v2.55.0 prior fixes hold: V2 non-causal, V5, GNA non-32-aligned, split-K-under-churn | ✅ (worst 1.0e-6) |

**SMOKE PASSED** — the published wheel IS the fixed+optimized binary. 0 orphan processes.

## Phase IV backlog — status

- **v2.56.0: SHIPPED.** A clean, stable, published reference (existing code hardened by the
  correctness review + optimized-to-the-floor) is now on PyPI.
- **Next (separate chantier): V6 NAX / dequant-in-GEMM** — the real performance frontier (exploit
  the M5 Neural Accelerators per the Day-J characterization). Green-lit on a published baseline; use
  `/mlx-mfa-nax-matmul2d-correctness` as the pre-kernel checklist.
- **Separate (when Marco publishes the m5max paper):** the investigation repo's git-init + the 3
  flagged re-measurements (matmul-table dedup/de-size-sweep, NA INT8 coverage re-run, bf16 accuracy
  column). Methodology fixes are applied to the working tree (no git there yet).
- **Diagnostic-only (low):** D-OPT-1 param-mask memoization (caller-usage-dependent).

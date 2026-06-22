# Docs / Knobs / Hygiene Reconciliation — Volet E

> Branch `fix/audit-remediation`, base HEAD `2d94b8c` (after A, C, B, D), host
> M5 Max / macOS 26.6 / MLX 0.31.2. One systematic reconciliation of every perf
> claim, routing claim, and knob against runtime/source ground truth. Numbers
> are re-measured-or-already-stamped only — none invented (RULE: "don't know" →
> remove). Line numbers verified at source (RULE 16 — several findings' line
> refs were stale, e.g. README:23 is V6-backward not conv3d, README:403 is
> "1.75×" not "21×").

## 1) Perf claims

Authority for stamped values: `docs/reference/PERF_CLAIMS.md` (telemetry-verified
rows) + `RESULTS.md` §2 + `docs/reference/dispatch-map.md`.

| # | Claim (location) | Was | Ground truth (source) | Provenance? | Action |
|---|---|---|---|---|---|
| CC-08 | conv3d denominator (README §Conv3D, ~L585) | "fp16 2.3–2.5×, bf16 1.4–2.7× **vs `mx.conv_general`**" | 2.3–2.5× is vs the **internal materialized-im2col** methodology baseline, NOT `mx.conv_general` (PERF_CLAIMS H-07 / ENV_VARS:114). The public win vs `mx.conv_general` = **median 1.64×** (SeedVR2 prod) | now y | denominators split: 2.3–2.5×=im2col (methodology), 1.64×=conv_general (public) |
| CC-09 | three conflicting conv3d ratios (README L30 "~1.2–1.35× vs legacy", L172 comment "2.3–2.5×", L585 "vs conv_general") | 3 different numbers/denominators for "the same path" | the bare "~1.2–1.35× vs legacy" matched **neither** authority (PERF_CLAIMS says 2.3–2.5× vs im2col) → unsubstantiated | n→y | collapsed to the two provenanced facts (1.64× vs conv_general; 2.3–2.5× vs im2col); header + example comment rewritten |
| CC-10 | V6-backward "Reachable via AUTO API" (README L151-157) | "D=64 … `MFA_ENABLE_V6_BACKWARD=1` → **1.81–1.82×**" | D=64 backward is **default-on, no env**, **2.16–3.05×** (PERF_CLAIMS `ii12_*`, FEATURE_COVERAGE:38, ENV_VARS:67); `MFA_ENABLE_V6_BACKWARD` is the D=128 opt-in; 1.81× **withdrawn** | now y | block rewritten to default-on 2.16–3.05×; env reframed as D=128-only |
| CC-11 | "up to ~21×" (README L414, FEATURE_COVERAGE:40) | bare "~21×", no provenance | measured **20.8× (M4 Max) / 18.4× (M1 Max)** at D=128 N=8192 win=256 (RESULTS.md §2) | n→y | stamped with the measured cell + hardware; "scales with mask sparsity" |
| — | (verified CLEAN, already stamped) README L24-26 V6-bwd 2.16–3.05× +date+MLX+hw; API_MANUAL:60; HARDWARE_SUPPORT:89; FEATURE_COVERAGE:38; RESULTS §2 table | — | full provenance present | y | none |

**4 perf claims corrected.** No number invented; the one unsubstantiated figure
(~1.2–1.35×) was replaced by a provenanced one (1.64×), not estimated.

## 2) Routing claims

Authority: `docs/reference/dispatch-map.md` (locked by `test_dispatch_map_lock.py`).
Runtime byteΔ spot-check (this M5): D=128 N≥2048 → NAX (Δ≠0); D=128 N<2048 → SDPA
(Δ=0); D=64 dense → SDPA (Δ=0). Matches the map.

| # | Claim (location) | Was | dispatch-map (authority) | Action |
|---|---|---|---|---|
| CX-09 | `flash_attention` docstring (attention.py ~L471) | "Dense causal **D=64/128 routes to MFA** on supported shapes" | M5: D=128 **N≥2048**→NAX, D=128 **N<2048**→SDPA, **all dense D=64**→SDPA (forward); the D=64 win is in the *backward* | docstring rewritten to the map (forward routing per (D,N); points to dispatch-map.md) |
| — | README L362-364 / RESULTS "D=64 stays SDPA" | — | matches map | CLEAN |
| — | dispatch-map.md itself (the authority) | — | matches runtime byteΔ | CLEAN (locked) |

**1 routing claim fixed** (the docstring); the README + map were already correct.

## 3) Knobs

`KNOWN_KNOBS` (116) / `REMOVED_KNOBS` vs non-comment read/alias/dispatch
appearances in `csrc/`+`mlx_mfa/` vs `ENV_VARS.md`.

| # | Knob(s) | Was | Verified | Action |
|---|---|---|---|---|
| CC-12 | `MFA_CONV3D_MPP`, `MFA_V6_BHND`, `MFA_V6_MATMUL_EXEC_SG`, `MFA_REQUIRE_MSL4`, `MFA_SUPPORTED_DTYPES`, `MFA_SUPPORTED_HDIMS` | in `KNOWN_KNOBS` → `validate_env(strict)` accepted them as valid DOF | **0 env-read sites** (comments / a `// MFA_REQUIRE_MSL4` MSL source marker / `_MFA_SUPPORTED_*` module constants) | **deleted** from `KNOWN_KNOBS` → now warn "unrecognized" not silently-accepted |
| CC-13 | `MFA_ENABLE_V4/V5`, `MFA_V5_FORCE_{BK,BD_TILE,BQ,WM}` | documented removed in ENV_VARS:47-51 but **absent** from `REMOVED_KNOBS` → miswarned "typo" | 0 read sites; ENV_VARS names them removed | **added** to `REMOVED_KNOBS` → now warn "REMOVED — no effect" |
| CC-16 | `MLX_MFA_HOOK_TELEMETRY` | ENV_VARS listed it without the import-timing caveat | captured into a module global at `_auto_hooks.py:68` (import-time) | ENV_VARS row now states "read once at import — set before `import mlx_mfa`" |
| — | other 110 `KNOWN_KNOBS` | — | each appears in a non-comment read/alias/dispatch line (V34→V6 names are live aliases via `getenv_aliased`); `MFA_KNOB_STRICT` read in `_knobs.py` | CLEAN |

**~12 knobs cleaned** (6 ghosts removed, 6 removed-knobs reclassified, 1 doc
caveat). No knob's runtime behaviour changed — registry/diagnostic only.

## 4) Packaging hygiene

| # | Item | Verified | Action |
|---|---|---|---|
| CC-18 | orphan sdist sources | `async_v2_noasm.metal` + `kernels/attention_forward.metal` = **0 build/test/CMake refs**; `mpp_int8_bench.mm` = **1 ref** (CMakeLists `MFA_BUILD_PROBES`) | excluded the 2 orphans from the sdist; **kept** mpp_int8_bench.mm (referenced — the safety rule forbids removing a referenced file) |
| CC-19 | `async_v2.metallib` CI gate | target-dead on macOS ≥26/M5 (`shader_cache.mm:97-103` → nullptr) but live on macOS 14/15 | still shipped; CI presence check **downgraded FATAL→advisory** (warning) so a target-dead artifact never blocks an M5 release; compile-critical sources stay FATAL |
| CC-20 | `MLX_MFA_METAL_PATH` (CMakeLists) | **0 consumers**; baked the build-machine abs path into `_ext` | **removed** the define (`MLX_BUILD_VERSION`, which IS live, kept); `_ext` rebuilds clean |
| CC-21 | `check_env.py` MLX floor | only reported the version; pip enforces `>=0.31.2` | added an **advisory** `<0.31.2` warning (pip remains the real gate) |

## Validation (bite-proven)
1. Full suite `2404 passed, 91 skipped, 0 failed, 0 XPASS` (collection ≥1800).
   `_ext` rebuilds clean after the CMakeLists change.
2. **Perf-claim locks** (`tests/test_volet_e_claims_knobs.py`): assert the
   reconciled numbers present + the stale/mis-denominated ones absent
   (1.64× present / "2.3-2.5× vs conv_general" absent; 2.16–3.05× present /
   "1.81-1.82× via env for D=64" absent; 20.8×/18.4× present / bare "~21x"
   absent). Flip a number → the corresponding assert fails. The existing
   `test_perf_claims_doc_sync.py` (parametrized SoT claims) stays green.
3. **Knob locks**: `test_every_known_knob_appears_in_real_code` (no comment-only
   ghost) + `test_knob_coverage_bites` (a no-appearance knob IS flagged) +
   `test_removed_ghosts_absent_from_registry` ×6 + `test_env_vars_removed_knob_in_removed_registry` ×6 + `test_removed_knob_warns_removed_not_typo`. Runtime-proven:
   `MFA_KNOB_STRICT=1 MFA_ENABLE_V5=1` → "knob 'MFA_ENABLE_V5' was REMOVED — no
   effect" (not typo); `MFA_CONV3D_MPP=1` → "unrecognized" (no longer accepted).
   `test_doc_accuracy_guards::test_env_vars_doc_knobs_in_registry` extended to
   skip "(not an env var)" rows (MFA_REQUIRE_MSL4).
4. **sdist manifest**: orphans absent (`async_v2_noasm.metal`,
   `attention_forward.metal` = 0); required present (`async_v2_kernel.metal`,
   `async_v2.metallib`, `mfa_steel_fwd_v2.cpp`, the TQ-paged kernel = 1);
   `mpp_int8_bench.mm` kept; `twine check` PASSED; version 2.61.0 consistent
   across pyproject / `__init__` / README / API_MANUAL / CHANGELOG.

---
*Docs / registry / packaging only; no kernel/dispatch/output change. Commit on
`fix/audit-remediation` only.*

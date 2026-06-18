# Audit Phase D — Documentation Rebuild from Verified Facts + Publication Cleanup

**Date:** 2026-06-18 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `f878e64`, M5 Max, macOS 26.6. Pre-flight: `repo-release-prep`,
`mlx-debug-forensics`. **DOC-ONLY** (+ the publish-surface guard + env-doc augment); no kernel/routing/
threshold change, no bug fix (Phase F), no perf re-measurement (Phase E).

## Stage 1 — Doc-surface inventory + classification

| Class | Count | What |
|---|---|---|
| **PUBLISH** (in the wheel/sdist) | 2 .md + LICENSE | `README.md`, `CHANGELOG.md` (+ LICENSE, THIRD_PARTY_LICENSES) — the only files the sdist `MANIFEST.in` whitelist ships |
| **RETAIN** (in-repo GitHub, wheel-EXCLUDED) | 326 tracked docs/+devnotes .md + the other root .md (NAMING/ENV_VARS/RESULTS/CLAUDE*/AUTORESEARCH*) | maintainer reference + the campaign journal (76 campaign/audit .md). All wheel-excluded by MANIFEST. |
| **already-RETAINED** (gitignored) | `.doc-archive/` (3 snapshots, untracked) | the established retained archive (2026-05-14 pattern) |
| **REWRITE** | 1 | `README.md` — augmented with the verified KNOWN_ISSUES + path-dependent env semantics (this phase) |

Total tracked .md = 336. **The publication split is structurally pre-existing**: `MANIFEST.in` is an
include-WHITELIST (README/CHANGELOG/LICENSE/CMakeLists/pyproject + csrc/mlx_mfa/examples) and
`wheel.packages=["mlx_mfa"]`, so docs/+devnotes/+.doc-archive are already excluded from the published
surface. "Stop publishing the journal" was already in force; Phase D makes it GUARDED + absorbs the
rationale into the published README.

## Stage 2 — Rebuilt published current-state docs (from verified A/B/C facts)
- **README KNOWN ISSUES** (new section, all [Verified] by runtime fingerprint): D=128 built-in-mask
  sparse → silent SDPA fallback (the `(long)p->NK` miscompile cause + the symmetric-mask workaround);
  work<2^31 → V1-scalar ~40× cliff; sparse-backward-dense-by-default; V5 ineligible; perf
  re-validation **pending (Phase E)** — current routing presented as **honest, not optimal**, each with
  a planned-fix pointer.
- **README env semantics + ENV_VARS.md**: the path-dependent `MFA_ENABLE_V6_BACKWARD` — **dense D=128 →
  full-native dQ/dK/dV; sparse hybrid → native-dV-only**; full-native sparse needs
  `MFA_V6_BWD_SPARSE_NATIVE=1` + bt≥64 (the B3/C2 finding, verified).
- **README correctness/coverage**: pointer to the 42 kernel locks + dispatch lock + fingerprint
  discipline (green-on-wrong-binary structurally caught), + the maintainer reference (`audit/` specs).
- The 4 per-kernel family specs + dispatch-map.md remain the durable maintainer reference in `audit/`.

## Stage 3 — Publication split (retain, absorb, don't lose the "why")
The wheel already excludes the journal (MANIFEST whitelist; `.doc-archive/` gitignored). The campaign
analyses' essential rationale is **absorbed into README's KNOWN ISSUES** (e.g. the `(long)p->NK`
investigation's conclusion is stated as the cause of the D=128 fallback) so retiring/not-publishing
them loses no "why". Provenance pointer for maintainers: `docs/v50/campaign-2026-06/audit/` (in-repo)
+ `.doc-archive/` (gitignored snapshots).

## Stage 4 — Anti-drift coupling + publish-surface guard

**doc-claim → executable lock map:**
| Published claim | Lock |
|---|---|
| routing: auto→SDPA, mfa→STEEL, D=128-asym-sparse→SDPA fallback, decode→SDPA, conv eligibility | `test_dispatch_map_lock.py` (11) |
| per-kernel correctness (sparse/dense/backward/GNA/conv/topk/sage/paged) | `test_{sparse_family,dense_steel_family,backward_family,b4_family}_*_lock.py` (42) |
| KNOWN ISSUE: D=128-API sparse silently runs SDPA; symmetric runs real kernel | `test_fingerprint_discipline.py` (wrong-binary locks + positive demo) |
| env: `MFA_ENABLE_V6_BACKWARD` dense→full-native vs sparse→dV-only | `test_fingerprint_discipline.py::...d128_backward_optin_is_native` + `test_backward_family_lock.py` (sparse hybrid = native-dV-only) |
| publish surface excludes the journal | `test_publish_surface_guard.py` (4, planted-leak self-test) |
| **un-locked, labeled:** work<2^31 → V1-scalar **41× perf** cliff | Deduced/perf — **Phase-E** (the byteΔ V1/V2 *identity* is locked via env-toggle; the 41× *magnitude* is perf, re-validated in E) |
| **un-locked, labeled:** V5 ineligibility | Assumed (no positive lock; noted) |

**Publish-surface guard** (`test_publish_surface_guard.py`): parses `MANIFEST.in`, FAILS if any
journal path (devnotes/docs/.doc-archive) is whitelisted, and asserts README/CHANGELOG/LICENSE ARE +
`wheel.packages=["mlx_mfa"]`. Self-test plants 3 forbidden includes and confirms the detector trips —
the journal cannot silently leak into a release.

## Disposition
Documentation rebuilt from runtime-verified ground-truth (honest about warts, perf Phase-E-pending);
the journal is separated from the published surface (retained in-repo + gitignored archive, NOT in the
wheel) with its rationale absorbed into README; every major published claim is coupled to an
executable lock (un-locked ones labeled Deduced/Assumed/Phase-E). The drift that caused the four
inversions cannot silently recur (dispatch + fingerprint + publish-surface guards). DOC-ONLY; suite
green; no orphans; not tagged. **Phase E (complete M5 re-bench: the 2^31 V1/V2 crossover, STEEL-vs-SDPA
M5-optimality, V5 reachability, sage int8 quality-worth) is next.**

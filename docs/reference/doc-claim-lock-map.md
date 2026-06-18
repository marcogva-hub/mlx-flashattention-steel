# Doc-claim → executable-lock map (current-state reference)

Every load-bearing published/reference claim is coupled to an executable lock so the
drift that caused the campaign's four which-binary inversions cannot silently recur.
Extracted from the Phase-D rebuild (audit) + extended through Phase F and the E-addendum.
Correctness claims are executable-locked; perf claims are Verified-at-date (timing is
CI-flaky — re-measure is the anti-drift), labelled as such.

| Published / reference claim | Lock |
|---|---|
| routing: dense auto→SDPA, mfa→STEEL, decode→SDPA, conv eligibility | `tests/test_dispatch_map_lock.py` |
| routing (Phase F): D=128 built-in masks → symmetric 32×32 → NAX-sparse (d≲ceiling) / SDPA (d≳ceiling); D∈{64,128} sparse → V2 (never V1-scalar default) | `tests/test_dispatch_map_lock.py` + `tests/test_decide_auto_version_shape_aware.py` |
| per-kernel correctness (sparse / dense-steel / backward / GNA / conv / topk / sage / paged) | `tests/test_{sparse_family,dense_steel_family,backward_family,b4_family}_*_lock.py` (42 cells) |
| dense NAX forward `v6_nax_forward` is a faithful FA-2 forward (default scale) | `tests/test_v6_nax_forward_lock.py` (9 cells) |
| green-on-wrong-binary is structurally caught (assert the BINARY — byteΔ vs SDPA — not just the MATH) | `tests/test_fingerprint_discipline.py` |
| env: `MFA_ENABLE_V6_BACKWARD` dense→full-native vs sparse→dV-only | `tests/test_fingerprint_discipline.py` + `tests/test_backward_family_lock.py` |
| publish surface excludes the journal (wheel MANIFEST **and** tracked repo tree) | `tests/test_publish_surface_guard.py` |
| **perf (Verified-at-date, NOT locked):** symmetric-NAX-sparse beats SDPA D=128 to d≈0.78; V1-scalar never fastest; STEEL legacy-on-M5; v6_nax-dense parity-or-win at D=128 | RESULTS.md + `docs/reference/BENCHMARKS.md` (re-measure is the anti-drift) |

## Maintainer reference set (this directory)
- `dispatch-map.md` — runtime-verified which-kernel-runs map (locked by `test_dispatch_map_lock.py`).
- `sparse-family-spec.md`, `dense-steel-family-spec.md`, `backward-family-spec.md`,
  `b4-family-spec.md` — the four per-kernel specs (fp32/oracle-correctness-locked).
- `API_MANUAL.md`, `ARCHITECTURE.md`, `FEATURE_COVERAGE.md`, `HARDWARE_SUPPORT.md`,
  `INVENTORY.md`, `SERVING_GUIDE.md`, `TRAINING_QUICKSTART.md`, `PERF_CLAIMS.md`,
  `RELEASE_PHILOSOPHY.md`, `HOOK_TELEMETRY.md`, `BENCHMARKS.md`, `INDEX.md` — current-state guides.

## Provenance / the journal
The campaign journal (phase reports, sprint decisions, design docs, the audit ledger,
devnotes, diagnostics) is RETAINED but OFF the public tracked tree — it lives in git
history and the gitignored `.doc-archive/` snapshot. It is intentionally not published to
either the wheel or the public repo surface (enforced by `test_publish_surface_guard.py`).

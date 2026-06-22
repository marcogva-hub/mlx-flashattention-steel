# Release-Gate Skip-Escape Enumeration (Volet D2 — CC-02)

Branch `fix/audit-remediation`, base HEAD `aabff7b` (after volet E2). Host M5 Max /
macOS 26.6 / MLX 0.31.2. Verify-first (RULE 16).

## CC-02 (the completeness oracle)

The publish-surface guard (GATE 3) routes its build precondition through
`_skip_or_fail` (`tests/test_publish_surface_guard.py:40`), which **hard-fails only
when `MFA_RELEASE_GATE` is set, else `pytest.skip`s**. CI never set the env, so a
missing `build` module or a non-zero `python -m build --sdist` → the gate **skips**
→ an all-skip run exits 0 → GATE 3 "passes" green while inspecting **no artifact**
(the 2.58.0 `.claude/settings.local.json` journal-leak class). Volet D wired the
gates into `publish.yml` but missed this internal skip-escape.

**Fix:** job-wide `env: MFA_RELEASE_GATE: "1"` on the `gates` job in
`.github/workflows/publish.yml` → the guard's skips become fails in the release
context. The env is read **only** by `test_publish_surface_guard.py` (verified
repo-wide grep), so nothing else changes — device-conditional M5-lock skips do
**not** route through `_skip_or_fail` and stay clean skips on the M1 runner.

## Full gate enumeration — fail-or-skip when precondition unmet (CI context)

### `publish.yml` → job `gates` (now `env: MFA_RELEASE_GATE=1`)

| gate | mechanism | precondition unmet → | escape? |
|---|---|---|---|
| GATE 1 — full suite | `pytest tests/` (exit code) | a test failure → non-zero exit → FAIL | none. Includes `test_publish_surface_guard.py`, whose `_skip_or_fail` now FAILS under the env. |
| GATE 2 — collection floor | shell: `COUNT<1800 → exit 1` | FAIL (exit 1) | none — shell exit-code, no pytest skip. |
| GATE 3 — publish-surface guard | `pytest test_publish_surface_guard.py` | **was SKIP** (build missing / sdist-build non-zero) → **now FAIL** via the env | **CLOSED (CC-02)**. |
| GATE 4 — release-audit equivalents | `pytest` over 5 doc/contract tests | a failure → non-zero exit → FAIL | none — 0 `pytest.skip`/`importorskip` sites in those 5 files (verified). |
| GATE 5 — M5/NAX fingerprint | `python scripts/check_m5_gate_fingerprint.py` | missing/stale receipt → `exit 1` → FAIL | none — script exit-code, no skip. |

Downstream `needs:` chain (GitHub's hard gate): `build → gates`,
`publish-testpypi → build`, `publish-pypi → build`. A `gates` failure → `build`
skipped → both uploads **unreachable**.

The only non-`_skip_or_fail` skip in the guard file is line 177
(`pytest.skip("not a git checkout …")`, the git-only tree-guard) — its precondition
(a git checkout) is **always met** under `actions/checkout@v4` (`fetch-depth: 0`), so
it runs in CI; not an escape. Left as-is (hardening it to fail would wrongly trip
when the gate is legitimately run from a non-git source export).

### `ci.yml` (regular CI — not a release path)

Does **not** run the publish-surface guard and does **not** set `MFA_RELEASE_GATE`.
Its only floor is the collection check (`COUNT<MIN → exit 1`, shell exit-code, no
skip). No release gate, no skip-escape. (Advisory `|| echo` greps in the packaging
job are re-gated FATAL for compile-critical members — pre-existing, CI-1/round-4.)

## Bite proof (skip → fail flip)

1. **Unit (mechanism):** `_skip_or_fail("simulated sdist-build non-zero")` →
   env unset → `Skipped` (green); env `"0"` → `Skipped`; `MFA_RELEASE_GATE=1` →
   `Failed` (red).
2. **End-to-end (forced build failure, monkeypatched `subprocess.run` → returncode 1):**
   without the env the fixture **skips**; with `MFA_RELEASE_GATE=1` it **fails**
   (`pytest.fail.Exception`).
3. **Happy path:** the real `test_publish_surface_guard.py` under `MFA_RELEASE_GATE=1`
   builds the sdist and runs the **real allowlist assertion** (7 passed, no all-skip)
   — the fix does not break the green path, and the assertion is genuinely reached.

## Dry-run

`act` is not installed locally and an actual publish is prohibited, so the gating is
established structurally: `publish.yml` parses as valid YAML, `gates.env` carries
`MFA_RELEASE_GATE=1`, and the `needs:` graph makes every upload depend (transitively)
on `gates` — a simulated guard sub-failure (bite #2) makes `gates` non-zero, which
GitHub Actions propagates by skipping `build` and therefore both publish jobs. No
real upload performed.

## Validation
- Local full suite (env unset): `2543 passed, 91 skipped, 0 failed, 0 XPASS` — local
  dev behavior unchanged (the escape is correct for offline local dev; it hard-fails
  only where `MFA_RELEASE_GATE` is set, i.e. the release CI).
- No gate weakened; only the skip-escape removed.  Commit on `fix/audit-remediation`.

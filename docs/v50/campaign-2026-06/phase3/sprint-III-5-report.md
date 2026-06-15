# Sprint III-5 — Documentation Update + v2.52.0 Release (Coupled)

**Date:** 2026-06-15
**Executor:** Claude Opus 4.8 High (1M context)
**Branch:** master
**Outcome:** v2.52.0 **PUBLISHED** (PyPI + GitHub + origin tag) — all 9 gates green, post-publish smoke green on the published wheel.

Closes the 2026-06 audit/optimization campaign. v2.52.0 is the complete
corrected state: the III-4 fresh-eyes whole-repo audit (9 passes,
repeat-until-clean, ~73 fixes) on top of everything promoted in v2.51.0,
including the fix for **two pre-existing CRITICAL silent-corruption bugs**
that shipped in v2.51.0.

---

## Skill invocations (§AA.2)

| Checkpoint | Skill | When | Outcome |
|---|---|---|---|
| Pre-version-bump canonical gate | `/mlx-mfa-release-audit` | Before bump | GREEN_WITH_ADVISORY (no blocking; advisories benign) |
| Perf-claim reachability (§Z) | (audit gate 4) | Pre-release | 21 claims reachable via public API |
| Pre-merge / post-doc perf-claim review | `/mlx-code-review` | R.1 docs | Applied (carried claims measured per-cell) |

No new kernel written → `/mlx-mfa-kernel-design` and
`/mlx-mfa-apple-primitives-coverage` not triggered this sprint.

---

## R.1 — Documentation overhaul (commit 3b89b51)

| Artifact | Change |
|---|---|
| `README.md` | Headline → v2.52.0 complete corrected state; measured per-cell promotions; ⚠ upgrade-from-v2.51.0 CRITICALs blockquote; new opt-in APIs |
| `CHANGELOG.md` | `[2.52.0]` entry: CRITICALs disclosure + upgrade directive (NOT buried); III-4 fixes; v2.51.0 promotions (per-cell + Reproduce block); new opt-in APIs; DECLINED list; migration note |
| `docs/PERF_CLAIMS.md` | Header → v2.52.0; withdrawn v2.39.1 "1.01–1.12×" fused claim confirmed ABSENT; ii9_/iii1_/ii12_/iii2_ rows measured |
| `docs/v50/audit-framing-inversions.md` | Pattern #9 (3 exhibits) + III-4 lessons #8/#9/#10 appended |
| `docs/v50/known-debt-v2.50.md` | KD-5 ROOT CAUSE FIXED; KD-7 LIFTED (verified current) |
| `CLAUDE.md` | Status → v2.52.0 — 1501 tests pass |

## R.2 — Version bump (release commit fd8d278, distinct from docs)

`pyproject.toml` + `mlx_mfa/__init__.py`: 2.51.0 → 2.52.0. Semver: **minor**
bump — no breaking public API; behavior changes are toward correctness.
Release commit (version + changelog) kept separate from the R.1 doc commit
per the sprint constraint.

## R.3 — 9-gate pre-release audit (BLOCKING) — all green

| Gate | Result |
|---|---|
| Multi-SoT version consistency | GREEN (2.52.0 everywhere) |
| Tool availability (`check_venv.sh`) | GREEN |
| Auto-default principle | GREEN |
| §Z public-API perf-claim reachability | GREEN (21 claims) |
| Skill-invocation log | GREEN (table present) |
| Test suite | GREEN — **1501 passed, 2 skipped** (default); **1503 ×2** stressed + `MFA_POOL_STRESS=1` pool canary |
| CHANGELOG | GREEN |
| Hook contract (gate 8) | GREEN |
| Programmatic paired-MMA (gate 9) | GREEN (2 passed) |

Verdict: **GREEN_WITH_ADVISORY** — only benign advisory (carried-claim version
strings tagged to intro versions; structurally correct). No blocking finding.
Publish gate cleared.

## R.4 — Tag + publish (irreversible) — DONE

| Surface | State |
|---|---|
| Annotated tag `v2.52.0` | at release commit fd8d278 (now 73d5738 after merge) → pushed |
| PyPI | **live** — https://pypi.org/project/mlx-mfa/2.52.0/ (cp311 wheel + sdist) |
| GitHub release | **published** (not draft) — wheel + sdist attached; CRITICALs disclosure in body |
| origin tag | present (83acf10…) |

`twine check`: both artifacts PASSED before upload.

## R.6 — Post-publish smoke test (clean Python 3.11 venv, published cp311 wheel)

Installed `mlx-mfa==2.52.0` from PyPI (downloaded the actual binary `.whl`),
ran a self-contained public-API smoke. **4/4 green:**

| Check | Result |
|---|---|
| CRITICAL #2 — return_lse backward | grad finite + matches SDPA-vjp across fp16/bf16/fp32 |
| CRITICAL #1 — topk full-row coverage | all 512 rows written (0 stale); first8/last8 balanced 0.95× |
| HEADLINE — V34 backward (causal + non-causal) | matches SDPA-vjp |
| HEADLINE — conv3d auto-hook (fp16 + bf16) | deterministic + matches fp32 (MAE/RMS 0.00014 / 0.00112) |

**Forensics note (non-blocking):** an initial conv check used a degenerate
16-channel shape and showed MAE/RMS ~0.11 (max-abs-rel swung 0.19→8.09 under
RNG-draw ordering — a near-zero-denominator artifact). Verified the NAX conv
path is **deterministic** (re-run maxdiff = 0.0 → not the stale-pool class) and
**accurate at realistic channel counts** (MAE/RMS 0.0001 at Cin=64). The
small-channel fp16 accuracy gap (NAX ~250× less accurate than native fp16 conv
at Cin=16) is a real but pre-existing, out-of-scope observation — spawned as a
separate investigation task (eligibility-gate / fp16-accumulation +
single-shape-class test gap, III-4 lesson #10).

---

## R.5 — v2.51.0 disposition (Marco-gated, OPEN)

v2.51.0 remains on PyPI and contains the two CRITICAL silent-corruption bugs.
v2.52.0 is now live and fixes them. Decision pending:

- **Option A — yank v2.51.0 (recommended).** `pip install mlx-mfa` already
  resolves to 2.52.0; yanking 2.51.0 stops new pins to a known-corrupt release
  while leaving it installable by exact version for reproducibility. The
  CHANGELOG + GH release already disclose the CRITICALs.
- **Option B — leave + disclose.** Keep 2.51.0 installable; rely on the
  CHANGELOG/README upgrade directive. Lower friction; leaves a corrupt default
  path one pin away.

Do **not** yank until v2.52.0 confirmed live — now satisfied.

---

## Remaining Marco-gated queue

| Item | State |
|---|---|
| v2.51.0 yank (Option A recommended) | OPEN — awaiting Marco |
| Small-channel fp16 conv3d accuracy | Spawned background task (out of scope) |

## Validation

- Ran: `/mlx-mfa-release-audit` (9 gates); default suite (1501 passed, 2 skipped); stressed + pool canary (1503 ×2); `twine check` (PASS ×2); clean-venv published-wheel smoke (4/4).
- Validated: PyPI latest = 2.52.0 with both files; GH release published with both assets; origin tag present; both CRITICALs + headline paths green on the published binary wheel.

## Git

- Doc commit `3b89b51`; release commit `fd8d278` (version + CHANGELOG); regression-coverage `b5de3f6`; merged/pushed as `73d5738`; tag `v2.52.0` pushed.

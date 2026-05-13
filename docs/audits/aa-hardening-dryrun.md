# §AA hardening dry-run — `/mlx-mfa-release-audit` first-fire validation

**Date:** 2026-05-13
**Sprint:** 4 (§AA hardening — recommended → mandatory blocking)
**Status:** PASSED — skill behaves as designed

## Purpose

Sprint 4 Section G requires validating the §AA enforcement mechanism
end-to-end before declaring it operational.  Without this validation,
`/mlx-mfa-release-audit` is theoretically a pre-tag gate but
operationally untested — i.e., the v2.37.0/v2.37.1 pattern in reverse
(institutional rule that "looks right" but doesn't actually fire).

This dry-run pretends `v2.99.0` is about to be tagged.  No actual
changes are made; we run the skill against the current `docs/aa-hardening`
branch state and verify the skill correctly identifies what's missing.

## Invocation

```bash
~/code/mlx-mfa-v2/.venv/bin/python \
  ~/.claude/skills/mlx-mfa-release-audit/audit_runner.py \
  --target-version 2.99.0
```

Exit code: `2` (BLOCKED — exactly as expected for a release-not-prepared
target).

## Output verdict

**`verdict: "BLOCKED"`** with 3 blocking findings + 1 advisory.

### Blocking findings (3) — all correctly identified

#### 1. `multi_sot_version_bump` — BLOCKED

```json
{
  "found": {
    "pyproject_toml": "2.37.3",
    "mlx_mfa_init": "2.37.3",
    "readme_header": "2.37.3"
  },
  "expected": "2.99.0"
}
```

Caught correctly.  All three Sources of Truth are at the current
released version (2.37.3) and don't match the synthetic
`target_version=2.99.0`.  This is the v2.33.x lesson encoded
mechanically.

#### 2. `test_suite` — BLOCKED

```
3 failed, 418 passed, 1 xfailed, 5 xpassed, 2 warnings in 17.06s
```

Caught correctly.  These are pre-existing svdquant + turboquant
failures observed in prior sessions (v2.37.2, v2.37.3).  The runner
correctly halts on them — they would block a real release until
either fixed or moved to a documented known-flakes list.

**Note:** the runner excludes `test_attention.py` and
`test_attn_bias_native.py` because those have documented FP16-noise
flakes in v2.37.2/v2.37.3 release notes.  The 3 remaining failures
are NOT in those excluded files, so they're "real" failures from
the runner's perspective.  This is the right conservative behavior.

#### 3. `changelog_entry` — BLOCKED

```json
{"reason": "CHANGELOG.md has no `## [2.99.0]` heading"}
```

Caught correctly.  No `## [2.99.0]` heading exists in CHANGELOG.md
(of course — we didn't write one).  The §Z + §AA.2 + Section 7
enforcement (CHANGELOG entry presence) fires as designed.

### Passing checks (3)

- **`tool_availability`** — PASS.  `bash scripts/check_venv.sh --no-install`
  exits 0; `.venv/bin/twine`, `pytest`, and `import build` all
  succeed.  §X.5 (Sprint 1) enforcement operational.
- **`auto_default_principle`** — PASS.  `mlx_mfa._auto_hooks._HOOKS_INSTALLED`
  is `True` after import; `_INSTALL_LOG` is non-empty.  The
  defensive-introspection probe added in Sprint 3 review correctly
  finds both attributes.  Sprint U auto-default principle
  operational.
- **`public_api_path_validation`** — PASS.  `pytest tests/test_release_notes_perf_claims.py`
  exits 0 with 6 claims audited.  §Z (Sprint 2) enforcement
  operational.

### Advisory findings (1)

**`skill_invocation_log` — ADVISORY**

```json
{
  "skills_seen": ["/metal-kernel-dev", "/mlx-code-review"],
  "missing_per_category": [
    {"category": "perf_claim_added", "expected_skill": "/mlx-mfa-perf-audit"},
    {"category": "pre_version_bump", "expected_skill": "/repo-release-prep"}
  ],
  "docs_audited": [
    "docs/audits/v37-systematic-audit.md",
    "devnotes/SESSION_LOG.md",
    "devnotes/sage-decode-productionization/sage_decode_productionization_task4_aot.md",
    "devnotes/sage-decode-productionization/sage_decode_productionization_task2_policy.md",
    "devnotes/sage-decode-productionization/sage_decode_productionization_task1.md"
  ]
}
```

Behavior is correct: the runner scans the 5 most-recently-modified
sprint deliverable docs for evidence of mandatory skill invocations.
It finds `/metal-kernel-dev` and `/mlx-code-review` traces (from
Sprint 2 + Sprint 3) but no `/mlx-mfa-perf-audit` (brand-new in
Sprint 3, not yet referenced in pre-existing deliverables) and no
`/repo-release-prep` (not yet invoked).

This ADVISORY (not BLOCKED) status is the §AA.2 "honest scope"
clause operating as designed: Check 5 enforces table presence
(passes here), but category coverage is on the sprint author.  The
advisory surfaces the gap for human review without halting the
release — appropriate for a doc sprint that doesn't itself add perf
claims.

## Interpretation

The skill correctly:

1. **Identifies blockers** — version mismatch, test failures, missing
   CHANGELOG entry — exactly the kinds of pre-release gaps the rule
   was designed to catch
2. **Passes legitimate state** — tool availability, auto-hooks,
   public API path validation — these are all operational and
   correctly reported as PASS
3. **Advises on category coverage** — surfaces the gap in Sprint 3
   skill invocation traces without false-blocking a doc-only sprint
4. **Exit codes match the verdict** — exit 2 on BLOCKED, suitable
   for CI gating

The §AA hardening is **operational**.  No follow-up implementation
work needed; the mechanism does its job.

## What would happen on a real release

For a real `v2.37.4` release flow, CC would:

1. Bump versions in pyproject.toml + __init__.py + README.md
2. Add `## [2.37.4]` CHANGELOG entry with Reproduce snippets per §Z
3. Fix or document the 3 pre-existing test failures (probably
   sage-decode-related per the recent devnotes)
4. Ensure new sprint deliverables include populated Skill invocations
   tables per §AA.2 templates
5. Re-invoke `/mlx-mfa-release-audit target_version=2.37.4`
6. Expect verdict `GREEN` (or `GREEN_WITH_ADVISORY` if category
   coverage still has gaps)
7. Only then proceed to `git tag v2.37.4`

If the skill returned `BLOCKED` for any reason, the release flow
halts per §AA.4 — no manual override.

## Skill invocations (per §AA.2)

| Skill | Decision point | Timestamp (ISO) | Findings | Action |
|---|---|---|---|---|
| /mlx-code-review | pre-merge of docs/aa-hardening branch (Sprint 4 Section F.1) | 2026-05-13T14:30Z | 2 MEDIUM (halt-protocol clarity, Check 5 enforcement boundary) + 1 LOW (concrete example) | All 3 fixed in commit `acde9c2` before this dry-run |
| /mlx-mfa-release-audit | Sprint 4 Section G dry-run (FIRST INVOCATION of the skill) | 2026-05-13T14:45Z | BLOCKED with 3 blocking findings + 1 advisory | Documented here; confirms skill operational |

## References

- `CLAUDE_V6_NAX.md` §AA.4 (canonical pre-tag gate spec)
- `CLAUDE_V6_NAX.md` §AA.1 (halt protocol — invoked successfully
  with verdict BLOCKED)
- `~/.claude/skills/mlx-mfa-release-audit/SKILL.md` (skill definition)
- `~/.claude/skills/mlx-mfa-release-audit/audit_runner.py` (~320 LOC
  Python implementation)
- `docs/skills/README.md` (Sprint 3 skill set inventory)
- `docs/skills/installation.md` (fresh-machine setup including
  MLX_MFA_REPO_ROOT override)

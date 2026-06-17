# mlx-mfa specialized skills

Sprint 3 (2026-05-13) created three mlx-mfa-specific Claude Code
skills that encode the institutional rules from Sprint 2's
amendment (`CLAUDE_V6_NAX.md` §Z + §AA) as executable automation.
These reduce dependency on "CC remembers to invoke the rule" — the
skills fire mechanically when invoked, removing the human-memory
failure mode demonstrated by the v2.37.0/v2.37.1 silent integration
bug.

## Skill inventory

| Skill | Purpose | When to invoke | Mandatory per §AA |
|---|---|---|---|
| `/mlx-mfa-perf-audit` | Verify a perf claim is reachable via the public API | After "X× speedup" discovery; pre-release; auditing past claims | **Yes** |
| `/mlx-mfa-release-audit` | 7-check pre-tag gate (§X + §X.5 + §Z + §AA + tests + CHANGELOG) | Before ANY version bump or PyPI upload | **Yes** |
| `/mlx-mfa-bench-methodology` | Auto-select §4-strict vs canonical protocol; cross-session ratio analysis | All sub-ms perf work; cross-session variance characterization | **Yes** |
| `/mlx-mfa-kernel-design` | New V6NAX kernel scaffolding + register budget pre-flight | New kernel write / register-pressure decisions | **Yes** (unblocked v2.38.x Phase B) |

Skill definitions live in `~/.claude/skills/<skill-name>/`.  See
`installation.md` for setup on a fresh dev machine.

## When to use each skill

### `/mlx-mfa-perf-audit`

**Mandatory trigger** (per §AA):
- A bench reports "X% speedup" or "X× faster" — verify reachable via
  public API before documenting in user-facing docs
- Pre-tag for any release adding perf claims to CHANGELOG
- Auditing an existing release's perf claims

**Output:** JSON verdict — `REACHABLE` / `UNREACHABLE` / `OVERSTATED` /
`NOISE_BAND` + `kernel_engaged` boolean + recommendation
(`preserve` / `retract` / `code_fix` / `reclassify_research_only`).

Detection strategy: differential gradient RMSE (per
`tests/test_release_notes_perf_claims.py` pattern).  V6NAX backward
vs SDPA-vjp gradients differ by FP16 rounding; SDPA fallback is
bit-identical.

**Reference:** v2.37.x silent fallback (`docs/v6-nax/v2.37.x-perf-claim-audit.md`)
documents the failure mode this skill prevents.

### `/mlx-mfa-release-audit`

**Mandatory trigger** (per §AA):
- Before ANY `git tag vX.Y.Z` or `pyproject.toml` version bump commit
- Before any `twine upload` or `gh release create`

**Output:** JSON verdict — `GREEN` / `GREEN_WITH_ADVISORY` / `BLOCKED` +
per-check status (7 checks total) + blocking findings list +
recommendation.

7 checks:
1. Multi-SoT version bump (`pyproject.toml` + `__init__.py` + README)
2. Tool availability (`scripts/check_venv.sh --no-install`)
3. Auto-default principle (`_HOOKS_INSTALLED` after import)
4. Public API path validation (`tests/test_release_notes_perf_claims.py`)
5. Skill invocation log audit (advisory)
6. Test suite full pass
7. CHANGELOG entry structurally complete

**Cross-references:** `CLAUDE_V6_NAX.md` §X / §X.5 / §Z / §AA;
`scripts/check_venv.sh` (Sprint 1); `tests/test_release_notes_perf_claims.py`
(Sprint 2).

### `/mlx-mfa-bench-methodology`

**Mandatory trigger** (per §AA):
- Any "is X faster than Y" question on mlx-mfa
- All sub-ms perf work
- Cross-session variance characterization (e.g., for auto-default
  graduation decisions like v2.36.1 V2 sparse)

**Output:** JSON results table — per-shape verdict `CONFIDENT`
(<10% cross-session range) / `BOUNDARY` (10-20%) / `HIGH_VARIANCE`
(≥20%) / `INELIGIBLE` + per-session ratios + protocol-used label
(`strict_cooldown` vs `canonical_warmup_continuous`).

Protocol selection follows §4.3:
- est. wall-clock ≥ 1.5 ms → §4-strict cooldown
- est. wall-clock < 1.5 ms → canonical warmup+continuous

**Cross-references:** `CLAUDE_V6_NAX.md` §4;
`docs/methodology/canonical-protocol.md`;
`bench/methodology/canonical_warmup_continuous_harness.py`.

## `/mlx-mfa-kernel-design` — UNBLOCKED v2.38.x Phase B (2026-05-13)

Sprint 3 deferred this skill pending the Apple helpers refactor
(M1-HIGH-01 + M3-HIGH-01 from Sprint 2 audit) which consolidated
~390 LOC of duplicated kernel helpers into the shared
`naxHelpersBlock()` static method in
`csrc/mfa/v6_nax/NAAttentionKernel.cpp`.

Sprint v2.38.x Phase B executed that refactor (commit `c509b4b`).
With patterns now in ONE location, this skill encodes:

- `naxHelpersBlock()` invocation pattern (no inline duplication)
- B+C+E+I mechanism bundle (Barriers / Compute / Exponent /
  Index math) marker template
- Register-budget pre-flight math (per-lane / per-SG / per-CU
  accounting against M5's 64 KB CU register file)
- Three-axis test scaffold per `CLAUDE_V6_NAX.md` §3.5
- Source-generator template (parameter resolution, raw-string
  marker selection, kernel-proper structure)

Sprint 3 deferral rationale quote:

> "Patterns are scattered across the codebase as of v2.37.3.
> Encoding them in a skill now would freeze incomplete patterns
> and create maintenance debt.  Decision: create this skill
> after Sprint 6 [Apple helpers refactor] consolidates the
> patterns."

Resolution: Sprint v2.38.x Phase B = "Sprint 6" in the Sprint 3
sequence numbering.  Skill created Phase D of same sprint.

**Smoke-test confirmation:**
- `--kernel-type backward_fused_dkdv --head-dim 64 BK=32 WM=4`
  correctly returns FAIL (would exceed M5 CU register file by
  ~26 KB) with mitigation recommendation pointing to the Phase 2.O2
  split-kernel pattern that historically resolved this.
- `--kernel-type forward --head-dim 128 BK=32 WM=4` correctly returns
  MARGINAL with ~14 KB margin — matches Sprint 2 audit M1-NON-ACT-01
  analytic finding.

## Skill invocation protocol

Per `CLAUDE_V6_NAX.md` §AA:

1. CC invokes the skill via slash-command syntax (`/mlx-mfa-perf-audit`,
   `/mlx-mfa-release-audit`, `/mlx-mfa-bench-methodology`)
2. CC acts on the JSON output (apply fixes, escalate findings,
   update sprint deliverable docs)
3. CC logs the invocation in the relevant sprint deliverable
   (commits to `docs/audits/`, `docs/sprints/`, `devnotes/`) —
   the institutional-memory trace future audits will check

Each skill's `SKILL.md` defines exact inputs/outputs.  Each skill's
`README.md` lists trigger phrases and example invocations.

## Installation on a fresh dev machine

See `docs/skills/installation.md`.

## References

- `CLAUDE_V6_NAX.md` §X (auto-default audit), §X.5 (tool availability),
  §Z (public API path testing), §AA (skill invocation checkpoints)
- `docs/RELEASE_PHILOSOPHY.md` (auto-default principle + public API
  validation subsection)
- `docs/audits/v37-systematic-audit.md` (Sprint 2 audit — drives
  Skill 4 deferral decision)
- `docs/v6-nax/v2.37.x-perf-claim-audit.md` (reference incident
  driving Sprint 2 + Sprint 3 institutional work)
- `tests/test_release_notes_perf_claims.py` (executable §Z
  enforcement; perf-audit skill generalizes this pattern)
- `scripts/check_venv.sh` (Sprint 1 deliverable; release-audit
  Check 2 invokes this)

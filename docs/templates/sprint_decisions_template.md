# Sprint <name> — decisions

**Branch:** `<branch-name>`
**Started:** YYYY-MM-DD
**Status:** active | shipped | abandoned

## Mandate

[One paragraph: what is this sprint trying to accomplish?  What is
the success criterion?  Reference the originating prompt or
finding (e.g., audit doc, FALSIFIED experiment, user-reported bug).]

## Design context

[Cross-references to prior sprints, design docs, audit findings.
List the artifacts a future reader needs to understand THIS
sprint's scope.]

## Decisions

### DC1 — <decision title>

**Question:** What's the choice being made?

**Options considered:**
- Option A: ...
- Option B: ...
- Option C: ...

**Decision:** Option B.

**Rationale:** [Why this option.  Cite empirical data, design
constraints, or scope discipline (§6).]

**Tradeoffs accepted:** [What we give up by choosing B.]

**Reversibility:** [How hard is it to revisit if we're wrong?
Cheap (toggle env var) / Medium (code change but no API break) /
Expensive (API change, requires deprecation).]

### DC2 — <next decision>

[...]

## Skill invocations

Per `CLAUDE_V6_NAX.md` §AA.2, every sprint deliverable doc MUST
include a populated Skill invocations table.  Missing or empty
section → audit fails (Check 5 of `/mlx-mfa-release-audit`).

| Skill | Decision point | Timestamp (ISO) | Findings count | Action taken |
|---|---|---|---|---|
| /mlx-code-review | <when invoked, e.g., pre-merge of audit_runner.py> | YYYY-MM-DDTHH:MMZ | <N findings> | <what was done with findings> |
| /mlx-mfa-perf-audit | <e.g., post-bench perf claim verification> | YYYY-MM-DDTHH:MMZ | <verdict> | <e.g., REACHABLE → claim preserved> |

If a §AA mandatory checkpoint did NOT apply during this sprint
(e.g., no perf claims discovered), document the inapplicability
explicitly with one row:

| (skill name) | (checkpoint that didn't fire) | N/A | N/A | inapplicable: <reason> |

This makes the audit trail complete — silent absence is not
distinguishable from forgetting.

## STATUS

[Current sprint status.  Update as the sprint progresses.  When
sprint completes, set Status header above to "shipped" and
document the merge commit + tag.]

### Open questions

- [ ] [Questions not yet resolved; carried into the sprint or
      flagged for follow-up sprint]

### Blockers

- [ ] [What's stopping progress?]

### Next actions

- [ ] [Concrete next step]

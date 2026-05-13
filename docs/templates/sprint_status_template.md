# Sprint <name> — status

**Branch:** `<branch-name>`
**Last updated:** YYYY-MM-DD
**Status:** active | blocked | shipped | abandoned

## Progress summary

[2-3 paragraph current-state summary.  What's done, what's
in-flight, what's blocked.  Include commit SHAs for milestones
already reached.]

## Section completion

| Section | Status | Commit | Notes |
|---|---|---|---|
| A — [section title] | ✓ done | `abc1234` | [one-line note] |
| B — [section title] | in progress | — | [why not done yet] |
| C — [section title] | pending | — | depends on B |

## Skill invocations

Per `CLAUDE_V6_NAX.md` §AA.2, every sprint deliverable doc MUST
include a populated Skill invocations table.

| Skill | Decision point | Timestamp (ISO) | Findings count | Action taken |
|---|---|---|---|---|
| /mlx-code-review | <e.g., pre-merge of feat/foo branch> | YYYY-MM-DDTHH:MMZ | <N findings> | <e.g., 2 MEDIUM fixed before commit> |

Empty section = audit fails.  See `docs/templates/sprint_decisions_template.md`
for inapplicability documentation pattern.

## Risks

- [ ] [Risk 1: description + likelihood + mitigation]
- [ ] [Risk 2: ...]

## Open questions

- [ ] [Question 1]

## Blockers (active)

- [ ] [What's stopping progress right now?]

## Next actions

- [ ] [Concrete next step]
- [ ] [...]

## Exit criteria

Per the originating prompt's exit criteria:

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Branch merged to master, pushed
- [ ] Skill invocation log populated (§AA.2)

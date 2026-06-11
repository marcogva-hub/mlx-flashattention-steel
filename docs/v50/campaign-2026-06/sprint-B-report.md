# Campaign 2026-06 — Sprint B report: institutional codification

**Date**: 2026-06-11 · **Type**: documentation/audit-tooling only (zero
runtime changes) · **Status**: COMPLETE

## Starting-point verification

Meta-prompt expected tip `f0f7688`; actual tip was `c83a18e` — one
commit ahead (`docs(review): whole-repo review 2026-05 final report`,
the review's own report doc).  Benign delta; proceeded from actual tip
per orchestration rules.

Pre-condition check: pattern catalogue held exactly Patterns #1–#8
(items 1–8 of the recurring-patterns list embed #5/#6/#7; #8 is a
standalone section) — baseline assumption CONFIRMED, proceeded.

## Changes shipped

| # | Change | File | Commit |
|---|---|---|---|
| 1 | **Pattern #9** — generator/dispatch hardcoded-constant mismatch (silent partial-write): symptom signature (`NK·BK_source` threshold), mechanism, why-it-hid (cell-coincidence masking), prevention rule, generalization (grep generators for cfg-name reassignments), linkage to #1/#5/#7 | `docs/v50/audit-framing-inversions.md` (appended after #8, matching its format) | `docs(patterns)` |
| 4 | **Pattern #6 reinforcement note** — 2026-05 external application: V3/V4/V5 perf promotion correctly declined on M1-era evidence despite 22/22 accuracy passes on M5 | same file, inside entry 7 | same commit |
| 2 | **Check 9** in `/mlx-mfa-release-audit` — source/dispatch block-dimension consistency; per-(kernel, constant) manual checklist enumerating STEEL V1/V2/V3/V4/V5, backward dQ/dKV, sparse/GNA/sage/paged/flash-decode, V34 fwd + 9 bwd, conv3d NAX; mismatch = CRITICAL; automation harness flagged as Sprint A/C candidate. Also fixed a PRE-EXISTING stale count ("all six checks" → "all nine"; file had 8) | `~/.claude/skills/mlx-mfa-release-audit/SKILL.md` (system folder — not committable to repo; state recorded here) | n/a (outside repo) |
| 2b | **Perf-audit upstream-gate note** — perf numbers from a kernel failing Check 9 are invalid by construction (fast AND wrong) | `~/.claude/skills/mlx-mfa-perf-audit/SKILL.md` | n/a (outside repo) |
| 3 | **§AA.7** — dispatch/source constant parity in audit scope; extends §AA.6 to stable kernels; KD-ledger mechanism-not-symptom corollary ("a KD entry without a mechanism is an open investigation, not a verdict") | `CLAUDE_V6_NAX.md` (after §AA.6) | `docs(governance)` |
| 5 | **KD-5 ledger refinement** — Pattern #9 xref, ledger lesson, Marco-gated `MFA_FORCE_NATIVE_BWD` reconsideration candidate (flag only; dispatch-policy change requires Marco + M5 re-bench) | `docs/v50/known-debt-v2.50.md` | `docs(kd-ledger)` |

## Validation

| Check | Result |
|---|---|
| Pattern count 8 → 9, #9 matches #8's entry format | ✅ |
| Release-audit skill: 9 checks listed, Check 9 unambiguous + CRITICAL severity | ✅ (header audit: Checks 1–7 `###`, 8–9 `##` — pre-existing style inconsistency, preserved) |
| §AA.7 present, cross-refs Pattern #9 + §AA.6 | ✅ |
| Pattern #6 carries the 2026-05 note | ✅ |
| KD-5 reclassified with mechanism + #9 xref | ✅ (was already ROOT-CAUSE-FIXED from the review; Sprint B added xref/lesson/candidate) |
| Full suite | **1346 passed, 0 xfailed, 0 xpassed** — unchanged, as expected for a docs-only sprint |
| No runtime code touched | ✅ (`git diff --stat` over the 3 repo commits: docs + CLAUDE_V6_NAX.md only) |

## Epistemic notes

- Pattern #9 content: **verified** (mechanism re-derived from
  `csrc/mfa_attention.cpp` + `csrc/mfa_steel_bwd.cpp` source during the
  2026-05 review; fix validated by 4 passing SDPA-VJP assertions).
- "Stale six-checks count" in the release-audit skill: **verified**
  (file inspection; the count predated both Check 8 and Check 9).

## Candidates surfaced for Sprint A / C

1. **Gate-#9 automation harness** — script introspecting generator
   preamble constants vs cfg-derived dispatch values per cell
   (Sprint A Phase A.5 / Sprint C).
2. **V3/V4/V5 M5 re-bench** — accuracy proven on M5; perf verdicts are
   M1-era (Sprint C).
3. **`MFA_FORCE_NATIVE_BWD` deprecation reconsideration** — Marco-gated
   (dispatch-policy change); needs STEEL-bwd vs V34-bwd vs SDPA-vjp M5
   bench first.

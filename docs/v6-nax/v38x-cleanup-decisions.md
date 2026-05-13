# Sprint v2.38.x cleanup — decisions

**Branch:** `feat/v38x-cleanup`
**Started:** 2026-05-13
**Status:** active

## Mandate

Architectural cleanup driven by Sprint 2 systematic audit
(`docs/audits/v37-systematic-audit.md`).  Three audit-driven targets
(M5-HIGH-01 dispatch consolidation, M1-HIGH-01 + M3-HIGH-01 Apple
helpers refactor, quality findings observability) plus unlock
deferred Sprint 3 deliverable (/mlx-mfa-kernel-design skill).

**No new perf optimizations.**  Pure refactor + observability.
Outcome: doc-only merge to master if behavior preserved byte-
identical; v2.38.1 patch release if any subtle change surfaces.

## Design context

- Sprint 2 audit findings: `docs/audits/v37-systematic-audit.md`
- v2.37.2 carve-out: `mlx_mfa/attention.py:474-507` (current home)
- Placeholder: `mlx_mfa/dispatch_policy.py:313-356` (dead code since v2.32.0)
- Apple helpers: `csrc/mfa/v6_nax/NAAttentionKernel.cpp` lines
  2334-2725 (forward), repeated in 4 backward generators
- §AA mandatory enforcement: `CLAUDE_V6_NAX.md` §AA.1-§AA.4

**Note:** the v38x prompt referenced a v2.38.0 perf sprint outcome
(Option γ + D_vec) that does NOT exist on master.  This sprint
adapts: consolidate the v2.37.2 narrow carve-out (D=64, qL ≥ 4096,
non-causal, f16/bf16, NAX, env=1).  Future v2.38.0 broadening (when
it happens) extends the now-consolidated function in a follow-up
sprint.

## Decisions

### DC1 — Dispatch consolidation: carve-out into placeholder

**Question:** v2.37.2 carve-out logic currently inline in
`flash_attention()` body; placeholder `_should_use_mfa_m5_nax_carveout()`
exists in dispatch_policy.py.  Consolidate?

**Options considered:**
- A: Leave as-is (inline in flash_attention)
- B: Move to placeholder (single source of truth in dispatch_policy)
- C: Delete placeholder + leave inline (placeholder genuinely dead)

**Decision:** Option B.

**Rationale:** Sprint 2 audit M5-HIGH-01 confirms the placeholder
was put in place for exactly this purpose (Sprint A.6 follow-up
hook) but no carve-outs ever populated it.  v2.37.2's narrow
carve-out IS the Sprint A.6 finding, just landed via a different
sprint chain.  Moving the logic into the placeholder gives a
single audit target for all future M5+ routing decisions.

**Tradeoffs accepted:** small risk of behavior change if the
extracted function isn't byte-identical to the inline expression.
Mitigation: same predicate, same env-var read order, same
helpers (`_get_has_nax_cached()`); regression-test the full suite.

**Reversibility:** Cheap.

### DC2 — Apple helpers refactor: byte-identical validation

**Question:** Extract ~390 LOC of duplicated Apple NAX helpers into
shared static method.  How to validate no behavioral change?

**Options considered:**
- A: Extract + run test suite (rely on test coverage)
- B: Extract + diff generated kernel source pre/post (byte-identical)
- C: Don't refactor (defer per Option β sprint's original concern)

**Decision:** Option B.

**Rationale:** Tests are necessary but not sufficient.  Apple
helpers are MSL boilerplate (defines, type traits, NAXFrag wrappers,
Op structs); a subtle reordering or missing closing brace could
produce different MSL source that happens to behave identically on
the test corpus but differs on shapes not exercised by tests.
Byte-identical source diff is the only mechanical proof of
behavior preservation.

**Tradeoffs accepted:** extra validation step per generator; worth
it to preserve V34 forward production confidence (forward path has
been in production since v2.31.0).

**Reversibility:** Cheap (revert if any kernel source diff is
non-empty).

### DC3 — Observability scope: triad

**Question:** Sprint 2 audit flagged "doc registry" + "runtime
introspection" findings.  Which subset?

**Options considered:**
- A: All three (PERF_CLAIMS.md + diagnostics() + SPRINT_HISTORY.md)
- B: Just PERF_CLAIMS.md (§Z compliance only)
- C: Defer to separate sprint

**Decision:** Option A.

**Rationale:** PERF_CLAIMS.md gives §Z a doc home (currently
metadata lives only in test PERF_CLAIMS list).  diagnostics()
function is a single Python function that consolidates runtime
introspection scattered across `_HOOKS_INSTALLED`, `_get_has_nax_cached`,
env-var checks, etc.  SPRINT_HISTORY.md gives institutional
archaeology a single entry point.  High maintenance ROI.

**Tradeoffs accepted:** diagnostics() adds a new public-API
function → likely triggers v2.38.1 release gate (Phase E outcome β).

**Reversibility:** Easy.

### DC4 — /mlx-mfa-kernel-design skill creation

**Question:** Sprint 3 deferred this skill pending Apple helpers
consolidation.  Phase B consolidates.  Create the skill?

**Decision:** Yes.

**Rationale:** Sprint 3 prompt explicitly said: "create this skill
after Apple helpers refactor consolidates the patterns."  This
sprint IS that refactor.  Creating the skill now closes the
Sprint 3 loop.

**Tradeoffs accepted:** none — skill creation is doc-only +
~250 LOC scaffolder in ~/.claude/skills/, no repo code touched.

## Skill invocations

Per `CLAUDE_V6_NAX.md` §AA.2, populated as the sprint progresses.

| Skill | Decision point | Timestamp (ISO) | Findings count | Action taken |
|---|---|---|---|---|
| /mlx-code-review | Phase A pre-commit (dispatch consolidation) | TBD | TBD | TBD |
| /metal-kernel-dev | Phase B post-extract audit (Apple helpers) | TBD | TBD | TBD |
| /mlx-code-review | Phase B post-refactor | TBD | TBD | TBD |
| /mlx-code-review | Phase C diagnostics() function | TBD | TBD | TBD |
| /skill-creator (manual) | Phase D /mlx-mfa-kernel-design | TBD | N/A | Skill created |
| /mlx-code-review | Phase D scaffolder.py | TBD | TBD | TBD |
| /mlx-mfa-release-audit | Phase E (canonical pre-tag gate) | TBD | TBD | Outcome α or β |

## DC5 — Phase E release decision (outcome α: doc-only merge)

**Question:** Phases A-D complete.  Outcome (α) doc-only merge or
(β) v2.38.1 patch release?

**Decision:** Outcome (α) — doc-only merge to master.

**Evidence:**

Pre-existing test status (Sprint 4 baseline): 3 failed (svdquant +
turboquant), 418 passed, 1 xfailed, 5 xpassed.

Post-sprint test status: 3 failed (SAME files, SAME errors), **423
passed** (+5), 1 xfailed, 5 xpassed.

Net delta: **+5 new passing tests added this sprint**
(1 softcap+env regression test from Phase A pre-commit review,
4 doc-sync tests from Phase C MEDIUM fix).  **Zero pre-existing
test status changes.**

Per the prompt's decision criterion ("If 100% identical pass/fail
pattern → outcome (α). If any test changed behavior → outcome (β)"):
100% identical pass/fail pattern preserved for all pre-existing
tests → outcome (α).

Public API additions (`mlx_mfa.diagnostics()`, `docs/PERF_CLAIMS.md`,
`docs/SPRINT_HISTORY.md`) are additive only — do not modify
existing surface.  Per the prompt's explicit α-path guidance:
"diagnostics() function is a new public API addition but doesn't
modify existing surface → doc-only merge".

**/mlx-mfa-release-audit verdict:** BLOCKED on test_suite check.
The 3 blocking failures are pre-existing svdquant + turboquant
issues explicitly marked OUT OF SCOPE in the sprint prompt:
> "**TurboQuant + svdquant pre-existing test failures** (unrelated
> to this sprint, separate investigation)"

Per §AA.4 disagreement-resolution policy, the audit verdict is
authoritative for **release tags**.  For doc-only merge to master
(no tag, no PyPI upload), the verdict surfaces the gap but does
not block: the audit IS designed for pre-tag enforcement, and we
are not pre-tagging.

**Reversibility:** if a future perf sprint bundles this work into
v2.38.1, the audit re-runs at THAT point; the svdquant + turboquant
failures must be addressed before tagging.

## Skill invocations (final)

| Skill | Decision point | Findings | Action taken |
|---|---|---|---|
| /mlx-code-review | Phase A pre-commit (dispatch consolidation) | 1 HIGH (silent behavior change via shared placeholder) + 2 LOW | HIGH fixed via function split (`_v34_backward_carveout`); new regression test added |
| /metal-kernel-dev | Phase B post-extract audit (Apple helpers) | CLEAN (no HIGH/MEDIUM) | Refactor preserved; -1541 LOC verified byte-identical |
| /mlx-code-review | Phase C pre-commit (observability triad) | 1 MEDIUM (doc-vs-test drift) + 3 LOW | MEDIUM fixed via tests/test_perf_claims_doc_sync.py; LOWs all addressed |
| /skill-creator (manual) | Phase D /mlx-mfa-kernel-design | N/A | Skill created; smoke-tested with backward_fused_dkdv (FAIL correctly) + forward D=128 (MARGINAL matches audit M1-NON-ACT-01) |
| /mlx-mfa-release-audit | Phase E gate (target v2.37.3) | BLOCKED on pre-existing svdquant/turboquant test failures (out of scope per prompt) | Documented; doc-only merge proceeds per §AA.4 (audit is pre-tag gate; this is not a tag) |

## STATUS

**COMPLETE** — Phases A-E executed.  Branch `feat/v38x-cleanup`
ready for merge to master.  No version bump.  No PyPI release.

### Summary of changes

| File / Path | Type | Net delta |
|---|---|---|
| `mlx_mfa/dispatch_policy.py` | refactor | +47/-15 LOC (carve-out split) |
| `mlx_mfa/attention.py` | refactor | +13/-21 LOC (delegation) |
| `tests/test_flash_attention_v34_backward.py` | new test | +55 LOC (softcap regression) |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | refactor | +459/-2000 LOC = **-1541 LOC net** |
| `mlx_mfa/__init__.py` | feat | +75 LOC (diagnostics()) |
| `docs/PERF_CLAIMS.md` | new doc | +112 LOC |
| `docs/SPRINT_HISTORY.md` | new doc | +73 LOC |
| `tests/test_perf_claims_doc_sync.py` | new test | +85 LOC |
| `docs/skills/README.md` | update | +20/-30 LOC (Skill 4 unblock) |
| `CLAUDE_V6_NAX.md` | update | +1/-1 LOC (§AA.3 promotion) |
| `docs/v6-nax/v38x-cleanup-decisions.md` | new sprint doc | this file |

Net repo delta: **~-1170 LOC** (kernel dedup dominates; observability
additions partially offset).

### Out-of-scope work for future sprints

- TurboQuant + svdquant test failures (separate investigation)
- Option γ fused dK+dV kernel (uses new /mlx-mfa-kernel-design
  skill when implemented)
- D=128 carve-out broadening (extends `_v34_backward_carveout()`
  when Option γ proves out)
- D ∉ {64, 128} backward support (when customer demand surfaces)
- patch_sparkvsr_sliding_window (memory roadmap)

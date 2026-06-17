# Prompt 5e — Dead Code + Tech Debt Analysis (v2.50 release prep)

**Branch tip**: `53c914c` (master). **Date**: 2026-05-14.
**Scope**: dead code, vestigial paths, doc proliferation, env-var
drift after v2.50 Prompt 5d Pattern #6 revert.

---

## 1. Findings table

| # | Path:line | Finding | Severity | Action | Rationale |
|---|---|---|---|---|---|
| F1 | `mlx_mfa/attention.py:422,1362,3649,4002,6232,6315,6397,6484` | 8 unused imports (`_ext`, `mfa_sage_forward`, `numpy as _np`, `mfa_attention_forward`, `mlx.core as mx` ×4) flagged by pyflakes | LOW | REMOVE | Pure dead imports; no behavioral impact |
| F2 | `mlx_mfa/attention.py:3445,3449,5792,6608-6638` | Unused locals (`BQ`, `BK`, `D`, `max_n_blocks`, `H_q`, `total_q`, `o_shape`) | LOW | REMOVE | Probably leftovers from refactors; verify each, then drop |
| F3 | `mlx_mfa/dispatch_policy.py:37 + 800` | `import math` shadowed by re-import at line 800 (pyflakes: "redefinition of unused") | LOW | REMOVE | Drop the duplicate at L800; the module-level import suffices |
| F4 | `mlx_mfa/dispatch_policy.py:440,444,451,459,473` | 5 f-strings with no placeholders | LOW | REMOVE | Replace `f"..."` with plain strings (lint hygiene) |
| F5 | `mlx_mfa/__init__.py:135` | `_invalidate_cached_env` imported but unused | LOW | REMOVE | Public-surface noise |
| F6 | `mlx_mfa/external_cache.py:13-14` | `mx`, `numpy` imported but unused | LOW | REMOVE | Dead at module head |
| F7 | `mlx_mfa/attention.py:2317-2395` (`_v6nax_sparse_hybrid_vjp` + factory) vs `:2461-2470` (`_v6nax_backward_vjp_sparse_full_native` + factory `_make_v6nax_sparse_full_native_vjp`) | Both paths coexist; per Pattern #6 (commit `cfacf55`) **hybrid is production default**, full-native is `MFA_V6_BWD_SPARSE_NATIVE=1` opt-in research path. Docstring at L2387 still says hybrid is "DEPRECATED in Prompt 5d Section A.4" — stale, reverted by Section B v3. | HIGH | DOCUMENT (fix docstring) + keep both | Hybrid is *not* deprecated; full-native is *opt-in research*. Docstring inversion confuses future maintainers. **Atomic fix**: rewrite docstring to reflect Pattern #6 revert. |
| F8 | `mlx_mfa/attention.py:3076-3119` Top-K dispatch | 4-path Top-K logic with overlapping env vars: `MFA_DISABLE_TOPK_NAX`, `MFA_DISABLE_TOPK_BISECT`, deprecated `MFA_TOPK_BISECT`, Python ref fallback | MEDIUM | DOCUMENT | All paths are intentional (bisection AUTO, mx.topk fallback, ref escape). `MFA_TOPK_BISECT` is back-compat only — the variable is read **nowhere** in source after Prompt 5c (verified: only mentions are in deprecation comments at L3079, L3093, and 6 test setenv calls). |
| F9 | `tests/test_v50_sprint_5b_section_b_topk_bisect.py:41,57,60,111,122,135` (6 setenv calls) | Tests still toggle `MFA_TOPK_BISECT=1` despite it being a no-op (default is now bisection) | MEDIUM | CONSOLIDATE | Re-target these tests: drop the setenv, validate the **default** (no env). The current tests assert "opt-in via env" which is no longer the production contract. **Atomic fix**: rename + flip semantics. |
| F10 | `ENV_VARS.md` (82 lines) | Missing all v2.40→v2.50 env vars: `MFA_ENABLE_V6_BACKWARD`, `MFA_V6_BWD_KERNEL`, `MFA_V6_BWD_SPARSE_NATIVE`, `MFA_DISABLE_TOPK_BISECT`, `MFA_DISABLE_TOPK_NAX`, `MFA_TOPK_BISECT` (deprecated), `MFA_DISABLE_ROPE_NAX`. Only `MFA_FORCE_NATIVE_BWD` (v2.36-era) is present. | HIGH | DOCUMENT | User-facing reference is incomplete. **Atomic fix**: extend ENV_VARS.md with a "v2.50 dispatch gates" table; cross-reference CHANGELOG `[Unreleased — for v2.50]`. |
| F11 | `README.md:7-43` | Narrative still anchored on v2.39.1 (Option γ outcome α, BK=32→16). No mention of v2.50 sprint outcomes (Pattern #6, sparse hybrid, Top-K AUTO, D=128 backward broadening). | HIGH | DOCUMENT | Pre-tag README rewrite required. Currently the first impression mis-states version. **Atomic fix**: add a "v2.50 highlights" section above the v2.39.1 block (or restructure as "Latest release: v2.50" with v2.39.1 as historical context). |
| F12 | `docs/PERF_CLAIMS.md` | Only v2.37.x + v2.38.1 + v2.39.1 entries. No v2.50 rows. Pattern #7 lesson (claims must carry numerical evidence) means each surviving v2.50 claim — sparse hybrid forward speedup (Sprint 1 density-threshold), Top-K bisection 3.85× — needs a registry entry + matching `tests/test_release_notes_perf_claims.py` parametrization. | HIGH | DOCUMENT | Blocking for v2.50 tag per `/mlx-mfa-release-audit` Check 4. **Atomic fix**: add 2-3 v2.50 active rows; demote Pattern-#6-falsified Sprint 5 "10× at d=0.1" projections to historical (it never landed as a claim). |
| F13 | `mlx_mfa/__init__.py:92-95,109,424-427,437` exports + `attention.py:3173-3411,3312,3363` | `flash_attention_speculative_verify`, `flash_attention_speculative_verify_paged`, `make_shared_prefix_cache`, `flash_attention_splitfuse`, `sage_output_correction` — all v1.1.0/v1.2.0 LLM-serving helpers that have **no v2.x test coverage**. `sage_output_correction` is explicitly a no-op per MEMORY.md (v1.2.0 finding). | MEDIUM | ESCALATE | Are these still part of the supported public surface? If yes → add tests (Rule 7 gap flag). If no → deprecate in v2.50 with a warning, remove in v2.60. **Question for Marco**: kept as untested API, deprecated, or removed? |
| F14 | `docs/v50/sprint1-backward-regression-status.md` + `sprint1-backward-regression-RESOLVED.md` | Two docs covering the same sprint, one superseded ("status" → "RESOLVED") | LOW | CONSOLIDATE (archive `-status`) | The `-status.md` is transient; keep only `-RESOLVED.md` as the canonical record. |
| F15 | `docs/v50/sprint-5*-status.md` (5 files: `sprint-5-prompt3-status`, `sprint-5-prompt5a-status`, `sprint-5c-section-a-status`, `sprint-5d-section-a-status`, `sprint-prompt5a-sectionB-xfails-status`) | All marked "status" — transient state docs from in-flight sprints, now complete | MEDIUM | CONSOLIDATE (archive) | Move to `docs/v50/.archive/`; keep landing-pad summary in a single `sprint5-status.md` (already exists at 47 LOC). |
| F16 | `docs/v50/*decisions*.md` (8 files) | Permanent decision-record docs — these are the institutional memory | KEEP | DOCUMENT | Per CLAUDE_V6_NAX.md retention policy: decisions = SOT. Index them in a `docs/v50/README.md`. |
| F17 | `docs/v50/audit-framing-inversions.md` + `phase-3b-architectures-comparison.md` + `section-a-v3-empirical-verification.md` + `section-b-v3-approach-5-empirical-skip-decision.md` + `known-issues-v2.50.md` + `test-cleanup-inventory.md` + `prompt-5e-audit-inventory.md` (this file's sibling) + `prompt-5e-dead-code-tech-debt.md` (this file) | Permanent reference docs: Pattern catalogue, architecture trade-offs, empirical-verification records | KEEP | DOCUMENT | Same as F16. These are the doc artifacts a future v2.60 reader will need. |
| F18 | `CHANGELOG.md [Unreleased — for v2.50]` | Sprint 5 "10× speedup at d=0.1" perf projection appears in CHANGELOG narrative + ⚠️ warning. Pattern #6 (Section B v3) falsified it. Reader must scan to "⚠️ PERF WARNING" to discover the inversion. | MEDIUM | CONSOLIDATE | Move the falsified projection to a "Falsified projections (historical record)" sub-section; lead with the empirical bench table + Pattern #6 finding. **Atomic fix**: re-order the section. |
| F19 | `MEMORY.md` | 378 lines; system reminder flags ">200 limit", "only part loaded" | LOW | CONSOLIDATE | Index entries to one-line summaries per Rule 0.5; move detail to per-topic files in `devnotes/`. Out of scope for v2.50 ship but should be flagged. |
| F20 | `mlx_mfa/dispatch_policy.py:373-388` (`_v6nax_backward_carveout`) vs `attention.py:4060-4115` (`_v6nax_eligible`) vs `csrc/mfa_v6_nax_primitive.cpp:625` (D-handling) | Three eligibility-gate sites for V6NAX backward. Per Pattern #5 multi-gate audit + Prompt 5b Section D broadening, all three are consistent (D ∈ {64,128}, qL≥2048, fp16/bf16, M5+ NAX, env-gated). **Verified**: comments explicitly cross-reference each other. | NONE | KEEP | Documented as intentional defense-in-depth. No drift detected. |
| F21 | Tests with `MFA_V6_BWD_SPARSE_NATIVE` coverage (search across `tests/`) | `tests/test_v50_sprint_5d_sparse_backward_native.py` covers the opt-in full-native path (11 tests). Hybrid default path covered separately in `test_v50_sprint_5c_sparse_backward_hybrid.py`. Both retained. | NONE | KEEP | Test split mirrors the production/research surface split. |

---

## 2. Recommended atomic commits (non-escalated)

1. **`chore(lint): drop unused imports + locals flagged by pyflakes`** — F1, F2, F3, F4, F5, F6 (all together; single-file changes; no behavioral impact). ~30 LOC delta.
2. **`docs(attention): fix stale hybrid-vjp deprecation docstring (Pattern #6 revert)`** — F7. Single docstring edit at `attention.py:2387-2395` to invert "DEPRECATED" claim. Reference `cfacf55`.
3. **`test(top-k): re-target bisection tests for AUTO default semantics`** — F9. Drop `setenv("MFA_TOPK_BISECT","1")` from 6 sites; assert default (no env) behaves as bisection; add 1 test asserting `MFA_DISABLE_TOPK_BISECT=1` opt-out path. ~20 LOC delta.
4. **`docs(env): extend ENV_VARS.md with v2.40→v2.50 dispatch gates`** — F10. New "v2.50 dispatch gates" section: `MFA_ENABLE_V6_BACKWARD`, `MFA_V6_BWD_KERNEL`, `MFA_V6_BWD_SPARSE_NATIVE`, `MFA_DISABLE_TOPK_BISECT`, `MFA_DISABLE_TOPK_NAX`, `MFA_TOPK_BISECT` (back-compat-only), `MFA_DISABLE_ROPE_NAX`. Each with type/default/scope/back-ref.
5. **`docs(readme): add v2.50 highlights block above v2.39.1 narrative`** — F11. ~40 LOC delta. Pattern #6, Top-K AUTO, sparse hybrid, D=128 broadening — one bullet each, with CHANGELOG link.
6. **`docs(perf-claims): register v2.50 active claims + falsify Sprint-5 projection`** — F12. 2-3 new rows in `docs/PERF_CLAIMS.md` + matching parametrizations in `tests/test_release_notes_perf_claims.py`.
7. **`docs(changelog): re-order v2.50 section to lead with empirical findings`** — F18. Move "10× projection" narrative to a "Falsified projections" sub-section beneath the Pattern #6 bench table.
8. **`docs(v50): archive 5 transient sprint-status docs`** — F14, F15. Move to `docs/v50/.archive/`; create `docs/v50/README.md` index categorizing remaining files as Decisions / References / Inventories.

---

## 3. Escalations (decisions required from Marco)

### E1 — F13: Vestigial LLM-serving helpers (`flash_attention_speculative_verify*`, `make_shared_prefix_cache`, `flash_attention_splitfuse`, `sage_output_correction`)

These were v1.1.0/v1.2.0 deliverables (Tracks J*, K*). No v2.x test coverage exists. `sage_output_correction` is a documented no-op (MEMORY.md). They remain in the public `__init__.py` surface.

**Binary trade-off**: 
- **(A) KEEP + RESURRECT**: Add v2.50 test coverage for each helper. Cost: +~200 LOC tests; preserves backward compat for downstream users (if any).
- **(B) DEPRECATE + REMOVE**: Add `DeprecationWarning` in v2.50, remove in v2.60. Cost: +20 LOC warnings; breaks downstream users who silently depend on them.

**Question for Marco**: do you know of any downstream user (mlx-lm fork, internal projects) consuming these APIs? If no → **B**. If unsure → **A** to be safe.

### E2 — F18: CHANGELOG positioning of Sprint 5 falsified projection

The "10× speedup at d=0.1" projection appears prominently in the current `[Unreleased]` section. Pattern #6 empirically falsified it. 

**Binary trade-off**:
- **(A) FULL TRANSPARENCY**: Keep the projection visible with the ⚠️ warning + Pattern #6 table side-by-side. Signals scientific rigor; reader sees the inversion narrative.
- **(B) CONSOLIDATED FALSIFIED-RECORD**: Move the projection to a sub-section "Empirical falsifications (institutional record)" at the bottom of the v2.50 section. Lead with what shipped + the Pattern #6 bench table.

**Recommendation**: **B**. Pattern #6 is the headline; the projection belongs in the audit-trail tail.

### E3 — STATUS docs proliferation strategy (F14, F15, F16, F17)

30 files in `docs/v50/`. Categorization:

| Category | Count | Examples | Disposition |
|---|---|---|---|
| Transient status (sprint in-flight) | 6 | `sprint1-backward-regression-status.md`, `sprint-5-prompt3-status.md`, `sprint-5-prompt5a-status.md`, `sprint-5c-section-a-status.md`, `sprint-5d-section-a-status.md`, `sprint-prompt5a-sectionB-xfails-status.md` | **Archive** to `docs/v50/.archive/` |
| Resolved-after-status | 1 | `sprint1-backward-regression-RESOLVED.md` | **Keep** (canonical) |
| Decisions (institutional memory) | 8 | `sprint1-decisions.md` … `sprint-5d-decisions.md`, `phase-3b-decisions.md`, `phase-4b-complete-decisions.md`, `phase-4b-complete-dv-residual-decisions.md`, `phase-3b-approach-5-decision.md` | **Keep** |
| References (architecture, patterns) | 5 | `audit-framing-inversions.md`, `phase-3b-architectures-comparison.md`, `section-a-v3-empirical-verification.md`, `section-b-v3-approach-5-empirical-skip-decision.md`, `known-issues-v2.50.md` | **Keep** |
| Audit / dispatch inventories | 4 | `sprint-5b-section-d-dispatch-audit.md`, `sprint-5d-section-a-dispatch-audit.md`, `test-cleanup-inventory.md`, `prompt-5e-audit-inventory.md` | **Keep** |
| This audit | 1 | `prompt-5e-dead-code-tech-debt.md` | **Keep** |
| Aggregated status | 5 | `sprint3-status-phase3b.md`, `sprint4-status.md`, `sprint4-status-phase4b-complete.md`, `sprint5-status.md`, `sprint3-status-phase3b.md` | **Consolidate** to one `sprint-completion-log.md` |

**Binary trade-off**:
- **(A) ARCHIVE AGGRESSIVELY**: Move 11 transient/aggregated docs → `.archive/`. End state: ~19 active files + index. Cost: -11 root-level entries.
- **(B) MINIMAL TOUCH**: Only delete clear duplicates (F14). Keep 5 sprint-completion status files. End state: ~29 files.

**Recommendation**: **A**. Future v2.60 reader is better served by 19 indexed files than 30 unindexed.

---

## 4. Decision summary table

| Action class | Findings | Effort | Blocking for v2.50 tag? |
|---|---|---|---|
| REMOVE | F1, F2, F3, F4, F5, F6 | S (1 commit, ~30 LOC) | No (hygiene) |
| CONSOLIDATE | F9, F14, F15, F18, F19 | M (3-4 commits) | F18 partial |
| DOCUMENT | F7, F8, F10, F11, F12, F16, F17 | M (4 commits) | **F10, F11, F12 BLOCKING** |
| ESCALATE | F13 (E1), F18 framing (E2), F14/F15 doc strategy (E3) | Decision-gated | E2 if affects CHANGELOG |
| KEEP / NO ACTION | F20, F21 | - | - |

**Critical-path for v2.50 tag**: F10 (ENV_VARS.md), F11 (README v2.50 narrative), F12 (PERF_CLAIMS.md registry) — all DOCUMENT-class — must land before any `git tag v2.50.0`. `/mlx-mfa-release-audit` Check 4 will FAIL on F12 alone.

---

*Word count: ~1480.*

# mlx-mfa institutional sprint history

Canonical index of architectural / procedural / methodology sprints
with deliverable docs.  Future archaeology entry point: read this
table top-to-bottom for the institutional context behind any version.

For per-commit history, see `git log`.  For per-release content, see
`CHANGELOG.md`.  This file fills the gap between commits and releases
— the **why** behind clusters of work.

---

## v2.36.x — auto-default principle + canonical methodology

| Sprint | Date | Type | Outcome | Primary deliverable docs |
|---|---|---|---|---|
| **U** (Unification) | 2026-05-12 | Architectural | **v2.36.0 SHIPPED** — auto-hook on import + Sprint U auto-routing through public `flash_attention*` / `sparse_attention*` / `conv3d_nax_forward` surfaces | `docs/RELEASE_PHILOSOPHY.md` (canonical statement) |
| **Canonical methodology** | 2026-05-12 | Methodology | **v2.36.1 SHIPPED** — V2 sparse graduates to shape-aware default; canonical §4.2 protocol (10-warmup + 100-continuous) calibrated via 7/7 shapes CONFIDENT or BOUNDARY | `docs/methodology/canonical-protocol.md`, `docs/methodology/canonical-bench-results.md` |

## v2.37.x — V34 backward + institutional amendment

| Sprint | Date | Type | Outcome | Primary deliverable docs |
|---|---|---|---|---|
| **V34 forward lse-write (BLK1)** | pre-2026-05-13 | Infrastructure | Merged, **v2.37.0 base** — V34 forward writes natural-log lse to device memory, enabling V34 backward kernel consumption (DC0 resolution) | `docs/v6-nax/v34-backward-decisions.md` |
| **V34 backward Option β** | pre-2026-05-13 | Architectural | **v2.37.0 SHIPPED** — V34 NAX-direct backward kernels (dQ + multi-SG dK/dV WM=4 Q-row partition + forward-fusion) ship as SHIP_OPT_IN via `MFA_ENABLE_V34_BACKWARD=1` | `docs/v6-nax/v34-backward-status.md`, `docs/v6-nax/v34-backward-option-gamma-design.md` (deferred follow-on) |
| **DC12 routing relaxation** | 2026-05-13 | Patch | **v2.37.1 SHIPPED** — `force_v34` param relaxes DC12 constraint; V34 backward eligible for D=64 small-Nk shapes | `docs/releases/v2.37.1-release-notes.md` |
| **v2.37.2 silent integration bug fix** | 2026-05-13 | Bugfix | **v2.37.2 SHIPPED** — narrow carve-out in `flash_attention()` body fixes silent SDPA fallback for V34 backward; D=64 qL≥4096 now 1.81-1.82× faster via documented public API | `docs/releases/v2.37.2-release-notes.md` |
| **Institutional amendment (§Z + §AA)** | 2026-05-13 | Procedural | **v2.37.3 SHIPPED** — §Z public API path testing rule + §AA skill invocation checkpoints + perf claim audit retraction of v2.37.1 qL=2048 1.44× claim + D=128 reclassification | `docs/v6-nax/v2.37.x-perf-claim-audit.md`, `docs/releases/v2.37.3-release-notes.md` |

## Sprint 1-4 — institutional hardening (all post-v2.37.3)

| Sprint | Date | Type | Outcome | Primary deliverable docs |
|---|---|---|---|---|
| **Sprint 1: venv consolidation** | 2026-05-13 | Infrastructure | doc-only merge to master | `CLAUDE.md` "Canonical Python environment" + `scripts/check_venv.sh` + `CLAUDE_V6_NAX.md` §X.5 |
| **Sprint 2: systematic audit** | 2026-05-13 | Audit | doc-only — 8 HIGH / 13 MEDIUM / 7 LOW / 8 NON-ACTIONABLE findings | `docs/audits/v37-systematic-audit.md` |
| **Sprint 3: mlx-mfa specialized skills** | 2026-05-13 | Skill creation | doc-only — created `/mlx-mfa-perf-audit`, `/mlx-mfa-release-audit`, `/mlx-mfa-bench-methodology`; deferred `/mlx-mfa-kernel-design` post-helpers-refactor | `docs/skills/README.md`, `docs/skills/installation.md` |
| **Sprint 4: §AA hardening** | 2026-05-13 | Procedural | doc-only — §AA upgraded from "recommended" to MANDATORY BLOCKING with halt protocol; templates pre-include Skill invocations table; first /mlx-mfa-release-audit dry-run validated | `CLAUDE_V6_NAX.md` §AA.1-§AA.4 + `docs/templates/` + `docs/audits/aa-hardening-dryrun.md` |

## v2.38.x — architectural cleanup → shipped releases

| Sprint | Date | Type | Outcome | Primary deliverable docs |
|---|---|---|---|---|
| **v2.38.x cleanup** | 2026-05-13 | Refactor + observability | (α) doc-only merge → folded into v2.38.0 release; Phase A/B/C/D infrastructure shipped. | `docs/v6-nax/v38x-cleanup-decisions.md` + this doc (`docs/SPRINT_HISTORY.md`) + `docs/PERF_CLAIMS.md` |
| **v2.38.0** | 2026-05-13 | Refactor cleanup release | Helper extraction (`_v34_eligible`, `_v34_backward_vjp`) + dead placeholder deletion + investigation foundation docs. **No perf claim.** | `CHANGELOG.md` `[2.38.0]` + `docs/v6-nax/v38-implementation-decisions.md` |
| **v2.38.1** | 2026-05-13 | Perf optimization release | D_vec precompute device buffer; **D=64 V34 backward 1.91× / 1.87× / 1.80× vs SDPA-vjp at qL∈{4096,8192,16384}** (was 1.75-1.79× v2.37.3); D=128 unchanged (carve-out D=64 hard-gated). | `CHANGELOG.md` `[2.38.1]` + `docs/v6-nax/v38-1-implementation-decisions.md` + `docs/v6-nax/v38-1-perf-claim-audit.md` |

Phases of original v2.38.x cleanup:
- **A: Dispatch consolidation** (M5-HIGH-01) — v2.37.2 carve-out moved into dedicated `_v34_backward_carveout()` function; a placeholder `_should_use_mfa_m5_nax_carveout()` was retained for genuine Sprint A.6 hooks but was **subsequently deleted in v2.38.0 P3 Phase C** (dormant since v2.32.0, no Sprint A.6 carve-outs ever materialized; if a future Sprint A.6 surfaces empirically-validated MFA-winning shapes on M5+ NAX canonical D, re-introduce a named function from the `head_dim ∈ {64, 128}` branch in `should_use_mfa()`). ✓ Committed `f7a04ce`.
- **B: Apple helpers refactor** (M1-HIGH-01 + M3-HIGH-01) — extracted 390-LOC helpers from 5 generators into `naxHelpersBlock()` static method; byte-identical forward kernel source verified; -1541 LOC net. ✓ Committed `c509b4b`.
- **C: Observability triad** (quality) — `docs/PERF_CLAIMS.md` registry (now extended with v2.38.1 D_vec rows) + `mlx_mfa.diagnostics()` runtime introspection + `docs/SPRINT_HISTORY.md` (this file).  ✓ Done; extended in v2.38.1.
- **D: /mlx-mfa-kernel-design skill** — unblocked by Phase B; encodes consolidated kernel-design patterns.  ✓ Created at `~/.claude/skills/mlx-mfa-kernel-design/`.
- **E: Release decision via `/mlx-mfa-release-audit`** — α doc-only chosen; Sprint v2.38.x infrastructure folded into v2.38.0 cleanup release. ✓ Decision DC5.

## v2.38.1 — D_vec precompute (M2-HIGH-01)

Single-phase optimization sprint.  D = rowsum(dO ⊙ O) precomputed once on host via MLX and passed as shared device buffer to V34 backward kernels (dQ + split-dK + legacy-fused-dKdV).  Eliminates 2 in-kernel rowsums per default-path V34 backward call.

| Phase | Status | Commit / artefact |
|---|---|---|
| A.1-A.6: D_vec wire-up (3 kernels + 3 Primitives + 3 bindings + `_v34_backward_vjp`) | ✓ Committed | `bf62af0` |
| A.7: Three-axis validation + /mlx-debug-forensics | ✓ HIGH SHIP | 5-axis byte-equivalence audit |
| A.8: /mlx-mfa-bench-methodology + /mlx-mfa-perf-audit + /mlx-mfa-release-audit | ✓ SHIP-green all gates | `3f4b786` (bench data + audit doc + CHANGELOG) |
| Release: tag + PyPI + GH | ✓ LIVE | https://pypi.org/project/mlx-mfa/2.38.1/ |

Perf delta (PUBLIC AUTO API, M5 Max NAX, 3 sessions × 4w+12i):
- D=64 qL=4096:  10.57ms → 9.59ms  (-9.3% wall, 1.75× → **1.91×** vs SDPA-vjp)
- D=64 qL=8192:  39.90ms → 38.27ms (-4.1% wall, 1.79× → **1.87×**)
- D=64 qL=16384: 170.81ms → 166.33ms (-2.6% wall, 1.75× → **1.80×**)
- D=128: unchanged (carve-out D=64 hard-gated; AUTO routes to SDPA-vjp)

Variance ratios all <1.15 per §AA.4.  Improvement decays with qL as eliminated rowsum work shrinks relative to K-loop time.

---

## Pattern observations across sprints

Three structural patterns emerge from the v2.36.x → v2.38.x arc:

1. **Architectural sprint → procedural amendment** — each major
   feature ship (v2.36.0 auto-hooks, v2.37.0 V34 backward) has
   surfaced an institutional gap that the next sprint patches.
   Sprint 4 codified this as §AA mandatory blocking (skills run
   at gate points; gaps surface immediately, not 3 sprints later).

2. **Audit-driven refactor** — Sprint 2's systematic audit produced
   a categorized findings registry; v2.38.x cleanup (this sprint)
   executes 3 of the 8 HIGH findings.  Audit doc is the canonical
   source for "what to refactor next."

3. **Two release tracks coexist** — across this history, PyPI
   shipments (6: v2.36.0, v2.36.1, v2.37.0, v2.37.1, v2.37.2, v2.37.3)
   carry architectural / bugfix changes, while doc-only merges
   (4: Sprints 1-4) carry institutional / procedural / infra
   additions that don't change runtime behavior.  Each track has
   its own gate (`/mlx-mfa-release-audit` for PyPI; sprint-deliverable
   review for doc-only).  v2.38.x cleanup outcome may take either
   path depending on Phase E gate verdict.

---

## Cross-references

- `CHANGELOG.md` — per-release content
- `CLAUDE_V6_NAX.md` — institutional rules (§3.5, §4, §X, §X.5, §Z, §AA)
- `CLAUDE.md` — top-level project context
- `docs/RELEASE_PHILOSOPHY.md` — auto-default principle
- `docs/PERF_CLAIMS.md` — §Z claim registry (created in this sprint)
- `docs/skills/README.md` — mlx-mfa-* skill inventory (Sprint 3)
- `docs/templates/` — sprint deliverable templates (Sprint 4 §AA.2)
- `docs/audits/` — sprint audit deliverables
- `docs/v6-nax/` — per-sprint decisions / results / status docs

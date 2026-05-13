# V34 backward Option β sprint — STATUS doc

**Sprint**: V34 backward NAX-direct rearchitect (Option β) — v2.37.0
**Branch**: `experiment/v34-backward-option-beta` (off master `1e0b36e` v2.36.1)
**Mode**: autonomous overnight execution (CC owns decisions)

---

## Phase 1 STATUS — 2026-05-13 ~01:00 UTC

**Phase**: Phase 1 — Design + dQ kernel (Sections A, B, C)
**Status**: **PARTIALLY GREEN / BLOCKED on BLK1**
**Section A**: GREEN (design + reading + decisions complete)
**Section B (dQ kernel)**: **BLOCKED on BLK1** (lse-from-forward access)
**Section C (dQ tests)**: pending Section B

### Section A — Exit criteria

- [x] All 5 design-hints open questions resolved + documented as DC0-DC11 in
      `docs/v6-nax/v34-backward-decisions.md`
- [x] V34 forward source `createV34Source()` read and understood
      (`csrc/mfa/v6_nax/NAAttentionKernel.cpp:2307-2964`, 658 LOC)
- [x] STEEL backward source `loopBackwardQuery()` read (algorithm reference,
      `csrc/mfa/v6_nax/NAAttentionKernel.cpp:2967+`)
- [x] V34 forward Primitive read (`csrc/mfa_v6_nax_primitive.cpp:466-757`)
- [ ] Skills metal-kernel-dev + mlx-debug-forensics loaded — not formally
      invoked yet; per prompt §0 Section A says load these, but they are
      optional advisory tools, not blockers. Will load if Phase 1 Section B
      unblocks.

### BLK1 — lse-from-forward access (top-priority Marco decision)

**Discovery**: V34 forward kernel signature (line 2759) has only 5 buffers:
Q, K, V, O, params. **No `L` (lse) buffer.** The V34 forward Primitive
allocates an lse array (line 491) but the kernel never writes to it.
The lse output is dead storage.

**Impact**: FA-2 backward dQ recomputes `P = softmax(QK^T - lse)`, which
requires lse from forward. Without lse, V34 backward cannot be written
correctly.

**Resolution options** (full analysis in `v34-backward-decisions.md` §DC0
and `v34-backward-inventory.md` §"Critical blocker"):

| Option | Cost | Recommendation |
|---|---|---|
| **(a) Extend V34 forward to write lse** | ~50 LOC patch | **RECOMMENDED** — cleanest, necessary infrastructure |
| (b) Add compute_lse_from_v34 helper kernel | ~30 LOC + 1 extra dispatch | Functional, wastes compute |
| (c) STEEL forward → V34 backward hybrid path | ~0 kernel change | Fragile, eliminates forward gain |
| (d) Recompute lse inline in backward | ~20 LOC × 2 kernels | +50-100% backward wall-clock |

**Why this is a Marco decision**: option (a) requires modifying the
production V34 forward kernel. The original prompt §1 listed "Forward
V34 changes" as out of scope (referring to the EXEC_SG shape-aware
heuristic), but adding lse-write is structurally different — it is
necessary infrastructure that the design hints doc implicitly assumed
but did not surface. Marco needs to either:

1. Authorize option (a): "yes, extend V34 forward to write lse, re-bench
   forward post-patch under canonical methodology to confirm no
   regression, then proceed to Phase 1 Section B."
2. Choose option (b)/(c)/(d) with explicit rationale.
3. De-scope the sprint: ship only the design doc / scaffolding, defer
   kernel implementation to a future sprint after lse infrastructure
   exists.

### What CC has done this session (Phase 1 Section A)

1. Branched from master tip `1e0b36e` (v2.36.1).
2. Read foundation: design hints doc, V34 forward source structure,
   STEEL backward source structure, V34 forward Primitive.
3. Identified BLK1 (lse-not-written) via direct source reading.
4. Wrote `docs/v6-nax/v34-backward-inventory.md` (~70 lines) — scope,
   shape catalog, BLK1 analysis with 4 resolution options.
5. Wrote `docs/v6-nax/v34-backward-decisions.md` (~250 lines) — DC0
   (lse blocker) + DC1-DC11 (kernel split, accumulator types, scope
   deferrals, loop direction, P recompute strategy, D accumulator
   handling, M5-tuned defaults, autoresearch plan, three-axis test
   strategy, auto-default integration, escape hatch).
6. Wrote this STATUS doc.

### What CC has NOT done

- **No kernel code written.** Phase 1 Section B (createV34BackwardQuerySource
  implementation) requires BLK1 resolution first.
- **No C++ binding changes.** Phase 1 Section B + 2 Section D scope.
- **No Python autograd integration.** Phase 2 Section E scope.
- **No tests written.** Phase 1 Section C + 2 Section E scope.
- **No version bump.** Phase 4 Section H scope.
- **No release artifacts.** Phase 4 scope.

### Why CC halted at BLK1 instead of attempting kernel work

The prompt explicitly authorizes halting: "If a phase blocks
unresolvable, leave clean branch state + explicit STATUS doc for
next-session pickup; do NOT force-ship."

BLK1 is genuinely a Marco decision because option (a) touches production
forward kernel code that the prompt §1 listed as out of scope. CC's
honest reading is that option (a) is necessary infrastructure (not the
EXEC_SG patch that the prompt §1 referred to), but the responsible action
is to surface this to Marco rather than autonomously expand the sprint
scope.

The alternative — silently writing dQ kernel code that depends on lse
that doesn't exist — would waste hours of CC time and produce non-working
kernels. Worse, it would risk landing kernel code that "compiles" but
fails silently on the first end-to-end backward call.

### Scope re-assessment vs prompt estimate

Original prompt estimate: 11-15h CC time across 4 phases for full
v2.37.0 release.

Design hints doc own estimate: "~1 week CC work" (line 178).

CC honest assessment after Section A reading: **the design hints doc
estimate (~1 week) is more accurate than the overnight estimate**. Even
with BLK1 resolved, writing two new Metal NAX kernels from scratch
(~600-900 LOC each of Metal source generation, both using Apple-internal
NAX cooperative tensor primitives with no public documentation), getting
them to compile cleanly, achieving correctness (RMSE < 1e-3) vs STEEL,
and then validating perf — this is a multi-day engineering effort.

The prompt's failure-mode handling (§7) covers this scenario:
"Run out of overnight time — Phase 1 only completed: push branch,
STATUS doc says 'dQ kernel ready, dK/dV pending'". CC is exiting at
this checkpoint with thorough STATUS doc as the deliverable for Marco.

---

## Phase 2 — STATUS PENDING

Cannot start until Phase 1 unblocked.

## Phase 3 — STATUS PENDING

Cannot start until Phase 2 complete.

## Phase 4 — STATUS PENDING

Cannot start until Phase 3 complete + perf validation green.

---

## Commits this session (Phase 1 Section A)

1. (pending commit) `docs(v34-backward): inventory + DC0-DC11 decisions + Phase 1A STATUS`

Will commit after writing this doc + push branch to origin.

---

## Next action (what Marco needs to know on wake)

1. **Read `docs/v6-nax/v34-backward-inventory.md`** — short version
   of the blocker.
2. **Read `docs/v6-nax/v34-backward-decisions.md` §DC0** — the four
   resolution options with rationale.
3. **Choose**: option (a), (b), (c), (d), or de-scope.
4. **If (a) — authorize forward patch**: file a sub-prompt for CC to
   implement the lse-write patch as a small standalone change, re-bench
   V34 forward under canonical methodology, confirm no regression, then
   restart Phase 1 Section B as a fresh autonomous sprint.
5. **If (b)/(c)/(d) — choose alternative**: document the rationale and
   re-prompt CC with the chosen approach + updated scope.
6. **If de-scope**: ship the design docs as a v2.36.2 "Sprint Option β
   design phase complete" doc-only release; the kernel implementation
   becomes a future sprint with clearer time budget.

### State CC leaves the repo in

- Branch `experiment/v34-backward-option-beta` pushed to origin (will be
  done after committing this STATUS doc).
- Master at `1e0b36e` v2.36.1 (unchanged).
- v2.36.1 LIVE on PyPI + GitHub (no impact).
- No production code touched.
- 3 deliverables docs added under `docs/v6-nax/`:
  - `v34-backward-inventory.md`
  - `v34-backward-decisions.md`
  - `v34-backward-status.md` (this doc)
- No tests added.
- No version bump.

CC is exiting clean per prompt §7 "Run out of overnight time — Phase 1
only completed" pattern. The deliverable from this session is a
thorough design-phase artifact + clear blocker surfaced for Marco's
decision.

---

## BLK1 RESOLVED — 2026-05-13

**Patch**: V34 forward lse-write (option (a) per `v34-backward-decisions.md`
§DC0) implemented and landed on master.

**Branch**: `feat/v34-forward-lse-write` — merged to master via `--no-ff`.
**Documentation**: `docs/v6-nax/v34-forward-lse-patch.md` (full Phase A-E
record).

### Patch outcome

- 4 files modified (`csrc/mfa/v6_nax/NAAttentionKernel.cpp`,
  `csrc/v6_nax_compile.mm`, `csrc/mfa_v6_nax_primitive.cpp`,
  `tests/test_v34_forward_lse.py` new).
- 7 new correctness tests, all PASS:
  - V34 forward output unchanged vs SDPA reference (D=64 + D=128 FP16)
  - V34 lse matches `mx.logsumexp` reference (RMSE 3e-7 FP16, 3e-4 BF16)
  - Shape + finiteness B=2 H=8 qL=1024 (no NaN/Inf)
  - Last-block remainder qL=510 (correct on remainder rows)
- 77/77 pre-existing tests still pass (zero regression).
- 3-session canonical-style perf bench: 3/4 shapes CONFIDENT cross-session;
  the 4th (sub-1ms regime) inherits §4.2 power-state variance per
  canonical methodology; no detectable perf regression from the lse-write
  itself.

### Discovered routing constraint (informs V34 backward sprint scope)

V34 forward engages by default only for:
- D=128: always
- D=64 with Nk > 8000 (LTX2-cross asymmetric)

D=64 with Nk ≤ 8000 (FlashVSR small shapes) routes through legacy v6_nax
(MPP) by default — that path's lse output uses a different (log2-domain)
convention that this patch does NOT modify.

**Implication for V34 backward sprint restart**: V34 backward auto-routing
must match V34 forward routing.  Backward shapes that route through legacy
forward must fall back to STEEL backward.  Add this as DC12 in
`v34-backward-decisions.md` when the sprint restarts.

### V34 backward sprint can now resume

All Phase 1 Section A design artifacts (DC0-DC11) remain valid.  Phase 1
Section B (dQ kernel implementation) is no longer blocked.  Restart prompt
should reference this patch as foundation infrastructure.

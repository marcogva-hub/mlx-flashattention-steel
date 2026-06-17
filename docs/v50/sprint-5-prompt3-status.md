# v2.50 Sprint 5 — V6NAX backward block-sparse NAX — DEFERRED (Prompt 3)

**Sprint date**: 2026-05-14
**Status**: **DEFERRED — dependency on Phase 4b-complete.B (K-parallel kernel dV residual)**

## TL;DR

Sprint 5 (V6NAX backward block-sparse) was scoped in the user's Prompt 3
prompt as "Section C: bundled with Phase 4b-complete per CC's recommendation
Prompt 2, dependency resolved" — assuming Section B Phase 4b-complete would
ship cleanly.

Section B Phase 4b-complete this session shipped PARTIAL:
- Critical `compile_v6nax_backward_pipeline` `isCausal=false` hardcoded bug
  fixed (Prompt 2's Phase 4b dQ was a silent no-op pre-fix — now WORKS)
- dQ kernel causal validated (RMSE 8.7e-6 at qL=2048 D=64 fp16)
- 4 K-parallel kernels (dV split, dK split, dKV legacy fused, dKdV fused)
  have causal infrastructure compiled in but produce dV with structural
  ~25× under-counting residual (RMSE 2.7e-3 vs 1e-3 bound)
- Production eligibility gates retained on `causal=True` for safety

Sprint 5 (block-sparse backward) would extend the SAME 4 K-parallel kernels
with block_mask buffer + per-block early-exit + per-element mask. The dV
residual that affects causal-only mode would similarly affect any K-parallel
backward operation, including block-sparse.

Per §AA.1 halt protocol, Sprint 5 deferred until Phase 4b-complete.B's
K-parallel kernel residual is resolved.  Same kernels need the fix; bundle
the resolution + sparse extension in one focused session (~3-4h CC total).

## Why Sprint 5 cannot ship cleanly this session

Sprint 5's design (per Prompt 3 Section C.1):
- Add `block_mask` device buffer to V6NAXBwd*Params
- Per-tile early-exit when block_mask says tile is inactive
- Per-element mask within partial tiles

These additions LAYER on top of the K-parallel kernel structure. If the
underlying K-parallel structure has a dV under-counting bug (Phase 4b-complete.B
finding), block-sparse would inherit it. Specifically:
- Block-sparse Q-loop iteration over ACTIVE blocks: works mathematically
  if dV per active block is correct
- Per-active block, dV += P^T @ dO computation: shares the same Stile→P
  pattern that produces the dV residual in causal mode
- The dV residual would manifest in block-sparse at high density (where
  many Q-blocks contribute) but mask at low density (only diagonal blocks
  active)

The user's Prompt 3 Section C explicitly notes this risk:
> Worth verifier que les deux patterns combinent sans conflit register
> pressure.

The conflict isn't register pressure — it's the underlying dV correctness.

## Recommended bundled future session

**Phase 4b-complete.B + Sprint 5 combined** (~3-4h CC):

1. **Phase 4b-complete.B kernel-debug** (~1.5-2h):
   - Add per-element sentinel writes to dV kernel
   - Compare V6NAX dV vs Python-reference at per-(r, c) level
   - Identify the structural under-counting root cause
   - Fix the bug in one kernel; replicate to all 4 K-parallel kernels
   - Validate dV RMSE drops below 1e-3 bound across kernel modes
   - Lift `_v6nax_eligible` and `_v6nax_backward_carveout` causal gates

2. **Sprint 5 block-sparse extension** (~1-2h, on top of working
   K-parallel kernels):
   - Add `block_mask` buffer to V6NAXBwd*Params
   - Add `#define V6NAXBWD*_SPARSE` macro
   - Per-tile early-exit + per-element mask block
   - Wire through to `flash_attention_sparse(backward='v6nax')` opt-in

3. **Three-axis validation + bench** (~30-60min):
   - Causal-only (Phase 4b-complete.B validation)
   - Sparse-only (Sprint 5 validation)
   - Causal + sparse combination (interaction test)
   - Non-causal non-sparse regression
   - 3-session §AA.4 cross-session bench

Bundling saves overhead — both work items touch the same 4 K-parallel
kernels and benefit from a single rebuild + test cycle.

## What's still preserved this session

- **Master tip 48372f1 (post-Section-A halt)** before Section B work
- Critical isCausal-plumbing fix (~50 LOC across 3 C++ files + 1 Python file)
  — DOES NOT regress anything, makes Prompt 2 Phase 4b dQ actually fire
- 4 K-parallel kernel mask blocks (compile-time gated off via eligibility gate)
- 1140 baseline tests pass (zero new regressions; 50 pre-existing
  v2.38.1 D_vec API mismatch unchanged — deferred to Prompt 4)

## §AA.1 halt protocol invocation

Sprint 5 halts cleanly:
- No code changes for Sprint 5 itself (only this STATUS doc)
- Section B Phase 4b-complete partial-state work ships at master
- Production behavior unchanged (causal callers still SDPA-vjp)
- Future session has clear scope: resolve K-parallel dV residual, then
  add block-sparse on top

## Master state post-Prompt-3 (full state)

- `master`: ~`<Section B partial merge>`
- Prompt 3 deliverables:
  - Section A: HALTED with STATUS doc (Top-K kernel deferred per §AA.5
    CONFIRMATION + `/metal-kernel-dev` NO_GO heap-maintenance design risk)
  - Section B: PARTIAL — Phase 4b-complete.A critical bug fix shipped;
    Phase 4b-complete.B K-parallel kernel residual deferred
  - Section C: DEFERRED with this STATUS doc (depends on Phase 4b-complete.B)
- Tests: 1140 passing (zero new regressions; 50 pre-existing unchanged)
- No version bump, no tag, no PyPI publication

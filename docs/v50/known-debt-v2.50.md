# mlx-mfa v2.50 known-debt registry

Items identified during Prompt 5e pre-release audit (Phases 1-3) that
require future investigation but are NOT BLOCKING v2.50 release.  Each
item carries: severity, current production impact, resolution roadmap.

---

## KD-1 — V34 backward sparse kernel mask shape mismatch (HIGH severity) — **RESOLVED v2.50.0 Prompt 5f Phase A**

**RESOLUTION DATE**: 2026-05-14 (Prompt 5f Phase A).  See CHANGELOG
v2.50 Phase A entry + `mlx_mfa/attention.py::_convert_mask_for_v34_bwd_kernel`
+ 17 tests in `tests/test_v50_prompt_5f_kd1_mask_shape_fix.py` +
C++ shape validation in 4 sparse Primitives (`csrc/mfa_v6_nax_primitive.cpp`).

**Identified by**: Phase 1.2 C++ code review (`docs/v50/prompt-5e-code-review-cpp.md`
HIGH-1 finding).

**Mechanism**: The 4 V34 backward sparse kernels (dQ + dV PoC + dK split
+ fused dKdV) index `block_mask` using V34 backward tile sizes
(`V34BWD_BQ`, `V34BWD_BK`):
- D=64 V34 bwd: BQ=32, BK=64 (NQ_v34 = qL/32, NK_v34 = kL/64)
- D=128 V34 bwd: BQ=64, BK=32 (NQ_v34 = qL/64, NK_v34 = kL/32)

But production callers (via `flash_attention_sparse` hybrid orchestrator
and `make_causal_block_mask`) pass a mask sized for FORWARD STEEL tile
sizes per `_steel_block_config`:
- D=64 forward STEEL: BQ=32, BK=64 (same as V34 bwd — no mismatch)
- D=128 forward STEEL: BQ=32, BK=32 (NQ_fwd=qL/32, NK_fwd=kL/32 — MISMATCH with V34 bwd D=128 NQ_v34=qL/64)

**Current production impact**:
- For D=64: NO MISMATCH.  V34 bwd dV BK=32 differs from V34 bwd D=64
  spec; let me verify... Actually `MFAV34BwdDVSparse::eval_gpu` hardcodes
  `BQ=64; BK=32` for dV regardless of D.  So mask access at
  `block_mask[qb * NK + tid.x]` with qb iterating to qL/64, tid.x in
  [0, kL/32).
- **Smoke tests pass** for all-True mask and block-causal because:
  - All-True: no positional indexing matters
  - Block-causal: first-row pattern happens to align across mask
    interpretations
- **Pathological sparse patterns** (e.g., block-diagonal, random low
  density) would silently produce wrong gradients in production hybrid
  path (where dV native sparse is invoked with the BT-shaped mask).

**Production safety mitigation**:
- Hybrid orchestrator (Prompt 5c, current production default per
  Pattern #6 revert): dQ/dK go through SDPA-vjp with bias mask
  (mathematically correct regardless of mask shape).  Only dV native
  sparse is affected — and its contribution is bounded by the per-row
  L_sparse normalization.
- Full native (opt-in `MFA_V34_BWD_SPARSE_NATIVE=1`): all 4 gradients
  affected.
- Section C wrapper (env unset): uses SDPA-vjp throughout — no V34
  sparse kernels invoked → NOT AFFECTED.

**Resolution roadmap** (Section A v4 follow-up):
1. Add Python-level mask conversion in `_v34_sparse_hybrid_vjp` and
   `_v34_backward_vjp_sparse_full_native` orchestrators:
   - Detect forward-shape mask vs V34-bwd-shape mask
   - Downsample/upsample as needed (logical OR across affected forward
     tiles for downsample; broadcast for upsample)
2. Add C++ shape validation back (currently documented-only) to catch
   future mask-shape regressions
3. Tests: add pathological sparse pattern coverage (block-diagonal +
   random low density) that would expose the bug

**Severity rationale**: HIGH because production sparse training
correctness depends on this.  NOT BLOCKING v2.50 release because:
- Current production default (Section C wrapper for env unset) is safe
- Hybrid path is documented as "preview" feature (env-gated by
  `MFA_ENABLE_V34_BACKWARD=1`)
- Pattern #6 finding showed V34 sparse backward is slower than
  SDPA-vjp anyway — most users won't enable it
- v2.50.1 patch release can ship the fix without breaking API surface

---

## KD-2 — `_v34_sparse_hybrid_vjp` and `_v34_backward_vjp_sparse_full_native` recompute forward (MEDIUM severity) — **RESOLVED v2.50.0 Prompt 5f Phase B**

**RESOLUTION DATE**: 2026-05-14 (Prompt 5f Phase B).  Both orchestrators
now consume `(O, L)` via the `outputs` parameter of `custom_function`,
matching `_make_mfa_sparse_custom` Section C pattern.  Bench at VSR
shape (B=1 H=12 qL=4096 D=128 fp16 BT=32 d=0.1): pre-fix 34.84ms,
post-fix 33.51ms (3-session range 33.35-33.58ms) = ~1.33ms saving.

**Identified by**: Phase 1.1 Python code review (H1 finding).

**Mechanism**: Both orchestrators recompute `sparse_attention_nax_with_lse`
inside the backward closure instead of passing O+L through the
`outputs` parameter of `custom_function`.  The pre-existing
`_make_mfa_sparse_custom._backward` (Section C wrapper) uses the
outputs pattern correctly.

**Current production impact**: 2× cost of sparse forward per backward
call.  For VSR shape at d=0.1, this is ~2-3ms overhead (sparse forward
is fast at low density).  Not a perf blocker; cosmetic.

**Resolution roadmap** (Section A v4 cleanup):
- Refactor both orchestrators to consume `outputs[0]` and `outputs[1]`
  from custom_function trace, matching `_make_mfa_sparse_custom` pattern

**Severity rationale**: MEDIUM (perf cosmetic, not correctness).

---

## KD-3 — Implicit D=128 fallthrough in dispatch (LOW severity) — **RESOLVED v2.50.0 Prompt 5f Phase C**

**RESOLUTION DATE**: 2026-05-14 (Prompt 5f Phase C).  `else: # D=128`
replaced with `elif head_dim == 128: ...; else: raise ValueError` in
`_v34_backward_vjp_sparse_full_native`.

**Identified by**: Phase 1.1 Python code review (H3 finding).

**Mechanism**: `attention.py:~2448` uses `else: # D=128` after `if head_dim == 64`.
If outer guard ever broadens to allow D=256, the `else` would silently
accept it.

**Current production impact**: Outer guard at `_v34_eligible` does
restrict D ∈ {64, 128}, so silently-accept-D=256 is currently
unreachable.  But brittle.

**Resolution roadmap**: change to `elif head_dim == 128: ...; else: raise`.

**Severity rationale**: LOW (defensive code; not currently a bug).

---

## KD-4 — `topk_ratio` parameter validation (LOW severity) — **RESOLVED v2.50.0 Prompt 5f Phase D (regression coverage; fix landed Prompt 5e Phase 1)**

**RESOLUTION DATE**: 2026-05-14 (Prompt 5f Phase D, regression tests).
Validation logic shipped in Prompt 5e Phase 1; Phase D adds 5 explicit
regression tests in `tests/test_v50_prompt_5f_kd4_topk_validation.py`.

**Identified by**: Phase 1.1 Python code review (H4 finding).

**Mechanism**: `flash_attention_topk(topk_ratio=0)` silently coerces
to `k_count=max(1, 0)=1` instead of failing loudly per CLAUDE.md
Rule 8.

**Resolution roadmap**: validate `0 < topk_ratio <= 1.0`; raise
ValueError otherwise.

---

## KD-5 — STEEL backward D=128 N≥2048 zeroed-blocks bug (DEPRECATED v2.50.0 Prompt 5f Phase E)

**v2.50.0 Phase E disposition**: `MFA_FORCE_NATIVE_BWD=1` now emits a
`DeprecationWarning` pointing to this entry.  V34 backward NAX-direct
(production default) is unaffected.  STEEL backward kernel itself
retained for research-only access via the env var; both production
xfails preserved.  Target removal: v2.51+.

**Identified by**: Prompt 5a Section B + clarified in Prompt 5b
Section D.  Documented in `docs/v50/known-issues-v2.50.md`.

**Mechanism**: `MFA_FORCE_NATIVE_BWD=1` routes through legacy STEEL
backward kernels (MFASteelBwdDQ/DKV).  At D=128 N≥2048, output zeroed
for query rows ≥1024 (16×BQ tile boundary).  Bug in legacy STEEL
backward path.

**Production safety**: V34 backward is the production path
(Section D Prompt 5b broadening).  STEEL backward is research-only.
Both production xfails (`TestNativeBackwardRouting[128-2048,
128-4096]`) preserved with accurate rationale.

**Resolution roadmap**: post-v2.50 dedicated investigation into STEEL
backward kernel.  Likely target for deprecation in v2.51+ since V34
is production.

---

## Summary

| ID | Severity | Production-active impact | Resolution sprint |
|---|---|---|---|
| KD-1 | HIGH | ~~Hybrid path silently wrong on pathological masks~~ | **RESOLVED v2.50.0 Prompt 5f Phase A** |
| KD-2 | MEDIUM | ~~2-3ms perf overhead in hybrid backward~~ | **RESOLVED v2.50.0 Prompt 5f Phase B** (~1.33ms saved) |
| KD-3 | LOW | ~~Defensive code (not currently a bug)~~ | **RESOLVED v2.50.0 Prompt 5f Phase C** |
| KD-4 | LOW | ~~Silent coerce of bad topk_ratio~~ | **RESOLVED v2.50.0 (Prompt 5e Phase 1 fix + Prompt 5f Phase D tests)** |
| KD-5 | (preserved xfail) | None (research-only path) | **DEPRECATED v2.50.0 Prompt 5f Phase E**; target removal v2.51+ |

None of KD-1 through KD-5 block v2.50 ship per scope analysis.

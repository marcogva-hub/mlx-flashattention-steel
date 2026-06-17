# v2.50 Sprint 4 — V6NAX backward causal NAX — HALTED (scope discovery)

**Sprint date**: 2026-05-13
**Branch**: `feat/v50-sprint4-bwd-causal` (Sprint 1+2 already merged to master)
**Status**: **HALT — scope discovery, audit underestimate**

## TL;DR

The v2.50-NAX-coverage audit estimated Sprint 4 (V6NAX backward causal NAX)
at **M effort (~1-2h CC)**.  Investigation reveals the real scope is
**L/XL (~5-7h CC)** because V6NAX **forward** does not yet support causal
either — V6NAX forward causal extension is an undocumented prerequisite.

Per §AA.1 failure-mode handling, halting Sprint 4 + documenting the
scope-correction.  Sprints 1+2 already merged to master successfully.
Sprint 4 deferred to a dedicated future session with the corrected
scope estimate.

## Discovery: V6NAX forward gates on `!isCausal`

`csrc/mfa/v6_nax/NAAttentionKernel.cpp:171`:

```cpp
if (useV6NAX && type.value == AttentionKernelType::forward
    && !isCausal && !masked && !isVarlen) {
  return createV6NAXSource();
}
// else: fall through to legacy STEEL source generator
```

The V6NAX forward kernel source generator (`createV6NAXSource()`) is only
emitted when `!isCausal`.  When causal is requested, V6NAX falls back to
the legacy STEEL `loopForward` template which has its own causal
implementation but emits **log2-domain lse**, not the natural-log lse
that V6NAX backward kernels expect.

Confirmed by code inspection:
- `_make_mfa_custom` at `attention.py:3744`: hardcodes `_v6_fwd(q, k, v, False, True)` —
  the `False` is causal, intentionally suppressed because V6NAX forward
  doesn't support it.
- `_v6nax_eligible` at `attention.py:3576-3577`: `if causal: return False  # DC3 deferred`.

## Why the audit underestimated

The audit's Sprint 4 mandate (`docs/audits/v50-nax-coverage/03-sprint-sequence.md`):

> Sprint 4: V6NAX backward causal NAX
> **Effort**: M (~2h CC realistic)
> **Files**: extend `createV6NAXBackwardQuerySource()` + `createV6NAXBackwardDKSource()`
> + `createV6NAXBackwardFusedDKDVSource()` in NAAttentionKernel.cpp to support
> causal masking; update `_v6nax_backward_carveout` to include causal.

The audit assumed V6NAX forward already supported causal and only the
backward kernels needed extension.  But the V6NAX forward kernel itself
does NOT support causal — that's the larger unaccounted scope.

## Real Sprint 4 scope (revised)

To ship V6NAX backward causal NAX properly, the work is:

### Phase 4a — V6NAX forward causal extension (~2-3h CC)
1. Extend `createV6NAXSource()` with causal block (mirror Apple SDPA NAX's
   `kb_lim`/`kb_min_causal` pattern, per `apple-sdpa-nax-analysis.md` §
   "How Apple handles the K-loop")
2. Lift the `!isCausal` gate at `NAAttentionKernel.cpp:171`
3. Verify V6NAX forward causal produces correct output + natural-log lse
4. Update `_make_mfa_custom` line 3744: `_v6_fwd(q, k, v, causal, True)`
5. /metal-kernel-dev pre-flight for register budget impact
6. Three-axis validation against STEEL causal forward

### Phase 4b — V6NAX backward causal (~1-2h CC, prerequisite Phase 4a)
1. Lift `_v6nax_eligible` causal gate at line 3576-3577
2. Lift `_v6nax_backward_carveout` causal exclusion in dispatch_policy.py
3. Backward kernels likely need NO source changes because the FA-2
   backward pattern handles causal via lse-encoded masking automatically:
   - V6NAX forward emits lse with causal masking (S[r,c]=-inf for r<c → P[r,c]=0)
   - Backward recomputes P from lse → P[r,c]=0 at masked positions
   - dS = P * (dP - D) → dS=0 at masked positions
   - dV, dK, dQ accumulations naturally inherit the mask
4. Three-axis validation: V6NAX backward causal matches `mx.vjp(SDPA_causal)`
   within FP16 ULP
5. /mlx-debug-forensics corruption audit
6. /mlx-mfa-bench-methodology + /mlx-mfa-perf-audit

### Phase 4c — Integration + tests (~0.5h CC)
1. Tests in `tests/test_v50_v6nax_backward_causal.py`
2. CHANGELOG entry
3. Pre-merge audit checklist

### Total revised: ~3.5-5.5h CC

This is the **L (~3-6h)** band per the audit's own effort categorization,
not M.

## Why halt instead of "just push through"

The §AA.1 failure-mode handling rule says:

> Sprint 4 blocker (e.g., causal correctness fails, register pressure
> resurface) :
> - Halt Sprint 4, STATUS doc, Sprints 1 + 2 préservés si déjà mergés

This Sprint 4 case is a scope-discovery blocker (audit underestimated
prerequisites), structurally similar to "correctness fails" — same
response: honest halt, STATUS doc, downstream sprints preserved.

Pushing through would either:
1. Burn ~5h CC on what was scoped as ~2h (budget overrun)
2. Force compromise scope (e.g., ship only Phase 4b without Phase 4a)
   — but that would break correctness because backward kernels would
   read log2-lse from STEEL forward instead of natural-log lse from V6NAX
   forward, producing wrong gradients silently

Either outcome violates v2.50's "production complete" mandate.  Honest
halt is the correct call.

## What's still preserved

- **Sprint 1 merged at master `be30352`**: density threshold recalibration
  (LCSA NAX wins at all densities on M5+, ~6× speedup on audit shape)
- **Sprint 2 merged at master `4601505`**: M5+ NAX rope path (mx.fast.rope +
  flash_attention, ~4× speedup on audit shape)
- **Tests passing**: 99/99 (90 baseline + 9 Sprint 2 + LCSA expansion)
  on master `4601505`.  Sprint 4 branch (`feat/v50-sprint4-bwd-causal`)
  has no committed changes — clean halt.

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| 4.1 read inputs | (no skill — direct reads + grep) | done |
| 4.2 scope discovery | (no skill — code inspection at NAAttentionKernel.cpp:171) | done — FOUND BLOCKER |
| 4.3-4.8 | not initiated due to scope-discovery halt | N/A |

## Recommended next steps for Sprint 4 (dedicated future session)

1. Read `apple-sdpa-nax-analysis.md` §"How Apple handles the K-loop"
   (the `kb_lim` / `kb_min_causal` pattern)
2. Read `createV6NAXSource()` body in `NAAttentionKernel.cpp` (Q-K-loop
   structure, masking infrastructure)
3. `/metal-kernel-dev` pre-flight for causal block design + register
   budget impact at D=64/D=128
4. Extend `createV6NAXSource()` with causal block (Phase 4a)
5. Verify V6NAX forward causal produces correct output + lse via
   bit-identical comparison to STEEL causal forward
6. Then proceed with Phase 4b backward gate lift + Phase 4c integration

Dedicated session estimate: **~5h CC** (3h Phase 4a, 1.5h Phase 4b, 0.5h
Phase 4c).

## Files in this halt

Only this STATUS doc.  No code changes committed on
`feat/v50-sprint4-bwd-causal`.  The branch will be pushed for record-
keeping then either deleted post-merge of this STATUS doc OR resumed in
the future dedicated session.

## Master state post-halt

- `master`: `4601505` (Sprint 2 merged)
- Sprints 1+2 of v2.50 Prompt 1 complete.
- Sprint 4 deferred with corrected scope estimate.
- CHANGELOG `[Unreleased — for v2.50]`: 2 entries (Sprint 1, Sprint 2).
- Tests: 99/99 pass.
- No version bump, no tag, no PyPI publication per internal-mode contract.

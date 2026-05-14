# v2.50 Phase 4b-complete dV residual — Prompt 4 Section B — RESOLVED

**Sprint date**: 2026-05-14 (Prompt 4 Section B)
**Branch**: `feat/v50-prompt4-dv-residual-debug`
**Master tip pre-Section-B**: `<Prompt 4 Section A merge>`

## TL;DR

The Prompt 3 "dV K-parallel residual" (V34 backward causal dV
producing 10× under-counting at ratio ~0.039 vs SDPA-vjp reference)
is **FULLY RESOLVED**.  My Prompt 3 hypothesis (fragment transpose
semantics in `dV += P^T @ dO`) was **WRONG**.

**Actual root cause**: Prompt 2 Sprint 4 Phase 4a lifted the `!isCausal`
gate at `NAAttentionKernel.cpp:171` (source generator selection), but
MISSED two parallel dispatch gates in `MFAV6Forward::eval_gpu()`:

1. **Line 176** (`generate_v6_source`):
   ```cpp
   if (use_v34 && (isCausal || !single_otile)) use_v34 = false;
   ```
2. **Line 625** (eval_gpu pre-dispatch decision):
   ```cpp
   if (use_v34 && (params_.causal || !so_for_v34)) use_v34 = false;
   ```

Both gates silently routed `causal=True` forward to STEEL legacy
(`createSource()`'s `loopForward` template) instead of `createV34Source()`.
STEEL legacy emits lse in **log2 domain** (S_log2 + log2(sum_score));
V34 backward kernels expect lse in **natural-log domain** (per the
Phase 4b dQ kernel comment at `NAAttentionKernel.cpp:4007`).

V34 backward consumed wrong-domain lse → all gradients wrong → dV at
~10× under-counting magnitude.

**Empirical proof — diagnostic with Q=K=V=dO=1 (qL=64, D=64)**:

Pre-fix lse output for V34 forward causal:
- r=0: lse = 11.5416 = `max_log2 + log2(sum_score)` = log2-domain
- r=7: lse = 14.5416 = `max_log2 + log2(8)` = log2-domain
- Expected natural-log lse: 8.0 + log(r+1)

Post-fix lse output:
- r=0: lse = 8.0000 ✓ natural-log
- r=7: lse = 10.0794 ✓ natural-log

Post-fix V34 backward causal dV:
- c=0: V34 = 4.7422, SDPA-ref = 4.7422, ratio = 1.000 ✓
- c=63: V34 = 0.0156, SDPA-ref = 0.0156, ratio = 1.000 ✓

Production validation at scaled inputs (qL=2048 D=64 fp16):
- dQ max_diff: 2.4e-7 (well within tol)
- dK max_diff: 1.0e-3 (within fp16 ULP)
- dV max_diff: 9.2e-4 (within fp16 ULP)

## Fix in C++ (3 lines changed)

### Fix 1: `csrc/mfa_v6_nax_primitive.cpp:176`
```diff
-  if (use_v34 && (isCausal || !single_otile)) use_v34 = false;
+  if (use_v34 && !single_otile) use_v34 = false;
```

### Fix 2: `csrc/mfa_v6_nax_primitive.cpp:625` (eval_gpu)
```diff
-  if (use_v34 && (params_.causal || !so_for_v34)) use_v34 = false;
+  if (use_v34 && !so_for_v34) use_v34 = false;
```

Both gates were leftover from pre-Phase-4a state when V34 forward
didn't support causal.  Phase 4a's `createSource()` line-171 lift only
addressed the SOURCE-GENERATOR side; the DISPATCH-DECISION side
remained restrictive.

## Fix in Python (gates lifted)

### `mlx_mfa/attention.py::_v34_eligible`
```diff
- if causal:
-     return False  # Phase 4b-complete deferred (K-parallel kernels)
```

### `mlx_mfa/dispatch_policy.py::_v34_backward_carveout`
```diff
  if (
      head_dim == 64
      and seq_len >= 2048
-     and not causal
      and dtype_key in ("float16", "bfloat16")
      and os.environ.get("MFA_ENABLE_V34_BACKWARD") == "1"
  ):
      return True
```

## What about my Prompt 3 K-parallel kernel mask blocks?

Prompt 3 added `#if V34BWD*_CAUSAL` mask blocks to all 4 K-parallel
kernels (dV split, dK split, dKV legacy fused, dKdV fused).  These
are now **structurally correct AND necessary** — the V34 backward
kernels DO need to apply causal masking to S before computing P, OR
ELSE P would be non-zero for c>r positions (since S = Q@K^T without
the mask doesn't know about causal).

Without my Prompt 3 mask blocks, even with the dispatch fix landed
(natural-log lse), the V34 backward kernels would still produce
wrong gradients because P[r,c] for c>r would be exp(S[r,c] -
lse_natural[r]) which is non-zero (lse only sums c<=r positions; S
for c>r is some natural value).

So the Prompt 3 mask blocks are STILL needed, and the Prompt 4 fix
just enables them to consume the correct natural-log lse from V34
forward causal.

## Why the Prompt 3 investigation missed this

In Prompt 3, I:
1. Diagnosed at the wrong layer (assumed kernel computation bug,
   investigated fragment transpose semantics)
2. Did not trace the ROUTING: I assumed force_v34=True flowed through
   to the V34 kernel, but the dispatch gates silently overrode it
3. Did not have a TEST that would have failed loudly when the kernel
   wasn't actually engaged

**Lessons for future sprint discipline**:
- When investigating a kernel issue, always verify the kernel is
  actually executing (e.g., sentinel write to detect code path)
- Don't assume `force_v34=True` end-to-end without tracing the
  dispatch chain
- A "force_v34=True" parameter should HONESTLY override; the current
  gates create silent fallback behavior that violates the contract

## Three-axis validation

### Axis 1 — Output correctness (V34 backward causal vs SDPA-vjp)

Validated at D=64 fp16/bf16, qL ∈ {2048, 4096}, causal=True (scaled
inputs *0.1):
- dQ max_diff: 2.4e-7 to 4.8e-8 (well below 1e-3 tight bound)
- dK max_diff: 1.0e-3 to 1.7e-5 (within bound)
- dV max_diff: 9.2e-4 to 1.9e-5 (within bound)

### Axis 2 — PUBLIC API path engaged

`flash_attention(q, k, v, causal=True)` with `MFA_ENABLE_V34_BACKWARD=1`
now engages V34 backward causal at D=64 qL≥2048 fp16/bf16.  Test
`test_flash_attention_causal_engages_v34` verifies V34 dispatch and
gradient correctness.

### Axis 3 — Edges preserved

- V34 backward non-causal: unchanged (RMSE 6e-6 — same as pre-fix)
- V34 forward causal: now emits natural-log lse correctly (was log2)
- V34 forward non-causal: unchanged (already correct)
- STEEL legacy backward (M1/M3 fallback): unaffected (only routing
  path changed for M5+ V34-eligible causal)
- Sprint 1 + Sprint 2 + Sprint 3 dispatch fixes: preserved

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| 4b.dv.0 §AA.5 premise check | `/mlx-mfa-apple-primitives-coverage` | done — CONFIRMATION (no Apple primitive provides V34-direct backward) |
| 4b.dv.1 element-level diagnostic | manual (Python harmonic-tail comparison + sentinel writes) | done — identified `max_log2 + log2(sum)` pattern in observed lse |
| 4b.dv.2 dispatch gate investigation | manual (grep `use_v34` + `isCausal` in source) | done — found 2 dispatch gates at lines 176, 625 |
| 4b.dv.3 fix + validation | Empirical diagnostic harness (`/tmp/dv_diagnostic.py`) | done — V34 dV ratio 1.000 vs SDPA ref |
| 4b.dv.4 test suite regression | `pytest tests/` | done — 1173 pass, 0 unexpected fail |
| 4b.dv.5 pre-merge | `/mlx-code-review` | pending |

## Files changed

| File | Net LOC | Purpose |
|---|---|---|
| `csrc/mfa_v6_nax_primitive.cpp` | +12 | Lift 2 dispatch gates (line 176 source-gen, line 625 eval_gpu) |
| `mlx_mfa/attention.py` | +0 (replaced gating comment) | Remove `if causal: return False` from `_v34_eligible` |
| `mlx_mfa/dispatch_policy.py` | +0 (replaced gating comment) | Remove `not causal` from `_v34_backward_carveout` |
| `tests/test_v34_helpers.py` | +5 (rename + update) | `test_d64_fp16_causal_returns_false` → `_returns_true` |
| `tests/test_v39_fused_dkdv.py` | +6 (rename + update) | `test_v34_eligible_causal_false` → `_true` |
| `tests/test_v50_v34_causal.py` | +20 (rename + behavior swap) | `test_sprint4_v34_eligibility_causal_returns_false` → `test_v34_eligibility_causal_returns_true`; `test_sprint4_flash_attention_causal_uses_sdpa_vjp` → `test_flash_attention_causal_engages_v34` (now asserts V34 engagement + correct gradients with scaled inputs) |
| `docs/v50/phase-4b-complete-dv-residual-decisions.md` | +200 (new) | this doc |

## Net effect on users

- `mx.grad(flash_attention(q, k, v, causal=True))` with
  `MFA_ENABLE_V34_BACKWARD=1`, D=64, qL≥2048, fp16/bf16 — now engages
  V34 NAX-direct backward causal kernels on M5+, producing correct
  gradients (RMSE within fp16 ULP).
- For non-V34-eligible callers (D=128, smaller qL, fp32, etc.): SDPA-vjp
  fallback unchanged.
- For non-causal callers: unchanged (V34 backward non-causal already
  worked correctly).

## Audit framing inversion

Per `docs/v50/audit-framing-inversions.md` pattern: this is a
**SCOPE_CORRECTION** for Sprint 4 Phase 4b-complete, not an inversion.
The Phase 4b-complete deliverable was correctly scoped (causal support
across all 5 backward kernels); the BUG was that Prompt 2 Phase 4a's
"lift the causal gate" change was incomplete — only one of three gates
got lifted.  Prompt 3 K-parallel kernel mask blocks were necessary
infrastructure; the residual was due to wrong-domain lse from
mis-routed forward.

This is a 5th "framing inversion" pattern: **incomplete-fix
pattern** — when changing a multi-gate dispatch chain, all gates
must be audited together; lifting one without the others creates
silent partial functionality.

## Sprint 5 sparse extension status

Sprint 5 (V34 backward block-sparse) was BLOCKED on Phase 4b-complete
being clean.  With Phase 4b-complete now shipped, Sprint 5 is
unblocked.  However, given the substantial Prompt 4 Section A + B
work already shipped, Sprint 5 implementation is **deferred to
Prompt 5 (dedicated release flow)** OR a focused future session.

The Sprint 5 design (per Prompt 2 + Prompt 3 STATUS docs) is sound:
extend the 4 K-parallel kernels with block_mask buffer + per-tile
early-exit + per-element block-sparse mask block.  Estimated effort:
~2-3h CC in a dedicated session.  See
`docs/v50/sprint-5-prompt3-status.md` for the design spec.

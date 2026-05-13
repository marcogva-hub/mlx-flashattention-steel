# Dispatch + `_make_mfa_custom` — multi-pass `/mlx-code-review`

**Status:** complete (2026-05-13)
**Sprint:** v2.38.0 investigation (Phase D)

## Targets

- `mlx_mfa/dispatch_policy.py` (1073 LOC; focus on `_should_use_mfa_m5_nax_carveout`
  + `_v34_backward_carveout` + `should_use_mfa`)
- `mlx_mfa/attention.py::_make_mfa_custom` (V34 backward branch, lines 3540-3680)
- `mlx_mfa/attention.py::flash_attention` body (carve-out delegation post-v2.38.x)

## Pass 1 — General tech debt

### DP1-MEDIUM-01 — `_should_use_mfa_m5_nax_carveout` is genuinely dead code

**File:** `dispatch_policy.py:313-355`
**Category:** D (Tech debt)
**Confidence:** High

The function returns `False` unconditionally and has done so since
v2.32.0.  Sprint v2.38.x Phase A added a clarifying docstring
("Sprint A.6 dormant pending genuine empirical findings") and split
the V34-backward concern into a separate `_v34_backward_carveout()`.

**However:** if no Sprint A.6 calibration work is planned for any
specific version, this function is a misleading scaffold — readers
see "carveout placeholder for future" but there's no actionable
"future" in the roadmap.  Sprint 2 audit M5-HIGH-01 originally
recommended consolidation; the split (v2.38.x Phase A) was the right
call for clarity but ALSO surfaces that one half is dead.

**Fix options:**
1. **Delete entirely** — `should_use_mfa()` at line 519 inlines
   `return False` for the canonical-path case.  Net: -45 LOC,
   removes a placeholder that's been dormant 6+ months.
2. **Document deletion target** — add to the docstring: "If no
   Sprint A.6 carve-outs materialize by v2.40.0, this function will
   be deleted."  Forces a concrete future review point.

**Recommendation:** option (1) — delete.  The risk is low
(`should_use_mfa()`'s caller flow unchanged; carve-out result is
False either way).  Keep `_v34_backward_carveout()` as the active
function.

### DP1-MEDIUM-02 — V34-eligibility predicate triplicated (Sprint 2 audit M4-MEDIUM-01 reconfirmed)

**File:** `attention.py:3554-3562, 3617-3625` + `dispatch_policy.py:392-399`
**Category:** D (DRY)
**Confidence:** High

The V34-eligibility predicate `(env=="1") + has_nax + D ∈ {64,128} +
fp16/bf16 + !causal` appears at THREE points:
1. `flash_attention()` carve-out delegation (calls `_v34_backward_carveout`)
2. `_make_mfa_custom::_impl` forward-fusion check (inline)
3. `_make_mfa_custom::_backward` eligibility check (inline)

Sprint 2 audit M4-MEDIUM-01 noted this in 2026-05-13.  Sprint v2.38.x
didn't fix it (out of scope for that sprint).  This Pass 1 reconfirms
it remains.

**Fix:** extract `_v34_eligible(q, k, v, causal)` helper:

```python
def _v34_eligible(q, k, v, causal) -> bool:
    """Single source of truth for V34 backward eligibility.
    Used by flash_attention() carve-out delegation AND _make_mfa_custom
    forward/backward predicates."""
    return (
        os.environ.get("MFA_ENABLE_V34_BACKWARD") == "1"
        and _get_has_nax_cached()
        and q.shape[3] in (64, 128)
        and q.dtype in (mx.float16, mx.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and not causal
    )
```

Then `_v34_backward_carveout()` can re-use the helper conceptually
(though it stays in `dispatch_policy.py` for separation of concerns),
and the two inline checks in `_make_mfa_custom` collapse to one call.

### DP1-MEDIUM-03 (NEW) — `_make_mfa_custom` V34 backward branch growing large

**File:** `attention.py:3540-3680`
**Category:** D (Maintainability)
**Confidence:** Medium

The V34-specific branch in `_make_mfa_custom::_backward` is currently
~140 LOC and growing.  Future additions (D_vec precompute, Option γ
fused kernel routing) will increase further.

**Fix candidate** for v2.39.0 cleanup: extract the V34 backward
computation into a dedicated module-level helper:

```python
def _v34_backward_vjp(q, k, v, O, L, dO, scale, *, use_fused=False):
    """V34 backward VJP — invoked from _make_mfa_custom._backward
    when V34 path is eligible.  Returns (dQ, dK, dV)."""
    from mlx_mfa import _ext
    dQ = _ext.v6_nax_backward_query(q, k, v, O, L, dO, scale)
    if use_fused or os.environ.get("MFA_V34BWD_USE_FUSED") == "1":
        dK, dV = _ext.v6_nax_backward_kv(q, k, v, O, L, dO, scale)
    else:
        wm = int(os.environ.get("MFA_V34BWD_WM", "4"))
        dVp = _ext.v6_nax_backward_dv_raw(q, k, v, L, dO, scale, wm)
        dKp = _ext.v6_nax_backward_dk_raw(q, k, v, O, L, dO, scale, wm)
        dV = mx.sum(dVp, axis=2).astype(q.dtype)
        dK = mx.sum(dKp, axis=2).astype(q.dtype)
    return dQ, dK, dV
```

Net: `_make_mfa_custom._backward` shrinks ~80 LOC.  V34 backward
logic is testable/refactorable independently of the custom-vjp wrapper.

### DP1-LOW-01 — `is_m3_plus` vs `has_nax` parameter redundancy in `should_use_mfa`

**File:** `dispatch_policy.py:359-371`

`should_use_mfa()` takes both `is_m3_plus` and `has_nax` as separate
parameters.  Since `has_nax` implies M5+ (NAX is M5+-only), and M5+
implies M3+, `has_nax=True` necessarily means `is_m3_plus=True`.  The
inverse isn't true (M3/M4 are `is_m3_plus=True` but `has_nax=False`).

Two parameters where one minus dependence exists.  Could collapse to
a single `hw_gen` enum (`M1`, `M2`, `M3`, `M4`, `M5`) but the current
boolean structure is simpler for the existing dispatch logic.  Defer;
not a real issue.

## Pass 2 — Compound improvements

### DP2-HIGH-01 (compound) — DP1-MEDIUM-02 + P2-HIGH-01 (D_vec precompute) — combined refactor opportunity

**Source:** DP1-MEDIUM-02 (predicate duplication) + P2-HIGH-01 from
V34 backward review (D_vec precompute is 2-4h, half-done).

The Python-side work for D_vec precompute MUST:
1. Check V34 eligibility before computing D_vec (waste otherwise)
2. Compute `D_vec = mx.sum(dO * O, axis=-1)` once
3. Pass D_vec to dQ + dV + dK binding calls

If the V34-eligibility predicate is extracted into `_v34_eligible()`
helper FIRST (DP1-MEDIUM-02 fix), then D_vec precompute slots into
the helper-using path cleanly:

```python
def _backward(primals, cotangents, output):
    q, k, v = primals
    dO, _ = cotangents
    O, L = output
    if _v34_eligible(q, k, v, causal):
        D_vec = mx.sum(dO * O.astype(mx.float32), axis=-1)
        return _v34_backward_vjp(q, k, v, O, L, dO, D_vec, scale)
    # ... fallback paths ...
```

**Compound win:** doing DP1-MEDIUM-02 + DP1-MEDIUM-03 + D_vec
precompute together yields a 60-80 LOC reduction in `_make_mfa_custom`
+ a much cleaner V34 backward integration surface.  ~3-4 hours CC total.

### DP2-MEDIUM-01 (compound) — Phase A TGP finding + Option γ scope revision

**Source:** Phase A TGP overhead measurement (~1µs/K-iter vs design
doc 100µs) + V34 backward review P3-HIGH-01 (Primitive boilerplate
consolidation).

Pre-Phase-A: Option γ was a 1-2 day implementation that maybe
delivered 7-40% backward improvement at D=128 (still ~2× SDPA-vjp).

Post-Phase-A: Option γ's main cost (TGP reduction overhead) is 100×
lower than feared.  The architectural ceiling on D=128 perf is NOT
the TGP reduction — it's the dK kernel's extra dO@V^T matmul work.

Implication: if Option γ is implemented, the perf win comes from
SOFTMAX FUSION (4-5ms savings), not from removing the partials-buffer
overhead.  The per-SG-slot pattern + Python mx.sum is fine; **TGP
streaming reduction would also be fine** per the empirical TGP
overhead measurement.

For v2.38.0 scope: Option γ can use EITHER per-SG-slot output OR TGP
streaming reduction.  Both are viable on M5 NAX.  Per-SG-slot is
simpler (matches existing Phase 2.O2 split pattern + Python mx.sum);
TGP streaming saves the partials-buffer memory + Python reduction
overhead.

Compound finding: Phase A measurement REMOVES the architectural
blocker on TGP streaming from the design doc.  Either implementation
choice is now viable on technical merits; pick based on code-size
simplicity (per-SG-slot wins) or memory-footprint optimization (TGP
streaming wins).

## Pass 3 — Architectural patterns

### DP3-MEDIUM-01 — Dispatch policy three-layer concern model is clean post-v2.38.x

After Sprint v2.38.x Phase A, the dispatch policy has three layered
concerns:

1. **Generic MFA-vs-SDPA shape thresholds** (M5_NAX_THRESHOLDS,
   M3_THRESHOLDS, DEFAULT_THRESHOLDS, _load_custom_table)
2. **Canonical-path M5+ NAX carve-out hook** (`_should_use_mfa_m5_nax_carveout`,
   currently dormant — DP1-MEDIUM-01)
3. **V34-backward-specific carve-out** (`_v34_backward_carveout`,
   env-var-gated, active)

These three concerns are well-separated.  No architectural reshape
needed.  The cleanup opportunities (DP1-MEDIUM-01 dead-code deletion,
DP1-MEDIUM-02 predicate extraction) are polish at the boundaries, not
structural redesign.

**Pass 3 verdict:** structurally clean.  Only polish remaining.

## Convergence — Pass 4 not needed

Passes 1-3 surface only LOW + MEDIUM polish.  No HIGH findings
introduced after Pass 1.  Pass 4 convergence check would be redundant.

**Stopped at Pass 3.**

## Synthesis — dispatch + integration

| Severity | Count | New vs Sprint 2 audit? | Compound? |
|---|---|---|---|
| HIGH | 1 | DP2-HIGH-01 is compound (NEW) | — |
| MEDIUM | 4 | DP1-MEDIUM-01 partial (audit M5-HIGH-01 went different direction); DP1-MEDIUM-03 NEW; DP2-MEDIUM-01 NEW | DP2-HIGH-01 + DP2-MEDIUM-01 compound |
| LOW | 1 | NEW | — |
| **Total** | **6** | **3 NEW** | **2 compound** |

**Most valuable findings:**

1. **DP2-HIGH-01 compound** — D_vec precompute + V34-eligibility
   helper + V34-backward-VJP extraction can be done in one ~3-4h
   Python refactor pass.  Bundles three audit findings (M2, M4-MEDIUM-01,
   plus new DP1-MEDIUM-03) into one coordinated change.

2. **DP2-MEDIUM-01 compound** — Phase A empirical TGP measurement
   REMOVES the design-doc blocker on TGP streaming reduction.  Option γ
   implementation has more flexibility than the original design doc
   suggested.

3. **DP1-MEDIUM-01** — Delete the genuinely-dead
   `_should_use_mfa_m5_nax_carveout` function (or commit to a deletion
   target version).  Sprint v2.38.x Phase A clarified the split, but
   now the one half that's dormant should be addressed concretely.

## Skill invocations log

| Pass | Skill | Findings count | Notes |
|---|---|---|---|
| 1 | /mlx-code-review (rubric applied) | 3 MEDIUM + 1 LOW | Tech-debt focus |
| 2 | /mlx-code-review (rubric applied) | 1 HIGH compound + 1 MEDIUM compound | Pass 1 baseline |
| 3 | /mlx-code-review (rubric applied) | 1 MEDIUM verification | Structural; clean |

3 passes total.  Convergence at Pass 3.

# V34 backward stack — multi-pass `/mlx-code-review`

**Status:** Pass 1 complete (2026-05-13)
**Sprint:** v2.38.0 investigation (Phase B)
**Methodology:** Marco's VSR-work multi-pass pattern — single-pass audit
is shallow; 2-4 passes catch compound findings. Stop at convergence.

## Target modules

| File | LOC | Function(s) |
|---|---|---|
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp` | 5275 | `createV34BackwardQuerySource` (3784), `createV34BackwardKeyValueSource` (4221, legacy), `createV34BackwardDVSource` (4651), `createV34BackwardDKSource` (4934) |
| `csrc/mfa_v6_nax_primitive.cpp` | ~1500 | `MFAV34BwdQ`, `MFAV34BwdKV` (legacy), `MFAV34BwdDV`, `MFAV34BwdDK` Primitives |
| `mlx_mfa/attention.py` | section ~3540-3680 | `_make_mfa_custom` V34 backward branch |

---

## Pass 1 — General tech debt + code smells

**Focus:** redundant computations, inconsistent patterns across the 3
backward kernels, fragile assumptions, stale comments.

### P1-HIGH-01 — `AttentionOperand::D` infrastructure half-wired (NEW finding, not in Sprint 2 audit)

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:532, 535, 576, 579`
**Category:** D (Maintainability) / C (Performance — unrealized win)
**Confidence:** High

`createBufferBindings()` and `operandLocationWithHeadOffsetValue()`
already list `AttentionOperand::D` in the V34 dQ + V34 dK/dV operand
sets:

```cpp
// Line 532:
operands = {AttentionOperand::Q, ..., AttentionOperand::D, ...,
            AttentionOperand::dQ};
// Line 535:
operands = {AttentionOperand::Q, ..., AttentionOperand::D, ...,
            AttentionOperand::dV, AttentionOperand::dK};
```

But:
- The kernel bodies (sed-verified via Sprint 2 audit) recompute
  `D[i] = rowsum(dO[i] ⊙ O[i])` INLINE each kernel
- The Python integration (`attention.py:3636-3651`) doesn't compute
  D_vec or pass it as an input
- `_ext.v6_nax_backward_query(q, k, v, O_v34, L_v34, dO, scale)`
  signature has 7 args; D not among them

The C++ infrastructure for D_vec as a precomputed operand is
**already scaffolded** (binding indices reserved, operand-location
math written) but **never wired to the Python layer**.

**Why it matters:** The Sprint 2 audit's M2-HIGH-01 estimate ("1 day
CC, ~300 LOC of work for D_vec precompute Primitive") **overstated
the effort**.  Half the work is already done in the C++ infra.
The remaining work is:
- Update kernel bodies to consume the D buffer (instead of inline
  recompute) — 1 edit per generator
- Add D as input to dQ + dV + dK Primitives' eval_gpu
- Update Python integration in `_make_mfa_custom` to compute
  `D_vec = mx.sum(dO * O, axis=-1)` once and pass through
- Add Python binding signatures for D input

Estimated effort: **2-4 hours** (not 1 day) — and the C++
binding-index reservation is the trickiest part already.

**Recommendation:** D_vec precompute work for v2.38.0 should start
from this half-wired infrastructure, not from scratch.  Save
significant effort.

### P1-MEDIUM-01 — Stale `FUTURE-CLEANUP` comment in dQ generator

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:3810-3811`
**Category:** D
**Confidence:** High

```cpp
ss << "// === Inline Apple NAX helpers (verbatim from V34 forward) ===\n";
ss << "// FUTURE-CLEANUP: extract to naxHelpersSource() and share with forward.\n";

ss << naxHelpersBlock();
```

The "FUTURE-CLEANUP" comment is **factually incorrect** — Sprint
v2.38.x Phase B extracted `naxHelpersBlock()`, and the next line
INVOKES it.  Comment says "future" for work that's done.

**Other generators** (dV, dK) don't have this stale comment — only
dQ does.

**Fix:** delete both lines or rewrite as "Apple helpers shared via
naxHelpersBlock() (extracted in v2.38.x Phase B)".

### P1-MEDIUM-02 — `SubOp` emitted in 3 separate generators

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:3821, 4252, 4963`
**Category:** D
**Confidence:** High

`SubOp` (subtraction functor for `row_bin_op<SubOp>`) is emitted
inline in:
- dQ generator (line 3821)
- Legacy fused dK/dV generator (line 4252)
- dK generator (line 4963)

`SubOp` is NOT emitted in:
- dV generator (~4651) — dV doesn't compute `dP - D_vec` (only dK does)
- Forward generator — no subtraction in forward

The duplication is the same pattern as the Apple helpers
(M1-HIGH-01) but smaller scope: 5 lines × 3 generators = 15 LOC
duplicated.

**Fix:** add SubOp to `naxHelpersBlock()` alongside MaxOp / SumOp /
MulOp / ExpSubOp.  The forward kernel doesn't use SubOp but emitting
it as an unused symbol in the helpers block is free (MSL compiler
strips unused).  Net: -10 LOC, one edit point for any future Op
struct changes.

### P1-LOW-01 — `TQ` vs `TQ_per_SG` naming inconsistency

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp`
**Category:** D
**Confidence:** High

| Generator | Variable | Formula | Notes |
|---|---|---|---|
| Forward (line 2722) | `TQ` | `BQ/(WM*kU)` | comment "expected = 1 per Apple's static_assert" |
| dQ backward (line 3790) | `TQ` | `BQ/(WM*kU)` | same formula |
| Legacy fused (line 4227) | `TQ` | `BQ/kU` | **different formula** (WM=1, no division) |
| dV (line 4658) | `TQ_per_SG` | `BQ/(WM*kU)` | comment "expected = 1" |
| dK (line 4940) | `TQ_per_SG` | `BQ/(WM*kU)` | same as dV |

The cosmetic split (`TQ` vs `TQ_per_SG`) reflects different sprint
authorship (forward + dQ were earlier; dV + dK were later in
Phase 2.O2 WM=4 split).  Normalize to one name: `TQ_per_SG` is more
descriptive when WM > 1; `TQ` was correct when WM was implicit (=1).

**Fix:** rename dQ's `TQ` → `TQ_per_SG` for consistency with the
WM-aware kernels.  Forward's `TQ` can stay if forward never moves to
WM > 1 (currently forward is single-SG... actually forward is also
WM=4 per the generator!  Forward's TQ = BQ/(WM*kU) with WM=4 → also
TQ_per_SG semantically).  Forward generator should also rename.

Three of five generators currently use the same semantics under
different names.  Normalize all to `TQ_per_SG`.

### P1-MEDIUM-03 — Legacy fused dK/dV kernel deletion candidate (Sprint 2 audit M3-HIGH-01 reconfirmed)

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:4221` (820 LOC)
**Category:** D
**Confidence:** High

Status update vs Sprint 2 audit M3-HIGH-01:
- Sprint v2.38.x Phase B preserved the legacy fused kernel as
  fallback gated by `MFA_V34BWD_USE_FUSED=1` env var
- Since v2.37.0 (Phase 2.O2 split landed), no production use case
  for the fused kernel has been documented
- This multi-pass review can't find any reference to it in
  production code paths

**Fix recommendation: deletion for v2.39.0** (out of scope per
v2.38.0 prompt's "Out of scope" section).  ~820 LOC + binding +
Primitive eliminated.  If a future Option γ (true fused dK+dV with
TGP reduction) lands, that's a NEW kernel, not the legacy WM=1
version.

### P1-LOW-02 — Per-kernel preamble redundancy

All 4 backward generators (and forward) emit the same `// MFA_REQUIRE_MSL4`
+ `#include <metal_*>` block.  Could be factored into
`emitKernelPreamble()`.  Minor; ~10 LOC per generator × 5 = 50 LOC
saved.  Low impact; defer.

---

## Pass 1 summary

| Severity | Count | Findings |
|---|---|---|
| HIGH | 1 | P1-HIGH-01 (D operand half-wired) |
| MEDIUM | 3 | P1-MEDIUM-01 (stale comment), P1-MEDIUM-02 (SubOp dup), P1-MEDIUM-03 (legacy kernel deletion) |
| LOW | 2 | P1-LOW-01 (naming), P1-LOW-02 (preamble dup) |

**Compound finding (this is the value of multi-pass over single-pass):**
P1-HIGH-01 + Sprint 2 audit M2-HIGH-01 together: D_vec precompute
implementation is **half-done already**.  The remaining work is much
less than the audit estimated.  This materially affects the v2.38.0
scope decision: D_vec precompute is a 2-4 hour task, not a day.

The single-pass Sprint 2 audit looked at kernel bodies and
identified D recomputation as duplicated; it didn't notice the
binding-level infrastructure was already prepared.  Multi-pass
catches the compound.

---

## Pass 2 — Compound improvements (post-Pass-1)

**Focus:** what's now visible with Pass 1 findings as conceptually-resolved
baseline? Opportunities exposed by Pass 1 fixes?

### P2-HIGH-01 (compound) — D_vec precompute total effort is FAR less than Sprint 2 audit estimated

**Source:** P1-HIGH-01 (operand half-wired) + Sprint 2 audit M2-HIGH-01
combined.
**Confidence:** High

Sprint 2 audit said: "1 day CC, ~300 LOC of work" for D_vec precompute
Primitive.  Now with P1-HIGH-01 in hand, the decomposition is:

| Task | Effort | Already done? |
|---|---|---|
| Reserve operand index for `D` in V34 dQ + dK/dV signatures | **DONE** in C++ (lines 532, 535, 576, 579) | ✓ |
| `operandLocationWithHeadOffsetValue` math for `D` | **DONE** in C++ (Pass 1 confirmed) | ✓ |
| Kernel bodies: switch from inline `rowsum(dO⊙O)` recompute to `D[i]` buffer read | ~3 edits (1 per generator), ~30 LOC each | ✗ pending |
| Primitive `eval_gpu`: bind D buffer at the reserved index | ~3 edits (1 per Primitive), ~10 LOC each | ✗ pending |
| Python `_make_mfa_custom`: compute `D_vec = mx.sum(dO * O, axis=-1)` ONCE before bwd calls + pass to v6_nax_backward_{query, dv_raw, dk_raw} | ~5 LOC + 3 binding-signature edits | ✗ pending |
| Python bindings (`csrc/bindings.cpp`): add D as input | 3 binding entries (`v6_nax_backward_query`, `v6_nax_backward_dv_raw`, `v6_nax_backward_dk_raw`) | ✗ pending |
| Tests: verify D_vec precompute produces same gradients as inline recompute | 1 new test, ~50 LOC | ✗ pending |

**Estimated effort: 2-4 hours CC** (not 1 day).  The expensive scaffolding
work (operand-index reservation, buffer-binding math) is already done.

### P2-MEDIUM-01 (compound) — D_vec precompute pairs with M4-MEDIUM-01 DRY-predicate fix

**Source:** P1-HIGH-01 (D_vec wiring point) + Sprint 2 audit M4-MEDIUM-01
(V34-eligibility predicate duplicated 3 times).
**Confidence:** Medium

Sprint 2 audit M4-MEDIUM-01 noted the V34-eligibility predicate (env
check + has_nax + D ∈ {64,128} + dtype + !causal) is duplicated in:
1. flash_attention carve-out (now `_v34_backward_carveout` per v2.38.x)
2. `_make_mfa_custom` forward-fusion check
3. `_make_mfa_custom` backward eligibility check

When implementing D_vec precompute (P2-HIGH-01), the Python code in
`_make_mfa_custom` MUST check V34-eligibility BEFORE computing D_vec
(it's wasted work otherwise).  Naturally suggests extracting the
predicate into a shared helper.

**Compound win:** doing D_vec precompute and M4-MEDIUM-01 DRY fix
together is cleaner than separately — one Python-side refactor pass
addresses both.

### P2-MEDIUM-02 (compound) — Legacy fused kernel deletion coordinates with Option γ implementation

**Source:** P1-MEDIUM-03 (legacy deletion candidate) + future Option γ
implementation.
**Confidence:** Medium

If Option γ ships as a NEW kernel (`createV34BackwardFusedDKDVOptionGammaSource`
or similar), the V34 backward source-generator inventory becomes:

| Kernel | Status | LOC |
|---|---|---|
| createV34BackwardQuerySource (dQ) | production | ~880 |
| createV34BackwardKeyValueSource (legacy WM=1 fused) | legacy fallback | ~820 |
| createV34BackwardDVSource (Phase 2.O2 dV, WM=4 split) | production default | ~330 |
| createV34BackwardDKSource (Phase 2.O2 dK, WM=4 split) | production default | ~440 |
| NEW: createV34BackwardFusedDKDVOptionGammaSource (Option γ) | future | ~700 (design est.) |

That's 5 kernels.  Deleting the legacy fused (Sprint 2 M3-HIGH-01) at the
same time as Option γ lands reduces to 4 kernels and aligns the
backward-stack mental model:
- dQ kernel
- dV kernel (split fallback if Option γ doesn't engage)
- dK kernel (split fallback if Option γ doesn't engage)
- Fused dK+dV (Option γ default when eligible)

Without this coordination, the inventory grows to 5+ kernels indefinitely.

---

## Pass 3 — Architectural patterns

**Focus:** abstraction-level issues, design coherence, boilerplate
patterns across multiple modules.

### P3-HIGH-01 — V34 backward Primitive boilerplate consolidation candidate

**File:** `csrc/mfa_v6_nax_primitive.cpp:832, 1046, 1245, 1431`
**Category:** D (Maintainability)
**Confidence:** High

Four V34 backward Primitive classes — MFAV34BwdQuery, MFAV34BwdKeyValue,
MFAV34BwdDV, MFAV34BwdDK — each ~200 LOC of similar scaffolding:

- ctor with stream + scale (DV/DK also take `wm` — inconsistency, see P4-LOW-01)
- `name()` returning class name string
- `eval_cpu()` throwing "CPU eval not supported"
- `eval_gpu()` with pipeline-cache key lookup + dispatch
- `is_equivalent()` comparing class members
- `print()` writing class name to stream

Estimated ~50 LOC of pure boilerplate per Primitive × 4 = ~200 LOC dedup
candidate.

**Fix:** introduce a base class `MFAV34BwdBase : public mlx::core::Primitive`
in `csrc/mfa_v6_nax_primitive.cpp` with:
- Common ctor (stream + scale; subclasses override if extra params needed)
- Common `eval_cpu()` (CPU not supported)
- Common `is_equivalent()` (delegate to virtual `kernel_specific_eq()`)
- Common `print()` (uses `name()`)
- Subclasses override only `name()` + `eval_gpu()` + (optional) `kernel_specific_eq()`

Same pattern as the Sprint v2.38.x Phase B Apple-helpers refactor but at
the C++ Primitive layer instead of MSL source layer.

**Effort:** ~1 day CC.  Risk: medium (Primitive class hierarchy
touches MLX internals; needs verification that base-class virtual
dispatch works under MLX's compute-graph optimizer).

### P3-MEDIUM-01 — Env var explosion (12+ V34BWD* env vars)

**Source files:** `csrc/mfa_v6_nax_primitive.cpp` (most), `mlx_mfa/attention.py`
**Category:** D / Sprint 2 audit Module 5 LOW-02 expansion
**Confidence:** High

Audit of `MFA_V34BWD*` env vars:

| Env var | Purpose | Practical use |
|---|---|---|
| `MFA_ENABLE_V34_BACKWARD` | enable V34 backward path | YES — primary user-facing |
| `MFA_V34BWD_USE_FUSED` | route to legacy fused WM=1 kernel | UNCLEAR (P1-MEDIUM-03 deletion candidate) |
| `MFA_V34BWD_BQ`, `BK`, `WM` | global tile overrides | DEBUG only |
| `MFA_V34BWDV_BQ`, `BK`, `WM` | dV tile overrides | DEBUG only |
| `MFA_V34BWDK_BQ`, `BK`, `WM` | dK tile overrides | DEBUG only |
| `MFA_V34BWDKV_BQ`, `BK`, `WM` | legacy fused tile overrides | DEAD when fused kernel deleted |
| `MFA_V34BWD_DUMP_SOURCE` | dump generated kernel source | DEBUG only |
| `MFA_V34_DUMP_SOURCE` | same scope, different name? | DEBUG, inconsistent naming |

Cleanup candidates when v2.39.0 lands:
- Unify MFA_V34BWD{,V,K,KV}_{BQ,BK,WM} into structured override:
  `MFA_V34_TILE_OVERRIDE='{"dV": {"BQ":64,"BK":32,"WM":4}, ...}'` (JSON in env var)
- Delete `MFA_V34BWDKV_*` when legacy fused deleted (P1-MEDIUM-03 / P2-MEDIUM-02)
- Pick ONE of `MFA_V34BWD_DUMP_SOURCE` / `MFA_V34_DUMP_SOURCE` (or unify)

Defer to a focused env-var-cleanup sub-sprint or a v2.39.0 doc-only
release.

### P3-MEDIUM-02 — Dispatch threshold tables + carve-out consolidation question

**Source:** `mlx_mfa/dispatch_policy.py` thresholds + `_v34_backward_carveout`
**Category:** D / question
**Confidence:** Low

Sprint 2 audit Module 5 flagged 3 threshold tables (M5_NAX_THRESHOLDS,
M3_THRESHOLDS, DEFAULT_THRESHOLDS).  Sprint v2.38.x Phase A added
`_v34_backward_carveout()` as a NEW separate function.

Architectural question for Pass 3: should the threshold tables and the
carve-out function be ONE concept?  They both decide "should MFA path
engage?" but at different decision points:
- Thresholds: generic MFA-vs-SDPA crossover by (D, qL, causal) shape
- Carve-out: V34-backward-specific routing override

After Pass 3 reflection: **keep them separate.**  The concerns are
genuinely orthogonal:
- Thresholds are general FA-vs-SDPA shape calibration
- Carve-out is V34-backward-specific (env-var gated, has additional
  V34-eligibility constraints)

Consolidating would obscure the orthogonal-concern boundary.  This
finding is a non-action; the question worth documenting for future
reviewers who might be tempted to "unify" without understanding the
two layers serve different purposes.

---

## Pass 4 — Convergence check

**Focus:** any remaining findings, or convergence?

### P4-LOW-01 — Primitive ctor signature inconsistency

`MFAV34BwdQuery`, `MFAV34BwdKeyValue` take only `(stream, scale)`.
`MFAV34BwdDV`, `MFAV34BwdDK` take `(stream, scale, wm)`.

Since all 4 are WM=4 in production (Phase 2.O2), the first two could
add `wm` param for consistency (default = 4).  Trivial polish.

### Pass 4 verdict: **CONVERGENCE REACHED**

Pass 4 surfaced 1 LOW finding (cosmetic).  No new HIGH or MEDIUM.
Multi-pass review converges.  Stop.

---

## Synthesis — V34 backward stack multi-pass

| Severity | Count | New vs Sprint 2 audit? | Compound? |
|---|---|---|---|
| HIGH | 2 | P1-HIGH-01 NEW + P3-HIGH-01 NEW | P2-HIGH-01 is compound of P1-HIGH-01 + M2-HIGH-01 |
| MEDIUM | 5 | 3 NEW + 2 reconfirm | P2-MEDIUM-01 + P2-MEDIUM-02 are compound |
| LOW | 3 | 3 NEW | — |
| **Total** | **10** | **8 NEW** | **3 compound** |

**Most valuable findings (from Marco's perspective — what actually
changes the v2.38.0 scope decision):**

1. **P1-HIGH-01 + P2-HIGH-01 compound** — D_vec precompute is a
   2-4 hour task, not 1 day, because the C++ binding infrastructure
   is already half-done.  This **materially reduces v2.38.0 effort
   estimate** for Section B (D_vec precompute) of the original sprint.

2. **P3-HIGH-01** — V34 bwd Primitive boilerplate consolidation
   (~200 LOC dedup) — independent quality win; ~1 day CC; medium risk.
   Defer to v2.39.0 cleanup unless coupled with Option γ implementation
   (which adds a 5th Primitive class).

3. **P2-MEDIUM-02** — Legacy fused kernel deletion should be
   COORDINATED with Option γ implementation, not done independently.
   Coordination saves rebuild overhead and clarifies the post-sprint
   inventory.

**Effort impact on v2.38.0 sprint:**
- D_vec precompute alone: 2-4h (was estimated 1 day) — **down**
- Option γ implementation: 1-2 days (per original design doc) — unchanged
- Total D_vec + Option γ: ~1.5-2.5 days CC (was estimated 2-3 days)

The investigation has REDUCED the v2.38.0 effort estimate by ~25%
while CONFIRMING the architectural outcome ceiling at (γ) per
Phase A TGP overhead measurement.

---

## Skill invocations log

| Pass | Skill | Findings count | Notes |
|---|---|---|---|
| 1 | /mlx-code-review (rubric applied) | 1 HIGH + 3 MEDIUM + 2 LOW | Tech-debt + code smells focus |
| 2 | /mlx-code-review (rubric applied) | 1 HIGH compound + 2 MEDIUM compound | Pass-1-resolved baseline |
| 3 | /mlx-code-review (rubric applied) | 1 HIGH + 2 MEDIUM | Architectural patterns |
| 4 | /mlx-code-review (rubric applied) | 1 LOW (convergence) | Stop criterion met |

4 passes total.  Convergence at Pass 4.


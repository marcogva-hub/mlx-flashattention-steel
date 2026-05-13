# V34 forward — multi-pass `/mlx-code-review`

**Status:** complete (2026-05-13)
**Sprint:** v2.38.0 investigation (Phase C)
**Disclaimer:** V34 forward is in production since v2.31.0 (March 2026).
Findings here are **ADVISORY ONLY** unless they're ship-blocking
correctness/perf bugs.  Do NOT queue major refactor of production
code without strong justification (commit `5bfd5c9` warning: Apple-
style single-Otile rewrite is stable and well-tested).

## Target

`csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV34Source()` (lines
2716-3037, ~321 LOC post-Sprint-v2.38.x Apple-helpers extraction).

## Pass 1 — General tech debt + code smells

### FP1-LOW-01 — Stale "Inline" comment misleading post-helpers-refactor

**File:** `NAAttentionKernel.cpp:2744`
**Category:** D
**Confidence:** High

```cpp
// === Inline Apple helpers (~17.7KB of verbatim Apple code) ===
ss << naxHelpersBlock();
```

The "Inline" prefix is now factually incorrect — Sprint v2.38.x Phase B
extracted the helpers to `naxHelpersBlock()`, which the next line
invokes.  Same scope as `P1-MEDIUM-01` in the V34 backward review (dQ
generator's "FUTURE-CLEANUP" stale comment).

**Fix:** rewrite as `// === Apple helpers (via shared naxHelpersBlock(),
extracted Sprint v2.38.x Phase B) ===` or simply delete the standalone
comment line.

### FP1-LOW-02 — TQ naming inconsistency

**File:** `NAAttentionKernel.cpp:2722`

```cpp
const int TQ = BQ / (WM * kU);   // expected = 1 per Apple's static_assert
```

Extends P1-LOW-01 from V34 backward review.  Forward uses `TQ`,
dV/dK use `TQ_per_SG`, dQ uses `TQ`.  Same fix: normalize all 5
generators to `TQ_per_SG`.

Forward is the longest-standing generator (since v2.31.0); changing
its variable name has the largest review surface.  Defer to a focused
"naming-normalize" sub-sprint or do it AT the next major refactor
that touches all five generators (e.g., when Option γ adds a 5th).

### Pass 1 verdict

**No production-blocking findings.**  Two LOW cosmetic items.

V34 forward has been validated since v2.31.0:
- Byte-identical source generation pre/post Sprint v2.38.x Phase B
  (`v34_probe_source()` diff EMPTY, 17688 chars both)
- 5 shapes bit-exact correctness (per commit `663be95` Phase 2-3
  acceptance criterion)
- Stable production deployment ~4 months

## Pass 2 — Compound improvements

**Focus:** any opportunities exposed by Pass 1 fixes?

The two Pass 1 LOW findings (stale comment, naming inconsistency) are
isolated polish.  They don't combine into a compound improvement.

The most-natural compound would be: extract the common kernel preamble
(`MFA_REQUIRE_MSL4` + `#include` block) into a helper.  But that's
extends V34 backward review's P1-LOW-02 (preamble redundancy) and
isn't forward-specific.

**Pass 2 verdict: NO new compound findings.**

## Pass 3 — Architectural patterns

The V34 forward generator at ~321 LOC (post-helpers-extraction) follows
the Apple `steel_attention_nax.h` reference pattern faithfully (per
the file-level comment at lines 2705-2715: "Apple file:line citations
are inline at each substitution site").

No architectural cleanup opportunity beyond what Sprint v2.38.x
already addressed (Apple helpers extraction via `naxHelpersBlock()`).

**Pass 3 verdict: NO findings.**

## Convergence — stopped at Pass 3

Total findings: 2 LOW (cosmetic).  Zero HIGH or MEDIUM.  Forward
generator is in clean production state per the byte-identical
verification done in Sprint v2.38.x.

## Classification

All findings classified per the V34-forward production-stability
caveat:

| Finding | Severity | Production-blocking? | Advisory action |
|---|---|---|---|
| FP1-LOW-01 | LOW | NO | defer to v2.39.0 polish |
| FP1-LOW-02 | LOW | NO | defer to next-touching-all-generators refactor |

**Net Phase C contribution to v2.38.0 scope decision: ZERO.**  V34
forward needs no changes for v2.38.0.

## Skill invocations

| Pass | Skill | Findings count | Notes |
|---|---|---|---|
| 1 | /mlx-code-review (rubric applied) | 2 LOW | Tech-debt focus; no production-blocking |
| 2 | /mlx-code-review (rubric applied) | 0 | No compound; converged |
| 3 | /mlx-code-review (rubric applied) | 0 | Architectural; clean; converged |

3 passes total.  Convergence at Pass 2 effectively (Pass 3 was
defensive verification).

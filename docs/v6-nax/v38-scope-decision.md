# v2.38.0 scope decision — post-investigation

**Status:** complete (2026-05-13)
**Sprint:** v2.38.0 investigation Phase E synthesis
**Verdict:** **P3** — alternative path
**Recommended scope:** D_vec precompute + DP2-HIGH-01 compound refactor.
DEFER Option γ to a focused v2.38.1.

## TL;DR

The original v2.38.0 perf sprint was halted in Phase A pre-flight per
§AA.4 — design doc cited "TGP streaming reduction NOT viable on M5"
as a blocker, but the underlying number was unverified.  This
investigation sprint resolved that ambiguity and surfaced a much
better v2.38.0 scope.

**Three findings drove the verdict:**

1. **Phase A empirical TGP overhead: ~1 µs/K-iter on M5 Max, NOT
   ~100 µs as design doc estimated** (100× overstated).  Total at
   qL=8192 NK=512: 0.5ms, not 51ms.  TGP streaming reduction IS
   viable on M5 NAX — but this doesn't unlock parity.

2. **Phase B P1-HIGH-01: `AttentionOperand::D` infrastructure is
   half-wired in C++** — `createBufferBindings` and
   `operandLocationWithHeadOffsetValue` already reserve D's
   binding-index, but the kernel bodies don't read it and the Python
   integration doesn't pass it.  D_vec precompute is therefore
   2-4h work (not 1 day as Sprint 2 audit estimated).

3. **Phase D DP2-HIGH-01 compound: D_vec precompute + V34-eligibility
   predicate extraction + V34-backward-VJP helper extraction can be
   done in ONE coordinated ~3-4h Python refactor pass**, bundling
   three audit findings (Sprint 2 M2-HIGH-01 + M4-MEDIUM-01 + new
   DP1-MEDIUM-03).

**Architectural floor remains:** dK kernel's extra `dO @ V^T` matmul
scales with D² and is the real architectural cost at D=128.  Even
with Option γ (now technically viable) + D_vec precompute, D=128
V34 backward is ~34-36ms vs SDPA-vjp 20ms = **1.7-1.8× slower**
ceiling.  (α) UNIVERSAL_AUTO_DEFAULT requires parity at all D —
mathematically unreachable.

## Decision verdict

### (P1) Option γ outcome (α) UNIVERSAL_AUTO_DEFAULT now reachable

**RULED OUT.**  Phase A removes the design doc's TGP blocker but
doesn't address the dK matmul architectural floor.  D=128 stays
1.7-1.8× SDPA-vjp regardless of Option γ + D_vec.  Parity at all D
requires reverse-engineering Apple SDPA-vjp's algorithm — out of
scope for any near-term sprint.

### (P2) Option γ outcome (γ) confirmed ceiling

**VIABLE but sub-optimal v2.38.0 scope.**  Option γ implementation
(~1-2 days CC, medium risk) delivers 7-40% backward improvement at
D=128 (still 1.7-1.8× SDPA-vjp).  Bundling Option γ into v2.38.0
doesn't unlock a better outcome class — same (γ) ceiling.

### (P3) Investigation surfaces alternative path

**CHOSEN.**  v2.38.0 scope reframed around the investigation's most
actionable findings:

| Task | Source | Effort | Risk | Outcome |
|---|---|---|---|---|
| D_vec precompute (via half-wired C++ infra) | P1-HIGH-01 + audit M2 | 2-3h | Low | 5-8% backward win |
| `_v34_eligible()` helper extraction | DP1-MEDIUM-02 / audit M4-MEDIUM-01 | 30min | Low | DRY; preps D_vec coord |
| `_v34_backward_vjp()` extraction | DP1-MEDIUM-03 (NEW) | 30min | Low | -80 LOC in `_make_mfa_custom` |
| Delete `_should_use_mfa_m5_nax_carveout` (dead) | DP1-MEDIUM-01 (NEW) | 15min | Low | -45 LOC; ends "Sprint A.6 dormant" placeholder |
| Stale comment cleanups (P1-MEDIUM-01, FP1-LOW-01) | multi-pass | 5min | Trivial | Doc accuracy |
| **Total** | — | **~3-4h CC** | **Low** | **~5-8% perf + ~125 LOC reduction + cleaner V34 integration** |

**Net v2.38.0 sprint:** quality + perf patch on top of v2.37.3 master.
Foundation for v2.38.1 Option γ when empirical customer demand
materializes.

## Why DEFER Option γ to v2.38.1

Three reasons:

1. **Outcome ceiling invariant.**  Whether Option γ ships in v2.38.0
   or v2.38.1, the achievable outcome is (γ) — D=64 broad
   auto-default + D=128 improved SHIP_OPT_IN at 1.7-1.8× SDPA-vjp.
   Packaging doesn't change the math.

2. **Risk/value asymmetry.**  D_vec + DP2-HIGH-01 are predictable
   wins on a coordinated 3-4h Python refactor surface.  Option γ
   is 1-2 days of new MSL kernel + Primitive + binding + tests with
   medium implementation risk (~700 LOC new kernel).  Coupling them
   in one sprint means the riskier piece can block the safer piece's
   ship.

3. **No empirical demand signal.**  No customer has surfaced D=128
   training as a critical bottleneck.  Shipping Option γ speculatively
   commits engineering time without a concrete user pulling on it.
   Per /metal-kernel-dev rubric: kernel work needs empirical
   justification for the architectural complexity.

## What Option γ becomes in v2.38.1 (recommended forward plan)

When implementation is scheduled (likely a future sprint following
empirical demand):

| Design choice | Pre-investigation recommendation | Post-investigation update |
|---|---|---|
| Per-SG-slot output vs TGP streaming reduction | Per-SG-slot only (TGP "not viable" per design doc) | EITHER — TGP streaming verified viable (Phase A); choose based on simplicity |
| Effort estimate | "1-2 days CC" (design doc) | Same (~700 LOC new kernel + integration) |
| Outcome ceiling | (γ) D=128 1.7-1.8× SDPA-vjp | Same |
| Integration | Builds on v2.38.0's D_vec foundation | Bonus: V34 backward integration cleaner post-DP2-HIGH-01 |

Option γ's value proposition is unchanged; the design choice flexibility
is improved.

## Compound findings worth noting for Marco's future planning

| Finding | Severity | Source | Impact on v2.38.0 |
|---|---|---|---|
| P1-HIGH-01 D_vec operand half-wired | HIGH | Phase B Pass 1 (NEW) | Enables (P3) verdict; 2-4h work instead of 1 day |
| DP2-HIGH-01 compound refactor | HIGH | Phase D Pass 2 (NEW) | Coordinates 3 audit findings into one pass |
| P3-HIGH-01 V34 bwd Primitive boilerplate consolidation | HIGH | Phase B Pass 3 (NEW) | ~200 LOC dedup; v2.39.0 cleanup candidate |
| P2-MEDIUM-02 legacy fused deletion coordinate with Option γ | MEDIUM | Phase B Pass 2 compound | When Option γ ships (v2.38.1+), delete legacy fused in same sprint |
| DP2-MEDIUM-01 TGP empirical removes design doc blocker | MEDIUM | Phase D Pass 2 (compound from Phase A) | Option γ design has more flexibility than design doc suggested |

## Skill invocations (per §AA.2)

| Skill | Phase | Findings count | Action |
|---|---|---|---|
| /metal-kernel-dev | A.1 — micro-bench design | Probe pattern + baseline subtraction | Implemented in `bench/v6_nax/tgp_overhead_microbench.py` |
| /mlx-mfa-bench-methodology | A.3 — canonical bench protocol | 3-session ratio, CONFIDENT @ NK=512 | Measurement: 0.976 µs/K-iter |
| /metal-kernel-dev | A.5 — reduction options consultation | 5 options evaluated, none significantly improve baseline | TGP overhead at M5 NAX hardware floor |
| /mlx-code-review × 4 | B — V34 backward stack | 2 HIGH + 5 MEDIUM + 3 LOW; 3 compound; converged Pass 4 | Multi-pass review per Marco's VSR methodology |
| /mlx-code-review × 3 | C — V34 forward | 2 LOW only; converged Pass 2 | Production-stable; no actionable cleanup |
| /mlx-code-review × 3 | D — dispatch + integration | 1 HIGH compound + 4 MEDIUM + 1 LOW; converged Pass 3 | DP2-HIGH-01 is the most actionable v2.38.0 finding |
| /metal-kernel-dev | E — final synthesis | (P3) verdict + scope recommendation | This document |

Total: **15 skill invocations** across 5 phases.

## Recommended next sprint prompt skeleton

(For Marco to launch the v2.38.0 scope confirmed by this investigation.)

```
Sprint title: v2.38.0 — D_vec precompute + DP2-HIGH-01 compound refactor
Type: quality + perf patch
Scope: ~3-4h CC, low risk

Phases:
1. Implement D_vec precompute (leverage half-wired AttentionOperand::D infra)
2. Extract _v34_eligible() helper from triplicated predicate
3. Extract _v34_backward_vjp() helper from _make_mfa_custom._backward
4. Delete _should_use_mfa_m5_nax_carveout dead placeholder
5. Stale comment cleanups across V34 forward + dQ generators
6. /mlx-mfa-perf-audit on the D_vec precompute claim (~5-8% backward win)
7. /mlx-mfa-release-audit pre-tag → v2.38.0 PyPI release
```

Reference: `docs/v6-nax/v38-investigation-decisions.md` for skill invocation log per §AA.2.

## References

- `docs/v6-nax/v38-investigation-decisions.md` (this investigation's overarching decisions doc)
- `docs/v6-nax/tgp-overhead-investigation.md` (Phase A empirical findings)
- `docs/v6-nax/v34-backward-multipass-review.md` (Phase B 4-pass findings)
- `docs/v6-nax/v34-forward-multipass-review.md` (Phase C 3-pass findings)
- `docs/v6-nax/dispatch-integration-multipass-review.md` (Phase D 3-pass findings)
- `bench/v6_nax/tgp_overhead_microbench.py` (Phase A micro-bench code)
- `docs/v6-nax/v34-backward-option-gamma-design.md` (now-superseded design doc; preserve as historical record)
- `docs/audits/v37-systematic-audit.md` (Sprint 2 single-pass audit — multi-pass surfaced 8 NEW findings beyond it)
- `CLAUDE_V6_NAX.md` §AA.4 (disagreement-resolution policy that drove this investigation)
- v2.38.0 perf sprint prompt (halted in Phase A pre-flight) — resolved by this investigation

# v2.38.0 investigation — decisions

**Branch:** `investigate/v38-tgp-overhead-and-multipass`
**Started:** 2026-05-13
**Status:** active
**Type:** investigation (NO code/kernel changes, NO release)

## Mandate

Resolve two unresolved questions from the halted v2.38.0 perf sprint:

1. **TGP overhead 10× discrepancy** — design doc says 51ms at qL=8192
   NK=512; CC pre-flight says 2.5ms. Which is correct? Empirical
   measurement required.
2. **Compound improvements** — multi-pass `/mlx-code-review` (per
   Marco's VSR-work methodology) on V34 backward + forward + dispatch
   until skill converges. Sprint 2 single-pass audit may have missed
   compound findings.

Output: definitive v2.38.0 scope decision (P1/P2/P3 verdict) backed
by data, not speculation.

## Design context

- v2.38.0 perf sprint halted in Phase A pre-flight per §AA.4
- Design doc: `docs/v6-nax/v34-backward-option-gamma-design.md` line
  56-73 (TGP streaming reduction analysis)
- CC pre-flight: 4 SG TGP writes (~10ns) + barrier (~75ns) +
  SG0 sum-and-write (~100ns) + barrier ≈ 200-300ns/row × 16 rows
  per K-tile = 3.2-4.8µs/K-iter; × 512 K-iter = ~2.5ms
- Design doc: ~100µs/K-tile × 512 = 51ms
- **10× discrepancy** — neither side has empirical data on M5 Max

## Decisions log

### DC1 — Phase A micro-bench design

(Filled in during Phase A.)

### DC2 — Phase A measurement result

(Filled in during Phase A.)

### DC3 — Phase B/C/D multi-pass convergence

(Filled in during Phase B-D.)

### DC4 — Phase E P1/P2/P3 verdict

(Filled in during Phase E.)

## Skill invocations

Per `CLAUDE_V6_NAX.md` §AA.2. Populated as the sprint progresses.

| Skill | Decision point | Timestamp (ISO) | Findings count | Action taken |
|---|---|---|---|---|
| /metal-kernel-dev | v2.38.0 sprint Phase A pre-flight (HALT — surfaced design ceiling) | 2026-05-13T~prior | scope-blocker | sprint pivoted to investigation per §AA.4 |
| /metal-kernel-dev | Phase A micro-bench design | TBD | TBD | TBD |
| /mlx-mfa-bench-methodology | Phase A overhead measurement | TBD | TBD | TBD |
| /metal-kernel-dev | Phase A reduction options consultation | TBD | TBD | TBD |
| /mlx-code-review × N | Phase B V34 backward stack | TBD | TBD | TBD |
| /mlx-code-review × N | Phase C V34 forward | TBD | TBD | TBD |
| /mlx-code-review × N | Phase D dispatch + integration | TBD | TBD | TBD |
| /metal-kernel-dev | Phase E final synthesis | TBD | TBD | TBD |

## STATUS

Active. Phase A in progress.

### Next actions

- [x] Branch + scaffold this doc
- [ ] Phase A.1: design micro-bench Metal kernel
- [ ] Phase A.2: implement + dispatch via mx.fast.metal_kernel
- [ ] Phase A.3: run canonical bench protocol
- [ ] Phase A.4: reconcile design vs measurement
- [ ] Phase A.5: /metal-kernel-dev reduction-options consultation
- [ ] Phase B-D: multi-pass /mlx-code-review (convergence-driven)
- [ ] Phase E: P1/P2/P3 verdict synthesis

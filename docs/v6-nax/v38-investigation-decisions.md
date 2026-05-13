# v2.38.0 investigation — decisions

**Branch:** `investigate/v38-tgp-overhead-and-multipass`
**Started:** 2026-05-13
**Status:** COMPLETE
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

Probe pattern: 4 SGs × BK=16 × D=128 fp32 TGP-write + barrier +
SG0-stream-sum + barrier (mirrors Option γ TGP reduction exactly).
Baseline: same outer shape, no TGP+barrier+reduce.  Subtraction
isolates TGP overhead from kernel dispatch floor.

Documented in `docs/v6-nax/tgp-overhead-investigation.md`.

### DC2 — Phase A measurement result

**At NK=512 (design doc reference shape):**
- Median per-iter overhead: 0.976 µs (3-session canonical §4.2)
- Cross-session range: 1.7% (CONFIDENT verdict)
- Total: 0.5 ms (NOT 51 ms as design doc claimed)
- **Design doc was overstated by ~100×**

### DC3 — Phase B/C/D multi-pass convergence

| Module group | Passes | New findings | Convergence |
|---|---|---|---|
| V34 backward stack | 4 | 8 NEW (2 HIGH + 3 MEDIUM + 3 LOW); 3 compound | Pass 4 |
| V34 forward | 3 | 2 LOW only (production-stable) | Pass 2 effectively |
| dispatch + integration | 3 | 6 NEW (1 HIGH compound + 4 MEDIUM + 1 LOW); 2 compound | Pass 3 |

Total NEW findings: **16** across 3 module groups (5 HIGH, 7 MEDIUM,
4 LOW); **5 compound improvements**.

The multi-pass methodology proved value: P1-HIGH-01 (D_vec operand
half-wired) was invisible to Sprint 2 single-pass audit because the
audit focused on kernel bodies, not binding-level scaffolding.
Multi-pass surfaced it on Pass 1 by examining the C++
infrastructure layer.

### DC4 — Phase E P1/P2/P3 verdict

**Verdict: P3** — alternative path.

(P1) UNIVERSAL_AUTO_DEFAULT: RULED OUT (dK matmul architectural floor).
(P2) Bundle Option γ in v2.38.0: viable but sub-optimal (same outcome
  ceiling; coupling risk).
(P3) D_vec precompute + DP2-HIGH-01 compound refactor; defer Option γ
  to v2.38.1: CHOSEN.

See `docs/v6-nax/v38-scope-decision.md` for full justification +
recommended v2.38.0 sprint plan.

## Skill invocations

Per `CLAUDE_V6_NAX.md` §AA.2.  Final log:

| Skill | Decision point | Findings count | Action taken |
|---|---|---|---|
| /metal-kernel-dev | v2.38.0 sprint Phase A pre-flight (prior session, halted) | scope-blocker | sprint pivoted to investigation per §AA.4 |
| /metal-kernel-dev | Phase A.1 micro-bench design (rubric applied) | probe + baseline methodology | implemented in `bench/v6_nax/tgp_overhead_microbench.py` |
| /mlx-mfa-bench-methodology | Phase A.3 canonical bench protocol (rubric applied) | 3-session CONFIDENT at NK=512 | measurement: 0.976 µs/K-iter; 100× lower than design doc |
| /metal-kernel-dev | Phase A.5 reduction-options consultation | 5 options evaluated | no major opt beyond baseline pattern; TGP overhead at M5 NAX floor |
| /mlx-code-review × 4 | Phase B V34 backward stack | 2 HIGH + 5 MEDIUM + 3 LOW; 3 compound | converged Pass 4 |
| /mlx-code-review × 3 | Phase C V34 forward | 2 LOW only | converged Pass 2 (Pass 3 defensive) |
| /mlx-code-review × 3 | Phase D dispatch + integration | 1 HIGH compound + 4 MEDIUM + 1 LOW; 2 compound | converged Pass 3 |
| /metal-kernel-dev | Phase E final synthesis | (P3) verdict + recommended v2.38.0 scope | `docs/v6-nax/v38-scope-decision.md` |

**Total: 15 skill invocations** across 5 phases.  All logged here
+ in per-phase deliverable docs per §AA.2.

## STATUS

**COMPLETE.**  Investigation deliverables ready for Marco's v2.38.0
sprint scoping.

### Completed actions

- [x] Branch + scaffold this doc
- [x] Phase A.1: design micro-bench Metal kernel
- [x] Phase A.2: implement + dispatch via mx.fast.metal_kernel
- [x] Phase A.3: run canonical bench protocol (CONFIDENT verdict)
- [x] Phase A.4: reconcile design vs measurement (design 100× overstated)
- [x] Phase A.5: /metal-kernel-dev reduction-options consultation
- [x] Phase B: V34 backward multi-pass (4 passes, converged)
- [x] Phase C: V34 forward multi-pass (3 passes, only LOW)
- [x] Phase D: dispatch + integration multi-pass (3 passes, 1 HIGH compound)
- [x] Phase E: (P3) verdict + scope decision doc

### Deliverables (all in `docs/v6-nax/`)

- `tgp-overhead-investigation.md` — Phase A empirical findings
- `v34-backward-multipass-review.md` — Phase B multi-pass
- `v34-forward-multipass-review.md` — Phase C multi-pass
- `dispatch-integration-multipass-review.md` — Phase D multi-pass
- `v38-scope-decision.md` — Phase E verdict + recommended v2.38.0 scope
- `v38-investigation-decisions.md` — this overarching decisions doc

### Code artifacts

- `bench/v6_nax/tgp_overhead_microbench.py` — Phase A micro-bench (reusable for future TGP overhead audits)

### NO code changes / NO version bump / NO release

Per investigation-only mandate.  v2.38.0 scope decision deliverable
ready for Marco's next sprint launch.

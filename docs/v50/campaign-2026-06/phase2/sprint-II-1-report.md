# Phase II — Sprint II-1 report: Exhaustive M5 Dispatch Re-Bench

**Date**: 2026-06-12 · **Status**: COMPLETE

## Headline

**74 cells measured; zero new inversions.**  The dispatch table is now
fully M5-validated: 44/44 dense-forward HOLD (SDPA), 24/24 backward
validated (II-0 V34 D=64-causal WIN confirmed in-grid at 2.19-2.71×
fp16 / 1.33-2.68× bf16; all else SDPA-vjp parity), 6/6 decode HOLD,
3/3 odd-D HOLD with diff=0.0.  See `M5-dispatch-map.md` (authoritative).

## Phase-I flagged targets — dispositions

| Target | Disposition |
|---|---|
| V3/V4/V5 M5 perf matrix | DONE Sprint C: 3-4× behind SDPA at D≤128 (V4>V2 at D=128 noted, academic).  HOLD |
| BK=32/16 @ D=128, D=256 BK=8 | MOOT on M5 — no dense forward cell routes to MFA after the D=256-causal fix; configs execute only on M1-M4 branches (hardware-gated, untouched per §AA.4) |
| D=512 "SDPA-optimal ceiling" | HOLDS on M5: auto=SDPA parity at all D=512 cells (re-probe 6.91 vs 6.90 — initial 8.25 reading was transient) |
| Decode-mode custom path | CLOSED: Apple sdpa_vector_2pass covers D≤256; 6/6 parity |
| Odd head dims 80/96/160 | No gap: parity + diff=0.0 (D=80 native NAX variant exists in metallib) |

## UNCERTAIN cells

Three forward cells flagged at 5-10% (D128-bf16-nc-gqa3-N4096,
D256-bf16-nc-N1024, D512-fp16-c-N4096) — all re-probed or classified
as wrapper/transient noise; auto routes to SDPA at all three and no
faster candidate exists.  No routing changes warranted.

## Promotions made

None needed beyond the already-landed Phase-I D=256-causal fix and the
II-0 V34-backward promotion — both confirmed live in this grid.

## M1 cross-check

M1-divergent branches (_DEFAULT_THRESHOLDS / _M3_THRESHOLDS) untouched
by any Phase-II change (all changes are has_nax-gated).  Physical M1
re-bench unnecessary for map validity; flagged available if Marco wants
belt-and-braces confirmation on the M1 Max secondary.

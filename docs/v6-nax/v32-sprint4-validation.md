# Sprint 4 cross-session re-validation — Phase 0 of v2.32.0 release

**Date:** 2026-05-06 13:24–13:52 (28 min wall clock)
**Session:** v2.32.0 release sprint (Phase 0)
**Branch:** `experiment/v6nax-forward-max` @ `8a389c3`
**Hardware:** M5 Max (`applegpu_g17s`), iStat performance fan profile active
**Bench:** `bench/v6nax_aba_wrapper.sh` — 5 production shapes × L/V/L (legacy → V6NAX → legacy) A/B/A
**Cooldowns:** 90s initial, 60s inter-round, 30s inter-shape
**Raw data:** [`v32-aba.json`](v32-aba.json)

## Why this exists

The user prompt for v2.32.0 mandated cross-session re-validation of Sprint 4
before merging. Sprint 4 reported a sign-flip — turning a "V6NAX −39%
regression on FlashVSR-dense" into a "V6NAX +20% win" via a tile-config
change (BK=64 → BK=32). Such inversions are exactly the methodology trap
flagged in `CLAUDE_V6_NAX.md` §2; cross-session re-validation in a fresh
session is the agreed defense.

## Results

### Per-shape Phase 0 measurements

| Shape | R1 legacy ms | R2 V6NAX ms | R3 legacy ms | R1↔R3 drift | SDPA stable? | V6NAX vs legacy avg | V6NAX/SDPA |
|---|---:|---:|---:|---:|:---:|---:|---:|
| FlashVSR-dense | 0.93 | 0.95 | 1.74 | **86.7% ❌** | 0.89→0.91→1.08 (drifted) | 0.713× (artifact) | 1.044× |
| LTX2-cross | 1.65 | 1.30 | 1.61 | 2.5% ✅ | 1.34/1.33/1.33 ✅ | **0.797× = +20% V6NAX** | **0.973×** |
| SeedVR2-small | 162.13 | 184.73 | 173.37 | 6.9% ✅ | 184.92/197.35/184.68 ✅ | 1.101× = **−10% V6NAX** | 0.936× |
| CogVideoX | 2335.85 | 2162.00 | 2352.43 | 0.7% ✅ | 2194/2187/2183 ✅ | **0.922× = +8% V6NAX** | **0.989×** |
| SeedVR2-large | 4073.61 | 3878.19 | 3891.03 | 4.5% ✅ | 3831/3797/3762 ✅ | 0.974× = **+3% V6NAX** | 1.021× |

### Acceptance verdict (per shape)

| Shape | Sprint 4 / v2.31.0 claim | Phase 0 finding | Verdict |
|---|---|---|---|
| FlashVSR-dense | Sprint 4 +20% V6NAX default | Drift 86.7% — measurement invalid. R1↔R2 cold comparison: V6NAX −2% | **NOT REPLICATED.** Sprint 4 +20% claim does not survive cross-session. |
| LTX2-cross | Sprint 4 +14% V6NAX default | Drift 2.5% — clean. V6NAX +20%, V6NAX/SDPA 0.973× | **REPLICATED ✅** (even better than Sprint 4 claim). |
| SeedVR2-small | v2.31.0 +36% V6NAX default | Drift 6.9% — clean. V6NAX −10% | **NOT REPLICATED ❌** — V6NAX is now slower than legacy. v2.31.0 +36% was thermally-inflated comparison. |
| CogVideoX | v2.31.0 +34% V6NAX default | Drift 0.7% — clean. V6NAX +8% | **PARTIALLY REPLICATED.** Real V6NAX win, much smaller than v2.31.0 marketed. |
| SeedVR2-large | v2.31.0 +40% V6NAX default | Drift 4.5% — clean. V6NAX +3% | **MARGINALLY REPLICATED.** Real V6NAX win, much smaller than v2.31.0 marketed. |

## Cross-session legacy drift

The most striking finding is not V6NAX's behavior but **how much faster
legacy is today than in the v2.31.0 v6nax-aba.json data** (which was the
basis for v2.31.0's headline claims):

| Shape | v2.31.0 legacy median ms | Today legacy avg ms | Δ (today vs v2.31.0) |
|---|---:|---:|---:|
| FlashVSR-dense | 1.115 | 0.93 (R1 cold) | −17% |
| LTX2-cross | 1.65 | 1.63 | −1% |
| SeedVR2-small | 275.6 | 167.75 | **−39%** |
| CogVideoX | 3669 | 2344 | **−36%** |
| SeedVR2-large | 6780 | 3982 | **−41%** |

D=128 legacy paths run 36–41% faster today than in v2.31.0 measurements.
Same hardware. Same code (no commits touched legacy path). Most likely:
macOS GPU power-management / pipeline-cache state shifted between v2.31.0
release and now.

This means **v2.31.0's headline "V6NAX wins +33-40% on D=128"** was the
ratio of (v2.31.0 V6NAX) over (v2.31.0 thermally-penalized legacy). Today's
ratio of (Sprint 5 V6NAX) over (today's faster legacy) is much smaller —
or, in SeedVR2-small's case, inverted.

## Implications for v2.32.0

### Sprint-by-sprint disposition

| Sprint | Outcome |
|---|---|
| 1 — Causal port | **Independent of Phase 0.** Real architectural extension. Ships. |
| 2 — LSE writeback | **Independent of Phase 0.** Silent bug fix. Ships. |
| 3 — align_Q/align_K compile-time gates | **Independent of Phase 0.** Apple-pattern infra; perf-neutral at our scale. Ships. |
| 4 — D=64 BK=64 → BK=32 default | **Robust improvement** — closes the real −39% regression on FlashVSR-dense (V6NAX BK=64 was 1.55ms → BK=32 is 0.95ms). Ships. |
| 4 — V6NAX always-on dispatch for D=64 | **Sprint 4 +20% claim does NOT replicate.** V6NAX is at parity (FlashVSR-dense) or wins big (LTX2-cross). Per Phase 0 spec, **revert** to v2.31.0 behavior: V6NAX default for D=64 N_kv > 8000 only; legacy default for D=64 small-N. |
| 5 — D=128 autoresearch sweep | **Confirms defaults are inside 1.3% noise of best.** Documents the search space. Ships. |

### v2.31.0 inherited issue (out of Phase 0 scope)

Phase 0 surfaced a v2.31.0 dispatch issue: V6NAX is currently the default
for **all D=128 shapes**, but cross-session data shows V6NAX regresses 10%
on SeedVR2-small while winning +8% on CogVideoX and +3% on SeedVR2-large.
The v2.31.0 release table that motivated "V6NAX universal D=128 default"
appears to have been measured against thermally-inflated legacy values.

Two paths exist for handling this in v2.32.0:

1. **Document only.** Note in CHANGELOG that SeedVR2-small specifically
   regresses ~10% under V6NAX vs legacy on M5 Max in current measurements;
   recommend `MFA_V6_USE_NAX=0` override for that shape. No code change
   beyond Sprint 4 dispatch revert.

2. **Code carve-out.** Add a per-shape D=128 dispatch refinement: V6NAX
   default only when `N >= 50000` (catches CogVideoX at 70k and
   SeedVR2-large at 111k; sends SeedVR2-small at 26.7k to legacy).
   More invasive, requires correctness validation across more shapes.

Recommended: **Option 1 (document only)** for v2.32.0. Treat the
SeedVR2-small finding as a measurement-methodology recalibration that
warrants its own follow-up sprint (re-bench v2.31.0's full 5-shape
table, decide whether v2.31.0 itself needs a v2.31.1 perf addendum).
Don't bake it into v2.32.0 without confirmation that the regression
is reproducible across multiple sessions on multiple machines.

### Final v2.32.0 dispatch policy (proposed)

Same as v2.31.0 (revert Sprint 4 dispatch change), with the Sprint 4
BK=32 fix preserved:

```cpp
// V6NAX dispatch logic — same as v2.31.0
if (D == 128) { use_v6nax = true; }
else if (D == 64 && N_kv > 8000) { use_v6nax = true; }
else { use_v6nax = false; }

// V6NAX D=64 tile defaults — Sprint 4 BK fix, KEPT:
v6nax_BQ = (head_dim == 64) ? 32 : 64;
v6nax_BK = (head_dim == 64) ? 32 : 32;  // ← was 64, now 32 (Sprint 4)
v6nax_WM = (head_dim == 64) ? 2 : 4;
```

## Methodology caveats

- **R1↔R3 thermal drift on FlashVSR-dense was 86.7%**, far above the 10%
  Phase 0 acceptance ceiling. The 1ms-scale workload is too small to
  amortize OS-scheduler / GPU-clock-state perturbations within a 60s
  cooldown. For meaningful FlashVSR-dense bench, consider longer
  cooldowns (180s+), longer iter counts (32+), or different methodology
  (sustained-rate measurement).
- **R2 SDPA on SeedVR2-small drifted +7%** (185 → 197) — its V6NAX
  measurement is therefore mildly thermally penalized. Adjusted for
  thermal trend (V6NAX ≈ 184.73 × 184/197 ≈ 173ms), V6NAX is still slightly
  slower than legacy avg 167.75ms (~3-4%). The "V6NAX loses" finding holds
  but margin shrinks from 10% to 3-4%.
- **Single session.** Phase 0 used one bench session. Reproducibility
  across multiple sessions over hours/days would strengthen any "V6NAX
  regresses on SeedVR2-small" claim. v2.32.0 is not blocked on this;
  treating as a follow-up sprint candidate.

## Decision summary

- ✅ **Ship v2.32.0** with Sprints 1, 2, 3, 5 + the Sprint 4 BK=32 tile-default change.
- ✅ **Revert** Sprint 4's V6NAX-always-on D=64 dispatch change to v2.31.0 behavior (V6NAX only for D=64 N_kv > 8000).
- 📄 **Document** the cross-session legacy-drift finding and the SeedVR2-small Phase 0 surprise in v2.32.0 CHANGELOG, deferring per-shape D=128 dispatch refinement to a future sprint.
- 🔬 **Backlog** — schedule a multi-session cross-validation of v2.31.0's perf table to decide whether a v2.31.1 perf-correction addendum is needed.

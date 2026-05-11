# Sprint G dispatch v6 — thermal-stable re-bench (revert confirmed)

**Date:** 2026-05-05
**Verdict:** **Scenario B — revert vindicated.** Dispatch v6's regression on
SeedVR2-large is architectural, not thermal-throttling artifact.

## Context

The v2.30.0 final session reverted Sprint G's dispatch v6 (`96daff7` →
`ca0fc44`) after a thermally-controlled re-bench showed regression
(SeedVR2-large +14.3%, SeedVR2-small +5.9%). Subsequent analysis suggested
the bench may have been confounded by Apple's default fan profile (carried
over via Migration Assistant from M1 Max), causing thermal throttling.

After Marco activated the iStat performance fan profile, this sprint
**re-tested dispatch v6 in thermally-stable conditions** to determine
whether the original revert was justified or a false positive.

## Methodology

**Cross-session A/B/A protocol** with subprocess-isolated rounds:

- R1: `feat/v6-nax` (dispatch v5, baseline)
- 120s inter-round cooldown
- R2: `experiment/sprint-g-rebench-thermal-stable` (dispatch v6 reapplied)
- 120s inter-round cooldown
- R3: `feat/v6-nax` (dispatch v5, thermal validation)

Each round: clean `git checkout` + `pip install -e . --force-reinstall` +
fresh subprocess. 3 runs × 4-8 iters median per shape, 30s inter-shape
cooldown for thermal stability.

Critical fix vs the mandate's draft script: this bench calls
`_ext.v6_nax_forward` **directly** (V6 NAX kernel) — the mandate's draft
used `mlx_mfa.attention()` which routes through `flash_attention()` to
STEEL/SDPA, **not V6 NAX**.

The dispatch v6 reapplication (commit `6ed6325`) committed all changes
**before** the wrapper started, fixing the prior session's confound where
uncommitted source mods carried across `git checkout`.

## Thermal stability validation

R1 vs R3 drift (both v5 baseline, ~25 minutes apart):

| Shape | R1 (ms) | R3 (ms) | drift |
|---|---:|---:|---:|
| FlashVSR-dense | 1.60 | 1.14 | -28.7% (R1 cold-start outlier) |
| LTX2-cross | 1.55 | 1.55 | 0.0% |
| SeedVR2-small | 266.59 | 268.75 | +0.8% |
| CogVideoX | 2933.53 | 2981.71 | +1.6% |
| SeedVR2-large | 5425.21 | 5752.86 | +6.0% |

**4 of 5 shapes ≤ 6 % R1↔R3 drift.** The new iStat profile is materially
better than the original session's bench (where drifts up to 50% were
observed on CogVideoX). The FlashVSR -28.7% drift is from R1's first-run
cold-start (2.15 ms before pipeline-cache warm); R3 was warmer.

**Verdict on thermal stability**: methodology validated. The benches are
now thermally trustworthy.

## Results

Dispatch v6 vs v5 baseline (`avg(R1, R3)`):

| Shape | v5 (avg) | v6 (R2) | Δ | Note |
|---|---:|---:|---:|---|
| FlashVSR-dense | 1.37* | 1.15 | -16.1%* | *R1 cold-start; warmed v5=1.14, v6=1.15 (neutral) |
| LTX2-cross | 1.55 | 1.77 | +14.2%† | †R2 cold-start; warmed v6=1.53 (neutral) |
| SeedVR2-small | 267.67 | 266.54 | -0.42% | unchanged config (v5=v6) |
| CogVideoX | 2957.62 | 2943.30 | -0.48% | unchanged config (v5=v6) |
| **SeedVR2-large** | **5589.04** | **6331.13** | **+13.27%** ⚠️ | **regression confirmed** |

**SeedVR2-large**: v6 runs were ['6057.47', '6391.72', '6331.13'] — tight,
no outliers. v5 R1 ['5425.21', '5352.39', '5460.73'] and R3
['5752.86', '5963.34', '5703.25'] — both inside [5350, 5970]. v6 (6300+)
is **well outside both v5 ranges**. The regression is real.

## What dispatch v6 actually changed

```
v5 → v6 modifications:
  D=64:                BK=64 SG=2  → BK=64 SG=4
  D=128, N >= 100000:  BK=32 SG=16 → BK=64 SG=8
  D=128, N < 100000:   unchanged
```

Per-shape impact:
- **D=64 (FlashVSR, LTX2)**: SG=2 → SG=4. Neutral (warmed v5 ≈ warmed v6).
  Sprint G's "-6.4 % on FlashVSR" claim is not reproducible.
- **D=128 N<100k (SeedVR2-small, CogVideoX)**: dispatch unchanged. Both
  benches show essentially identical numbers (-0.42 % and -0.48 % within
  variance).
- **D=128 N≥100k (SeedVR2-large)**: BK=32 SG=16 → BK=64 SG=8. **Regresses
  +13.3 %**. The original Sprint G "-11.7 %" claim was a within-session
  pipeline-cache artifact.

## Conclusion

The original revert decision in v2.30.0 final was correct. Dispatch v6's
modifications are either **neutral** (on D=64 + D=128 small N) or
**regress** (on D=128 large N). There's no thermal-throttling-hidden gain
to recover.

**Action**: keep dispatch v5 as production default. Close the question.
The branch `experiment/sprint-g-rebench-thermal-stable` can be kept for
historical traceability but should NOT be merged.

## Methodology lessons confirmed

1. **iStat performance fan profile is required** for stable benches on
   M5 Max. Apple's default profile causes ≥50 % drift on long-running
   D=128 shapes; iStat profile reduces that to ≤6 %.
2. **Cross-session A/B/A with committed source state** is the trustworthy
   methodology. Both prior issues (Sprint G original bench and
   exec_sg cache-key bug) involved within-session cache pollution. With
   committed branch state + subprocess-isolated rounds + fresh rebuild,
   the bench is finally measuring real signal.
3. **Sprint G's original "wins" were systematic within-session
   contamination**, not thermal artifacts. The within-session A/B
   compared two pipelines compiled in the same Python session with shared
   cache; the second config (v6) inherited warmth from the first config
   (v5)'s warmup, making it look 6-12 % faster purely from cache state.
   The cross-session bench cleanly shows v6 is neutral or worse.

## Files

- `docs/v6-nax/sprint-g-revert-confirmed.md` (this)
- `docs/v6-nax/sprint-g-rebench-thermal-stable.json` (raw bench data)
- `outputs/sprint_g_rebench.log` (execution log; not committed)
- `bench/sprint_g_round_bench.py`, `bench/sprint_g_aba_wrapper.sh`
  (scripts; copied from /tmp for archival)

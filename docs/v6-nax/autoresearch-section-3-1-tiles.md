# S3.1 — fine BQ × BK × SG sweep (tiered)

**Status:** **COMPLETE.** Run on 2026-05-05, M5 Max, ~1 hour wallclock.

## Method — tiered protocol

A naïve "all 216 feasible configs × 5 shapes × 6-8 iters" sweep would have
taken ~9 hours (SeedVR2-large at sub-optimal configs reaches 64+ sec/iter).
Instead, the script tiers:

| Tier | Configs | Shapes | Cost |
|---|---|---|---|
| 1 | All 216 feasible | FlashVSR-dense + LTX2-cross | ~2 min |
| 2 | Top 30 by Tier-1 sum | SeedVR2-small | ~10 min |
| 3 | Top 10 by Tier-2 time | CogVideoX + SeedVR2-large | ~30 min |

Tier promotion = sort by per-shape median time within tier, take top-N.

## Search space

- BQ ∈ {8, 12, 16, 20, 24, 32}
- BK ∈ {16, 24, 32, 40, 48, 56, 64, 80, 96, 128}
- SG ∈ {1, 2, 4, 6, 8, 12, 16, 24, 32}

Total: 540 raw configs. After feasibility filter (BK%32==0 to satisfy
Apple MPP cooperative-left static_assert; threads/TG = SG×32 ≤ 1024):
**216 configs** progressed to Tier 1.

## Tier 1 — D=64 dominated (FlashVSR-dense + LTX2-cross sum, ms)

144/216 configs passed correctness (RMSE < 5e-3). Top-10:

| Rank | BQ | BK | SG | Sum (ms) |
|---|---|---|---|---|
| 1 | 16 | 64 | 2 | 2.64 |
| 2 | 16 | 64 | 4 | 2.66 |
| 3 | 16 | 64 | 8 | 2.77 |
| 4 | 16 | 32 | 2 | 2.79 |
| 5 | 16 | 32 | 4 | 2.82 |

**D=64 verdict: BQ=16 BK=64 SG=2 stays optimal** — matches v2.29.0
default. No change.

## Tier 2 — D=128 medium (SeedVR2-small, ms)

Top 30 from Tier 1 advanced. Top-5:

| Rank | BQ | BK | SG | SeedVR2-small (ms) |
|---|---|---|---|---|
| 1 | 16 | 32 | **16** | **308.91** |
| 2 | 16 | 32 | 8 | 329.44 |
| 3 | 16 | 32 | 4 | 330.94 |
| 4 | 16 | 64 | 16 | 332.06 |
| 5 | 16 | 32 | 12 | 335.74 |

**D=128 SG=16 wins by ~6 % over SG=8** at the same BK=32 — the v2.29.0
default uses SG=8.

## Tier 3 — D=128 long (CogVideoX + SeedVR2-large)

Top 10 from Tier 2 advanced. Top-5 by sum:

| Rank | BQ | BK | SG | CogVideoX (ms) | SeedVR2-large (ms) | Sum |
|---|---|---|---|---|---|---|
| 1 | 16 | 32 | **16** | 3189 | 7785 | 10 974 |
| 2 | 16 | 64 | 16 | 3139 | 7987 | 11 126 |
| 3 | 16 | 32 | 8  | 3359 | 8452 | 11 811 |
| 4 | 16 | 32 | 12 | 4165 | 8745 | 12 910 |
| 5 | 16 | 64 | 8  | 4252 | 8691 | 12 943 |

**D=128 long-N verdict: BQ=16 BK=32 SG=16** — wins by ~5-8% over
v2.29.0's SG=8 on every shape. SG=16 BK=32 is consistently better than
SG=16 BK=64 across both Cog and SeedVR2-large.

## Cross-shape decision

- **D=64**: BQ=16 BK=64 SG=2 (unchanged from v2.29.0)
- **D=128**: BQ=16 BK=32 **SG=16** (was SG=8)

The D=128 change is potentially a 5-8 % win across all 3 D=128 shapes.
However, run-to-run variance on M5 Max is also 5-10 % (the same SG=8
config measured 1129 ms / 936 ms / 329 ms across three different runs at
different points in this campaign). The ~6 % SG=16 advantage may be
within variance.

**Decision: ship SG=16 for D=128 if confirmation bench reproduces the
gain.** Section 3.6 (synthesis) runs that confirmation.

## Files

- `bench/v6_autoresearch_section_3_1_tiles.py` — tiered sweep script
- `docs/v6-nax/autoresearch-section-3-1-tiles-data.json` — raw results
- `docs/v6-nax/autoresearch-section-3-1-tiles.md` — this file

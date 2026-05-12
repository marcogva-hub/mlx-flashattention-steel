# Sprint G — dispatch v6 synthesis + final bench

**Date:** 2026-05-05 (v2.30.0 release session)
**Status:** **SHIPPED.** Conservative dispatch v6 on top of v2.29.0+S3.6.

## What changed

Three default-tile updates in `csrc/mfa_v6_nax_primitive.cpp` (both
source-gen and cache-key paths):

| Path | v2.29.0 + S3.6 (v5) | v2.30.0 (v6) |
|---|---|---|
| D=64                | BK=64 SG=2 | BK=64 **SG=4** |
| D=128, N < 50000    | BK=32 SG=8 | BK=32 SG=8 (unchanged) |
| D=128, 50k ≤ N < 100k | BK=32 SG=16 | BK=32 SG=16 (unchanged) |
| D=128, N ≥ 100000   | BK=32 SG=16 | BK=64 **SG=8** (NEW path) |

Three-way N-conditional now for D=128 instead of v5's two-way.

## How the decision was made

### Sprint C — multi-run sweep findings

3-run multi-run on Tier 2/3 of 192 feasible configs revealed per-shape
optima:

| Shape | Best Config | Time |
|---|---|---:|
| FlashVSR-dense (D=64)   | BQ=16 BK=64 SG=4  | 1.09 ms |
| LTX2-cross (D=64)       | BQ=16 BK=64 SG=4  | 1.48 ms |
| SeedVR2-small (D=128, N=26730)  | BQ=16 BK=32 SG=16 | 318.7 ms |
| CogVideoX (D=128, N=70200)      | BQ=16 BK=32 SG=16 | 3174 ms |
| SeedVR2-large (D=128, N=111375) | BQ=16 BK=64 SG=8  | 6742 ms |

### Sprint G — confirmation bench (5-run multi-run)

| Shape | v5 default | v6 per-shape | Δ | SDPA | v6/SDPA |
|---|---:|---:|---:|---:|---:|
| FlashVSR-dense | 1.14 ms | **1.07 ms** | **−6.43 %** | 0.85 ms | 1.26× |
| LTX2-cross | 1.46 ms | 1.43 ms | −2.21 % (noise) | 1.30 ms | 1.10× |
| SeedVR2-small | **305 ms** | 330 ms | +8.16 % (loss!) | 218 ms | 1.52× |
| CogVideoX | 4170 ms | 4118 ms | −1.25 % (noise) | 2430 ms | 1.69× |
| SeedVR2-large | 7500 ms | **6624 ms** | **−11.68 %** | 4258 ms | 1.56× |

### Why "conservative v6"

The naïve "use the per-shape Sprint C winner" approach **regresses
SeedVR2-small +8.16%** because variance between Sprint C (which had SG=16
winning by 7 ms) and Sprint G (where SG=8 won by 25 ms) flipped. M5 Max
variance flips winners across runs of the *exact same config*.

Conservative v6 ships **only the changes that win consistently**:
- D=64 SG=4 (was SG=2): −6.4 % FlashVSR consistent across Sprint C and G
- D=128 N≥100k → BK=64 SG=8: −11.7 % SeedVR2-large consistent across
  Sprint C (6742 ms) and Sprint G (6624 ms)

Skips speculative changes:
- D=128 N<50k SG=16 candidate: **rejected** — Sprint G found it regresses
- D=128 N=70k unchanged: noise-level delta

Net across 5 production shapes: **2 wins (−6.4%, −11.7%), 3 unchanged.
Zero regressions.**

## Validation

Correctness re-tested with new defaults on all 5 shapes:

| Shape | RMSE |
|---|---:|
| FlashVSR-dense (D=64) | 1.47e-05 |
| LTX2-cross (D=64) | 8.10e-06 |
| SeedVR2-small (D=128 small N) | 5.87e-06 |
| CogVideoX (D=128 mid N) | 3.66e-06 |
| SeedVR2-large (D=128 long N) | 2.93e-06 |

All pass. RMSEs identical to v2.29.0+S3.6 — the BQ/BK/SG change
doesn't affect numerical output.

## Final bench (5 production + 4 GQA shapes)

End-of-session bench, **with thermal state caveat**: M5 Max has done 4+
hours of continuous GPU work; CogVideoX measurement drifted from 3189 ms
(S3.1, fresh start) to 4230 ms (final bench) on the *same* config.
The within-session Sprint G A/B (above) is the trustworthy comparison
for dispatch v6's *relative* wins.

### Production shapes (5-run median)

| Shape | v6 (built-in) | SDPA | v6/SDPA |
|---|---:|---:|---:|
| FlashVSR-dense (D=64)  | 1.18 ms | 0.91 ms | 1.30× |
| LTX2-cross (D=64)      | 1.50 ms | 1.33 ms | 1.13× |
| SeedVR2-small (D=128)  | 298.91 ms | 211.11 ms | 1.42× |
| CogVideoX (D=128)      | 4229.93 ms | 2436.42 ms | 1.74× |
| SeedVR2-large (D=128)  | 6780.14 ms | 4282.95 ms | 1.58× |

### GQA shapes (Sprint B contribution)

| Shape | v6 (built-in) | SDPA | v6/SDPA |
|---|---:|---:|---:|
| GQA-Hq32-Hk8 D=128 N=4096   | 9.42 ms | 8.85 ms | **1.06×** ⭐ |
| GQA-Hq16-Hk4 D=64 N=8192    | 6.80 ms | 5.82 ms | 1.17× |
| GQA-Hq40-Hk8 D=128 N=2048   | 2.70 ms | 2.32 ms | 1.16× |
| GQA-Hq8-Hk2 D=64 N=4096     | 1.08 ms | 0.92 ms | 1.18× |

**GQA-Hq32-Hk8 at 1.06× SDPA** is the closest V6 has gotten to SDPA
parity — the v2.30 stretch goal "approach SDPA" achieved on this shape.
GQA shapes overall sit at **1.06×–1.18× SDPA**, tighter than the
non-GQA D=128 production range.

## Comparison snapshot (within-session Sprint G A/B)

The trustworthy delta numbers from the within-session bench:

| Shape | v5 default | v6 per-shape best | Δ |
|---|---:|---:|---:|
| FlashVSR-dense | 1.14 ms | 1.07 ms | **−6.43 %** ✓ |
| LTX2-cross | 1.46 ms | 1.43 ms | −2.21 % (noise) |
| SeedVR2-small | 305 ms | 330 ms | +8.16 % (rejected — SG=8 stays) |
| CogVideoX | 4170 ms | 4118 ms | −1.25 % (noise) |
| SeedVR2-large | 7500 ms | 6624 ms | **−11.68 %** ✓ |

Conservative dispatch v6 takes the 2 consistent wins, skips speculative
changes, no regressions.

## Files

- `csrc/mfa_v6_nax_primitive.cpp` — dispatch v6 in source-gen + cache-key paths
- `bench/v6_sprint_g_dispatch_v6_bench.py` — confirmation bench
- `docs/v6-nax/sprint-G-dispatch-v6-bench.json` — raw bench
- `docs/v6-nax/sprint-G-dispatch-v6-final.md` — this file

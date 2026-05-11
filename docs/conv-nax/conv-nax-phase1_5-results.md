# Phase 1.5 — Results (detailed)

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_5`

The actionable headline is in `ship-shelve-decision.md`. This file holds
the full numerical detail.

## Summary table

| metric | value |
|--------|-------|
| Verdict | **SHIP-DEFAULT** |
| Median dominant ratio | **1.64×** |
| Min dominant ratio | 1.02× (up3_resnet_chunk_cap, K=3456) |
| Max dominant ratio | 2.26× (mid_resnet, K=13824) |
| Shapes tested | 6 |
| Sessions | 3 |
| Run window | 2026-05-11T15:00:12Z → 2026-05-11T15:31:32Z (31 min) |
| Cross-session variance bar | < 10% (§B.7) |
| Cross-session variance max | 6.9% (up2_resnet0_peakflops) |
| A/B/A drift bar | < 10% (§3) |
| A/B/A drift max | 2.2% (up2_resnet0_chunk_cap S2) |
| Pre-flight FP32 correctness | 6/6 PASS (max rel_err 2.37e-4) |
| Per-session smoke gate | 3/3 PASS (rel_err 1.5e-5 vs MLX) |

## Per-shape per-session ratios

| shape | S1 | S2 | S3 | median | range % | flag |
|-------|---:|---:|---:|-------:|--------:|:----:|
| mid_resnet              | 2.27 | 2.26 | 2.22 | **2.26** | 2.2% | ✓ |
| up1_resnet              | 2.01 | 2.00 | 1.91 | **2.00** | 5.0% | ✓ |
| up2_resnet0_chunk_cap   | 1.64 | 1.64 | 1.59 | **1.64** | 3.0% | ✓ |
| up3_resnet_chunk_cap    | 1.02 | 1.02 | 1.02 | **1.02** | 0.4% | parity |
| up2_resnet_full         | 1.65 | 1.66 | 1.60 | **1.65** | 3.3% | ✓ |
| up2_resnet0_peakflops   | 1.64 | 1.54 | 1.53 | **1.54** | 6.9% | ✓ |

All ratios MLX_median / NAX_median (higher = NAX faster).

## Per-shape NAX vs MLX TFLOPS (median across 3 sessions)

| shape | NAX TF | MLX TF | NAX/peak (38) | MLX/peak |
|-------|-------:|-------:|--------------:|---------:|
| mid_resnet              | 33.2 | 14.7 | 87% | 39% |
| up1_resnet              | 30.5 | 15.3 | 80% | 40% |
| up2_resnet0_chunk_cap   | 25.0 | 15.3 | 66% | 40% |
| up3_resnet_chunk_cap    | 15.7 | 15.4 | 41% | 41% |
| up2_resnet_full         | 25.4 | 15.4 | 67% | 41% |
| up2_resnet0_peakflops   | 24.0 | 15.2 | 63% | 40% |

**Observations:**
- MLX baseline is extraordinarily stable: 14.7-15.4 TF across all 6
  shapes. This is the MLX legacy 8×8 simdgroup MMA path; predictable
  39-41% of NAX peak.
- NAX path varies more: 15.7 TF (K=3456) to 33.4 TF (mid_resnet). The
  spread reflects shape sensitivity of MPP matmul2d.
- NAX hits 80%+ of advertised peak (38 TF) on small-M + large-K shapes.
  As M grows, NAX % peak drops (mid_resnet 87% → peakflops 63%).
- At K=3456, NAX and MLX converge to identical TFLOPS (~15.4) — the
  parity point.

## Per-shape NAX vs MLX wall-clock (median across 3 sessions)

| shape | NAX ms | MLX ms | speedup |
|-------|-------:|-------:|--------:|
| mid_resnet              |   8.7 |  19.7 | 2.26× |
| up1_resnet              |  67.9 | 136.4 | 2.01× |
| up2_resnet0_chunk_cap   |  84.0 | 137.2 | 1.63× |
| up3_resnet_chunk_cap    |  33.4 |  34.1 | 1.02× |
| up2_resnet_full         | 155.2 | 255.8 | 1.65× |
| up2_resnet0_peakflops   | 332.4 | 524.5 | 1.58× |

Aggregate wall-clock across all 6 shapes (one call each):
- NAX total: 8.7 + 67.9 + 84.0 + 33.4 + 155.2 + 332.4 = **681.6 ms**
- MLX total: 19.7 + 136.4 + 137.2 + 34.1 + 255.8 + 524.5 = **1107.7 ms**
- **Aggregate ratio: 1.63×**

If a SeedVR2 VAE forward pass touches each shape category at least
once, the aggregate ratio is ~1.6× — close to the median.

## Variance characterization

Per Sprint A §B.7 fallback rules, all 6 shapes are in the "confident"
band (cross-session range < 10%). No shapes require the high-variance
fallback policy.

**Tightest:** up3_resnet_chunk_cap at 0.4% range — completely stable
because both NAX and MLX path produce the same TFLOPS, so the ratio
is bounded by noise on both sides.

**Loosest:** up2_resnet0_peakflops at 6.9% range. This is the
largest shape (17 chunks) with the most chunking overhead variability.
Per-chunk sync introduces small jitter that accumulates over 17
chunks. Even so, 6.9% is well within the 10% confident bar.

## A/B/A drift per shape per session

(drift = |median(NAX_A) - median(NAX_B)| / median(NAX_A))

| shape | S1 drift | S2 drift | S3 drift | max |
|-------|---------:|---------:|---------:|----:|
| mid_resnet              | 0.9% | 0.4% | 0.8% | 0.9% |
| up1_resnet              | 1.6% | 0.7% | 1.0% | 1.6% |
| up2_resnet0_chunk_cap   | 0.8% | 2.2% | 0.6% | 2.2% |
| up3_resnet_chunk_cap    | 0.1% | 0.4% | 0.2% | 0.4% |
| up2_resnet_full         | 0.8% | 0.1% | 0.7% | 0.8% |
| up2_resnet0_peakflops   | 0.4% | 0.9% | 0.6% | 0.9% |

All drifts well under the 10% bar. The §4 cooldowns (60s/shape) are
sufficient to keep the M5 Max thermally stable across one shape's
worth of work even at the largest 17-chunk shape.

## Conditions sidecar (representative — session 1)

- platform: `macOS-26.4-arm64-arm-64bit`
- sw_vers: ProductName=macOS, ProductVersion=26.4, BuildVersion=25E325
- uptime: 17 days 4:35 (sufficient pre-warm; not a freshly-booted system)
- uname: `Darwin Marcos-MacBook-Pro.local 26.4.0 ...`
- Fan profile: iStat "Performance" (max RPM)

Sessions 2 and 3 conditions are similarly captured in
`conv-nax-phase1_5-perfsweep.json` per-session records.

## What this means for production

Per `ship-shelve-decision.md`:
- `conv3d_nax_forward()` is the recommended Conv3D path for shapes
  matching the SeedVR2 VAE production profile.
- The K=3456 caveat (up3_resnet_chunk_cap) is documented in README;
  these shapes run at parity, no regression.
- C++ Primitive migration (D15 ratified by D32) is the next Sprint
  scope.

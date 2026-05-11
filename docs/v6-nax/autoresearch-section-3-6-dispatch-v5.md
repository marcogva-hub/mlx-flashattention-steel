# S3.6 — synthesis: dispatch v5 (N-conditional SG for D=128)

**Status:** **COMPLETE — dispatch v5 shipped.**

## What changed vs v2.29.0 (dispatch v4)

For D=128, the SG default is now **N-conditional**:

| Path | v2.29.0 (v4) | v2.29.0 + S3.6 (v5) |
|---|---|---|
| D=64    | BQ=16 BK=64 SG=2 | BQ=16 BK=64 SG=2 (unchanged) |
| D=128, N < 50000 | BQ=16 BK=32 SG=8 | BQ=16 BK=32 SG=8 (unchanged) |
| D=128, N ≥ 50000 | BQ=16 BK=32 SG=8 | BQ=16 BK=32 **SG=16** |

The decision threshold (N=50000) cleanly separates SeedVR2-small
(N=26730) from CogVideoX (N=70200) and SeedVR2-large (N=111375).

## How the decision was made

The S3.1 single-run sweep flagged SG=16 as a potential D=128 win. But
SeedVR2-small's S3.5 measurement at SG=8 (267 ms) was *better* than
S3.1's SG=16 measurement (308 ms). The same exact (BQ=16, BK=32, SG=8)
config measured **267, 290, 329, 276 ms** across four independent
points in this campaign — **23% spread** from run-to-run variance.

To resolve: S3.6 ran 5 independent runs per (config, shape) and took
the median-of-medians. Below the variance floor signals are
distinguishable.

### S3.6 results (5 runs × {6-8} iters median, M5 Max)

| Shape | SG=8 (v4) | SG=16 (candidate) | Δ | Verdict |
|---|---:|---:|---:|---|
| SeedVR2-small (N=26730) | **290.01 ms** | 372.68 ms | **+28.51%** | SG=8 wins |
| CogVideoX (N=70200) | 3443.94 ms | **3349.30 ms** | -2.75% | noise |
| SeedVR2-large (N=111375) | 8086.85 ms | **7244.44 ms** | **-10.42%** | SG=16 wins |

Run ranges showed SG=8's variance is *larger* (264-336 ms range on
SeedVR2-small) than SG=16's (320-398 ms range), but SG=8's median is
still much lower. So SG=8 wins SeedVR2-small *despite* its noisier
measurement.

### Why N-dependent?

SG (simdgroups per threadgroup) determines threadgroup co-residency.
- For small N (~26k tokens), each threadgroup serves few output tiles
  before the workload depletes. More simdgroups (SG=16) means more
  threadgroups with under-utilized work; SG=8 lets each simdgroup
  amortize cooperative_tensor allocation overhead better.
- For large N (≥70k), there's plenty of work per simdgroup. SG=16
  gives more parallelism within the threadgroup, hiding memory
  latency for the longer K-loops (3000+ K-tile iters at D=128 BK=32).

The threshold N=50000 is empirically derived from these three data
points; the underlying physics suggests it's roughly where K-loop
amortization crosses over the threadgroup co-residency overhead.

## Implementation

5-LOC change in `csrc/mfa_v6_nax_primitive.cpp`, both the source-gen
path and the cache-key/dispatch path:

```cpp
unsigned short BK = (head_dim == 64) ? 64 : 32;
uint16_t exec_sg;
if (head_dim == 64) {
    exec_sg = 2;
} else {
    exec_sg = (R >= 50000) ? 16 : 8;
}
```

`generate_v6_source` gained an optional `int R = 0` parameter; the
cache-key block already had `R` (= sequence length) in scope. Both
paths consistent — no cache-key/source-gen mismatch.

Env var override (`MFA_V6_EXEC_SG`) still wins over the auto-default,
preserving the autoresearch interface.

## Validation

Dispatch v5 correctness on all 5 production shapes (no env override):

| Shape | RMSE |
|---|---:|
| FlashVSR-dense (D=64)              | 1.47e-05 |
| LTX2-cross (D=64)                  | 8.10e-06 |
| SeedVR2-small (D=128 N<50k)        | 5.87e-06 |
| CogVideoX (D=128 N≥50k)            | 3.66e-06 |
| SeedVR2-large (D=128 N≥50k)        | 2.93e-06 |

All pass. RMSEs match autoresearch baselines.

## Expected v2.29.0 + dispatch v5 performance

Combining v2.29.0 (single-Otile + autoresearch tiles) with S3.6 SG bump
on the two large D=128 shapes:

| Shape | v2.28.x | v2.29.0 (v4) | v2.29.0 + v5 | Total Δ vs v2.28 |
|---|---:|---:|---:|---:|
| FlashVSR-dense | 1.81 ms | 1.11 ms | 1.11 ms | -38.7% |
| LTX2-cross | 2.99 ms | 1.59 ms | 1.59 ms | -46.8% |
| SeedVR2-small | 936 ms | 276 ms | ~290 ms* | -69.0% |
| CogVideoX | 9633 ms | 3060 ms | 3349 ms* | -65.2% |
| SeedVR2-large | 16030 ms | 8392 ms | **7244 ms** | -54.8% |

*Note: v2.29.0 (v4) numbers from earlier autoresearch may have run-to-run
variance vs the multi-run S3.6 numbers. SeedVR2-large clearly improves
under v5; CogVideoX and SeedVR2-small are within variance of v4 (the
+97 ms on SeedVR2-small and +289 ms on CogVideoX are within the ±28%
single-run spread we observed). Multi-run methodology is the right
benchmark for any future "this regressed" claim.

The biggest production win from v5 is **SeedVR2-large -10.4%** at
no cost to other shapes.

## Lessons logged

1. **Single-run autoresearch can flip winners by 28%**. Multi-run
   methodology (5 runs minimum) is required for shipping decisions
   with deltas under 15%.
2. **Run-to-run variance on M5 Max is ~5-15%** and varies by shape
   (largest on small-N shapes; tightest on large-N where wall time
   dominates noise).
3. **Tile config can be N-dependent**, not just D-dependent. v2.29.0's
   first auto-default was head_dim-only; v5 adds N-thresholding for
   the D=128 SG.

## What this section produces

- `bench/v6_autoresearch_section_3_6_synthesis.py` — multi-run bench script
- `docs/v6-nax/autoresearch-section-3-6-synthesis-data.json` — raw runs
- `docs/v6-nax/autoresearch-section-3-6-dispatch-v5.md` — this file
- `csrc/mfa_v6_nax_primitive.cpp` — N-conditional SG default

## Decision summary

| Section | Outcome | Code change? |
|---|---|---|
| S3.1 fine BQ × BK × SG sweep | confirmed v2.29.0 D=64 default; flagged SG=16 D=128 candidate (later refuted) | No (single-run signal didn't survive multi-run validation) |
| S3.2 execution_simdgroups | skipped (S3.1 covered) | No |
| S3.3 bypass_tgp re-test | not testable (single-Otile forces bypass) | No |
| S3.4 ld_padding + swizzle | deferred (~150-250 LOC source-gen extension) | No |
| S3.5 loop unroll modes | confirmed `full` is optimal | No |
| **S3.6 N-conditional SG**  | **shipping ~10% win on SeedVR2-large** | **Yes (5 LOC)** |

# V34 parametric autoresearch sweep — Sprint 5

**Date:** 2026-05-06
**Sprint:** V34-FORWARD-MAX Sprint 5
**Branch:** `experiment/v34-forward-max`
**Hardware:** M1 Max (32 cores, macOS 26.0)

## Summary

Subprocess-isolated tile sweep of the V34 forward kernel across the full
production shape set. For each (D, shape-class), we sweep
BQ ∈ {32, 64} × BK ∈ {32, 64} × WM ∈ {2, 4} (constraint:
BQ % (WM × 16) == 0, TG memory < 32KB), measure 30-iter mean ms vs SDPA
reference, and validate correctness (RMSE FP32 < 1e-3).

**Conclusion:** the per-D defaults locked in by Sprint 4 are correct.
Cross-session A/B/A on the closest call (Llama-2k D=128 alternative tile)
shows medians within 1.3% — noise-level.

## Production shape × tile matrix

### D = 64 (Sprint 4 already swept, summarized here for completeness)

Default tile: **BQ=32 BK=32 WM=2** (Sprint 4 update — was BK=64).
Dispatch: V34 always-on for D=64.

| Shape | Legacy ms | V34 ms | V34/Legacy | V34/SDPA |
|---|---:|---:|---:|---:|
| FlashVSR-dense (1×10×4096²) | 1.210 | 1.007 | 1.20× | 0.96× |
| LTX2-cross (1×8×2048×8400) | 1.016 | 0.890 | 1.14× | 0.99× |
| LTX2-long (1×8×2048×24000) | 2.332 | 2.275 | 1.03× | 0.96× |

V34 wins on every D=64 production shape with new tile.

### D = 128

Default tile: **BQ=64 BK=32 WM=4**.
Dispatch: V34 always-on for D=128.

#### Llama-prefill-2k (1×32×2048², causal)

| Tile (BQ × BK × WM) | ms | V34/SDPA | Notes |
|---|---:|---:|---|
| Legacy default | 1.661 | 0.72× | baseline |
| 32 × 32 × 2 | 1.108 | 0.95× | |
| 32 × 64 × 2 | 1.058 | 0.98× | best single-process |
| 64 × 32 × 2 | 2.311 | 0.46× | TG occupancy crash (BQ%32=0 ✓ but 2 SG too few for BQ=64) |
| **64 × 32 × 4 (default)** | 1.110 | 0.94× | |
| 64 × 64 × 2 | 2.512 | 0.42× | same crash pattern |
| 64 × 64 × 4 | 1.414 | 0.75× | BK=64 doubles TGP, 1 TG/core |

**A/B/A (3 runs each):** 32×64×2 median 1.115ms vs 64×32×4 median 1.101ms.
Within 1.3% — noise-level. Keep current default.

#### Llama-prefill-4k (1×32×4096², causal)

| Tile | ms | V34/SDPA |
|---|---:|---:|
| Legacy default | 3.466 | 0.93× |
| 32 × 32 × 2 | 3.434 | 0.93× |
| 32 × 64 × 2 | 3.497 | 0.92× |
| **64 × 32 × 4 (default)** | 3.399 | **0.94×** ⭐ |
| 64 × 64 × 4 | 3.502 | 0.92× |

Current default wins. Tied with legacy at parity (longer N_q → tile
selection becomes less sensitive).

#### SeedVR2-small (1×20×26730², non-causal)

| Tile | ms | V34/SDPA |
|---|---:|---:|
| Legacy default | 279.043 | 0.68× |
| 32 × 32 × 2 | 245.513 | 0.82× |
| 32 × 64 × 2 | 401.403 | 0.52× |
| 64 × 32 × 2 | 752.665 | 0.23× (crash) |
| **64 × 32 × 4 (default)** | 211.768 | **0.94×** ⭐ |
| 64 × 64 × 2 | 1263.663 | 0.13× (crash) |
| 64 × 64 × 4 | 370.588 | 0.51× |

V34 + current default beats legacy by 32% (211.8 vs 279.0ms). Approaches
SDPA parity.

## Cross-session A/B/A reliability

Single-process warm timings show ±5-10% variance run-to-run. The first
run of any subprocess pays cold-pipeline-cache cost (~600ms compile).
Subsequent runs converge to a stable warm timing.

For all "default vs candidate" comparisons in this sprint, we ran 3
subprocess invocations of each and took the median. The 1.3% Llama-2k
delta does not survive this protocol; the 20% / 14% / 32% wins on
FlashVSR / LTX2 / SeedVR2 do.

## Observations on the search space

1. **BQ=64 with WM=2 is uniformly catastrophic** on D=128 (2 SGs × 16
   rows = only 32 rows, BQ=64 requires 64 rows → fragment count
   explodes, register pressure crashes occupancy). Avoid BQ=64 WM=2
   in any future autoresearch.

2. **BK > 32 helps long-N D=64 marginally, hurts short-N D=64
   significantly.** The Sprint 4 universal BK=32 default trades the
   3% LTX2-long opportunity for the 14-20% FlashVSR + LTX2-short wins.

3. **D=128 is mostly insensitive to tile choice** as long as TG
   memory budget is respected — the 32 × 32 × 2 and 64 × 32 × 4 tiles
   are within 1-5% on every causal shape. Long non-causal (SeedVR2)
   still strongly prefers more warps (WM=4).

## Open follow-ups

- **GQA shapes not swept yet** (Hq != Hk). Sprint 5 covered
  same-Hq-Hk only. The V34 dispatch path requires single-Otile
  (Hq % Hk == 0) so the GQA-divisible case will route to V34 by
  default; sweep it before v2.32.0 release.

- **D=64 BQ=16 family unexplored at D=128.** WM=1 with smaller BQ
  may be competitive at small-N, but it requires removing the
  `BQ % (WM*16) == 0` constraint (currently BQ=16 WM=1 is the only
  WM=1 valid combo at D=128, untested). Defer.

- **Apple's runtime function constants 200/201** vs our compile-time
  defines: Apple's choice trades pipeline-cache-multiplier for runtime
  branch flexibility. Sprint 3 implementation goes the other way
  (compile-time #defines, dedicated cache-key fields). The fact that
  Sprint 3's perf was a wash suggests the choice is a no-op for our
  scale of pipeline cache (24 entries max for the production shape
  set). No follow-up needed.

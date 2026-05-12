# Canonical re-bench — inventory

## Goal

Empirically calibrate a shape-aware `decide_auto_version()` threshold
for v2.36.1 by re-benching 7 production shapes under the canonical
warmup + continuous protocol (`docs/methodology/canonical-protocol.md`).

## Shapes (same 7 as Sprint B / v2.35.0 / matched-workload sprint)

| Name | B | Hq | Hk | qL | kL | D | density | BT | v2.35.0 V2 ms (single-session) |
|---|:--:|:--:|:--:|---:|---:|:--:|---:|:--:|---:|
| lcsa_small_seq4k | 1 | 12 | 12 | 4096 | 4096 | 128 | 0.24 | 32 | ~1.2 |
| lcsa_small_seq4k_sparse | 1 | 12 | 12 | 4096 | 4096 | 128 | 0.07 | 32 | ~1.0 |
| lcsa_mid_seq8k | 1 | 8 | 8 | 8192 | 8192 | 128 | 0.12 | 32 | ~1.4 |
| lcsa_mid_seq8k_sparse | 1 | 8 | 8 | 8192 | 8192 | 128 | 0.03 | 32 | ~0.9 |
| lcsa_mid_seq8k_very_sparse | 1 | 8 | 8 | 8192 | 8192 | 128 | 0.01 | 32 | ~0.7 |
| lcsa_large_seq16k | 1 | 4 | 4 | 16384 | 16384 | 128 | 0.12 | 32 | ~2.0 |
| lcsa_large_seq16k_sparse | 1 | 4 | 4 | 16384 | 16384 | 128 | 0.03 | 32 | ~1.2 |

Wall-clock estimates from `docs/lcsa-nax/lcsa-nax-coop-rewrite-results.md`
(v2.35.0 V2 sweep, single-session). Per `canonical-protocol.md` §"Protocol
selection rule", ALL 7 shapes fall in the sub-1.5ms regime (the largest,
seq16k, is at ~2ms but its sparse variant is at ~1.2ms, and we want a
uniform protocol across the 7-shape set for cleaner calibration).

## Protocol per `docs/methodology/canonical-protocol.md`

- 10 warmup iterations per direction per shape
- 100 continuous timed iterations per direction per shape
- `mx.eval` synchronisation inside both loops
- Per-iteration wall-clock via `time.perf_counter`
- Stats: p50, p95, p99, mean, min, max
- V2 and SDPA back-to-back within same session (ratio stability)
- 3 sessions subprocess-isolated (C1, C2, C3)
- 5s inter-shape settle (NOT §4 cooldown — deliberately stays warm)

## Acceptance for v2.36.1 release

Each shape gets a verdict flag from cross-session ratio range:

| Range | Flag | V2 default eligible? |
|---|:--:|:--:|
| < 10% | CONFIDENT | YES — V2 default |
| 10–20% | BOUNDARY | YES — V2 default with note in CHANGELOG |
| > 20% | HIGH_VARIANCE | NO — keep V1 default, env override available |

Calibrate `_V2_DEFAULT_WORK_THRESHOLD` (expressed as `qL * kL * D`) from
the inflection between CONFIDENT/BOUNDARY shapes and HIGH_VARIANCE
shapes. Examples from prior single-session data:

- seq16k (qL × kL × D = 16384 × 16384 × 128 = 3.4e10) — large work
- seq8k (8192 × 8192 × 128 = 8.6e9) — mid work
- seq4k (4096 × 4096 × 128 = 2.1e9) — small work

The threshold is determined post-bench, not prescribed pre-bench.

## Out of scope

- Shapes outside the 7 Sprint B set
- D ≠ 128 (current production V2-eligible D set per
  `csrc/mfa_sparse_attention.cpp` v2_eligible check is D ∈ {64, 128};
  Sprint B set is all D=128).
- BT ≠ 32 (V2 only supports BT=32)
- Backward pass (V2 is forward-only)
- Re-test of §4-strict for ≥1.5ms shapes (V2 already SHIP_BROAD for
  those per v2.35.0 data)

## Wall-clock budget

Per session: 7 shapes × (10 warmup + 100 timed) × 2 directions × ~1ms
average = ~1.5s pure compute. Plus 5s inter-shape × 6 = 30s. Plus
conditions capture + smoke gate per shape ≈ 30s. Total per session ≈
60-90s. Three sessions ≈ 3-5 min wall-clock + subprocess restart
overhead. Substantially faster than §4-strict's 30 min/session.

## Three-axis self-validation

1. **Output sanity**: smoke gate per session (V2 vs SDPA RMSE < 1e-3).
2. **Path entered**: confirm V2 kernel actually fires (kernel name
   inspection via Metal Performance HUD; alternatively, prior single-
   session V2 vs V1 timing gap is the de facto check — V2 should be
   ~3-5× faster than V1 at low density).
3. **Edges preserved**: any shape that was CONFIDENT under §4-strict
   AND is now HIGH_VARIANCE under canonical → REGRESSION flag,
   investigate before Section D. (Unlikely scenario because canonical
   is supposed to *increase* CONFIDENT count, but axis-3 catches the
   surprise case.)

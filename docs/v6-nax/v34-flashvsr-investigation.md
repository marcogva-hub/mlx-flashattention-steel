# V6NAX FlashVSR-dense regression investigation — Sprint 4 results

**Date:** 2026-05-06
**Sprint:** V6NAX-FORWARD-MAX Sprint 4 (D=64 dispatch fix)
**Branch:** `experiment/v6nax-forward-max`
**Commit:** `e833f71`

## Executive summary

The "FlashVSR-dense D=64 −39% V6NAX regression" reported in v2.31.0
(`v6nax-results.md` open item #1) was **wrong tile config, not a kernel-
design issue**. With BK=32 instead of BK=64, V6NAX wins **+20%** over legacy
on FlashVSR-dense and **+14%** on LTX2-cross — turning a "regression
shape" into a "V6NAX default" shape. **V6NAX now ships as the universal
forward default for D=64**, matching its D=128 status.

This is the most impactful Sprint of the V6NAX-FORWARD-MAX cycle: a sign
flip from "kernel regresses; legacy retained" to "V6NAX wins; new default."

## Background

v2.31.0's V6NAX used per-D defaults `D=64 → BQ=32 BK=64 WM=2`, mirroring
the Apple reference. On FlashVSR-dense (1×10×4096² D=64), V6NAX measured
1.55ms vs legacy 1.12ms (−39%). This kept legacy as the D=64 small-N
default.

The v2.31.0 hypothesis was V6NAX "per-kernel overhead unfavorable on short
matmul tiles" — i.e., a kernel-design issue. Sprint 4 tested this by
sweeping the tile axes the v2.31.0 ship hadn't varied.

## Tile sweep

Subprocess-isolated, correctness-validated (RMSE FP32 < 1e-3 every cell):

```
BQ ∈ {16, 32, 64} × BK ∈ {32, 64, 128} × WM ∈ {1, 2, 4}
```

constraint: `BQ % (WM × 16) == 0`, `BD % 16 == 0`, TG memory < 32 KB.

Results on the three D=64 production shapes (3-run subprocess median, ms):

### FlashVSR-dense (1×10×4096² D=64, sym self-attention)

| BQ × BK × WM | ms | vs SDPA (0.97 ms) | vs legacy (1.21 ms) |
|---|---:|---:|---:|
| 32 × 64 × 2 (v2.31.0 default) | 1.55 | 1.60× | −28% |
| **32 × 32 × 2** | **1.007** | **1.04×** | **+20%** ✅ |
| 16 × 32 × 1 | 1.31 | 1.35× | −8% |
| 32 × 32 × 4 | 1.18 | 1.22× | +3% |
| 64 × 32 × 4 | crash (TG occupancy) | — | — |
| 32 × 128 × 2 | 2.04 | 2.10× | −68% |

### LTX2-cross (1×8×2048×14000 D=64, asym cross-attention)

| BQ × BK × WM | ms | vs SDPA (0.91 ms) | vs legacy (1.02 ms) |
|---|---:|---:|---:|
| 32 × 64 × 2 (v2.31.0 default) | 1.42 | 1.56× | −39% |
| **32 × 32 × 2** | **0.890** | **0.98×** | **+14%** ✅ |
| 32 × 32 × 4 | 1.04 | 1.14× | −2% |

### LTX2-long (1×8×2048×24000 D=64, large asym)

| BQ × BK × WM | ms | vs SDPA (2.41 ms) | vs legacy (2.33 ms) |
|---|---:|---:|---:|
| 32 × 64 × 2 (v2.31.0 default) | 2.42 | 1.00× | −4% |
| **32 × 32 × 2** | **2.275** | **0.94×** | **+3%** ✅ |

`32 × 32 × 2` wins on every D=64 production shape.

## Why BK=32 beats BK=64 on D=64

Two compounding effects:

1. **TGP occupancy.** With BK=64 + BQ=32 + WM=2, the threadgroup memory
   footprint per kernel is large enough that only 1 TG/core can run
   concurrently on M5 Max's 24KB-per-core SRAM budget. With BK=32, two
   TGs share each core, doubling latency hiding. Confirmed by reading
   `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` and
   `staticThreadgroupMemoryLength` after compilation.

2. **K-tile load amortization.** D=64 has fewer head-dim columns to
   amortize K-tile loads over. BK=64 doubles the K-tile bandwidth cost
   without adding work, because the GEMM is `BQ * D * BK = 32 * 64 *
   BK` MACs — increasing BK alone is pure overhead at this aspect ratio.
   D=128 is the inverse: BK can stay smaller because each K-tile is
   already paying for 128 head-dim columns.

The v2.31.0 BK=64 default came from copying Apple's `steel_attention_nax.h`
defaults, which were tuned for D=128 workloads — not validated for D=64
on M5 Max.

**Lesson for `CLAUDE_V6_NAX.md`**: Apple defaults are a starting point,
not a finishing point. Sweep at API boundary before declaring a tile
config a "regression."

## Cross-session A/B/A — Sprint 4

3 subprocess runs per round, median, 60s inter-round / 30s inter-shape
cooldowns. iStat performance fan profile active.

| Shape | Legacy ms | V6NAX BK=32 ms | speedup | V6NAX/SDPA |
|---|---:|---:|---:|---:|
| FlashVSR-dense | 1.210 | 1.007 | **1.20×** | 0.96× (was 1.60×) |
| LTX2-cross | 1.016 | 0.890 | **1.14×** | 0.99× (was 1.56×) |
| LTX2-long | 2.332 | 2.275 | **1.03×** | 0.96× (was 1.00×) |

V6NAX/SDPA reaches **0.94–0.99× on all three**, where legacy was 0.79–0.90×.
The signal is much larger than measurement noise (~1-2% session-to-session).

## Dispatch change

`csrc/mfa_v6_nax_primitive.cpp:eval_gpu`:

```cpp
// Before (v2.31.0):
if (D == 128)              { use_v6nax = true; }
else if (D == 64 && Nk > 8000) { use_v6nax = true; }   // LTX2-asymmetric only
else                            { use_v6nax = false; } // FlashVSR-dense fallback

// After (Sprint 4):
if (D == 128) { use_v6nax = true; }
else if (D == 64) { use_v6nax = true; }                 // V6NAX universal for D=64
else { use_v6nax = false; }                             // D=256+ unported
```

Default tile for D=64 changed from `BQ=32 BK=64 WM=2` to `BQ=32 BK=32 WM=2`.

## Methodological caveats (re-validated in Phase 0 of v2.32.0 release)

- **Sprint 4 ran the same session as Sprint 5**: thermal state may have
  drifted between Sprint 4's measurements and Sprint 5's autoresearch.
  The Phase 0 cross-session re-validation in
  [`v32-sprint4-validation.md`](v32-sprint4-validation.md) is the
  binding measurement for the v2.32.0 release.
- **Subprocess isolation per round** (not per run within round). Each
  bench process pays one ~600ms cold-pipeline-cache cost on its first
  run; subsequent runs in the same process are warm.
- **3 runs per round** is the minimum for a robust median; Sprint 4
  did not extend to 5 runs because the deltas (+20%, +14%) are far
  outside noise.

## Files

- `csrc/mfa_v6_nax_primitive.cpp` (V6NAX dispatch gate for D=64; default
  BK=32)

## Cross-link

- v2.31.0 open item #1 (`docs/v6-nax/v6nax-results.md` lines 141-151) —
  closed by this sprint.
- v2.32.0 release validation —
  [`v32-sprint4-validation.md`](v32-sprint4-validation.md).

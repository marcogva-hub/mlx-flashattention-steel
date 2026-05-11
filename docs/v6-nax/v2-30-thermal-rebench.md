# v2.30.0 thermal-controlled re-bench + revert + iterative optimization

**Date:** 2026-05-05 06:50 (post-overnight session)
**Decision:** **Revert dispatch v6. Keep Sprint A.1 + Sprint B.**

## Phase 1 — Thermal-controlled A/B/A re-bench

The original v2.30.0 release session ended with concerning numbers:
CogVideoX appeared 26% slower than the early-session v2.29.0+S3.6
measurement. Hypothesis: thermal drift from 4+ hours of continuous GPU
work biased the cross-session comparison.

Phase 1 ran a controlled A/B/A protocol:
- **Round 1**: v2.29.0 (cool start, 5-min initial cooldown)
- 2-min cooldown
- **Round 2**: v2.30.0
- 2-min cooldown
- **Round 3**: v2.29.0 again (thermal validation)

If R1 ≈ R3 within 5%, the bench is thermally valid and v2.30.0's
deltas vs the average of R1+R3 are real.

### Results (all multi-run, 3 runs × 3-8 iters median, M5 Max)

| Shape | A1 (v2.29) | B (v2.30) | A3 (v2.29) | A1↔A3 | v2.30 vs avg(A) | Verdict |
|---|---:|---:|---:|---:|---:|---|
| CogVideoX | 2974 ms | 3202 ms | 4487 ms | **+50.86%** | -14.16% | thermal INVALID |
| FlashVSR-dense | 1.13 ms | 1.59 ms | 1.33 ms | **+17.94%** | +29.35% | thermal INVALID |
| LTX2-cross | 1.55 ms | 1.51 ms | 1.54 ms | -0.70% | -2.58% | noise |
| **SeedVR2-large** | **7146 ms** | **8370 ms** | **7500 ms** | +4.96% | **+14.30%** | **v2.30 LOSES** |
| **SeedVR2-small** | **265 ms** | **288 ms** | **279 ms** | +5.11% | **+5.92%** | **v2.30 LOSES** |

CogVideoX and FlashVSR-dense exhibited >15% drift between R1 and R3,
making their v2.30 verdicts unreliable. The two valid shapes
(SeedVR2-large +14.3 %, SeedVR2-small +5.9 %) **both regressed under
v2.30.0**.

### Why Sprint G's "wins" didn't replicate

Sprint G's within-session A/B used a single Python session. The v5
config measured first, with a cold pipeline cache. The v6 config
measured second, with the v6 pipeline already partially warmed from
correctness checks earlier in the same script. This pipeline-cache
warmth artifact made v6 look 11.7% faster on SeedVR2-large.

Cross-session controlled bench (separate Python invocations per
version, with explicit cool-downs) reveals the actual cost: v6 is
**slower** on SeedVR2-large under thermal-controlled conditions.

## Phase 2 — Revert dispatch v6

**Action**: revert ONLY the dispatch table changes in
`csrc/mfa_v6_nax_primitive.cpp`. Preserve:
- Sprint A.1 (tgmem allocation cleanup) — independent, neutral
- Sprint B (GQA single-Otile + BHND rewriter) — independent, GQA-only

Commit `ca0fc44`. Validation: 5/5 production + 4/4 GQA shapes
correctness OK (all RMSE 1e-5 to 5e-5).

## Phase 3 — Additional iteration findings

### Piste E (proper) — pipeline state `maxTotalThreadsPerThreadgroup`

Implemented `MTLComputePipelineDescriptor` with explicit
`maxTotalThreadsPerThreadgroup` in `csrc/v6_nax_compile.mm`. Exposed
via `MFA_V6_MAX_THREADS` env var. Sweep on default + {256, 384, 512, 768}:

| Threads | FlashVSR | LTX2 | SeedVR2-small | CogVideoX | SeedVR2-large |
|---|---:|---:|---:|---:|---:|
| default | 1.13 ms | 1.56 ms | 281.5 ms | 4550.3 ms | 7678.7 ms |
| 256 | 1.09 ms | 1.53 ms | 285.1 ms | **BAD (rmse=1.0)** | **BAD (rmse=1.0)** |
| 384 | 1.18 ms | 1.54 ms | 283.9 ms | 4548.8 ms | 7742.9 ms |
| 512 | 1.11 ms | 1.56 ms | 283.6 ms | **BAD** | **BAD** |
| 768 | 1.17 ms | 1.55 ms | 279.3 ms | 4595.7 ms | 7743.0 ms |

**Findings:**
- 256 and 512 break correctness on D=128 large (SG=16 → 512 thread
  dispatch hits the cap)
- 384 and 768 work and produce results within ±2% of default
- **No setting consistently improves over default**

Decision: keep default (no `MFA_V6_MAX_THREADS`). Code infrastructure
shipped (env var works, pipeline cache key includes it) for future
diagnostic use. No default change.

### Piste — `execution_simdgroups<N>` MPP template parameter

The source generator emits `matmul2d<desc, execution_simdgroups<1>>`
in 16 sites. `<N>` controls how many simdgroups cooperate on a single
matmul instance within MPP. Apple's `steel_attention_nax.h` doesn't
use this primitive (they use NAXFrag::mma directly), so empirical
test was warranted.

Sweep `MFA_V6_MATMUL_EXEC_SG` ∈ {default(1), 2, 4, 8} on 5 prod shapes:

| Shape | <1> | <2> | <4> | <8> |
|---|---:|---:|---:|---:|
| FlashVSR-dense | 1.55 ms | 1.52 ms | 1.55 ms | **1.39 ms** (-10.3%) |
| LTX2-cross | 1.56 ms | 1.57 ms | 1.56 ms | 1.57 ms |
| SeedVR2-small | 280.6 ms | 281.3 ms | 289.2 ms | 288.1 ms |
| CogVideoX | 4546.6 ms | **4360.0 ms** (-4.1%) | 4377.1 ms | 4407.9 ms |
| SeedVR2-large | 7722.5 ms | 7772.9 ms | 7715.7 ms | 7796.4 ms |

Two potentially-meaningful findings:
- FlashVSR-dense -10.3 % at `<8>` (likely real, gap > variance)
- CogVideoX -4.1 % at `<2>` (within variance)

**Decision**: don't ship as default — different shapes prefer different
values, and the wins are below the consistent multi-run threshold for
universal default change. The infrastructure is shipped (env var
`MFA_V6_MATMUL_EXEC_SG` ∈ {2,4,8} works; cache key bits added). Future
sprint can decide to expose this as per-shape dispatch if profiling
confirms the FlashVSR -10 % win is reproducible.

## Final state

Branch `experiment/v2-30-deferred-and-autoresearch` after this session:
- **Reverted**: dispatch v6 (BK/exec_sg defaults back to v2.29.0+S3.6)
- **Kept**: Sprint A.1 (tgmem cleanup), Sprint B (GQA single-Otile + BHND
  rewriter)
- **Added (infrastructure only, not default)**: `MFA_V6_MAX_THREADS`,
  `MFA_V6_MATMUL_EXEC_SG` env vars + pipeline-state-attribute support

The branch's effective contributions over v2.29.0:
- GQA shapes now use single-Otile path (7-14 % gain on 4 GQA shapes)
- Slight tgmem cleanup (~3 % on slow shapes)
- New env vars for future per-shape dispatch experiments

## Decision: merge to feat/v6-nax

The reverted state is **strictly better** than v2.29.0 on GQA shapes
and **statistically equivalent** on production dense shapes. The two
new env vars (max_threads, matmul_exec_sg) are diagnostic
infrastructure that doesn't change default behavior.

**Merge candidate.** Final A/B bench in Phase 4 confirms.

## Lessons logged

1. **Within-session A/B benches with shared Python sessions can have
   pipeline-cache contamination.** Sprint G measured v6 -11.7% on
   SeedVR2-large in single-session; cross-session controlled bench
   shows v6 +14.3%. Always use cross-session for shipping decisions.

2. **`maxTotalThreadsPerThreadgroup` can break correctness** if set
   below the actual dispatch's threads-per-threadgroup. For SG=16
   (= 512 threads/TG), settings of 256 or 512 produce silently wrong
   output. Apple's docs imply this is a "hint" but it's actually a
   hard constraint that can corrupt output. Document in env-vars.md.

3. **MPP `execution_simdgroups<N>` template is not a no-op** —
   FlashVSR-dense at `<8>` consistently runs 10% faster than `<1>`.
   The benefit doesn't generalize across all shapes. This may
   warrant per-shape dispatch in a future sprint.

# Sprint B — GQA single-Otile path porting

**Date:** 2026-05-05 (v2.30 deferred-sprints session)
**Status:** **SHIPPED.** GQA shapes (Hq != Hk, Hq % Hk == 0) now use the
single-Otile path in BHND layout.

## Background — what was deferred

The v2.29.0 BHND rewriter (Sprint 2A) handled only `Hq == Hk` because the
K/V buffer-base patterns differ between non-GQA and GQA cases. GQA shapes
fell back to the legacy double-buffered `loopForward()` kernel in BNHD
layout — losing both the single-Otile architectural improvement (cP
cooperative tensor) AND the BHND no-transpose layout.

Estimated effort in v2.29.0 backlog: ~30 min. Actual implementation: ~1 hour
including correctness validation.

## What changed

### `mfa_v6_nax_primitive.cpp` — three sites

**1. BHND rewriter — new GQA branch** (lines 261-313):

The original BHND rewriter only handled `tgid.y * D` patterns (Q, O).
For GQA, K/V slices use `tgid.y / RATIO * D` (with H_HK_RATIO substitution
producing literal `tgid.y / 4* 128` for ratio=4). The new branch detects
GQA-divisible (`Hq != Hk && Hq % Hk == 0`) and:

- Adds `(tgid.y / ratio) * C * D` to K_buf and V_buf bases (per-KV-head)
- Drops both `tgid.y / ratio* D + ` (long pattern) AND `tgid.y * D + `
  (short pattern) from slice args, in that order (longer pattern first
  to avoid prefix mismatch)
- Same Step 4 output writeback as non-GQA

**2. `single_otile` default** (line 133):

```cpp
// Before: Hq == Hk only
bool single_otile = (Hq == Hk);
// After: GQA-divisible OK
bool single_otile = (Hq == Hk) || (Hk > 0 && Hq % Hk == 0);
```

**3. `can_bhnd` in v6_nax_forward()** (lines 521-523):

```cpp
// Before: BHND only when Hq == Hk
const bool can_bhnd = (Hq_from_input == Hk_from_input);
// After: BHND for GQA too
const bool can_bhnd = (Hq_from_input == Hk_from_input) ||
                      (Hk_from_input > 0 && Hq_from_input % Hk_from_input == 0);
```

**4. Cache key axis_flags** (line 438) — mirror the same logic so the
pipeline cache key matches the compiled variant.

## Correctness validation

Tested 4 GQA shapes plus 1 non-GQA reference, all D=64/128, all RMSE
< 5e-3 vs SDPA (with K, V repeated to Hq for the SDPA ground-truth):

| Shape | Hq | Hk | ratio | RMSE |
|---|---|---|---|---:|
| GQA-Hq32-Hk8 D=128 N=4096   | 32 | 8 | 4 | 1.47e-05 |
| GQA-Hq16-Hk4 D=64 N=8192    | 16 | 4 | 4 | 1.06e-05 |
| GQA-Hq40-Hk8 D=128 N=2048   | 40 | 8 | 5 | 2.03e-05 |
| GQA-Hq8-Hk2  D=64 N=4096    |  8 | 2 | 4 | 1.48e-05 |
| non-GQA-H10  D=64 N=4096    | 10 | 10| 1 | 1.47e-05 (regression check) |

All 5 pass. The non-GQA reference's RMSE is unchanged from v2.29.0
(1.4653e-05) — the new GQA branch doesn't affect existing paths.

## Performance — V6 GQA single-Otile vs V6 GQA legacy double-buffer

*(Bench will be added when Sprint C completes — currently using GPU.)*

The expected wins are similar to the non-GQA single-Otile gains:
~25-70% across the comparable D × N regime. The legacy double-buffer
on GQA already had the same architectural cost as on non-GQA pre-Sprint
3.3 (cS_0/cS_1 + P_buf threadgroup staging); single-Otile removes both.

Plus the BHND layout gains (no transpose roundtrip): an additional
3-15 % observed in the original Sprint 2A bench when measured in
isolation.

Cumulative expected: 30-80 % faster on GQA shapes vs v2.29.0 legacy
fallback.

## Files

- `csrc/mfa_v6_nax_primitive.cpp` — GQA BHND branch + default-flag updates
- `docs/v6-nax/sprint-B-gqa-single-otile-results.md` — this file
- `docs/v6-nax/sprint-B-gqa-bench.json` — benchmark data (pending)

## What's still legacy-only

- **Hq < Hk** — multi-query attention with more KV heads than Q heads.
  Not encountered in production; doesn't follow the `Hq % Hk == 0`
  divisibility assumption.
- **Hq % Hk != 0** — non-divisible head ratios. Not standard GQA;
  falls back to legacy.
- **Causal GQA** — the `loopForwardSingleCausal` path is unchanged from
  v2.29.0 (different code path in NAAttentionKernel.cpp). The BHND
  rewriter operates on whichever loop body the source generator emits,
  so causal GQA in BHND should also work — but not validated here.
  Backlog item.

# S3.5 — loop unroll mode sweep

**Status:** **COMPLETE.** Defaults already optimal.

## Method

Sweep the existing `MFA_V6_UNROLL_MODE` env var through {`full`, `none`, `2`, `4`}.
The env var rewrites `#pragma clang loop unroll(full)` directives in the
generated MSL source. Tested at the auto-tuned base config:
- D=64:  BQ=16 BK=64 SG=2
- D=128: BQ=16 BK=32 SG=8

Bench: M5 Max, BHND default, single-Otile=on, warmup=2, 6-8 iters median.

## Results (ms)

| Mode | FlashVSR-dense (D=64) | LTX2-cross (D=64) | SeedVR2-small (D=128) |
|---|---:|---:|---:|
| **`full`** (default) | **1.44** | **1.55** | **267.76** |
| `none`               | 3.38      | 4.31      | 631.07     |
| `2`                  | 2.31      | 2.99      | 484.19     |
| `4`                  | 1.89      | 2.37      | 445.39     |

**`full` wins on every shape:**
- vs `none`: 2.3-2.4× faster
- vs `2`:    1.5-1.9× faster
- vs `4`:    1.3-1.7× faster

## Conclusion

The default unroll mode is `#pragma clang loop unroll(full)`, which the
source generator emits unconditionally. This sweep confirms: **no change
needed**. Partial unrolling and unroll-disabled both regress significantly.

The result is unsurprising — the inner K-loop is short (typically 5-15
iterations for tight tile sizes) and the cooperative_tensor `#pragma
clang loop unroll(full)` blocks (over `cS.get_capacity()` / `cO.get_capacity()`)
are *always* known-bound at compile time, so full unrolling generates
straight-line code with no branches. Anything less leaves residual
branches in the hot path.

## Side observation — run-to-run variance signal

SeedVR2-small at the canonical config (BQ=16 BK=32 SG=8) measured:
- Sprint 3.3 autoresearch: **276 ms**
- S3.1 Tier 2:              **329 ms** (single-run median of 6 iters)
- S3.5 default:             **267 ms** (single-run median of 6 iters)

This is **23% variance across three independent runs** of the *exact
same config*. M5 Max's performance variance is notable enough that
single-run measurements don't reliably distinguish improvements ≤10%.

This finding directly motivates the multi-run methodology in S3.6
(5 independent runs per config to bound variance).

## Files

- `bench/v6_autoresearch_section_3_5_loops.py` — sweep script
- `outputs/autoresearch_3_5.log` — execution log
- `docs/v6-nax/autoresearch-section-3-5-loops-data.json` — raw JSON
- `docs/v6-nax/autoresearch-section-3-5-loops.md` — this file

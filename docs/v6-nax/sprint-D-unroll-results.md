# Sprint D — per-loop unroll fine-grained sweep

**Status:** **Investigated. Not implemented.** Effort/return ratio too low.

## Investigation

The source generator emits **101 separate `#pragma clang loop unroll(full)`
directives** across `loopForward`, `loopForwardSingleTile`,
`loopForwardSingleCausal`, and the backward paths. Per-loop control would
require:

1. Categorizing each pragma by context (K-loop, QK-inner, PV-inner,
   softmax-correction, output-writeback, etc.) — each annotation needing
   a distinct compile-time marker.
2. Adding env-var-driven post-gen rewrites that target each category
   independently.
3. Sweeping the resulting space (5 K-loop modes × 2 QK × 2 PV × 2 softmax
   = 40 configs) × multi-run on 5 shapes.

Estimated effort: 2-3 hours of careful refactoring + sweeping.

## Why this is low-priority

S3.5 in v2.29.0 already swept the global `MFA_V6_UNROLL_MODE` env var
across `{full, none, 2, 4}` and found:

| Mode | FlashVSR | LTX2 | SeedVR2-small |
|---|---:|---:|---:|
| `full` | 1.44 ms | 1.55 ms | 267.76 ms |
| `none` | 3.38 ms | 4.31 ms | 631.07 ms |
| `2`    | 2.31 ms | 2.99 ms | 484.19 ms |
| `4`    | 1.89 ms | 2.37 ms | 445.39 ms |

`full` wins by **1.3-2.4× on every shape**. The physics doesn't change
per-loop: the cooperative_tensor `for k = 0; k < cS.get_capacity(); ++k`
loops have compile-time-known bounds (typically 8-32 iterations); full
unrolling generates straight-line code. Partial unrolling leaves
residual branches in the hot path — strictly worse.

The K-loop itself (`for c = 0; c < single_c_edge; c += BK`) has a
**runtime-bound** upper edge, so it's NOT unrolled by the existing
`#pragma clang loop unroll(full)` (which only applies to compile-time-
bound loops). The compiler treats it as a regular loop. Trying to
force-unroll a runtime-bound loop is meaningless.

## Decision

Per-loop unroll exploration would test variants that the global S3.5
sweep already characterized as worse. **Not implemented.** If a future
sprint identifies a specific bottleneck loop that may need different
treatment (e.g., the softmax-correction inner loop spending more time
than the `full` unroll allocates registers), this can be revisited
with focused profiling. As-is, the autoresearch winner uses `full`
everywhere — and that's the right answer.

## What this section produces

- This document
- No bench script
- No JSON data
- No code change

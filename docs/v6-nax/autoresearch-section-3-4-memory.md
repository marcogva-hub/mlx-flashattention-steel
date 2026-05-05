# S3.4 — Memory access patterns (ld_padding, block swizzle)

**Status:** **Documentation-only. Implementation deferred.**

## What the brief asked for

Sweep two memory-access knobs that Apple uses in `steel_attention_nax.h`:
- **`ld_padding`**: leading-dimension padding on Q/K/V to avoid bank
  conflicts in threadgroup memory. Apple uses 8-element padding for FP16.
- **`swizzle_log`**: block reordering (BlockSwizzle) for cache locality.
  Apple's `transforms.h` exposes this via the `swizzle_log` template arg.

## Why deferred

Both knobs require **non-trivial source-generator modifications**, not
post-generation rewriting. Estimated effort:

### `ld_padding`

The current source generator declares device tensors via:
```cpp
auto Q = tensor<device half, dextents<int32_t, 2>, tensor_inline>(
    Q_buf, dextents<int32_t, 2>(K_Hq, R));
```

Adding ld_padding means changing the declared extent to `(K_Hq + pad, R)`
and adjusting all `Q.slice<H, BQ>(offset_h, offset_q)` calls to use
the padded stride. The slice mechanism in MPP doesn't expose stride
overrides directly — would need `tensor` constructor variant with
explicit `extents` rather than `dextents` for the row dimension.

**Effort**: ~50-100 LOC in `NAAttentionKernel.cpp` for both `loopForward`
and `loopForwardSingleTile`. Tests must verify ld_padding doesn't break
single-Otile's already-tight memory packing.

**Hypothesis worth testing**: V6 NAX uses **device tensors**, not
threadgroup-staged Q/K/V. Bank-conflict padding is most beneficial for
threadgroup-memory accesses. Single-Otile already eliminated P_buf
threadgroup staging. Q/K/V live in device memory through the matmul2d
view abstraction. **Bank-conflict padding may not help meaningfully**
in the single-Otile path because threadgroup memory isn't on the hot
path anymore.

### `swizzle_log` / BlockSwizzle

Block swizzling is Apple's technique to interleave threadgroup IDs across
output tiles for cache locality:
```cpp
const int swizzled_x = tgid.x ^ ((tgid.y & swizzle_mask) >> ...);
```

To implement this in V6, we'd need to:
1. Compute swizzled coordinates inside the kernel based on `swizzle_log`
   template arg
2. Change all `tgid.x * BQ` references to `swizzled_x * BQ`
3. Threading the BHND rewriter (which already operates on `tgid.y`) is
   tricky — they're orthogonal but the rewriter assumes one mapping

**Effort**: ~100-150 LOC. Touches both `loopForward` paths AND the BHND
post-gen rewriter.

**Hypothesis worth testing**: swizzle helps L2 cache locality when
multiple threadgroups concurrently read the same K/V regions. For
single-Otile + BQ=16 (small tiles, more threadgroups in flight), this
*could* help. But the gain depends heavily on the GPU's specific
cache-line size and L2 capacity — M5 Max is unknown territory.

## Decision

**Not implemented in this campaign.** The autoresearch winner from S3.1
(BQ=16 tiled, single-Otile) is already at 1.2-2.06× SDPA. Closing the
remaining gap with these levers would require:
- ~150-250 LOC of source-generator extension
- Non-trivial test coverage to ensure correctness across BQ/BK/SG
  combinations
- Unknown gain (possibly 0% if Apple's MPP already does the equivalent
  internally; possibly 5-15% if it doesn't)

This is out of the v2.29.0 budget. Logged for a future sprint.

## What this section produces

- This document
- No bench script
- No JSON data
- No code change

## Suggested follow-up sprint

If the residual D=128 gap (1.35-2.06× SDPA) becomes a priority:

1. **First**: instrument with Metal frame capture to see if M5's L2 is
   under-utilized (cache miss rate via `MTLCounterSample` if exposed).
   If cache misses are not the bottleneck, swizzle won't help.
2. **Then**: implement swizzle_log as a post-gen rewriter (insert
   computation lines before the existing tgid.x usages — easier than
   modifying the source generator template).
3. **Skip ld_padding** unless threadgroup memory access becomes
   demonstrably hot — single-Otile mostly avoids it.

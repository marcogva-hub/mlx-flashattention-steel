# V6 NAX — BHND Layout Migration Plan & Layout Analysis

**Date:** 2026-05-04
**Branch:** `feat/v6-nax`
**Status:** Analysis + plan documented; implementation attempt time-boxed in this sprint.

---

## Layout discovery — non-obvious finding

The Draw Things v2 kernel template comment says it operates on `[B, N, H, D]`
("BNHD") layout. Verifying this required understanding how MSL's
`tensor<device T, dextents<int32_t, 2>, tensor_inline>` interprets a
2D extent.

### Slice-math reconstruction

From `csrc/mfa/v6_nax/NAAttentionKernel.cpp`:

- **Line 433**: `constant uint K_Hq = HEAD_DIMENSION * Hq;`  (= D * H_q)
- **Line 751**: `auto Q = tensor<...>(Q_buf, dextents<int32_t, 2>(K_Hq, R));`
- **Line 845**: `Q.slice<BD, BR>(tgid.y * D + k, tgid.x * BR);`

For this slice to produce a valid `BD × BR` patch from Q, the column-major
interpretation of MSL `tensor` must apply:

> `dextents<int32_t, 2>(K_Hq, R)` lists `(innermost contiguous, outer slow)`
> dimensions. **Element [i, j] is at buffer offset `j * K_Hq + i`.**

This is the OPPOSITE of NumPy's `(outer, inner)` convention, but it is
required for the slice math to work.

### Per-batch layout

Combining slice arithmetic with the per-batch base offset
(`Q_buf += tgid.z * Q_batch_stride`, line 545):

```
address(b, n, h, d) = b * Q_batch_stride + n * K_Hq + h * D + d
                    = b * (N*H_q*D) + n * (H_q*D) + h * D + d
```

This is **`[B, N, H_q, D]` row-major** — the BNHD layout the comment
claims. The kernel is correct; the column-major dextents is the trick
that makes the slice math line up.

### Why this matters for migration

To migrate to BHND `[B, H_q, N, D]`, we want:
```
address(b, h, n, d) = b * (H_q*N*D) + h * (N*D) + n * D + d
```

The kernel's slice arithmetic `(head*D + k, c)` cannot directly produce
this — the `(N, head)` dim ordering requires a different addressing
strategy. Two viable approaches:

1. **Per-head binding**. Bind Q as `tensor(Q_buf + b*H_q*N*D + h*N*D, dextents(D, N))`.
   The per-head extents are `(D, N)` column-major = `(N, D)` row-major
   = the per-head BHND block. Slice arg becomes `(k, c)` (drop head*D).

2. **Stride-aware binding**. Pass strides as kernel parameters and use
   them in addressing. Requires moving away from `tensor_inline` to a
   stride-aware tensor type.

Approach 1 is simpler and matches Apple's per-head pointer-arithmetic
pattern in `steel_attention_nax`.

---

## BHND migration plan (Sprint 2A)

### Scope (forward path only, no backward/varlen)

The forward attention kernel:
- `loopForward(...)` (`NAAttentionKernel.cpp:760+`)
- Uses tensor declarations at line 751-755
- Uses slice ops in lines 766, 845, 854, 1027, 1046, 1084, 1093, 1280, etc.
- Output writeback at line 1293, 1310, 1315, 1348, 1353

### Required changes

| Change | Location | Old → New |
|--------|----------|-----------|
| Add per-head offset to per-batch buffer base | `createAdjustOffsets` (line 539-547) | `Q_buf + tgid.z * Q_batch_stride` → `+ tgid.y * R * D` (Q, O); `+ (tgid.y/ratio) * C * D` (K, V) |
| Tensor extents | Lines 751-755 | `dextents(K_Hq, R)` → `dextents(D, R)` for Q/O; `dextents(K_Hk, C)` → `dextents(D, C)` for K/V |
| Slice arguments | Multiple (lines 766, 845, 854, 1027, 1046, 1084, 1093, 1280) | Drop `tgid.y * D + ` (Q) or `tgid.y/ratio * D + ` (K/V) — leave just `k` or `D_remainder` |
| Output writeback base | Line 1293 | `O_buf + tgid.x * BR * K_Hq + tgid.y * D` → `O_buf + tgid.x * BR * D` (head offset is in O_buf) |
| Output cell store row stride | Lines 1310, 1315, 1348, 1353 | `idx[1] * K_Hq` → `idx[1] * D` |
| compute_d kernel (backward) | Line 691-695 | analogous; defer until forward works |

### Implementation strategy: post-generation source rewriting

Rather than modifying `NAAttentionKernel.cpp` (the Draw Things-derived
source generator), apply textual rewrites to the generated source string
in `mfa_v6_nax_primitive.cpp::generate_v6_source()`, gated by
`MFA_V6_BHND=1`. This is the same pattern used for Axes 4/5/6.

Advantages:
- No invasive changes to the upstream-derived code
- Cleanly togglable for benchmarking
- Cache key already includes `axis_flags` → fresh pipeline per variant

Disadvantages:
- String matching is fragile (must be exact)
- Cannot adapt to upstream NAAttentionKernel.cpp format changes silently

### Validation plan

1. **Build**: clean compile.
2. **Sentinel**: `MFA_V6_BHND=1 MFA_V6_SENTINEL_FILL=1 python bench/v6_coverage_diagnostic_v2.py`. Expect 0 sentinels.
3. **RMSE**: V6/SDPA RMSE should remain ~3e-4. If RMSE > 1e-2, BHND broken.
4. **All shapes**: FlashVSR-dense, SeedVR2-small, CogVideoX, SeedVR2-large, LTX2-cross. All must pass.
5. **Benchmark**: vs. baseline (BNHD path), expect 5-12% time on small
   shapes + 4× peak memory reduction.

### Failure-mode triage

If sentinel passes but RMSE explodes:
- Likely cause: address arithmetic bug in slice rewriting
- Diagnostic: compute V6 vs SDPA per-head RMSE; if different heads have
  different errors, head-offset is wrong

If sentinel shows uncovered cells:
- Likely cause: dispatch grid config bug, or kernel reads wrong per-head pointer
- Diagnostic: compare which (h, n, d) cells contain sentinel; pattern
  reveals the geometry bug

If correctness OK but perf regresses:
- Likely cause: per-head pointer is non-aligned; cache miss penalty
- Diagnostic: compare full-V6 time and kernel-only time before/after

---

## Sprint 2B — Chunked-K dispatch (DEFERRED)

Chunked-K requires the kernel to support runtime-variable `C` (KV-length
within a chunk). Currently `C` is a function constant set at compile time
(`v6_nax_compile.mm:48`). Two paths to enable chunking:

1. **Recompile per chunk size**: works but adds compile latency to first call.
2. **Convert C to runtime kernel argument**: requires kernel signature change.

Either approach requires kernel binding modification. Combined with the
LSE-weighted reduction that combines partial outputs, the work is
similar in scope to Sprint 2A.

**Recommendation**: Sequence Sprint 2A first; revisit 2B as a separate
session. The chunked-K dispatch ALSO benefits from the BHND layout
(per-chunk K slicing without transpose materialization), so the order
matters: BHND first, then chunked-K layered on top.

---

## Risk assessment

| Risk | Severity | Mitigation |
|------|---------:|-----------|
| Post-gen rewrite misses a slice site → silent corruption | HIGH | Sentinel diagnostic + RMSE check |
| Per-head offset interacts with Morton-order grid | MEDIUM | Verify by running on small shape first |
| GQA case (Hq ≠ Hk) needs different per-head offset for K/V | MEDIUM | Only LTX2-cross is GQA; defer GQA-specific BHND if needed |
| Variance/measurement noise hides regression | LOW | Run multiple iters, p50 reporting |
| BHND breaks varlen path | UNCLEAR | Forward-only scope; varlen is separate codepath |

---

## Decision: this sprint

Given the kernel-modification depth required and the constraint of one
session, the right scope for this sprint is:

1. **This document** — layout analysis + migration plan (delivered).
2. **Time-boxed BHND attempt** — post-gen rewrite behind `MFA_V6_BHND=1`,
   ~45 min budget. If correctness validates: benchmark. If not: revert.
3. **Defer Sprint 2B** — Chunked-K to a separate session for
   kernel-parameterization work.

The plan above is the canonical reference for whoever continues this work.

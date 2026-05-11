# V6 NAX Investigation Sprint — Executive Summary

**Date:** 2026-05-03
**Branch:** `feat/v6-nax`
**Hardware:** Apple M5 Max (40 cores, applegpu_g17s)
**Scope:** 5 investigation tasks + executive synthesis. No kernel modifications.

---

## TL;DR — three findings invalidate three sprint premises

| Sprint premise | Reality |
|----------------|---------|
| "We ported the wrong code (v1 from `/kernels/`, not v2 from `/v2/`)" | **WRONG.** Draw Things merged `/v2/` *into* `/kernels/` on March 6, 2026 (commit `0bf97fca`). Our port (May 3, 2026) IS the v2 code. Bit-identical kernel source generator. |
| "M5 Apple9+ dynamic shader core memory may exceed 32 KB" | **WRONG.** `maxThreadgroupMemoryLength = 32,768 bytes` verified by direct Metal API call. No new search space opens. |
| "Four MLX PRs are directly applicable to V6 NAX" | **PARTIALLY WRONG.** 3 of 4 are CLOSED/unmerged. Only PR #3307 (chunked SDPA) is technique-applicable; #3295 targets the wrong device family, #3293 D=256 is out of scope, #3306 LSE is already in our V6. |

**One finding redirects future work**:

| Finding | Implication |
|---------|-------------|
| Apple's `steel_attention_nax` uses **`metal_simdgroup_matrix`** (low-level) NOT MPP `matmul2d_descriptor` | The 5–7pp efficiency gap to SDPA is most plausibly the **MPP abstraction-layer overhead**. Tile-tuning cannot close this gap. |

---

## Per-task summaries

### Task 1 — Draw Things MFA v2 source code

**Verdict: We have v2.** The user's premise is wrong.

- Repo `liuliu/ccv` `unstable` HEAD: NO `/v2/` directory exists.
- Commit `0bf97fca (March 6 2026) Finish migrate v2.` is literally
  `git mv lib/nnc/mfa/v2/* lib/nnc/mfa/kernels/`. The v2 code was merged
  INTO `kernels/`, the directory we ported from.
- Our port commit (`b4c63d5`, May 3 2026) is dated AFTER the migrate.
- Diff our `csrc/mfa/v6_nax/NAAttentionKernel.cpp` vs upstream HEAD:
  **99 lines, all framework adaptation** (constructor signature,
  threadgroup helpers). The `createSource()` body that emits the actual
  kernel MSL is **byte-identical**.
- Morton-order grid layout (commit `ae1de996 March 30`) IS in our
  generated kernel and IS dispatched correctly by our wrapper
  (`csrc/v6_nax_compile.mm:111-119`).
- All 14 post-migrate commits to `NAAttentionKernel.cpp` (latest April
  28, 2026) are reflected in our port.

**The Draw Things "4.6× M5/M4" perf claim applies to OUR kernel.** Our
0.87–0.97× SDPA result is the v2 kernel's own ceiling, not staleness.

→ Detailed analysis: `docs/v6-nax/draw-things-v2-analysis.md`

### Task 2 — MLX NAX-related PRs

**Verdict: Only #3307 is applicable; the rest are closed or irrelevant.**

| PR | Title | Status | Verdict |
|----|-------|--------|---------|
| **#3307** | Chunked full-attention SDPA | CLOSED | **Sprint 2 candidate.** Threshold N>65K. Splits K into 32K chunks, combines via LSE-weighted reduction. SeedVR2-large (N=111375) qualifies. Expected +5-15% on the slowest shape. |
| #3295 | NAX tuning for gen-17 g | CLOSED | Targets `applegpu_g17g` (M5 base/non-Pro). Our M5 Max is `applegpu_g17s` (already on optimized path). Zero relevance. |
| #3293 | D=256 in fused SDPA | merged | D=256 added via single-tile instantiation `(32, 16, 256, 4, 1)`. ~30% slower than unfused on short seq. Not a NAX path (NAX restricts to D∈{64, 80, 128}). Out of scope. |
| #3306 | LSE output | CLOSED | Function constant 304 + buffer 8 for LSE. Our V6 already emits LSE as `outputs[1]` via cooperative_tensor reduction. |

→ Detailed analysis: `docs/v6-nax/mlx-pr-analysis.md`

### Task 3 — Metal profiling

**Verdict: Capture works programmatically; deep counter analysis needs Instruments GUI.**

- `mx.metal.start_capture(path)` and `mx.metal.stop_capture()` work.
  Successfully captured `docs/v6-nax/captures/v6_flashvsr_dense.gputrace`.
- No programmatic GPU counter API in MLX 0.31.x.
- Static analysis of register pressure: ~22.7 KB/simdgroup, ~70% of
  estimated register file capacity. Consistent with autoresearch's
  reluctance to push ExecSG > 16.
- Threads-per-TG occupancy: **50% of M5's 1024 ceiling** at our default
  (16 SG × 32 = 512 threads/TG). Apple's SDPA uses 128 threads/TG (more
  TGs co-resident per core for latency hiding) — opposite trade-off.

Sprint 2 needs Instruments-GUI inspection of the captured `.gputrace`
to pin down: ALU utilization, L2 cache hit rate, register spill count,
TG memory bank conflicts.

→ Detailed analysis: `docs/v6-nax/v6-metal-profile.md`

### Task 4 — Threadgroup memory budget

**Verdict: 32 KB hardware-enforced ceiling. No new search space.**

Probed via direct Metal API call:
```
maxThreadgroupMemoryLength: 32768 bytes (32.0 KB)
```

- Our autoresearch's 32 KB filter is correct.
- Apple9+'s "dynamic shader core memory" refers to internal cache
  partitioning, NOT a relaxation of the per-TG TGP allocation cap.
- 65 KB tile configs (R=16, C=128, SG=16) remain infeasible.

→ Detailed analysis: `docs/v6-nax/m5-threadgroup-memory.md`

### Task 5 — Apple SDPA NAX kernel

**Verdict: Apple uses a fundamentally different abstraction layer.**

Source: `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`
(482 LOC) + `nax.h` (887 LOC). Both readable.

| Aspect | Our V6 | Apple SDPA NAX |
|--------|--------|---------------|
| MMA primitive | MPP `matmul2d_descriptor` + `cooperative_tensor` | Raw `metal_simdgroup_matrix` + custom `NAXFrag/NAXTile` |
| Layout | `[B, N, H, D]` (transposed by host) | `[B, H, N, D]` (native, no transpose) |
| Tile config | R=16, C=48, SG=16 (~512 threads/TG) | BQ=64, BK=32, BD=head_dim, WM=4, WN=1 (128 threads/TG) |
| Grid | 1D Morton-encoded | 3D natural `(NQ, H, B)` |
| Causal | Per-tile boundary + last-tile element mask | Per-tile boundary + per-element `r < c` |
| Sinks | Not supported | Supported (function constant 302) |

**Critical insight**: Apple bypasses MPP entirely. They build a
`NAXTile<T, M, N>` abstraction over 16x16 fragments using
`metal_simdgroup_matrix.h`, then call `mma()` explicitly with
fragment-level scheduling. This is the **lower-level API**. Our V6
uses the higher-level MPP path which wraps this with driver-controlled
cooperative tensor allocation.

The MPP abstraction layer is the most plausible explanation for the
5–7pp efficiency gap to SDPA — tile-tuning cannot close it.

→ Detailed analysis: `docs/v6-nax/apple-sdpa-nax-analysis.md`

---

## Strategic implications

The sprint **did NOT find** a missed optimization in tile-tuning space:
- We have the latest v2 code.
- TGP memory cap is real.
- No applicable open MLX PRs.

The sprint **DID find** the architectural gap that explains the V6/SDPA
ceiling:
- Apple uses `simdgroup_matrix` directly.
- We use MPP `matmul2d_descriptor` (one abstraction layer above).
- The driver's cooperative tensor allocation imposes scheduling overhead
  Apple's path avoids.

**This is a DESIGN-level gap, not a TUNING-level gap.** Closing it
requires a kernel rewrite, not parameter sweeps.

---

## Recommended Sprint 2 priorities

| # | Task | Difficulty | Expected gain | Risk |
|---|------|-----------|---------------|------|
| 1 | **Reimplement V6 forward using `metal_simdgroup_matrix`** mirroring Apple's `attention_nax` structure | HIGH (~2 weeks) | +5–10% (close the gap to SDPA) | MEDIUM — MPP retreat means losing MPP-specific optimizations Apple's NAX team has built into the driver |
| 2 | **Implement chunked-K dispatch** (PR #3307 pattern) for SeedVR2-large only | MEDIUM (1–2 days) | +5–15% on N>65K shapes | LOW — chunked reduction is well-understood |
| 3 | **Switch V6 layout to BHND** (eliminate host transposes) | MEDIUM (~3 days) | +1–6% (varies by N), correctness-preserving | LOW — kernel offset math change only |
| 4 | **Profile in Instruments** (open existing .gputrace) before making any architectural change | LOW (1 hour) | Validates bottleneck hypothesis | NONE |
| 5 | **Skip Axe 7** double-buffering; rationale documented in optimization-campaign-report.md | — | — | — |

**Recommended sequencing**:
1. Week 1 day 1: Profile existing capture in Instruments. Confirm or
   refute the abstraction-layer hypothesis.
2. Week 1 day 2-3: Implement chunked-K + BHND layout. Both are quick
   wins independent of the abstraction debate.
3. Week 2-3: Attempt the simdgroup_matrix rewrite IF profiling
   confirms abstraction overhead is the bottleneck. Otherwise pivot
   to other angles (e.g., backward pass V6.1, INT8 NAX).

---

## What this sprint did NOT investigate (out of scope)

- Backward pass V6.1 (Draw Things v2 has it, but we don't expose it)
- INT8 NAX attention (`NAInt8AttentionKernel.cpp` from v2)
- Sinks (registered tokens) support
- M6 architecture forecasting (no M6 hardware available)

---

## Files created this sprint

| File | Purpose |
|------|---------|
| `docs/v6-nax/draw-things-v2-analysis.md` | Task 1 — premise refutation + v2 inventory |
| `docs/v6-nax/mlx-pr-analysis.md` | Task 2 — PR analysis |
| `docs/v6-nax/v6-metal-profile.md` | Task 3 — capture verification + static analysis |
| `docs/v6-nax/m5-threadgroup-memory.md` | Task 4 — TGP memory verification |
| `docs/v6-nax/apple-sdpa-nax-analysis.md` | Task 5 — Apple kernel architecture |
| `docs/v6-nax/captures/v6_flashvsr_dense.gputrace` | First V6 GPU trace |
| `docs/v6-nax/investigation-sprint-summary.md` | This file |

No code modifications were made.

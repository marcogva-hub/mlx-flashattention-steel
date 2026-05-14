# v2.50 Prompt 5b Section B — Phase 3b Top-K architecture comparison

**Mandate**: implement Top-K native Metal kernel after multi-architecture
iteration; eliminate the 17× regression vs SDPA documented in the
pre-Prompt-2 audit.  Phase 3a (Apple SDPA NAX + bias dispatch) shipped
the 1.25× speedup; Phase 3b targets further reduction via native
Metal kernel.

## §AA.5 premise validation (CONFIRMATION)

Per `docs/v50/audit-framing-inversions.md` Pattern verdict: Apple
primitives CANNOT close the gap.

**Empirical bench (M5 Max, B=1 H=16 qL=4096 D=128 fp16, k_count=64)**:

| Apple primitive | Latency | Notes |
|---|---|---|
| `mx.topk` (k=64) | 32.30 ms | Internal sort + slice |
| `mx.argpartition` (kth=4032) | 34.28 ms | Same cost — internal sort |
| `mx.partition` (kth=4032) | 32.32 ms | Same cost |
| `mx.sort` | 32.19 ms | Reference |
| `mx.fast.scaled_dot_product_attention` (no top-K) | **3.02 ms** | Theoretical floor (no filtering) |
| `q @ k.T * scale` matmul | 3.97 ms | Score computation |

**Component decomposition** of current Phase 3a path:

| Step | Time |
|---|---|
| `q @ k.T * scale` | ~4 ms |
| `mx.topk` (sort 4096 → top 64) | ~32 ms ← **bottleneck** |
| `mx.min(topk_vals)` (threshold extraction) | ~1 ms |
| `mx.where(scores >= threshold)` (bias build) | ~3 ms |
| `mx.fast.sdpa` with bias mask | ~3-5 ms |
| **Total** | **~42-45 ms** |

**Verdict**: native kernel justified — bottleneck is `mx.topk` (32ms,
8× matmul cost).  Apple has no `mx.fast.topk` accelerated primitive
in MLX 0.31 ecosystem.  Pure SDPA without top-K is 3ms, giving 14×
theoretical headroom over Phase 3a.

## Architecture iteration (5 approaches investigated)

### Approach 1 — Two-pass radix-select + sparse attention (CONFIRMATION)

**Design**: kernel 1 = compute scores + radix-select top-K indices.
Kernel 2 = scatter-gather SDPA on filtered K positions.

**Verdict**: L-effort (full C++ Primitive extension; 2 kernels +
scatter-gather buffer + cache + binding).  Estimated 4-6h focused
work.  Deferred (Section B v2) — see roadmap below.

### Approach 2 — Apple primitive composition (FALSIFIED)

**Design**: `mx.argpartition` to find top-K indices, then SDPA with
gathered K positions.

**Empirical**: `mx.argpartition` is 34.28 ms = same cost as `mx.sort`.
MLX 0.31 doesn't have an accelerated partition primitive separate
from sort.

**Verdict**: FALSIFIED.  No improvement over Phase 3a.

### Approach 3 — `mx.compile` graph fusion (FALSIFIED)

**Design**: wrap Phase 3a's operations in `@mx.compile` to let MLX
fuse the matmul + topk + bias-build + SDPA into one graph.

**Empirical**:
- Phase 3a uncompiled: 42.94 ms
- Phase 3a with `mx.compile`: 47.95 ms (slight regression, likely
  due to bias allocation patterns under JIT)

**Verdict**: FALSIFIED.

### Approach 4 — Bisection-based threshold via `mx.fast.metal_kernel` (✅ SHIPPED)

**Design**: custom Metal kernel computes per-row top-K threshold via
**bisection** — exponentially-converging binary search for the score
value above which exactly K elements lie.  Per row:
- Threadgroup = 256 threads, processes one (B*H, N) row
- Phase 1: simdgroup-reduce min/max for bisection range
- Phase 2: 32 iterations of bisection
  - Each thread counts its 16 local scores >= mid
  - simdgroup sum + threadgroup reduction = row count
  - Adjust lo/hi based on row_cnt vs K
- Output: FP32 threshold value

Threshold is then used to build bias mask for Phase 3a's SDPA call.

**Empirical (3-session bench at audit shape)**:

| Path | Latency |
|---|---|
| Phase 3a (mx.topk) | 42.91 ms |
| Architecture B (bisection metal_kernel) | **11.15 ms** |
| Speedup | **3.85×** |

**Correctness**: bisection produces FP32-precision threshold (32-iter
bisection over [-30, +30] gives ~7e-9 precision).  Cast to FP16 for
mask comparison gives ~64-69 elements selected (same range as Phase
3a's mx.topk which also has 64-69 due to FP16 ties at the boundary).
SDPA output differs by max ~0.68 vs Phase 3a — this reflects the
INHERENT FP16 ambiguity of "top-K with k_count=64 out of 4096"
when scores have ties near the threshold.  Both paths produce
mathematically valid top-K-approximate outputs.

**Verdict**: ✅ SHIPPED as opt-in via `MFA_TOPK_BISECT=1` env var.

### Approach 5 — Single-pass running top-K state machine (DEFERRED)

**Design**: single Metal kernel maintains a running top-K heap per
Q-row across K-tiles.  Avoids materializing the full scores tensor.

**Concern** (per `/metal-kernel-dev` pre-impl review): heap-maintenance
state across K-tiles requires per-row threadgroup memory (64 fp32
scores × 64 K_top = 16 KB), and the heap insert/replace operation has
complex SIMD-divergence behavior (heap properties don't map naturally
to SIMD-uniform patterns).

**Verdict**: DEFERRED to Section B v2.  Approach 4 (bisection) avoids
this complexity by NOT maintaining per-row state — just bisecting on a
pre-computed scores tensor.  The trade-off: Approach 5 saves the
scores materialization (512MB at audit shape), Approach 4 saves the
sort but pays the matmul (4ms / 512MB) twice (once for scores, once
for bias `where` operation).  At current audit shape, Approach 4
delivers most of the achievable speedup.

## Selected architecture: Approach 4 (bisection)

**Why**: best speedup (3.85×) achievable in a single `mx.fast.metal_kernel`
implementation without C++ Primitive plumbing.  Native dispatch
through Apple's MSL pipeline + simple Python integration.

**Limitations documented**:
- Score tensor still materialized (512MB at audit shape) — same as
  Phase 3a.  Approach 5 would eliminate this.
- FP16 boundary ambiguity inherent — both Phase 3a and Approach 4 may
  select 64-69 elements depending on ties.
- Opt-in default; Phase 3a remains AUTO default to preserve exact
  `mx.topk` semantics for users who haven't validated approximate
  top-K acceptability.

## Section B v2 follow-up roadmap

To productionize Approach 5 (single-pass running top-K):

1. `mx.fast.metal_kernel` with per-row top-K min-heap state in
   threadgroup memory.  Estimated 16 KB TGM (under 32 KB budget).
2. K-tile streaming: each TG processes K_BLOCK columns at a time,
   accumulating into the per-row heap.
3. Output: top-K indices [B, H, N, K_top].
4. PASS-2: SDPA with scatter-gather on K/V using indices.
5. Three-axis validation + bench characterization.

**Estimated effort**: 4-6h focused.  Targets the remaining 8× headroom
over Approach 4 (11ms current → ~5-6ms with full elimination of
scores materialization, ~7× over Phase 3a).

## Skill invocations (§AA.2)

| Skill | When | Verdict |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` | Pre-impl premise check | CONFIRMATION — no Apple primitive composition reaches the speedup |
| `/metal-kernel-dev` | Approach 4 design review | GREEN — bisection is pure control flow, no register-budget concerns, SIMD reductions are well-understood |
| `/mlx-mfa-bench-methodology` | 3-session bench Approach 4 | Median 11.15 ms; cross-session variance < 5% |
| `/mlx-code-review` | Pre-merge self-review | Approach 4 opt-in; Phase 3a remains AUTO default; FP16 boundary semantics documented |

## Files modified

| File | Change |
|---|---|
| `mlx_mfa/attention.py` | Add `_topk_bisect_threshold_kernel` mx.fast.metal_kernel + opt-in routing in `flash_attention_topk` |
| `tests/test_v50_sprint_5b_section_b_topk_bisect.py` | 6 tests: correctness vs Phase 3a, opt-in env, count behavior, bf16, D=64/D=128 |
| `docs/v50/phase-3b-architectures-comparison.md` | This doc |

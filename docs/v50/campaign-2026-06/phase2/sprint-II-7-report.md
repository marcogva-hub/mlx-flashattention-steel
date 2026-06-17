# Sprint II-7 — Profiling-Driven Bottleneck Hunt (2026-06-12)

**Status**: COMPLETE.
**Headlines**: (1) LCSA dynamic mask build was a pure-Python/numpy
bottleneck costing **3x the attention it serves** — GPU-vectorized,
**15.4x faster** (full LCSA loop 3.3x).  (2) **`mlx.nn.Conv3d` never
reached the conv auto-hook** (it calls `mx.conv3d`, not
`mx.conv_general`) — every standard MLX model silently bypassed NAX
conv; fixed.  (3) Decode-loop ladder quantified: the TQ attend kernel
is the kernel-bound floor (14x dense SDPA) — feeds the II-5 decode
ledger.  (4) Non-causal D=64 backward measured at **1.88x** via the
clean V6NAX split kernel — promotion decision data for Marco.

## Harnesses (benchmarks/profile_ii7_harnesses.py, committed)

Per-call instrumentation separates graph-build (Python/dispatch) from
eval (GPU) time.  M5 Max, fp16, medians of 30 (warmed).

| Harness | Cell | build ms | eval ms | build % |
|---|---|--:|--:|--:|
| DiT attn | N=4096 fwd | 0.016 | 1.41 | 1.1% |
| DiT attn | N=8192 fwd | 0.022 | 4.66 | 0.5% |
| DiT attn | N=4096 causal bwd (V6NAX split) | 0.081 | 8.80 | 0.9% |
| DiT attn | N=4096 non-causal bwd (SDPA-vjp) | 0.069 | 17.92 | 0.4% |
| VAE conv3d | T8 32x32 C256 | 0.034 | 1.42 | 2.3% |
| VAE conv3d | T8 64x64 C128 (K=3456) | 0.038 | 2.06 | 1.8% |
| VAE conv3d | T16 64x64 C128 | 0.049 | 3.82 | 1.3% |
| TQ paged decode | S0=4096 GQA4 | 0.80 | 3.94 | 16.8% |
| TQ paged decode | S0=16384 GQA4 | 0.81 | 15.87 | 4.9% |
| sparse | topk k=64 N=4096 D=128 | 0.051 | 11.17 | 0.5% |
| sparse | LCSA mask build (AFTER) | 0.22 | 0.56 | 28% |
| sparse | LCSA attend | 0.014 | 3.67 | 0.4% |

Attention/conv dispatch layers are healthy (<2.5% build) — the Phase-I
decision-cache work held.

## Optimization 1 — LCSA mask build: 11.19 ms → 0.73 ms (15.4x)

cProfile attribution of the old `make_lcsa_mask`: `np.einsum` 4.7 ms
(CPU matmul), `pool_tiles` 4.4 ms (385 `np.mean` calls in a Python
loop), `np.array(q/k)` 2.3 ms (two full GPU→CPU copies, 16 MB each).
The FlashVSR LCSA pattern rebuilds the mask per layer per step, so this
was a per-call production cost larger than the attention it gates
(11.19 vs 3.7 ms).

Rewrite (committed): on-GPU MLX throughout — reshape+mean tile pooling
(ragged tail handled), per-head GPU matmul + (B,H)-mean, and the
per-row variable-k top-k selection via dense descending ranks
(`rank < top_k & spatial`; non-spatial entries sort last at -inf).
Lazy output.  **Full-loop: mask+attend 14.9 → 4.5 ms = 3.3x.**  57
LCSA tests pass; tie-breaking among exactly-equal pooled scores may
differ from np.argpartition (top-k contract and keep-counts preserved).

## Optimization 2 — `mx.conv3d` hook coverage (Auto-default violation)

Telemetry under the VAE loop read **0 executed / 0 fallback after 20
`mlx.nn`-style conv3d calls**: `mlx.nn.Conv3d.__call__` invokes
`mx.conv3d` directly, never `mx.conv_general` — the only surface
`install_hooks()` patched.  Plain-MLX users therefore had NO path to
NAX conv (the Phase-96 SeedVR2 engagement worked because that pipeline
calls `conv_general`).  Fix (committed): `install_hooks()` also patches
`mx.conv3d`, delegating into `_patched_conv_general` (same eligibility,
telemetry, KD-6 dtype contract, fallback).  Verified: engagement
counted, routing parity max_err 0.008 (fp16-grade at K=3456),
`uninstall_hooks()` symmetric.  At the T8-64x64-C128 cell the NAX win
over MLX 0.31.2 baseline is ~1.03-1.08x (consistent with II-1's
zero-inversion map; the marginal value is shape-dependent).

## Quantified floors (documented, not optimized)

**Decode ladder (S=4096, GQA4, D=128, one step):**
| Path | ms | vs dense |
|---|--:|--:|
| dense `mx.fast.sdpa` (fp16 KV resident) | 0.33 | 1.0x |
| plain paged (`PagedInferenceContext.step`) | 1.00 | 3.0x |
| TQ-3bit paged | 4.67 | 14x |
| TQ-3bit, `tq_v=False` | 4.16 | 12.5x |
| TQ-3bit, `wht_in_kernel=True` | 4.60 | 14x |

The TQ attend kernel dominates (append is 0.42 ms incl. its eval; step
Python build 0.80 ms).  K-dequant/QK in-kernel path is the cost
(`tq_v` toggle moves only 0.5 ms).  **Kernel-bound** — this is the
repo-owned surface where the II-5 cider-style KV-group scheduling and
a dequant-path overhaul would compound; feeds the Marco-gated decode
ledger.  Memory note: `_v_pool_fp16` (50 MB at this config) stays
allocated and written per token even when `tq_v=True` binds the packed
V — a memory/traffic saving candidate for the decode sprint.

**Other floors:** DiT forward = Apple SDPA NAX (closed); VAE conv3d
eval = II-4's measured 63-104%-of-peak GEMM + im2col split (the
fused-im2col / MPP-conv2d leads from II-4/II-5 are the path); topk
k=64 N=4096 D=128 = 11.2 ms production path (Architecture B; the
XL streaming variant remains Marco-gated at its 1.6x ceiling).

**Decode step Python build (0.80 ms, 16.8% at S=4K)**: dominated by
`append`'s 5 sliced pool `__setitem__`s + the TQ pack graph build —
diminishing returns vs the 4 ms kernel; documented, not chased.

## Promotion-decision data for Marco (NOT acted on)

Non-causal D=64 backward (the DiT-training cell, deliberately excluded
from the II-0 causal-only promotion): default SDPA-vjp 17.07 ms vs V6NAX
split opt-in 9.07 ms = **1.88x**, unit-scale max errs dQ/dK/dV =
4e-4/2e-3/1e-3 (clean split kernel, paired-MMA fix in).  With II-6's
guard + unit-scale locks in place, the original promotion-blocker
evidence is materially stronger than at II-0 time.  Decision: Marco.

## Validation

- Full-loop after-state: all 4 harnesses monotonically non-worse
  (DiT/VAE/decode unchanged within noise; sparse 3.3x better).
- Suite: 1391 passed (x3 runs across the sprint's commits).
- Three-axis on both promotions: outputs vs reference (LCSA mask
  contract via 57 tests; conv3d routing parity 0.008), path-entered
  (telemetry/timing), edges (ragged tails, per-head masks, uninstall).

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-bench-methodology` (protocol) | all numbers | medians of 30, warmed, build/eval split |
| `/mlx-mfa-perf-audit` (protocol) | LCSA + hook claims | full-loop before/after on the committed harness; no public-API claim ships without it |
| `/mlx-code-review` (protocol) | LCSA rewrite | Category-C checklist item ("manual numpy instead of mx ops") was the exact finding |

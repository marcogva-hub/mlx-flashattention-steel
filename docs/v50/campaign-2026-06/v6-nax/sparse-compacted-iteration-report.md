> **⚠ CORRECTION (2026-06-17, `compacted-kernel-increment-0-report.md`):** this report's premise is
> **RETRACTED**. Its measurements used an ASYMMETRIC mask (BQ=32/BK=16) which on M5/26.6 routes to
> **dense Apple SDPA**, not a sparse kernel — so the "flat 3.8ms", the "compaction floor 5–15×", and
> the "GO-scale" verdict were SDPA artifacts (the floor was just SDPA on a shorter sequence). The
> REAL mlx-mfa sparse kernel (symmetric mask) **already tracks density and already wins** (4.7× @
> d=0.03 vs SDPA) — there is NO compacted kernel to build (FULL INVERSION). The real lever is
> ROUTING (get asymmetric masks onto the working symmetric kernel). See the increment-0 report.

# Block-Sparse Compacted-Iteration — Gated Prototype (issue 2: make the skip translate to wall-clock)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `abaee24`, macOS 26.6, M5 Max 128GB, mlx 0.31.2. Probe (committed):
`benchmarks/methodology/sparse_compaction_probe.py`. Pre-flight: `benchmark-measurement-correctness`
(effective-FLOP, plausibility-gated ≤51.8 TFLOPS, regime-checked, 3-rep median, ablate the REAL
kernel — no standalone proxy). **No production kernel or routing change (keep-all-paths).**

## Headline: **GO-scale.** The flat ~3.8ms is the **loop-walk over inactive blocks** — compaction-addressable — and removing it makes wall-clock **track density** (5.7× at d=0.125, 3.6× at d=0.25, up to 14.8× at d=0.008), measured on the production kernel.

The diagnostic (`abaee24`) showed the block-sparse forward runs in density-independent time. This
sprint decomposes that term and gates a fix. **Gate PASSED:** the dominant cost is the
per-threadgroup walk over all NK K-blocks (`for kb in 0..cNK { if !mask continue }`,
`mfa_sparse_attention.cpp:524`), which compaction (iterate only active blocks) removes.

## Phase 0 — ∝N² decomposition + GATE (marked ablations on the REAL routed kernel)

Decoupling qL (→NQ grid) from kL (→NK walk), at active=2 K-blocks (≈zero compute):

| Ablation A — loop-walk (fix NQ=4, vary NK) | Ablation B — grid (fix NK=128, vary NQ) |
|---|---|
| NK=64 → 0.28ms; 128 → 0.39; 256 → 0.63; 512 → 1.11; **1024 → 1.97ms** | NQ=4 → 0.39ms; 64 → 1.14; 256 → 3.54; **512 → 6.65ms** |
| per inactive K-block ≈ **1.4µs** (uncoalesced device mask read + branch latency) | per query-block ≈ 12µs (includes its own NK-walk) |

**Control:** at qL=kL=4096, active=2 → 3.74ms ≈ active=full → 3.77ms. **Equal ⇒ neither compute nor
active-block memory dominates** — it is the density-independent walk over inactive blocks.

**GATE VERDICT: PASS — compaction-addressable (loop-walk dominant).** Both axes scale, but the
walk over inactive blocks is the removable term (the floor below removes it at fixed NQ and t
collapses). Mask-gated load (`continue` before K/V load, `:524`) already prevents masked-block
streaming, so memory is not the wall — confirmed by the control. **Fix directed: 1D compacted
active-block iteration** (loop only active K-blocks per query-block; precompute the active-index
list / bounding range host-side — an in-kernel scan would re-introduce the O(NK) walk).

## Phase 0-floor — the decisive prototype evidence (does the skip translate to wall-clock?)

Compacted-iteration cost is **∝ active_count regardless of block layout** — the `K_kb` pointer jumps
per active index (`:528`), so contiguous and scattered active blocks cost the same. Therefore the
**production kernel run at `kL = active·BK`** over the production query length (qL=4096, NQ=128 grid
held fixed) IS a faithful compacted-iteration measurement — the real optimized kernel, not a
standalone proxy (footgun #3), and representative for any mask structure:

| active | density | current (full-walk) | **compacted floor** | **potential** |
|---|---|---|---|---|
| 2 | 0.008 | 3.76 ms | **0.25 ms** | **14.8×** |
| 8 | 0.031 | 3.73 ms | 0.36 ms | 10.5× |
| 32 | 0.125 | 3.77 ms | 0.66 ms | **5.7×** |
| 64 | 0.250 | 3.75 ms | 1.05 ms | 3.6× |
| 128 | 0.500 | 3.79 ms | 1.94 ms | 2.0× |
| 256 | 1.000 | 3.72 ms | 3.72 ms | 1.0× |

**The flat 3.8ms becomes a curve that tracks density** (0.25 → 3.72 ms). The skip translates to
wall-clock. All floor eff-TFLOPS ≤ 37 (≤ 51.8 peak; plausibility OK — at low density the floor is
fixed-overhead-bound, not throughput-bound, as expected).

## Phase 1 — (N, memory) frontier (public-library justification + routing input)

The deterministic memory differentiator is the **mask representation**: SDPA needs an N×N additive
mask, the block-sparse kernel an [NQ,NK] bool block-mask (**512× smaller**):

| N | N×N fp16 mask (SDPA-specific) | block mask | 
|---|---|---|
| 8192 | 1.0 GB | 2 MB |
| 16384 | 4.0 GB | 8 MB |
| 32768 | **16.0 GB** | 32 MB |

Total peak is qkv-dominated (~1.2× higher for SDPA at large N; the per-call peak figure is
allocator-pool-noisy and not relied on here). **Honest framing:** memory is a *secondary, long-N*
justification — the N×N mask (GBs at N≥16384) is real overhead a constrained Mac (16/24/32GB,
*projected* — not measured on such a machine) cannot afford alongside a model, but qkv dominates at
multi-head configs. **The PRIMARY public-library win is the compute curve above (5.7× at d=0.125).**

## Phase 2 — prototype scope (the GO build, NOT shipped this sprint)

The floor is the achievable curve. The full kernel converts it to a routed reality. Smallest real
form: per-query-block active-index list (or `[kb_lo,kb_hi]` bounding range for the banded/causal
family — exact compaction there), precomputed host-side from `block_mask`, passed as buffers; the
K-loop iterates only those. Keep-all-paths: the current sparse kernel stays (baseline + memory
fallback); the compacted kernel is additive. One residual the floor does not capture, to measure
first in the build: **scattered-mask gather overhead** (active-index read + non-contiguous K/V
locality) — bounded (the floor shows the ceiling), and the GO increment's first three-axis check.

## Phase 3 — verdict

1. **Does the skip translate to wall-clock?** YES — t tracks density once the inactive-block walk is
   removed: **5.7× at d=0.125, 3.6× at d=0.25, 2.0× at d=0.5** vs the current flat 3.8ms (absolute:
   0.66ms vs 3.77ms at d=0.125).
2. **(N, memory, density) → best-path map:** dense **SDPA** where it fits and density is high
   (unconstrained, d≳0.5: SDPA 3.05ms beats even compacted-sparse). **Compacted-sparse** at low
   density (d≲0.25: 0.7–1.0ms beats SDPA's 3.05ms) AND at long N on constrained memory (where the
   N×N mask OOMs). **Current-sparse** retained as the memory fallback + baseline.
3. **GO-scale.** Build the full compacted block-sparse kernel + memory-aware AUTO routing (by N,
   density, available memory), keep-all-paths, each increment three-axis (mask-ndim 2/3/4, GQA,
   all-masked query-block, partial/boundary blocks) + Pattern #6, gather-overhead measured first.
   The floor is the GO evidence: the achievable low-density win is large and compaction-addressable.

## Validation / discipline
- Phase 0 ablations on the REAL kernel (no proxy); gate = compaction-addressable; fix = 1D compacted
  iteration. Floor measured on the production kernel at compacted-equivalent shape (∝active_count,
  layout-independent → representative). All eff-TFLOPS ≤ 37 ≤ 51.8 peak; 3-rep median; effective
  FLOP; causal-½ avoided (non-causal). Memory frame on the deterministic mask-size delta (peak is
  noisy — flagged); constrained budgets projected — flagged.
- No production kernel/routing change (current sparse retained). No new Metal kernel built this
  sprint — the prototype evidence is the real kernel at compacted-equivalent shape (a legitimate,
  footgun-free measurement). No orphan processes. Not tagged/published.

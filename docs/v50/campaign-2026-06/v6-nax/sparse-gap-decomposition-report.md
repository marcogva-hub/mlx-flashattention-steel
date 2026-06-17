> **⚠ CORRECTION (2026-06-17, `compacted-kernel-increment-0-report.md`):** the measurements in this
> report were taken with an ASYMMETRIC block mask (BQ=32/BK=16), which on M5/26.6 routes
> `flash_attention_sparse` to **dense Apple SDPA** (`_sparse_fallback_sdpa_perhead`), NOT an mlx-mfa
> sparse kernel. The "flat ~3.8ms density-independent / skip wall-clock-inert" finding is therefore
> **RETRACTED** — it was SDPA's mask-independence, not the sparse kernel. The REAL sparse kernel
> (symmetric mask) already tracks density (4.7× @ d=0.03 vs SDPA). Which-binary must be confirmed at
> the runtime dispatch, not the C++ source. See the increment-0 report.

# Sparse NAX Gap Decomposition — Diagnostic (read-mostly + marked micro-probes)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `532bca8` (post-V6-rename), macOS 26.6, M5 Max 128GB, mlx 0.31.2.
Probe: `benchmarks/methodology/sparse_gap_probe.py` (committed) + inline verification probes
(below). Pre-flight: `benchmark-measurement-correctness` (effective-FLOP, plausibility-gated vs
51.8 TFLOPS fp16 NAX peak, regime-checked, 3-replicate median, fp32-sanity). **No production
kernel or routing change.**

## Headline: the addressable gap is NOT matmul2d-setup — it is that **block-skip does not translate to wall-clock**. The `matmul2d→raw-simdgroup` lever's premise is **FALSIFIED**.

The routed block-sparse NAX-matmul2d forward runs in **density-INDEPENDENT time** (~3.8ms at
N=4096, *flat* from d=0.008 to d=1.0). It honors the mask (correct output) but computes/streams as
if dense. So the "11.4 TFLOPS, 4× below SDPA" was never a slow *per-FLOP* kernel — it was
active-block effective-FLOPs ÷ a near-dense runtime. The per-work (slope) term, where matmul2d
cooperative-tensor setup lives, is **0.7%** of the cost; the density-independent term is **99.3%**.

## Phase 0 — per-path completeness map (READ-ONLY, fresh code)

| Path | Compute backend (file:line) | Routing (file:line) | Verdict |
|---|---|---|---|
| dense fwd D=64/128 (V1/V2/V2-sk/V2-dsplit/V3/V4/V5/decode) | STEEL `simdgroup_matrix` (`mfa_attention.cpp:207-845`) | `backend="mfa"` expert; dense default→SDPA (`dispatch_policy.py _M5_NAX_THRESHOLDS`=999999) | **STEEL-by-design** (SDPA owns dense NAX) |
| V6NAX dense fwd primitive (`v6_nax_forward`) | NAX `matmul2d` (`mfa_v6_nax_primitive.cpp`) | recompute-only for V6 backward O,L (`attention.py:5298`); causal→STEEL gate (`:4928`) | **DECLINED/semi-orphaned** as a forward |
| **block-sparse / LCSA fwd** | **NAX `matmul2d`** coop-tensor (`mfa_sparse_attention.cpp:284-312` `BaseNAXFrag::mma`; tiled via `:548/:604`, gen `sparse_kernel_source:641`) | **ROUTED** (`lcsa_nax.py:198` → `_ext.sparse_attention_forward_with_lse`) | **ROUTED — diagnostic subject** |
| windowed (V3) | STEEL `simdgroup_matrix` | routed conditional-auto (`mfa_attention.cpp:821`) | STEEL-by-design |
| GNA native | STEEL `MFAMMAFrag::mma` (`mfa_gna_fwd.cpp:373/509`) | routed D=128 3D | STEEL-by-design (small-N overhead-bound) |
| paged / TQ decode+prefill | gather/dequant + Apple SDPA | routed default | by-design (sync-floor-bound; IV-D1/D2) |
| conv3d-nax | NAX `matmul2d` (`mfa_conv_nax.hpp:39`) | routed (auto-hook) | ROUTED |
| backward dQ/dK/dV/fused | NAX `matmul2d` (`NAAttentionKernel.cpp`) | routed D=64 causal default-on; D=128 opt-in (`MFA_ENABLE_V6_BACKWARD`) | ROUTED |
| sparse backward (hybrid) | NAX fwd + Apple SDPA-vjp dQ/dK | **DEFAULT** (`attention.py:3196`) | ROUTED |
| sparse backward (full-native) | NAX `matmul2d` native sparse | `MFA_V6_BWD_SPARSE_NATIVE=1` opt-in (`attention.py:3189-3194`) | **DECLINED-on-perf** (Pattern #6: native < SDPA-vjp; `docs/v50/section-a-v3-empirical-verification.md`) |

**Completeness verdict (answers Marco's "is anything missing"): NOTHING is MISSING, NOTHING is
ORPHANED.** Every path is routed-NAX, DECLINED-on-perf-with-documented-reason (V6 dense forward;
full-native sparse backward), or STEEL-by-design (dense=SDPA territory, windowed, GNA-small-N,
decode-sync-bound). The V6 NAX kernel family is **complete**. The block-sparse forward HAS NAX and
is routed; the open question is purely whether its gap is *addressable*, answered below.

## Phase 1 — the realistic sparse ceiling (NOT dense-44.9)

Anchors (N=4096 D=128 B2 H8 f16, **non-causal**, effective-FLOP, 3-rep median, cv≤0.02):
SDPA dense **3.051 ms = 45.05 TFLOPS** (register cooperative_tensor; the achievable per-work rate).
STEEL dense `backend="mfa"` 12.40 ms = 11.08 TFLOPS (known M5 STEEL-dense regression; not the M5 path).

**Ideal-sparse(d)** = active-compute-floor + unavoidable-gather = `d · t_SDPA` + (small mask-read).
| d | measured | ideal-sparse (d·SDPA) | gap (×) |
|---|---|---|---|
| 0.25 | 3.77 ms (9.1 TF) | 0.76 ms (45 TF) | **4.94×** |
| 0.50 | 3.76 ms (18.3 TF) | 1.53 ms (45 TF) | **2.46×** |

The addressable gap vs the *realistic* ceiling is large and **grows as density falls** — the
signature of "sparsity not exploited," not of a per-FLOP inefficiency.

## Phase 2 — decomposition (marked ablation micro-probes)

**Density sweep** (rect mask, each Q attends first ⌈d·NK⌉ K-blocks; t = a + b·d):
d=0.125→3.751, 0.25→3.826, 0.375→3.772, 0.5→3.777, 0.625→3.816, 0.75→3.792, 0.875→3.778,
1.0→3.803 ms. Fit: **t(d) = 3.778 ms (intercept) + 0.021 ms·d**, i.e. ~flat (the per-work slope is
0.7% of cost). fp32-sanity: sparse(d=1.0)==dense max_abs_err **0.0**; sparse(d=0.25) vs independent
masked-SDPA fp32 ref **3.81e-6** (honors the mask — correct sparse output, not a dense fallback).

**Extreme-sparsity + fixed-active N-sweep** (the mechanism probe):
- active=2 of 256 K-blocks: t = **3.83 ms** = same as d=1.0 (3.80 ms). Attending 2 blocks costs the
  same as 256.
- fixed active=2, vary N: 1.32 (N1024) / 1.68 / 3.69 / **15.0 ms** (N8192) — **t scales with total N²
  at zero compute**, and t(active=2) ≈ t(d=1.0) at every N (N8192: 15.0 vs 15.2 ms).

**Bucket attribution of the 11.4→ideal-sparse gap:**
| bucket | share | simdgroup-addressable? |
|---|---|---|
| **density-independent term** (iterate-all-tiles + K/V stream ∝N², launch) | **~99%** at low d | **NO** — this is the gap |
| matmul2d cooperative-tensor setup (per-`mma` ct copy-in/run/out, `:296-311`) | ~0.7% slope; 1.25× form-penalty visible only at d=1.0 (36 vs 45 TF) | yes, but negligible absolute |
| online softmax / normalization | in the density-independent term | n/a |

**The `continue`-skip (`mfa_sparse_attention.cpp:524/724`) does not eliminate wall-clock cost.** The
per-Q-tile threadgroup iterates all NK K-tiles; the skip drops the MMA but the dominant ∝N² cost
(per-tile loop/address-gen and/or K/V streaming) survives. Root mechanism (loop-overhead vs memory
traffic) is **DEDUCED**, not yet isolated by a profiler trace — that is the first step of any fix,
not assumed here.

## Phase 3 — lever verdict

1. **Addressable gap?** YES, and large: at d=0.5 the kernel is 2.46× slower than ideal-sparse, at
   d=0.25 4.94×, growing as density falls (the wasted-FLOP-savings signature).
2. **Which lever:**
   - **`matmul2d→raw-simdgroup`: NOT WARRANTED — premise FALSIFIED.** It targets the 0.7% per-work
     slope; the form penalty is 1.25× only at full density and negligible at the sparse densities
     VSR/LCSA use. The diagnostic existed to test this intuition on data; the data refutes it.
   - **The real lever is "make block-skip translate to wall-clock"** (compacted active-tile
     iteration / grid launched only for active (q,k) pairs / true early-exit), a tiling/grid rewrite,
     NOT an MMA-form change.
3. **The honest bar (Pattern #6 reflex):** the relevant baseline is **dense SDPA (3.05 ms)**, not
   ideal-sparse. Today the sparse kernel (3.8 ms) **LOSES to SDPA-with-additive-mask** even at d=1.0
   (1.25×) and at every density (flat 3.8 ms vs SDPA 3.05 ms) — SDPA does the same masked work
   faster. A skip-to-wall-clock fix only pays off where `d · 3.05 ms` < SDPA's effective cost, i.e.
   **low density (d ≲ 0.25**, the typical LCSA/VSR regime). This mirrors the sparse-backward Pattern
   #6 finding (native sparse < SDPA-vjp dense) on the forward side.

**Verdict: the `matmul2d→raw-simdgroup` chantier is CLOSED (premise falsified).** There is a large
addressable gap, but the lever is a **skip-to-wall-clock tiling/grid rewrite**, gated on first
root-causing *why* the existing `continue` is wall-clock-inert (a ~1-day profiler/sentinel
investigation — NOT a kernel commitment), and justified only at low density vs the dense-SDPA bar.
If that root-cause shows the ∝N² term is unavoidable memory traffic (K/V must be resident), the
gap is **inherent** and **v2.57.0 (the rename release) is the terminus** for the sparse forward —
production should prefer dense SDPA-with-mask at these shapes. Clean either way: a measured
"premise falsified, narrow conditional lever, root-cause-gated" is as first-class as "prototype
warranted."

## Validation / discipline
- All effective TFLOPS ≤ 45 (≤ 51.8 peak) — plausibility gate passed, no artifacts. Causal-½ avoided
  (all non-causal, apples-to-apples). Regime valid (Δt tracks total-N at fixed compute). 3-rep
  median cv≤0.04. fp32-sanity on output-producing probes (0.0 / 3.8e-6). Provenance: probe script +
  raw numbers above.
- No production kernel/routing change (keep-all-paths). Micro-probes marked + throwaway/committed
  for reproducibility. No orphan processes. Not tagged/published.

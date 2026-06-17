# Compacted Block-Sparse Kernel — Increment-0: NO-GO (FULL INVERSION + which-binary correction)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `276b746`, macOS 26.6, M5 Max 128GB, mlx 0.31.2. Pre-flight:
`metal-kernel-dev`, `benchmark-measurement-correctness`, `mlx-mfa-nax-matmul2d-correctness`.
**No kernel built, no routing changed (keep-all-paths).**

## Headline: **NO-GO on the build — the compacted kernel ALREADY EXISTS and already wins.** And a which-binary error invalidates the prior three sprints' core measurements.

Before building, the which-binary check (lesson #14, run at the RUNTIME dispatch this time, not just
the C++ source) revealed: **`flash_attention_sparse` with an asymmetric mask on M5+ routes to dense
Apple SDPA** (`_sparse_fallback_sdpa_perhead`, `attention.py:3239` — the V1 STEEL sparse kernel is
disabled on M5/26 by the `(long)p->NK` bug). The prior three sprints all used asymmetric masks
(`make_causal_block_mask`/`_steel_block_config` → BQ=32/BK=16), so **they benchmarked dense SDPA, not
any mlx-mfa sparse kernel.** The REAL sparse kernel (symmetric mask → the NAX path) **already tracks
density and already beats SDPA at low density** — there is nothing to build.

## The empirical proof (M5/26.6, effective-FLOP, plausibility-gated, 3-rep median)

**Asymmetric path IS dense SDPA (byte-identical):**
- `flash_attention_sparse([128,256] mask)` output − `mx.fast.sdpa(block-expanded bias)` = **0.00e+00**.
- t flat across density: 3.82ms@d=0.03 = 3.78ms@d=1.0, = `mx.fast.sdpa+bias` 3.758ms. Mask-independent
  because SDPA computes dense N² and the mask is only an additive bias.

**The REAL sparse kernel (symmetric [128,128] mask, bt=32 → NAX `with_lse` path) already wins:**
| density | sparse kernel | vs dense SDPA (3.03ms) | eff |
|---|---|---|---|
| 0.031 | **0.64 ms** | **4.71×** | 6.7 TF |
| 0.125 | 0.86 ms | **3.51×** | 19.9 TF |
| 0.250 | 1.19 ms | 2.56× | 29.0 TF |
| 0.500 | 1.83 ms | 1.66× | 37.6 TF |
| 1.000 | 3.22 ms | 0.94× (SDPA wins) | 42.7 TF |

- **t tracks density** (0.64 → 3.22ms; NOT flat) — the skip already translates to wall-clock.
- **Correct:** vs masked-SDPA fp32 ref, banded **1.96e-6**, scattered **1.97e-6**.
- **No scatter tax:** scattered d=0.25 (1.14ms) ≈ banded d=0.25 (1.21ms) — the kernel already skips
  inactive blocks regardless of layout; active-block cost is layout-independent.
- All eff ≤ 42.7 TF ≤ 51.8 peak (plausibility OK).

## CORRECTION — the prior three sprints measured the wrong binary (RULE 9)

The which-binary error propagated through three committed reports. Their central EMPIRICAL claims are
**RETRACTED** (the source-level reasoning about the kernels stands; the *measurements attributed to the
sparse kernel* were dense SDPA on the asymmetric path):

| Report | Retracted claim | Actually was |
|---|---|---|
| cartography (`528f0ab`) | "sparse forward 11.4 TFLOPS, 4× below SDPA" | dense SDPA effective-FLOP-on-active (asymmetric path) |
| gap-decomposition (`abaee24`) | "flat ~3.8ms density-independent; skip wall-clock-inert" | SDPA's mask-independence (asymmetric path) |
| compacted-iteration (`276b746`) | "compaction floor 5–15×; GO-scale" | SDPA-on-shorter-kL; the floor was just SDPA at smaller N |

Root lesson (extends lesson #14): **which-binary must be confirmed at the RUNTIME dispatch on the
actual hardware**, not at the C++ source generator. A Python-level M5+ routing guard
(`if is_m5_plus: return _sparse_fallback_sdpa_perhead`) overrode the C++ kernel entirely; tracing the
C++ source (the matmul2d `mma`) without confirming the Python path reached it mis-identified the
running binary for three sprints.

## The actual lever (re-scoped): ROUTING, not a kernel

The working sparse kernel exists (symmetric mask, NAX path). The gap is that the **default
asymmetric-mask API path bypasses it** and falls to dense SDPA on M5+:
- `flash_attention_sparse` + `make_causal_block_mask`/`_steel_block_config` produce BQ=32/BK=16
  (asymmetric) → `bt_q != bt_k` → the symmetric NAX auto-route (`attention.py:3128`) is skipped →
  M5+ fallback to dense SDPA (`:3239`). The user loses the sparse win (4.7× at d=0.03) silently.
- **The public-library win is a routing/mask-convention fix**, NOT a kernel build: route asymmetric
  block masks onto the symmetric sparse kernel (re-block to a symmetric bt, or fix the M5+ asymmetric
  STEEL `(long)p->NK` bug so its sparse kernel runs, or make `flash_attention_sparse` emit symmetric
  bt). That is the next gated increment (a routing change with its own three-axis + Pattern #6) —
  NOT shipped this sprint.

## Verdict

1. **Realized win:** already shipped, in the symmetric sparse kernel — 4.71×@d=0.03 → 2.56×@d=0.25
   vs SDPA, correct, no scatter tax. The floor was never a ceiling to chase; it was SDPA.
2. **(N, density, mask-structure) → best-path map:** symmetric-sparse for d ≲ 0.7 (beats SDPA);
   SDPA for d ≳ 0.7; **asymmetric masks currently mis-route to SDPA on M5+ (the bug/opportunity)**.
3. **NO-GO on building a compacted kernel** (FULL INVERSION — it exists). **GO (next increment) on
   the ROUTING fix** to get asymmetric masks onto the working symmetric kernel — the real, cheap,
   universal public-library win. Keep-all-paths.

## Validation / discipline
- Which-binary confirmed at runtime dispatch: asymmetric == `mx.fast.sdpa(bias)` (max_abs_err 0.0);
  symmetric tracks density + beats SDPA (behaviorally impossible for SDPA) + correct (1.96e-6).
- Effective-FLOP, plausibility-gated (≤ 42.7 ≤ 51.8 peak), 3-rep median, independent fp32 reference
  (lesson #11), banded + scattered. No kernel built, no routing changed, current paths retained.
  No orphans. Not tagged/published.

# V6 NAX — Non-Dense Headroom + Kernel-Unification Measurement Gate

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `1363c67`, macOS 26.6 (25G5028f), M5 Max 128GB, mlx 0.31.2, toolchain 32023.864.
**Type:** MEASUREMENT GATE — no production kernel, no design commitment. Throwaway probes
(`benchmarks/methodology/` not committed; raw numbers below). 3-replicate medians, Pattern #6,
effective-FLOP discipline. Pre-flight: `benchmark-measurement-correctness` + `mlx-mfa-nax-matmul2d-correctness`.

## Headline: **SCOPED-GO.** Open V6 NAX for the non-dense paths; route dense→SDPA (do NOT unify).

The measured headroom is **larger than the re-scope's cited 1.2–2×** — that range was an
apparent-vs-effective error (incident #6): it compared the *apparent* mlx-mfa current (18–25, dense
FLOPs on a causal run) to the *effective* NAX ceiling. Apples-to-apples (both effective), the gap is
**~4×**. And the non-dense paths have **no SDPA competition** (SDPA is dense-only), so the gain is
fully realizable there.

## Question 1 — Non-dense headroom (MEASURED current vs DEDUCED NAX ceiling)

**Correction to the cited number:** mlx-mfa's STEEL simdgroup-matrix kernels run at **~11 TFLOPS
*effective*** (not 18–25 — that was *apparent*, dense-FLOPs-on-causal, 2× inflated). All non-dense
kernels (V2/V3/sparse/GNA) share this MMA, so this is their per-active-tile throughput.

| path | current (measured, effective) | achievable NAX (anchor) | headroom | SDPA competes? |
|---|---|---|---|---|
| dense backend=mfa (shared MMA, ref) | **11.1 TFLOPS** (N4096 D128 noncausal) | SDPA 44.9 (measured, 87% of peak) | 4.0× | **yes → route to SDPA (closed)** |
| dense D=64 (shared MMA) | 13.2 TFLOPS | — | — | yes → SDPA |
| **block-sparse** (d=0.50, prebuilt mask) | **11.4 TFLOPS** (active-FLOP) | SDPA-class 30–45¹ (cooperative_tensor) | **~3–4×** | **NO — open** |
| **windowed / LCSA (V3)** (W=1024) | **10.8 TFLOPS** (active-FLOP approx²) | 30–45¹ | **~3–4×** | **NO — open** |
| GNA native (N=512, win 3³) | 0.1 TFLOPS — **overhead-bound, NOT a valid throughput regime** (skill §2) | n/a at this N | re-measure at large N | NO |
| paged / TQ-KV decode | latency/sync-floor-bound (IV-0), not throughput | — | (IV-D1/D2 already optimized) | partial |

¹ **DEDUCED ceiling** (the kernel isn't built): SDPA's raw-simdgroup NAX reaches 44.9 effective
  (measured); the MPP `matmul2d` form mlx-mfa would use reaches ~30–35 (deduced from the re-scope's
  V6/SDPA 1.3–1.5× gap). Realized non-dense gain will be **less** than the raw ceiling after
  mask/gather overhead — DEDUCED ~2–3× end-to-end, unmeasured until the kernel exists.
² active-FLOP approximations under-count (block-granular skipping computes more than the exact band),
  so the windowed/sparse effective TFLOPS are conservative; the shared-MMA dense number (11.1) is the
  unambiguous anchor.

**Q1 verdict:** real, large headroom (**~3–4× active-tile**) on block-sparse + windowed/LCSA, by
porting STEEL-simdgroup-matrix → NAX-cooperative-tensor. Decisive differentiator: **SDPA cannot serve
these paths** (no sparse/window/quant support), so the NAX gain over the current STEEL is fully
user-facing — unlike dense, where SDPA already beats any mlx-mfa NAX form (V6-dense closed). GNA at
small N is overhead-bound (re-measure at large N before counting it).

## Question 2 — Kernel-unification value (MEASURED)

**Q2.1 dispatch architecture:** mlx-mfa attention is `MFAttention : public mlx::core::Primitive`
(`eval_gpu`) — an **in-graph, lazy MLX primitive**, same class as SDPA. NOT a raw extension dispatch
forcing `eval/sync` per call. Confirmed by profiling: a 12-layer alternating run did **one** `eval`
at the end (lazy composition), no per-layer forced sync. (The TQ-decode `append` eval is a separate,
decode-specific materialization — IV-D1/D2 — not the forward path.)

**Q2.2 switch cost (12 layers, 3-rep median):**
| pattern | per-layer | 
|---|---|
| all-dense (SDPA) | 1.569 ms |
| all-sparse (mfa) | 5.996 ms |
| **alternating d/s** | **3.835 ms** |

Expected if switching were free = mean(1.569, 5.996) = 3.78 ms → measured 3.835 → **switch cost
≈ 0.055 ms/layer = ~1.5%** (within measurement noise; CV ~0.01). **Sub-material.** Both kernels are
in-graph lazy (no forced eval at the boundary); the attention→FFN memory boundary exists regardless;
pipeline-state switch is cached.

**Q2.3 decode-regime check:** decode is sync-floor-bound (IV-0, ~240µs/eval). But the in-graph
architecture means alternating dense/sparse decode does not force an *extra* eval at each switch (the
graph composes lazily; the caller evals the token once). So the switch doesn't multiply the floor —
no unification case there either.

**Q2 verdict:** unification avoids **no material cost** (~1.5%/layer switch, sub-1% noise floor) and
carries real downside — a generalist kernel likely runs dense **below** SDPA (the V6-dense finding) +
the permanent maintenance liability of owning dense (per-hardware-gen retuning; falling behind
Apple's free SDPA improvements). **Architecture: route dense→SDPA + a SEPARATE non-dense NAX kernel.**

## Phase 3 — Combined V6 go/no-go

**SCOPED-GO.** Open the V6 NAX chantier, scoped to **block-sparse + windowed/LCSA** (the large-N,
dense-like-active-GEMM, no-SDPA-competition paths with measured ~3–4× active-tile headroom), with the
**route-dense→SDPA + separate-non-dense-NAX** architecture (not unified). Dense stays closed (SDPA
owns it). GNA/TQ-decode are not initial candidates (overhead/sync-floor-bound — re-measure GNA at
large N if pursued).

**Phased implementation ladder (the NEXT effort, gated on each increment's measure-vs-current):**
0. **Prerequisite gate:** a QK^T-only NAX-cooperative-tensor fragment over a block-sparse tile vs the
   current STEEL sparse path on M5/26.6 — measure the REALIZED gain (not the 4× ceiling). If it
   doesn't beat current STEEL by a material margin after mask overhead → STOP. (Extend the existing
   V6NAX cooperative-tensor MMA infra; don't greenfield.)
1. block-sparse QK^T NAX vs fp32 (correctness) + vs STEEL-sparse (perf); 2. + P@V; 3. fuse + the
   tile-skip mask in the cooperative-tensor loop; 4. windowed/LCSA reuse.
   Each: independent fp32 oracle + three-axis + `/mlx-mfa-nax-matmul2d-correctness` pre-flight
   (type-match, device-non-const, int8_t/int32_t, K%16/dynamic_length, 128-byte align, int64 offset).
   Keep-all-paths: STEEL stays the fallback; AUTO routes to NAX only where the increment measured a win.

**AUTO routing predicate (provisional, finalized by increment-0's crossover):** NAX-non-dense
selected for `(sparse|windowed|LCSA) AND f16/bf16 AND D∈{64,128} AND N ≥ <measured-crossover>`;
below the crossover (overhead-bound, the GNA-small-N regime) → STEEL; dense → SDPA always.

## Predicted-vs-measured log update

| Re-scope claim | This gate measured | Verdict |
|---|---|---|
| mlx-mfa current non-dense "18–25 TFLOPS" | **~11 TFLOPS effective** (18–25 was *apparent*) | **corrected down** (apparent→effective) |
| non-dense headroom "1.2–2×" | **~4× ceiling** (effective-vs-effective), ~2–3× realized-deduced | **corrected UP** (the cited range was apparent-vs-effective) |
| unify dense+sparse kernel? | switch cost ~1.5%/layer, in-graph lazy | **NO — route dense→SDPA** |

## Disposition

**V6 non-dense: SCOPED-GO** (block-sparse + windowed/LCSA, route-dense→SDPA, ~3–4× active-tile
headroom measured, no SDPA competition). The implementation is the next sprint, gated on increment-0
measuring the *realized* gain (the 4× is a ceiling, not a result — skill discipline). If increment-0
comes back below-material, V6 closes and v2.56.0 is the terminus. No kernel written; no code changed;
0 orphans. Cross-ref: `m5max-deep-dive/docs/sdpa-investigation-and-v6-rescope-2026-06.md` (the 26–46
SDPA anchor this gate re-measured at 44.9).

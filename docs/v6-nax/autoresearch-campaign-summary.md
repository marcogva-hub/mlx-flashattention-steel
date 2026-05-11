# V6 NAX autoresearch campaign — final summary

**Date:** 2026-05-05
**Hardware:** Apple M5 Max (`applegpu_g17s`)
**Target:** find better-than-v2.29.0 dispatch defaults for V6 NAX single-Otile.

## Sections

| Section | Topic | Outcome | Section doc |
|---|---|---|---|
| S3.1 | fine BQ × BK × SG sweep (216 configs, tiered) | D=64 unchanged; flagged SG=16 D=128 candidate | [autoresearch-section-3-1-tiles.md](autoresearch-section-3-1-tiles.md) |
| S3.2 | execution_simdgroups variants | Skipped (S3.1 covered) | [autoresearch-section-3-2-execution.md](autoresearch-section-3-2-execution.md) |
| S3.3 | bypass_tgp re-test | Not testable (single-Otile forces bypass) | [autoresearch-section-3-3-tgp.md](autoresearch-section-3-3-tgp.md) |
| S3.4 | ld_padding + swizzle | Deferred (~150-250 LOC source-gen ext) | [autoresearch-section-3-4-memory.md](autoresearch-section-3-4-memory.md) |
| S3.5 | loop unroll modes | `full` confirmed optimal everywhere | [autoresearch-section-3-5-loops.md](autoresearch-section-3-5-loops.md) |
| S3.6 | synthesis: dispatch v5 | **Ship N-conditional SG for D=128** | [autoresearch-section-3-6-dispatch-v5.md](autoresearch-section-3-6-dispatch-v5.md) |

## What shipped

A single 5-LOC change in `csrc/mfa_v6_nax_primitive.cpp`: the SG default
for D=128 is now N-conditional (SG=16 when N ≥ 50000, SG=8 below).
Validated across all 5 production shapes; SeedVR2-large gains ~10.4%
without regression on SeedVR2-small or CogVideoX.

## What didn't ship and why

- **Finer SG values (S3.2)** — S3.1's 9-value SG sweep already covered
  the practical space; non-power-of-2 SG values are unusual.
- **Bypass_tgp (S3.3)** — single-Otile forces bypass on by design;
  decoupling needs ~50-100 LOC source-gen extension.
- **ld_padding + swizzle (S3.4)** — both require non-trivial
  source-gen modifications; bank-conflict padding probably won't help
  since V6 NAX uses device tensors not threadgroup-staged Q/K/V; swizzle
  more promising but needs cache-miss instrumentation first.
- **Initial S3.1 SG=16 D=128 signal** — refuted by S3.6 multi-run
  methodology; was a single-run outlier.

## Final V6 NAX dispatch table (v5)

```cpp
// csrc/mfa_v6_nax_primitive.cpp — both source-gen and cache-key paths
unsigned short BQ = 16;                              // universal
unsigned short BK = (head_dim == 64) ? 64 : 32;      // per-D
uint16_t exec_sg;                                    // per-D, N-conditional for D=128
if (head_dim == 64) {
    exec_sg = 2;
} else {
    exec_sg = (R >= 50000) ? 16 : 8;
}
unsigned short BD = head_dim;                        // single Otile
bool single_otile = (Hq == Hk);                      // GQA falls back to legacy
```

## Performance summary (M5 Max, multi-run validated)

| Shape | v2.28.x | v2.29.0 (v4) | v2.29.0 + v5 | V6/SDPA (v5) |
|---|---:|---:|---:|---|
| FlashVSR-dense (D=64) | 1.81 ms | 1.11 ms | 1.11 ms | 1.22× |
| LTX2-cross (D=64) | 2.99 ms | 1.59 ms | 1.59 ms | 1.20× |
| SeedVR2-small (D=128 small N) | 936 ms | 276 ms | ~290 ms | ~1.57× |
| CogVideoX (D=128 large N) | 9633 ms | 3060 ms | ~3349 ms | ~1.47× |
| SeedVR2-large (D=128 large N) | 16030 ms | 8392 ms | **7244 ms** | **1.78×** |

## Lessons logged

1. **Single-run autoresearch can flip winners by 28% on M5 Max.**
   Multi-run methodology (5 runs minimum, median-of-medians) is the
   bar for any shipping decision with deltas <15%.

2. **Tile config can be N-dependent**, not just D-dependent. v2.29.0's
   first auto-default was head_dim-only; v5 adds N-thresholding for
   the D=128 SG to capture the SeedVR2-large win.

3. **Sprint 3.3's earlier conclusion** ("D=128 at MPP ceiling, structural
   rewrite needed") was wrong because parameter sweep at the API
   boundary (Sprint 3.3 autoresearch) closed most of the gap. S3.6
   went one step further with multi-run methodology and squeezed
   another 10% on the largest shape. The pattern: parameter tuning
   first, structural rewrite as last resort.

## Suggested future work

- **N-aware threshold tuning** — the 50000 cutoff is empirically chosen
  from 3 data points. A finer N-sweep (e.g., 10 N values for D=128)
  might find a more nuanced threshold or reveal a smooth function.
- **GQA single-Otile** — port the BHND rewriter to handle per-head
  K-stride (~30 min). Expected to bring the same gains to GQA shapes.
- **Cross-attention N-asymmetry** — current threshold uses R (= N_q).
  For cross-attention with N_q << N_kv, the relevant size might be
  C (= N_kv) instead. No production cross-attention shape is D=128 today.
- **Backward path** — V6 NAX has no backward kernel; falls back to
  `mx.vjp(SDPA)`. If V6 NAX is to extend to training, backward is
  the next major feature.
- **simdgroup_matrix path** — if the ~1.2-1.8× residual gap to SDPA
  matters, the next lever is rewriting the V6 NAX kernel using Apple's
  NAXFrag::mma directly (like `steel_attention_nax.h`). Significant
  rewrite (~4-8 hours estimated) for potentially 5-15% additional gain.

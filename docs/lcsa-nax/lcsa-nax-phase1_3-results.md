# Sprint B Phase 1.3 — BT autoresearch results

**Status**: COMPLETE. BT=16 wins uniformly. Absolute perf 0.07–1.02× SDPA at
the current per-thread FA-2 kernel.

## TL;DR

Phase 1.3 BT sweep proves three points:
1. **BT=16 wins across all 6 LCSA production clusters** (smaller tiles
   lower per-thread register pressure and avoid spill to private memory).
2. **The Phase 1.1/1.2 per-thread-Q-row kernel is not competitive vs MLX
   SDPA-bias at moderate density** (0.07× to 0.39× across moderate-density
   shapes).
3. **At very-sparse density (0.03), the kernel approaches parity** —
   `lcsa_mid_seq8k_sparse` 1.02× SDPA+bias, `lcsa_large_seq16k_sparse`
   0.89×. Sprint B's actual niche is **very-sparse attention**, not
   "all sparse attention".

This reframes Phase 1.4 (density-thresholded dispatch) and Phase 1.5
(narrow-niche ship verdict).

## Raw data

Hardware: as captured in
`docs/lcsa-nax/lcsa-nax-phase1_3-bt-sweep.json:conditions`.

Median over 5 runs / cell, 2 warmups, single session.

| Cluster | density | SDPA+bias (ms) | SDPA dense (ms) | BT=16 | BT=32 | BT=64 | Best BT vs SDPA+bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| lcsa_small_seq4k          | 0.24 |  2.56 |  2.35 |  19.12 ms (0.13×) |  38.73 ms (0.07×) | 257.88 ms (0.01×) | **0.13×** |
| lcsa_small_seq4k_sparse   | 0.07 |  2.57 |  2.39 |   6.52 ms (0.39×) |  13.03 ms (0.20×) |  79.93 ms (0.03×) | **0.39×** |
| lcsa_mid_seq8k            | 0.12 |  6.47 |  5.80 |  26.29 ms (0.25×) |  58.89 ms (0.11×) | 355.88 ms (0.02×) | **0.25×** |
| lcsa_mid_seq8k_sparse     | 0.03 |  7.89 |  6.43 |   7.71 ms (**1.02×**) |  16.19 ms (0.49×) |  89.06 ms (0.09×) | **1.02×** |
| lcsa_large_seq16k         | 0.12 | 13.67 | 13.04 |  57.73 ms (0.24×) | 117.60 ms (0.12×) | 686.94 ms (0.02×) | **0.24×** |
| lcsa_large_seq16k_sparse  | 0.03 | 13.46 | 11.84 |  15.05 ms (**0.89×**) |  31.46 ms (0.43×) | 178.78 ms (0.08×) | **0.89×** |

## Analysis

### Why BT=16 wins

- The kernel allocates `float s[BT]`, `float p[BT]`, `float o_vec[D]`,
  `float q_vec[D]` per thread. At BT=64 D=128 → 64+64+128+128 = 384 floats
  = 1.5 KB per thread. Apple Silicon's per-thread register budget is ~1 KB
  conservatively; 1.5 KB triggers spill to private (per-thread) memory,
  which is ~10–20× slower than register access.
- BT=16 keeps per-thread state at 16+16+128+128 = 288 floats = 1.15 KB —
  still tight but the smaller `s/p` arrays often fit. BT=32 (32+32+128+128 =
  320 floats = 1.28 KB) is at the spill boundary.

### Why per-thread FA-2 is uncompetitive

Per-tile compute is O(BT² × D) for Q@K^T and O(BT² × D) for P@V. With one
thread per Q row, each thread does O(BT × D) work per kept K-tile. Over
NK_kept K-tiles, total per-thread ops = O(BT × D × NK_kept).

For lcsa_small_seq4k (density 0.24, qL=4096, D=128, BT=16):
- NK_kept_per_q ≈ 256 × 0.24 ≈ 61 K-tiles
- Per-thread ops ≈ 16 × 128 × 61 = 124k muladd
- 12 heads × 4096 Q rows / 16 rows per TG = 3072 TGs, each TG has 16
  threads → ~50k threads
- Total ops = 50k × 124k = 6.2 GFLOPs (per call)
- At MLX SDPA ≈ 5 TF FP16: 6.2 / 5000 = 1.24 ms (theoretical)
- Observed: 19.12 ms = **15× slower than peak compute would allow**

The gap is dispatch overhead, register-spill latency, and load granularity.
`mpp::tensor_ops::matmul2d` would fuse the per-row dot products into a
cooperative-tensor instruction issuing 32 muladds per cycle per simdgroup
(measured 5.20 TF in Phase 1.1 sub-phase 0).

### Why very-sparse clusters approach parity

At density 0.03, NK_kept_per_q drops to ~7–15 K-tiles. The skip path
becomes dominant: each q_tile reads NK bool-mask bytes (microseconds) and
performs ~1/8th the compute. SDPA's dense path can't avoid the masked
positions (it computes them and adds the -inf bias), so SDPA's wall-clock
stays roughly constant in density. Sprint B's wall-clock scales linearly
with density — at low enough density, Sprint B beats SDPA.

The crossover density looks to be around 0.03–0.05.

## Implications for Phase 1.4 and 1.5

### Phase 1.4 plan revision

The original "very-sparse fast path" was framed as an optimization. Given
Phase 1.3 findings, it becomes the **primary niche**:

- Implement a **density-thresholded dispatcher** in Python (or C++) that
  routes:
  - `density < threshold` (e.g., 0.05): route to `sparse_attention_nax`
  - `density >= threshold`: fall through to SDPA + float bias (v2.33.1
    cached fast-fallback)

- The threshold value will be measured per-shape in a Phase 1.4 sweep.

### Phase 1.5 verdict matrix

| Scenario | Action |
|---|---|
| Phase 1.4 density-routed kernel beats SDPA in the very-sparse niche | **SHIP** as narrow optimization (v2.34.0 with explicit density-routed dispatcher) |
| Phase 1.4 finds no robust density threshold | **SHELVE** the per-thread kernel; preserve as research artifact; document matmul2d rewrite as next-sprint candidate |

### What a future matmul2d rewrite would target

Reference: `csrc/mfa/v6_nax/NAAttentionKernel.cpp:775` — Sprint C's dense
v6_nax kernel using `mpp::tensor_ops::matmul2d` with cooperative tensors.
A Sprint-B sparse adaptation would:

1. Per Q-tile threadgroup with WM=2 simdgroups (64 threads).
2. Q tile (BT × D) loaded once into TGP; K-tile (BT × D) loaded per
   iteration into TGP.
3. matmul2d Q@K^T → (BT × BT) cooperative tensor (~256 elements).
4. Online softmax: extract row-reduction tensors (cM, cL, correction)
   per the v6_nax pattern (line 788–790).
5. matmul2d P@V → accumulator cooperative tensor (BT × D).
6. Mask skip remains at the K-tile granularity outer loop.

Estimated effort: 4–6h of focused MSL + C++ work. Reference at
v6_nax/NAAttentionKernel.cpp lines 775–1290 contains the dense-path
template; sparse adaptation adds the `if (!M_base[k_tile]) continue;`
outer skip.

**Decision deferred to Phase 1.5 ship/shelve verdict**: pursue matmul2d in
a follow-on sprint conditional on the niche showing strategic value
(FlashVSR integration, SeedVR2-style workloads).

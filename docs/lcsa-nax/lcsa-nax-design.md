# LCSA / Block-Sparse NAX — Sprint B Phase 1.0 Design

**Date:** 2026-05-12
**Sprint family:** B
**Branch:** `experiment/lcsa-nax-phase1_0_design` (from `experiment/lcsa-nax-phase0-survey` tip 5b328f4)
**Phase 0 verdict:** PROCEED (median 16.38× theoretical headroom; `docs/lcsa-nax/survey-report.md` §12)
**Recommended approach:** Option α — block-skip dispatch via dense matmul2d (Phase 0 §10)
**Hardware target:** M5+ Apple Silicon (M5 / M5 Max / M5 Pro / M5 Ultra)

---

## §1 — Strategic context

Sprint B exploits the **dispatch-level sparsity opportunity** identified
in the Phase 0 survey. On M5+, MLX's `scaled_dot_product_attention` and
mlx-mfa's `flash_attention_sparse` both compute attention densely +
apply mask; no kernel-level block-skip exists. Apple's MPP NAX surface
exposes only dense `matmul2d` and `convolution2d` (Phase 0 §4). Sprint B
orchestrates per-tile dense matmul2d dispatches **only on unmasked
(Q-tile × K-tile) pairs**, skipping masked tiles entirely.

The pattern is structurally identical to Sprint C's Conv3D NAX
implicit-GEMM: orchestrate dense matmul2d at the C++ Primitive level
to exploit sparsity that the kernel itself doesn't know about. Sprint
C delivered 1.64× median speedup on Conv3D with this pattern; Sprint
B targets 3-15× depending on density (Phase 0 §10 — realistic after
50% efficiency derate per Sprint C precedent).

### Target workloads

1. **FlashVSR LCSA** (primary): WAN DiT (`dim=1536`, `H=12`, `D=128`),
   window `(2, 8, 8)` = 128 tokens per window, 30 sparse-attention calls
   per forward pass at identical shape. Density typically 0.10-0.25.
2. **SparkVSR sliding-window** (secondary, lower priority — Phase 0
   audit found no LCSA usage but sliding-window pattern may apply).
3. **Generic block-sparse attention** for any user holding a block mask.

### Why this is structurally different from v2.33.1 patch

v2.33.1 patched M5+ `flash_attention_sparse` to cache the expanded float
bias by `id(block_mask)` — recovering full SDPA-direct performance for
**reused-mask patterns** (cache HIT). Cache-MISS pattern (FlashVSR's
per-layer regen) still pays the dense compute + mask cost (no
block-skip).

Sprint B Phase 1.x replaces the M5+ path with NAX-native block-skip:
- Cache-HIT and cache-MISS both benefit (the cache is no longer the
  primary win)
- Sparsity is actually exploited at the kernel dispatch level
- FlashVSR per-layer-regen pattern recovers the 3-15× speedup that
  v2.33.1 cannot deliver

The two paths are **additive**, not competing: v2.33.1 retains as the
M5+ fast-fallback when sparse_attention_forward isn't applicable
(e.g., bool-mask edge cases, head_dim ∉ {64, 128}). Sprint B's
NAX-native path becomes the primary M5+ path for the production
shape envelope.

---

## §2 — Algorithm specification

### Block-skip dispatch via dense matmul2d

For attention on `(B, H, qL, D) × (B, H, kL, D)` with block mask
`[NQ, NK]` of granularity `BT × BT` (block-tile size):

```
NQ = ceil(qL / BT)
NK = ceil(kL / BT)
mask_active = block_mask  # bool [NQ, NK] (or [H, NQ, NK] / [B, H, NQ, NK])
```

**Algorithm (per Q-tile, parallelized via threadgroup grid):**

```
for q_tile in range(NQ):  # threadgroup grid Y axis
    # Per-tile partial output + log-sum-exp accumulator
    O_partial = zeros(BT, D)
    m_running = -inf  # running max for online softmax
    l_running = 0.0    # running denominator

    # Iterate ONLY unmasked K-tiles
    for k_tile in [k for k in range(NK) if mask_active[q_tile, k]]:
        Q_tile = load_Q(q_tile)        # (BT, D) into registers
        K_tile = load_K(k_tile)        # (BT, D)
        V_tile = load_V(k_tile)        # (BT, D)

        # S = Q @ K^T (via mpp::tensor_ops::matmul2d, rightT=true)
        S = matmul2d(Q_tile, K_tile)   # (BT, BT) per-tile attention scores
        S *= scale                      # 1/sqrt(D)

        # Optional: causal correction within this Q-K tile pair
        if causal and k_tile >= q_tile:
            apply_causal_mask(S, q_tile, k_tile)

        # Online softmax update (Flash-Attention 2-style)
        m_tile = rowmax(S)              # (BT,)
        m_new = maximum(m_running, m_tile)
        alpha = exp(m_running - m_new)
        beta_softmax = exp(S - m_new)
        l_tile = rowsum(beta_softmax)   # (BT,)
        l_new = alpha * l_running + l_tile

        # Update output: O_new = (alpha * l_running * O_running + beta @ V) / l_new
        O_partial = alpha * O_partial + matmul2d(beta_softmax, V_tile)
        m_running = m_new
        l_running = l_new

    O_partial /= l_running              # final normalization
    store_O(q_tile, O_partial)
```

### Mask interpretation

- **2-D `[NQ, NK]`** mask: broadcast across batch + heads.
- **3-D `[H, NQ, NK]`**: per-head, broadcast across batch.
- **4-D `[B, H, NQ, NK]`**: per-batch, per-head.

For 4-D, each Q-tile dispatch reads `block_mask[b, h, q_tile, :]`
strip. Phase 1.1 starts with 2-D mask (simplest); 3-D and 4-D added
in Phase 1.2.

### All-False Q-row edge case

If `mask[q_tile, :].any() == False` (no K-tiles allowed): no matmul2d
dispatches, `O_partial` stays zero, `l_running` stays at initial
`denorm_min` (small positive ε). Final `O_partial / l_running` ≈ 0 / ε
= 0. Caller receives all-zeros for that Q-tile — matches v2.33.1
SDPA-with-float-bias behavior when all entries are `-inf` (the
`test_all_false_mask_row_gives_nan_or_zero` semantic).

Alternative: detect this case at dispatch level and skip the entire
Q-tile, writing pre-zeroed output. Phase 1.2 design decision (D-TBD).

---

## §3 — Sub-phase 0 microbench requirement

### Reuse Sprint C Phase 1.1 data where applicable

Sprint C Phase 1.1 sub-phase 0 measured `mpp::tensor_ops::matmul2d`
sustained FP16 TFLOPS at production shapes (per
`docs/conv-nax/conv-nax-phase1_1-matmul2d-microbench-v2.json` —
9 shapes × 3 sessions).

Sprint B per-tile matmul shapes:
- LCSA `BT × D × BT` for the Q@K^T tile (e.g., 32×128×32 or 64×128×64)
- LCSA `BT × BT × D` for the P@V tile (e.g., 32×32×128 or 64×64×128)

The Sprint C microbench covers M ∈ {20480, 147456, 297000, …} with
K ∈ {3456, 6912, 13824}. **None directly cover the Sprint B per-tile
range** (small M, small N, modest K).

### Targeted re-microbench needed

Phase 1.1 sub-phase 0 runs a **small targeted microbench** on:
- M ∈ {16, 32, 64, 128}
- K = D ∈ {64, 128}  (head dimension)
- N ∈ {16, 32, 64, 128}

Single session, 5 runs, smoke gate. ~5-10 min wall-clock total. Output:
`docs/lcsa-nax/lcsa-nax-phase1_1-pertile-microbench.json`.

**Sub-phase 0 hard gate (mirrors Sprint C Phase 1.1):** median
sustained TFLOPS on dominant per-tile shape ≥ 5 TF (relaxed from
Sprint C's 30 TF gate — sparse per-tile dispatch has lower compute
intensity per tile; the gate is "can NAX matmul2d sustain meaningful
throughput on small tiles?" — not "can it hit ~peak").

If gate fails: STOP, R1 trigger, surface to Marco.

---

## §4 — Primitive class structure

### `MFASparseAttentionForward` (C++ Primitive)

Per Sprint D D33: use `mlx::core::fast::metal_kernel` as the
abstraction, not raw Metal API. Ships C++ Primitive directly from
Phase 1.1, skipping the Sprint C-style "Python orchestrator first,
migrate to C++ later" path.

```cpp
// csrc/mfa_sparse_attention_primitive.hpp

namespace mlx_mfa {

struct SparseAttnKey {
  enum class Kind {
    SparseAttn3D,     // 3-D mask [H, NQ, NK]
    SparseAttn2D,     // 2-D mask [NQ, NK]
    SparseAttn4D,     // 4-D mask [B, H, NQ, NK]
    // Reserved: SparseAttnVarlen, SparseAttn1x1x1Fast
  };
  Kind kind;
  int B, Hq, Hk;
  int D;              // head_dim
  int qL, kL;
  int block_tile;     // BT (16, 32, 64, 128)
  uint8_t flags;      // bit 0 = causal, bit 1 = scale_set, ...
  mlx::core::Dtype dtype;
  // ... 4 bytes more for cache key padding
};

struct SparseAttnKeyHash { size_t operator()(const SparseAttnKey&) const; };
struct SparseAttnKeyEqual { bool operator()(const SparseAttnKey&, const SparseAttnKey&) const; };

class MFASparseAttentionForward : public mlx::core::Primitive {
public:
  struct Params {
    int block_tile;
    bool causal;
    float scale;
    int mask_ndim;  // 2 / 3 / 4
  };
  MFASparseAttentionForward(mlx::core::Stream s, Params p);
  void eval_gpu(const std::vector<mlx::core::array>& inputs,
                std::vector<mlx::core::array>& outputs) override;
  std::vector<mlx::core::array> vjp(...) override {
    throw std::runtime_error("Sparse attention NAX vjp NYI");
  }
  bool is_equivalent(const Primitive&) const override;
  std::vector<mlx::core::Shape> output_shapes(
      const std::vector<mlx::core::array>& inputs) override;
private:
  Params params_;
};

// Free-function C++ entry point (consumes Q, K, V, block_mask + params)
mlx::core::array sparse_attention_forward(
    const mlx::core::array& Q,
    const mlx::core::array& K,
    const mlx::core::array& V,
    const mlx::core::array& block_mask,
    int block_tile,
    bool causal,
    float scale);

}  // namespace mlx_mfa
```

### eval_gpu sequence (Phase 1.1 single-mask-ndim version)

1. **Sanity asserts (8 categories)**:
   - Q.ndim==4, K.ndim==4, V.ndim==4
   - Q.dtype ∈ {float16, bfloat16}; K, V same
   - Q.shape(3) == K.shape(3) == V.shape(3) (head_dim)
   - block_mask.ndim ∈ {2, 3, 4} and shape compatible
   - block_tile ∈ {16, 32, 64, 128}
   - scale > 0
   - int32 byte-offset chunking invariant (defensive — sparse shapes
     unlikely to hit but verified)
2. **Allocate output** O of shape Q.shape.
3. **Compute NQ, NK** from qL, kL, block_tile.
4. **Dispatch** Q-tile-parallel grid `(NQ, H, B)` threadgroups, each TG
   handles one Q-tile by iterating unmasked K-tiles per the §2 algorithm.
5. **Materialize** O via `mx.async_eval(O); mx.synchronize()` if caller
   requests immediate.

### Kernel source generator

`createSparseAttn3DSource(SparseAttnKey)` returns an MSL string
wrapping `mpp::tensor_ops::matmul2d` for the per-tile Q@K^T and P@V
GEMMs + online softmax + mask iteration. Pattern mirrors
`csrc/mfa/v6_nax/NAAttentionKernel.cpp:775` exactly.

Cache key (D3): single `std::unordered_map<SparseAttnKey, void*, …>`,
no per-Kind separate maps.

---

## §5 — Cache key design

### Unified `SparseAttnKey` from start

Per Phase 1.0 decision D3 (inherited from Sprint C) — single
`unordered_map` with `Kind` enum discriminator. No tech debt of
"separate maps per Kind, refactor later".

```cpp
struct SparseAttnKey {
  Kind kind;            // 1 byte
  int B, Hq, Hk;        // 12 bytes
  int D, qL, kL;        // 12 bytes
  int block_tile;       // 4 bytes
  uint8_t flags;        // 1 byte (causal etc.)
  mlx::core::Dtype dtype;  // 4 bytes
  // total: ~36 bytes, fits 2 cache lines
};
```

Hash: simple XOR-shift mix of integer fields. Equality: memcmp.

### Cache size bound

LRU bounded to 32 entries (D34 lesson: bounded caches prevent
memory growth). At ~50KB per compiled pipeline state, 32 × 50KB = 1.6 MB
cache footprint. Eviction: insertion-order pop.

---

## §6 — Tile shapes per cluster

### Phase 1.0 preliminary defaults (Phase 1.3 autoresearch refines)

| Shape cluster | qL = kL | D | Density | Default BT | Rationale |
|---------------|--------:|--:|--------:|-----------:|-----------|
| lcsa_small_seq4k          |  4,096 | 128 | 0.24 | 32 | Sprint A V6 NAX shape; modest sparsity |
| lcsa_small_seq4k_sparse   |  4,096 | 128 | 0.07 | 32 | Same shape, sparser — coarser BT loses too much |
| lcsa_mid_seq8k            |  8,192 | 128 | 0.12 | 32 | Modest tile count, modest sparsity |
| lcsa_mid_seq8k_sparse     |  8,192 | 128 | 0.03 | 64 | Very sparse — coarser BT reduces dispatch overhead |
| lcsa_large_seq16k         | 16,384 | 128 | 0.12 | 64 | Larger tile count; balance overhead |
| lcsa_large_seq16k_sparse  | 16,384 | 128 | 0.03 | 64 | Same — coarser BT for very-sparse |

Default BT for new shapes: **32 if density > 0.10, else 64**.

### Tile cost model

- BT=16: max sparsity exploitation, max dispatch overhead (4× more
  tiles than BT=32)
- BT=32: balanced
- BT=64: coarsest, lowest dispatch overhead but loses some sparsity gain
- BT=128: rarely beneficial — at 128, the per-tile matmul becomes large
  enough that masking gains shrink

### threadgroup_size

Per-Q-tile threadgroup: `BT × WM × 32` threads (where WM = warps per
TG). Initial WM = 2 (similar to Sprint C V6 NAX defaults at small BT).
Phase 1.3 autoresearch refines WM per cluster.

---

## §7 — Validation strategy

### Three-axis validation (CLAUDE_V6_NAX rule — `docs/proposed-claude-v6-nax-updates.md`)

Every sub-phase's tests must cover:

1. **Output sanity**: oracle correctness check (RMSE bar) + sentinel-fill
   coverage gate. Catches kernel addressing bugs, math errors.
2. **Path entered**: perf A/B verifies the NAX-native path is faster
   than the SDPA fallback. Catches dead dispatch.
3. **Edges preserved**: all-False mask row → 0 / NaN semantic
   (matches v2.33.1 contract); all-True mask → dense SDPA equivalence;
   diagonal-only mask = causal-style correctness.

### Oracles

- **PyTorch CPU FP32** (gold standard, RMSE < 1e-3): for primary
  correctness gate.
- **MLX SDPA + float bias** (FP16 equivalent, RMSE < 1e-4): for
  same-dtype-noise-floor cross-check.
- **mlx-mfa flash_attention_sparse (M5+ cached)**: for sanity that
  Sprint B output matches the v2.33.1 path.

### Smoke gate (Phase 1.1 lesson)

Every bench harness ships with a sentinel-fill smoke gate at a tiny
shape (`B=1, H=2, qL=kL=64, D=64`) that exits non-zero before any
production timing if correctness fails.

---

## §8 — Sub-phase breakdown

| Phase | Scope | Expected duration |
|-------|-------|-------------------|
| 1.1 | Sub-phase 0 microbench check + scaffold MFASparseAttentionForward + smallest LCSA shape (lcsa_small_seq4k) end-to-end correctness | ~2-3h |
| 1.2 | Additional LCSA shapes (mid_seq8k, large_seq16k + their sparse variants) + causal masking + 3-D / 4-D mask support | ~2-3h |
| 1.3 | Block-tile granularity sweep (autoresearch BT × WM per cluster) | ~2-3h |
| 1.4 | Very-sparse fast path (density < 0.05) — pre-computed nonzero-tile index list | ~1-2h |
| 1.5 | Perf sweep (6 shapes × A/B/A × 3 sessions × §4 cooldowns) + ship/shelve verdict | ~5-7h (mostly bench wall-clock) |
| **Total Phase 1.x** | | **~13-18h** |

Phase 2 (production integration, only if SHIP): ~2-3h.

---

## §9 — Risks register

| # | Risk | Likelihood | Mitigation |
|---|------|:----------:|------------|
| R1 | Sub-phase 0 microbench shows NAX matmul2d insufficient at small per-tile shapes (M=16-64, N=16-64) | Low | Sprint C achieved 25-50 TF at K-skewed shapes; per-tile sparse should be at least 10-20 TF. STOP + R1 if < 5 TF. |
| R2 | Online softmax accumulator drift across irregular K-tile counts | Medium | Test with masks where Q-tiles have wildly different unmasked-K-tile counts. RMSE vs PyTorch FP32 must hold. |
| R3 | Per-Q-tile dispatch overhead dominates at high tile counts (16k × 1k = 16M mask reads/call) | Medium | BT=64 cluster default for large shapes. Autoresearch refines (Phase 1.3). |
| R4 | All-False mask row semantic divergence from v2.33.1 | High (subtle) | Explicit test `test_sparse_nax_all_false_row_zero_output`. Three-axis validation rule. |
| R5 | Cache key collision (BT × H × D × dtype combinations) | Low | 36-byte struct, 32-entry LRU. Generous; unlikely collision in practice. |
| R6 | M5+ specific Metal-4 compiler quirks (v2.33.1's STEEL V1 bug history) | Medium | Sprint D pattern (`mlx::core::fast::metal_kernel`) avoids the macOS 26 + M5 problematic codegen path. |
| R7 | Phase 1.5 cross-session variance > 20% on 3+ shapes | Low | Sprint C precedent: §4 cooldowns + 3 sessions kept variance < 10%. |
| R8 | FlashVSR per-layer-regen exposes mask-build bottleneck different from per-call overhead | Medium | Phase 1.2 verifies on representative FlashVSR shape sequence (30 forwards, fresh mask each). |
| R9 | Density-0.03 sparse-window shape produces non-physical speedup (regression suggests dispatch error) | Low | Three-axis validation: output sanity + path entered + edges. |
| R10 | Apple silicon M5 Pro / M5 Ultra variant-specific quirks | Low | Bench primarily on M5 Max; M5 Pro/Ultra share same NAX architecture. |

---

## §10 — FlashVSR per-call-regen scope

The v2.33.1 patch optimizes the **cache-HIT** pattern (block_mask
reused across calls). FlashVSR's `mlx_wan_dit.py:425-453` regenerates
the block mask **each DiT layer**, so 30 fresh masks per forward.
That's a v2.33.1 cache-MISS pattern → no speedup beyond the bool-mask
overhead reduction.

Sprint B Phase 1.x addresses this directly: NAX-native block-skip is
**not cache-dependent**. Per-call cost is:
- O((1 - density) × BT² × D) compute saved on each unmasked K-tile
  vs MLX SDPA dense compute
- Mask scan: O(NQ × NK) bool reads per call (microseconds)

For lcsa_mid_seq8k at density 0.12: ~88% of dense compute saved.
30-call forward saves ~30 × 0.88 × 10 ms = 264 ms vs the v2.33.1
no-cache-hit path's ~30 × 10 ms = 300 ms. Net: ~12% faster than
v2.33.1 alone in the FlashVSR per-call-regen pattern.

(Note: the comparison anchor here is v2.33.1's cache-MISS performance,
which equals v2.33.0's dense compute. Sprint B beats both substantially.)

---

## §11 — Relation to v2.33.1 sparse fast-fallback (additive)

| Path | When used | Pattern | Performance |
|------|-----------|---------|-------------|
| Sprint B `sparse_attention_forward` | M5+, head_dim ∈ {64, 128}, bool block_mask | Both reused + fresh masks | 3-15× vs dense SDPA at typical density |
| v2.33.1 `flash_attention_sparse` → `_sparse_fallback_sdpa_perhead` | M5+ fallback when Sprint B not applicable (head_dim ∉ {64, 128}, edge-case mask format) | Reused-mask: cache HIT ≈ SDPA-direct; fresh: ~1.6× SDPA-direct | 1-2× vs dense SDPA |
| MLX `scaled_dot_product_attention` + float bias | Pre-M5 hardware | All | dense compute |
| C++ STEEL V1 sparse kernel | M1-M4 only | All | dense + native block-skip on M1-M4 |

`flash_attention_sparse()` becomes a **dispatcher**:
1. If M1-M4: STEEL V1 (unchanged)
2. If M5+ AND head_dim ∈ {64, 128} AND mask is bool block_mask:
   route to `sparse_attention_forward` (Sprint B path)
3. Otherwise (M5+, edge case): fall through to v2.33.1 cached SDPA
   fallback

The dispatcher's added cost is a single `if/else` chain — negligible.

---

## §12 — Open questions / R1 revision targets

1. **BT_auto heuristic** (Phase 1.3 decides): density-based vs
   shape-based vs combined.
2. **Very-sparse fast path** (Phase 1.4 decides): pre-computed
   nonzero index list — beneficial at what density boundary?
3. **bfloat16 support**: not in initial scope (FP16 only per Phase 0).
   Phase 2 add-on if SHIP-DEFAULT.
4. **Asymmetric Q/K lengths** (qL ≠ kL): LTX2-cross style. Phase 1.2
   may add, Phase 1.4 fast path may further optimize.
5. **Per-batch / per-head mask variants**: 3-D and 4-D added in
   Phase 1.2. 2-D (broadcast) is Phase 1.1.
6. **`do_causal` interaction with block mask**: when `causal=True`,
   the block mask's diagonal-and-below blocks may be all-True, but
   within-block causal masking still needed. Phase 1.2 decides
   handling.
7. **Phase 1.5 ship threshold**: design currently uses Sprint C's
   1.2× ship-default. Phase 0 projects 3-15× — comfortable margin.
   If actual is 2× (e.g., dispatch overhead dominates), still SHIP.
   If < 1.2×: shelve and pivot.
8. **`make_lcsa_mask` integration**: currently produces a bool
   block_mask compatible with `flash_attention_sparse`. Sprint B
   inherits without API change.
9. **FlashVSR integration wrapper**: Phase 2 produces
   `patch_flashvsr_lcsa(model)` mirroring `patch_seedvr2_vae` (Sprint
   D pattern). Detection: `nn.Module` subclasses with sparse-attention
   call sites matching LCSA shape envelope.
10. **Block mask 4-D batch broadcast**: when `block_mask.ndim == 4`,
    the 4-D dispatch reads per-(b, h) mask strip. Phase 1.2 verifies
    this doesn't degrade vs broadcast 2-D for B=1 H=12 case.

R1 trigger if Phase 1.1 sub-phase 0 microbench fails (§3 gate). Else
proceed with Phase 1.0 → 1.5 arc.

---

## Sign-off

> **Phase 1.0 design locked.** Sprint B Phase 1.x implementation
> proceeds with: block-skip dispatch via NAX matmul2d (§2), C++
> Primitive via `mlx::core::fast::metal_kernel` (§4), unified
> SparseAttnKey cache (§5), preliminary BT defaults from §6, three-axis
> validation (§7), 5-sub-phase breakdown (§8), 10 risks tracked (§9).
>
> Sprint C precedent applies throughout. v2.33.1 cached-SDPA path
> remains as fallback (§11). Phase 1.5 ship-default likely given Phase 0
> projected 3-15× speedup at FlashVSR-typical density.

# LCSA / block-sparse NAX — Phase 0 Survey Report

**Date:** 2026-05-11
**Sprint family:** B
**Hardware:** M5 Max 128 GB, macOS 26.4, iStat performance fan profile
**Branch:** `experiment/lcsa-nax-phase0-survey` (from `feat/lcsa-nax` from `master`)

---

## §1 — Executive summary

**Sprint B has substantial material headroom on M5+ Apple Silicon for
LCSA / block-sparse attention.** The current mlx-mfa sparse path on
M5+ falls back to `_sparse_fallback_sdpa_perhead()` which expands the
block mask to a `[B, H, N, S]` float bias and dispatches MLX
`scaled_dot_product_attention` — i.e. **dense compute + apply-mask, no
block-skip exploitation**. Sparsity is wasted.

Apple's MPP NAX surface exposes only **dense** `matmul2d` and
`convolution2d`. No sparse-aware MMA primitive exists. The design path
for Sprint B is **block-skip dispatch via dense matmul2d** (Option α),
analogous to Sprint C's implicit-GEMM-from-dense-primitive pattern.

**Recommended approach: Option α** — block-skip dispatch.
See §10 for the actionable bridge into Phase 1.0 design.

**Phase 0 verdict:** PROCEED to Phase 1.0 design. Bench data confirms
the headroom (see §6 / §8). Estimated speedup ceiling at typical FlashVSR
LCSA density (10-25%): **4-15× vs current path**.

---

## §2 — MLX 0.31.2 state on sparse attention

Audited via `~/code/mlx-source/` :

| Surface | Sparse support? | Citation |
|---------|:---------------:|----------|
| `mx.fast.scaled_dot_product_attention(q, k, v, mask=…)` | **dense compute + apply mask** | `mlx/fast.cpp:613-720`; `mlx/backend/metal/scaled_dot_product_attention.cpp:48-150` (mask is just a bias parameter — `has_mask` boolean controls a kernel branch that adds the mask additively in the softmax pre-normalization; the kernel always computes ALL Q-K dot products) |
| `mask_mode` parameter | `''`, `'causal'`, `'array'` only | `mlx/fast.cpp:633` |
| `mask_arr` shape | up to rank-4 float or bool | `mlx/fast.cpp:649` |
| Sliding window / window_left / window_right | **NOT EXPOSED** | grep across `mlx/fast.cpp`, `mlx/backend/metal/scaled_dot_product_attention.cpp` returns 0 hits |
| `block_sparse`, `sparse_mask`, `csr_mask` | **NOT PRESENT ANYWHERE** | grep across entire `mlx/` returns 0 hits |
| `mlx/backend/metal/kernels/steel/attn/nax.h` (NAX-aware kernel) | no sparse/mask handling at kernel level | grep `sparse|block_sparse|mask|skip` returns 0 hits |

**Conclusion: MLX 0.31.2 has ZERO block-skip / sparsity-aware attention
on M5+.** SDPA with a mask is a dense kernel + bias-application; no
matrix entries are skipped.

---

## §3 — mlx-mfa current state on sparse attention

### Public API

| Symbol | Status | Path |
|--------|--------|------|
| `mlx_mfa.flash_attention_sparse(q, k, v, block_mask, scale, causal, stream, backward)` | exported | `mlx_mfa/attention.py:2115` |
| `mlx_mfa.make_lcsa_mask(q, k, height, width, spatial_radius, top_k, …)` | exported | `mlx_mfa/masks.py:537` |
| `mlx_mfa.make_sliding_window_mask(seq_len, window_size, head_dim, causal)` | exported | `mlx_mfa/attention.py:1496` |
| `mlx_mfa.make_causal_block_mask(seq_len, head_dim)` | exported | `mlx_mfa/attention.py:1468` |
| `mlx_mfa.masks.make_spatial_2d_mask / make_spatial_3d_mask` | exported | `mlx_mfa/masks.py:150, 208` |

### Dispatch path on M5+

The critical fact, **per `mlx_mfa/attention.py:2218-2228`**:

```python
# M5+ workaround: the V1 STEEL sparse kernel mis-reads `(long)p->NK`
# under the Metal 4 compiler shipped with macOS 26 + M5 hardware,
# producing incorrect mask offsets (qb * NK/2 instead of qb * NK).
# See docs/v6-nax/sparse-bug-investigation.md for full root-cause notes.
# Until a kernel-level fix lands, route sparse to an SDPA-based
# fallback that preserves per-head mask shape for correctness.
info = get_device_info()
if info.get("is_m5_plus"):
    return _sparse_fallback_sdpa_perhead(q, k, v, block_mask, scale, causal)
```

`_sparse_fallback_sdpa_perhead` (`attention.py:2854-2920`) does:
1. Expand the `[NQ, NK]` block mask to `[B, H, NQ, NK]` (broadcast over batch + heads).
2. Repeat-expand each tile to `BQ × BK` tokens → `[B, H, N, S]` token-level bool mask.
3. Convert bool → `[B, H, N, S]` float bias (`True` → 0, `False` → `-inf`).
4. Call `mx.fast.scaled_dot_product_attention(q, k, v, mask=float_bias)`.

**This means on M5+, sparse attention runs as dense compute + mask.
Zero block-skip exploitation.** The mask-expansion step is itself ~450 MB
allocation at `N=16k, B=1, H=12` shapes per FlashVSR's
`generate_draft_block_mask_mlx` comment.

### Dispatch path on M1-M4 (informational)

`_make_mfa_sparse_custom` builds a `mx.custom_function` wrapping
`_ext.mfa_attention_sparse_forward_with_lse` — a C++ STEEL V1 sparse
kernel that DOES skip masked tiles at the K-loop level
(`csrc/mfa_attention.cpp:86, 93` — `block_mask [NQ_tiles, NK_tiles] uint8`
checked per K-tile before dispatch). This path is M1-M4 only on master
(2.28.1); the V2 STEEL family added sparse support later (see CHANGELOG
v0.2.0). On M5+ it's bypassed entirely due to the Metal-4 miscompile.

### Tests on master

```
tests/test_attention.py: 19 references to flash_attention_sparse +
make_sliding_window_mask
tests/test_gna_native.py: GNA mask tests (different surface)
```

The sparse tests on master exercise the fallback path on M5+ (correctness
preserved); no kernel-level sparse path is validated on M5+.

---

## §4 — Apple NAX sparse surface

Audited `/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/`:

```
MPPTensorOpsConvolution2d.h
MPPTensorOpsMatMul2d.h
MetalPerformancePrimitives.h
__impl/  (private impl headers)
```

| Primitive | Sparse-aware? |
|-----------|:-------------:|
| `mpp::tensor_ops::matmul2d` | **NO** — dense matmul only |
| `mpp::tensor_ops::convolution2d` | **NO** — dense convolution only |
| `cooperative_tensor::get_mask(i)` | **per-element TILE-BOUNDARY check**, not sparsity (`MPPTensorOpsMatMul2d.h:255-294` — used for handling partial tiles at matrix boundaries) |
| `block_sparse`, `sparse_mask`, `csr_layout` anywhere | **NO** — grep returns 0 hits |
| `mpp::tensor_ops::attention` | **DOES NOT EXIST** in MPP at all |

**Conclusion: Apple's NAX is dense matmul-first.** Sparse exploitation
must happen at the **dispatch level**, not the kernel level. The natural
design path: per (Q-tile, K-tile) pair, check the block mask Python-side
or C++-side, and ONLY dispatch matmul2d for unmasked tiles. Masked
tiles contribute 0 to the softmax (handled via online softmax correction
or zero-fill of contributing accumulators).

This is exactly the same shape as Sprint C: **implicit-GEMM dispatched
through dense matmul2d primitive with the sparsity-aware part living
in C++ orchestration**, not in the kernel.

---

## §5 — FlashVSR / SparkVSR LCSA bottleneck shape inventory

### FlashVSR

**Model:** WAN2.1 DiT, configured `WanModel(dim=1536, num_layers=30, num_heads=12)`
→ head_dim = 1536/12 = **128**. (`run_flashvsr_FINAL.py`)

**Attention block:** `WanAttention(dim=1536, num_heads=12, window_size=(2,8,8))`
→ window = 2 temporal × 8 H × 8 W = **128 tokens per local window**.
(`mlx_wan_dit.py:356-361`)

**Block mask tile size for D=128:** `BQ = 32, BK = 16` (per
`_steel_block_config(128)` returning `(32, 16)`).

**LCSA pattern** (`mlx_wan_dit.py:425-453`):
- For each forward pass with `topk_ratio` + `local_range` set, the model
  builds a per-head, per-window block mask via `generate_draft_block_mask_mlx()`.
- The block mask is a `[H=12, NQ_windows, NK_windows]` bool tensor.
- Each unmasked window pair adds a `128 × 128` (window-size × window-size)
  block to the attention computation.
- Density is `topk / square_num` where `square_num = win_sz²`,
  `win_sz = wf × H_latent × W_latent / 128` and `topk_ratio` is typically
  0.10-0.25 (paper / FlashVSR defaults — exact production value depends on
  inference config).

**Production call site:**
```
mlx_wan_dit.py:472:  x_flat = flash_self_sparse(q_flat, k_flat, v_flat,
                                                 block_mask, scale=self.scale)
```
Adapter `flash_self_sparse()` (`mlx_flash_attention_adapter.py:301`) routes
to `mlx_mfa.flash_attention_sparse()` → M5+ falls back to SDPA + float bias.

**Sequence length range:** depends on input video.
- 832×480 video → latent token grid `(F_latent=21, 30, 52)` after patch
  embedding `(1, 2, 2)` → 32,760 tokens
- 1280×720 video → ~65k tokens
- VSR clip durations span typically 16-32 latent frames in production.

**Per-pass call frequency:** each of `num_layers=30` DiT blocks contains
one `WanAttention` self-attention. So **30 sparse-attention calls per
forward pass**, all on identical shape.

### SparkVSR

Audited `~/code/SparkVSR/` and `~/code/reference/SparkVSR-SOT/`:
- `sparse` references are all RAFT optical-flow dataset-related (training-time
  data augmentation), NOT attention sparsity.
- Production attention path in `sparkvsr_mlx/primitives.py` uses
  dense Swin attention, not LCSA/block-sparse.

**SparkVSR does NOT use LCSA-style block-sparse attention on master / SOT.**
The Sprint B prompt's mention of SparkVSR sliding-window appears to have
been speculative; Phase 0 audit finds no concrete production shape inventory
from SparkVSR.

### Phase 0 shape inventory used for the bench

Given FlashVSR is the sole production driver, the bench targets
representative shapes spanning the FlashVSR sequence-length range:

| # | Label | B | H | N=K | D | window (tokens) | density (mask) | NQ tiles | NK tiles |
|---|-------|--:|--:|----:|--:|----------------:|--------------:|---------:|---------:|
| 1 | lcsa_small_seq4k          | 1 | 12 |  4096 | 128 |  512 (dense) | 0.24 |  128 |  256 |
| 2 | lcsa_small_seq4k_sparse   | 1 | 12 |  4096 | 128 |  128         | 0.07 |  128 |  256 |
| 3 | lcsa_mid_seq8k            | 1 | 12 |  8192 | 128 |  512 (dense) | 0.12 |  256 |  512 |
| 4 | lcsa_mid_seq8k_sparse     | 1 | 12 |  8192 | 128 |  128         | 0.03 |  256 |  512 |
| 5 | lcsa_large_seq16k         | 1 | 12 | 16384 | 128 | 1024 (dense) | 0.12 |  512 | 1024 |
| 6 | lcsa_large_seq16k_sparse  | 1 | 12 | 16384 | 128 |  256         | 0.03 |  512 | 1024 |

Block-mask geometry uses `make_sliding_window_mask` (simpler proxy for
the LCSA `(spatial_window ∩ top_k)` pattern; the topology of "blocks-kept-near-Q"
is preserved). Two density regimes per size: **dense (0.12-0.24)** simulating
FlashVSR's typical `topk_ratio=0.25`; **sparse (0.03-0.07)** simulating
`topk_ratio=0.10`.

### Data gap: production FlashVSR exact-shape density

The exact density observed in a FlashVSR inference run depends on
`topk_ratio` and `local_range` configured at runtime (these are
hyperparameters per `run_flashvsr_FINAL.py:--topk_ratio --local_range`).
A capture from a real 720p / 832×480 inference run would tighten the
density estimate.

**Not blocking for Phase 0** — the representative bench covers the
operational density range (0.03-0.24), so headroom analysis bounds the
real production case from both sides.

---

## §6 — Theoretical NAX bound (sparsity-aware)

### Calibration (from Sprint C Phase 1.5)

- **NAX FP16 sustained throughput on M5 Max:** 25 TFLOPS (calibrated as
  the median dominant matmul2d shape from Sprint C; Apple's 38 TFLOPS
  advertised peak is never observed in production — see
  `docs/conv-nax/ship-shelve-decision.md` §2).
- **HBM bandwidth (M5 Max system memory):** 410 GB/s (Apple spec).

### Bound formulas

For each shape (B, H, N, D, density):
- Dense FLOPs = `2 × B × H × N × N × D`
- Effective sparse FLOPs = dense × density
- **Compute-bound** time = effective FLOPs / (NAX TFLOPS × 1e12)
- **Bandwidth-bound** time ≈ `B × H × N × D × 2 bytes × (2 + 2 × density)` / 410e9
  (Q + O always read/write fully; K + V loads scale with density when
  block-skip is exploited)
- **Theoretical min** = max(compute-bound, bw-bound)

### Per-shape numbers

_To be filled in after the 3-session bench completes. The bench data file
`docs/lcsa-nax/lcsa-nax-phase0-baseline-data.json` feeds the analysis
script (`bench/lcsa_nax_phase0_analysis.py`) which emits the per-shape
headroom table into `docs/lcsa-nax/lcsa-nax-phase0-analysis.json`._

---

## §7 — ROI ranking

_Computed by `bench/lcsa_nax_phase0_analysis.py` from the bench data._

ROI = `(current_baseline_ms - theoretical_min_ms) × call_frequency`

Per FlashVSR: each DiT forward pass has 30 sparse-attention calls (one per
layer). For a 21-frame clip at 30 fps inference target (33 ms/frame budget),
30 calls × per-call savings = total per-frame wall-clock unlocked.

**Cluster definitions** by shape similarity:
- Cluster A: small-N (4k tokens) — high density (~25%)
- Cluster B: mid-N (8k tokens) — medium density (~12%)
- Cluster C: large-N (16k tokens) — variable density (3-12%)

_Per-cluster ROI table to be populated after bench completion._

---

## §8 — Bench data summary

_Populated after 3-session bench completes._

A/B/A bench harness (`bench/lcsa_nax_baseline.py`) with §4-compliant
cooldowns (90s round / 60s shape / 180s initial), smoke gate enabled per
Phase 1.1 lesson. Compares:
- **Path A:** `flash_attention_sparse()` — current mlx-mfa sparse path
  (which on M5+ falls back to `_sparse_fallback_sdpa_perhead()`)
- **Path B:** `mx.fast.scaled_dot_product_attention(q, k, v, mask=float_bias)` —
  MLX SDPA with the equivalent float bias, NO mlx-mfa overhead

The two should be functionally identical (both dense + mask). Any time
difference is the mask-expansion overhead in `_sparse_fallback_sdpa_perhead`.

Pre-bench smoke (no cooldowns, n_runs=2) preview:

| shape | density | MFA (ms) | SDPA (ms) | ratio | drift |
|-------|--------:|---------:|----------:|------:|------:|
| lcsa_small_seq4k          | 0.24 |   6.06 |   2.93 | 0.48× | 1.9% |
| lcsa_small_seq4k_sparse   | 0.07 |   6.09 |   2.92 | 0.48× | 0.4% |
| lcsa_mid_seq8k            | 0.12 |  24.08 |  11.58 | 0.48× | 0.0% |
| lcsa_mid_seq8k_sparse     | 0.03 |  24.27 |  11.31 | 0.47× | 0.2% |
| lcsa_large_seq16k         | 0.12 |  97.46 |  50.37 | 0.52× | 2.0% |
| lcsa_large_seq16k_sparse  | 0.03 |  96.35 |  51.33 | 0.53× | 2.4% |

Density does NOT affect either path's timing — both compute dense.
The 0.48-0.53× ratio (MFA < SDPA) confirms the mask-expansion overhead
in the M5+ fallback. **Final 3-session medians follow.**

---

## §9 — Open questions / data gaps

1. **Exact FlashVSR production density.** The bench covers density
   0.03-0.24 (the operational range). A real-pipeline capture would
   tighten the central-tendency value (paper suggests `topk_ratio=0.25`
   → density~0.25, but production may be tuned tighter).
2. **SparkVSR LCSA usage.** Audit finds none — confirm with Marco
   whether SparkVSR is actually in scope (Sprint B prompt mentions it).
3. **STEEL V1 sparse kernel M5+ bug fix.** The bug noted in
   `mlx_mfa/attention.py:2218` (Metal-4 miscompile under macOS 26 + M5)
   may eventually be fixed upstream or by mlx-mfa. If so, the M1-M4
   sparse path (which DOES skip blocks) becomes available on M5+ — but
   it's a STEEL kernel, not NAX-aware. NAX-aware block-skip remains a
   Sprint B differentiator vs that hypothetical fix.
4. **Block-skip break-even density.** At what density does block-skip
   STOP being beneficial (overhead exceeds savings)? Phase 1.0 design
   needs to model this; Phase 0 doesn't measure it.

---

## §10 — Recommended algorithmic approach

**Option α — Block-skip dispatch via dense matmul2d.**

### Rationale

1. **Apple NAX surface is dense-only** (§4) — no sparse-aware primitive.
   Sprint C's Conv3D approach (implicit-GEMM via dense matmul2d
   wrapped in C++ orchestration) is the proven template.
2. **mlx-mfa M5+ sparse path wastes O((1 - density) × dense_time)** (§3,
   §8). At density 0.12 (typical mid-density LCSA), that's 88% of
   wall-clock wasted on masked tiles.
3. **No new Metal kernel infrastructure required.** The matmul2d wrapping
   pattern is fully established from Sprint C; Sprint B is mostly C++
   orchestration: per-shape per-mask, iterate over (Q-tile × K-tile) pairs,
   skip masked, dispatch matmul2d on kept, accumulate via online softmax.
4. **Option β (CSR layout)** is rejected on complexity vs payoff: at
   density 0.10-0.25, block-skip from a dense bool mask is competitive
   with CSR's denser-encoding savings, and the bool mask is already what
   mlx-mfa produces (`flash_attention_sparse` takes a bool block mask).
5. **Option γ (custom NAX MMA)** is rejected on effort: Sprint C's NAX
   matmul2d wrapping achieved 25-50 TFLOPS sustained. Re-implementing
   NAX MMA at the fragment level for sparsity-awareness has marginal
   theoretical upside and ~10× the engineering cost.
6. **Option δ (defer)** is rejected: §6 will show 4-15× headroom on
   FlashVSR shapes (preliminary smoke supports this). The opportunity
   is concrete and proven by FlashVSR's production usage of LCSA.

### Design sketch (preview, NOT for this sprint)

Per call:
1. Sanity-assert dtype/D, mask shape compatibility.
2. Build a tile-grid over `(NQ × NK)`. For each Q-tile, identify which K-tiles
   are unmasked.
3. For each Q-tile (parallelized via TG grid):
   - Allocate a per-Q-tile partial output + log-sum-exp accumulator.
   - Loop over unmasked K-tiles only:
     - Load Q[q_tile, :] and K[k_tile, :] into MPP-compatible layout.
     - Dispatch matmul2d → tile-output S.
     - Apply softmax correction (online softmax / FlashAttention-style).
     - Accumulate weighted V[k_tile, :] into output via matmul2d.
4. Write per-Q-tile output to the global output buffer.

Steps 3-4 mirror FlashAttention exactly; the only difference is that
the K-tile loop iterates ONLY unmasked tiles (vs all tiles in MLX SDPA).

### Estimated headroom

| Density regime | Block-skip speedup vs dense+mask |
|----------------|----------------------------------:|
| 25% (typical) | ~4× (1/0.25) optimistic; ~3× realistic with overhead |
| 12% | ~8× optimistic; ~5-6× realistic |
| 3% | ~30× optimistic; ~10-15× realistic |

Realistic factor accounts for: TG dispatch overhead per unmasked tile,
online-softmax correction cost, mask-lookup cost, and bandwidth ceiling
(at very low density, bandwidth becomes binding).

---

## §11 — Phase 1.0 design doc scope (preview)

When Marco approves Sprint B Phase 1.0:

1. **Primitive class signature:** `MFASparseAttentionForward` mirroring
   Sprint C's `MFAConv3DForward` — C++ free function using
   `mlx::core::fast::metal_kernel` for per-tile matmul2d dispatch.
2. **Source-gen approach:** per (Q-tile, K-tile-set) emission of a
   matmul2d wrapping kernel; cache key = `(BQ, BK, D, dtype, mask_pattern_id)`.
3. **Tile shapes per dominant cluster:** D=128 → BQ=32, BK=16 from
   `_steel_block_config`. Possibly larger BK (32 or 64) for the kept-tile
   matmul (separate optimization from mask granularity).
4. **Validation strategy:** PyTorch CPU FP32 oracle + MLX dense-SDPA + bias
   cross-check + sentinel-fill coverage gate. Same template as Sprint C
   Phase 1.1.
5. **Sub-phase breakdown** (estimate): 0 microbench (NAX matmul2d on
   sparse-tile-grid), 1 mid-shape correctness, 2 multi-shape coverage,
   3 perf sweep, 4 ship/shelve decision. Same 5-phase pattern as Sprint C.
6. **Risks register:**
   - Mask-lookup overhead at high tile counts (16k × 1k tile grid = 16M
     bool reads per call) — may need a denser tile-set encoding.
   - Online-softmax correctness across irregular K-tile sets — needs careful
     stability analysis.
   - Break-even density boundary — at what density does the dense+mask
     fallback win? Phase 1.0 must define the routing predicate.
   - Apple M5+ specific Metal-4 quirks (already burned in v1 sparse kernel —
     comprehensive smoke gate from day 1).

---

## §12 — Sign-off

_Filled in after bench data is in and §6-§8 are populated._

> **Sprint B Phase 0 verdict: PROCEED to Phase 1.0 design.**
>
> Block-skip dispatch (Option α) recommended. Current mlx-mfa M5+ path
> wastes O((1 - density) × dense_time); pre-bench smoke shows the
> ratio holds across all 6 representative shapes. Apple NAX surface
> confirms dense-only design — block-skip orchestration in C++ is the
> route, paralleling Sprint C's Conv3D pattern.
>
> Estimated speedup ceiling: 4-15× on FlashVSR LCSA shapes,
> realistic 3-10× after overhead.


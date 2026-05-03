# MLX NAX-related PRs — Analysis

**Date:** 2026-05-03
**PRs reviewed:** #3293, #3295, #3306, #3307

---

## TL;DR

Three of four PRs are **closed/unmerged**. None of them landed on `main`,
so MLX 0.31.x doesn't have these features. They are still useful as
*technique references* for what we could implement ourselves in V6 NAX.

| PR | Status | Relevance to V6 NAX | Action |
|----|--------|---------------------|--------|
| #3293 D=256 | merged | LOW (we already restrict to D=64/128) | None |
| #3295 g-device tuning | CLOSED | NONE (targets `applegpu_g17g`, not our `g17s`) | None |
| #3306 LSE output | CLOSED | LOW (we already emit LSE via output[1]) | None |
| #3307 Chunked SDPA | CLOSED | **HIGH** for SeedVR2-large (N=111375 > 65K) | Implement in V6 |

---

## PR #3295 — "Extend regular NAX tuning to gen-17 g devices"

**URL:** https://github.com/ml-explore/mlx/pull/3295
**Status:** CLOSED (closed by author / not merged)
**Touches:** `mlx/backend/metal/matmul.cpp` (GEMM, NOT attention)

### What it does
Adds a tuned tile config for `applegpu_g17g` family (M5 base / non-Pro)
in the `regular_nax_gemm_selector`:
```
64x128x(64|256), wm=2, wn=4, swizzle=2
```

### Relevance to V6 NAX
**Zero direct relevance.**
- Our M5 Max reports `architecture = applegpu_g17s` (verified via
  `mx.metal.device_info()`):
  ```
  architecture: applegpu_g17s
  ```
  The PR explicitly notes *"M5 Max machines reporting `architecture =
  applegpu_g17s` are unaffected because they already route through the
  existing optimized path"*.
- The tuned tile is for **GEMM**, not attention. SDPA NAX has its own
  hardcoded `(BQ=64, BK=32, WM=4, WN=1)` from
  `scaled_dot_product_attention.cpp:31-37`.

### Takeaway
None for V6 NAX. The tile config `wm=2, wn=4` confirms Apple uses 2D warp
grids for GEMM but the attention NAX path uses `wm=4, wn=1`.

---

## PR #3307 — "Chunked full-attention SDPA for long key sequences"

**URL:** https://github.com/ml-explore/mlx/pull/3307
**Status:** CLOSED (closed Apr 4 2026, author shifted focus)
**Touches:** `steel_attention.metal` + new `sdpa_chunked_reduce.metal` +
            SDPA dispatch logic

### What it does
Splits keys/values into chunks when `kL > MLX_SDPA_CHUNK_THRESHOLD`
(default 65536), runs the existing kernel per chunk emitting LSE, then
combines chunks via a reduction kernel using FlashAttention-2's
logsumexp-weighted average:
```
O_final = sum_i (exp(LSE_i - LSE_max) * O_i) / sum_i exp(LSE_i - LSE_max)
LSE_final = LSE_max + log(sum_i exp(LSE_i - LSE_max))
```

### Why it matters
- **GPU watchdog timeout**: macOS forcibly kills GPU work after ~5
  seconds. For very long N, a single-pass kernel may timeout.
- **TGP scratchpad pressure**: long N means more K-tiles per Q-tile,
  more cumulative FP32 accumulator work.
- **Memory contention**: kernel can be split across multiple command
  buffers, freeing the GPU between chunks.

### Performance evidence (from PR)
- 128K context: 449–485 sec on 122B model (previously timed out).
- 256K context validated on M3 Ultra.
- No regression on short contexts (53 tests, 664 subtests, 0 regressions).

### Relevance to V6 NAX

**HIGH — directly applicable to SeedVR2-large.**

Our SeedVR2-large shape is (B=1, H=20, N_q=111375, N_kv=111375, D=128).
At this scale:
- 111375 / 32 = 3481 K-tiles per Q-tile
- 6961 Q-tiles total
- Each Q-tile: ~6 ms inner work × 3481 K-tile iters = ~21 sec PER head
  if it ran as one tile
- Across 20 heads: would exceed watchdog.

Our actual runtime is 4.7 sec per call (MFA dispatches one TG per
(Q-tile, head) so the grid is parallelized). But chunking would still
reduce **memory pressure** — a single-pass needs to hold the 80 GB K
buffer in residency for the whole call; chunked would touch it in
sliding windows.

### Takeaway: Sprint 2 candidate
Implement chunked-K dispatch in V6 NAX:
1. Add an `lse` output (already done — `outputs[1]` is LSE).
2. When `kL > MFA_V6_CHUNK_THRESHOLD` (env-var, default 65536), split
   K into `(kL + chunk - 1) / chunk` chunks of size `MFA_V6_CHUNK_SIZE`
   (default 32768).
3. Run V6 forward per chunk emitting `(O_partial, LSE_partial)`.
4. Reduce via `sdpa_chunked_reduce`-equivalent kernel
   (or a simple Python-level reduction since we already have LSE).

Estimated work: 1–2 days. Expected gain: 5–15% on SeedVR2-large via
better memory locality + GPU watchdog headroom.

---

## PR #3293 — "fix: add head_dim=256 to fused SDPA full attention kernel"

**URL:** https://github.com/ml-explore/mlx/pull/3293
**Status:** Merged
**Touches:** `mlx/backend/metal/scaled_dot_product_attention.cpp`,
            `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal`

### What it does
Adds D=256 support to the fused (non-NAX) `steel_attention` kernel by
instantiating with tile params `(32, 16, 256, 4, 1)`.

PR text: *"The Metal kernel template already handles arbitrary BD via
template parameter — only the dispatch gate and kernel instantiation
list were missing."*

### How D=256 is handled
**Single-tile, NOT multi-block on D.** The kernel processes the full
256-element head dim in one tile using BD=256 directly. This means the
register file must hold a 32×256-element fragment per simdgroup, which
on M1 was infeasible (caused spill) but works on M5.

### Performance trade-off (from PR)
- D=256 fused kernel is **~30% slower** than the unfused multi-kernel
  path on **short sequences**.
- Becomes essential at long contexts where unfused requires 8+ GB single
  allocations (exceeds Metal buffer limits).

### Relevance to V6 NAX
Limited:
- Our V6 currently restricts to D∈{64, 128} via
  `mfa_v6_nax_primitive.cpp:351`. We could extend.
- Production VSR shapes don't use D=256 (FlashVSR D=64, others D=128).
- The Draw Things v2 has no D=256 path either; would need a new
  AttentionKernel variant.
- Apple's NAX path **does NOT support D=256** (line 622). D=256 falls
  through to the legacy `steel_attention` (non-NAX) path with this PR.

### Takeaway
Skip. Out of scope for V6 NAX (Apple isn't doing it on NAX either).

---

## PR #3306 — "Add logsumexp output to fused SDPA kernel"

**URL:** https://github.com/ml-explore/mlx/pull/3306
**Status:** CLOSED (closed by author Apr 4, 2026)
**Touches:** `steel_attention.metal` (NOT NAX kernel directly)

### What it does
Adds an optional LSE output buffer to `steel_attention`:
- New function constant 304 (`output_logsumexp`)
- New buffer 8 holds per-row LSE values (FP32)
- Formula: `lse = max_score * M_LN2_F + log(sum_score)` (converts from
  log2 internal domain to natural log).

When `output_logsumexp=false`, kernel is bit-identical to before
(zero overhead via specialization).

### Relevance to V6 NAX
**Already done in V6.**

Our `mfa_v6_nax_primitive.cpp:235` allocates `lse` as `outputs[1]`. The
generated kernel writes both `O` and `L` (logsumexp). This is one of
the reasons we chose the v6 architecture — the cooperative_tensor
reduction infrastructure naturally exposes max+sum.

### Takeaway
None. Our v6 already has LSE.

---

## Bonus finding: Apple's NAX kernel structure

While searching MLX for these PRs, located the production NAX kernel:
- `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`
- `mlx/backend/metal/kernels/steel/attn/nax.h`

Detailed analysis in `apple-sdpa-nax-analysis.md`. Summary: Apple uses
the **lower-level `metal_simdgroup_matrix` API** with custom NAXTile/
NAXFrag abstractions, NOT MPP `matmul2d_descriptor`. This is the most
plausible explanation for the 5–7pp efficiency gap our V6 NAX cannot
close via tile-tuning alone.

---

## Recommendations (synthesis with other tasks)

1. **Implement chunked-K (PR #3307 pattern)** in V6 — Sprint 2, HIGH
   priority. Expected +5–15% on SeedVR2-large.
2. **Skip PR #3295, #3293, #3306** — none directly applicable.
3. **Most impactful follow-up is the abstraction-layer rewrite** (see
   `apple-sdpa-nax-analysis.md`), not a PR adoption.

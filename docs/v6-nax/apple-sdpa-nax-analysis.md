# Apple SDPA NAX Kernel Analysis

**Date:** 2026-05-03
**Source:** `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h` (482 LOC)
            `mlx/backend/metal/kernels/steel/attn/nax.h` (887 LOC)
            `mlx/backend/metal/scaled_dot_product_attention.cpp:18-164` (NAX dispatcher)
**MLX version:** main / 0.31.x

---

## TL;DR — The architectural gap is the abstraction layer, not the algorithm

| Aspect | V6 NAX (Draw Things v2) | Apple SDPA NAX |
|--------|-------------------------|---------------|
| MMA primitive | MPP `matmul2d_descriptor` + `cooperative_tensor` | Raw `metal_simdgroup_matrix` + custom `NAXFrag/NAXTile` |
| MMA call site | `matmul_qk_op.run(mQ, mK_0, cS_0)` | `stile_t::NAXFrag_t::mma(...)` (manual fragment scheduling) |
| Fragment placement | Driver-controlled (cooperative_tensor allocation) | Programmer-controlled (explicit `Stile.frag_at(iq, ik)`) |
| Layout | `[B, N, H, D]` (transposed by host) | `[B, H, N, D]` (MLX native, no transpose) |
| Tile config | BQ=16-32, BK=32-64, ExecSG=4-16 (1D warp grid) | BQ=64, BK=32, BD=head_dim, WM=4, WN=1 (2D warp grid) |
| Threads/TG | 128–512 (4–16 simdgroups) | 128 (4 simdgroups, fixed) |
| Grid layout | Morton-order 1D × 1 × B | Natural 3D `(NQ, H, B)` |
| K/V staging | Threadgroup memory (Path B) | Direct device-memory `NAXTile.load()` |
| Causal | Per-tile boundary check + last-tile element mask | Per-element `(r < c) ? -inf : score` |
| Sinks (registered tokens) | Not supported | Supported (function constant 302) |

**The key takeaway**: Apple uses a **lower-level abstraction layer** with
explicit fragment management. We use **MPP's higher-level cooperative
tensors**. The MPP layer adds scheduling overhead Apple's path avoids.

---

## File map

- `nax.h:30-62` — `BaseNAXFrag` definition: 16x16 simdgroup matrix
  fragment with explicit element layout `(kElemRows=2, kElemCols=4,
  kElemRowsJump=8)` matching M5 NAX's hardware tile shape.
- `nax.h:200-400` — `NAXTile<T, M, N>` template: array of M×N
  16x16 fragments with `frag_at(i, j)` accessor and `load()`/`store()`
  helpers.
- `nax.h:500-700` — `BaseNAXFrag::mma(...)` static method calls the
  hardware MMA intrinsic.
- `steel_attention_nax.h:75-85` — kernel entry point declaration:
  ```cpp
  template <typename T, int BQ, int BK, int BD, int WM, int WN, ...>
  [[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
  void attention_nax(...)
  ```
- `steel_attention_nax.h:200-250` — Q@K^T loop (Stile = QK accumulator).
- `steel_attention_nax.h:255-280` — element-wise scale (S *= 1.44269504).
- `steel_attention_nax.h:285-330` — KV-length-tail mask (inactive on
  aligned tiles; replaces non-K elements with `-inf`).
- `steel_attention_nax.h:332-360` — causal mask (per-element `r < c`).
- `steel_attention_nax.h:360-440` — generic mask (additive or boolean).
- `steel_attention_nax.h:440+` — online softmax + P@V loop (similar
  structure to Draw Things, register-resident accumulator).

---

## Tile and dispatch geometry

From `scaled_dot_product_attention.cpp:31-37`:
```cpp
int wm = 4;       // simdgroups along Q tile rows
int wn = 1;       // simdgroups along K tile cols (always 1)

int bd = q.shape(-1);  // head_dim
int bq = 64;
int bk = 32;
```

From line 162–163:
```cpp
MTL::Size grid_dims = MTL::Size(NQ, H, B);
MTL::Size group_dims = MTL::Size(32, wm, wn);  // = (32, 4, 1) = 128 threads
```

So Apple dispatches:
- Grid: 3D `(ceil(qL/64), H_q, B)`
- Threadgroup: `(32, 4, 1)` = 128 threads = 4 simdgroups
- BQ=64 (4 simdgroups × 16 rows/simdgroup = 64 rows per tile)
- BK=32 (each simdgroup processes 1 column tile of 32 cols)
- WN=1 means each simdgroup owns the full K tile width.

**Compare to our V6**:
- We dispatch `(grid_x_morton, 1, B)` with TG `(32 × ExecSG, 1, 1)`.
- For our best D=64 config: BQ=16 × ExecSG=16 = 256 rows per logical tile,
  but each simdgroup processes 16 rows (TQ=1, no Q sub-tiling) and
  BK=64 cols. We use 16 simdgroups vs Apple's 4.

**Why does Apple use only 4 simdgroups (128 threads)?**
- Smaller TGs → more TGs co-resident per core → better latency hiding via
  hardware-level wavefront overlap.
- Larger BQ (64 vs 16) per TG → fewer Q-tile boundaries to handle, better
  amortization of K-loop fixed cost per Q-tile.
- The arithmetic per simdgroup is more saturating with BQ=64 worth of work.

Our autoresearch found `ExecSG=16` empirically wins on M5. This is
counter-intuitive vs Apple's choice. Possible explanations:
1. MPP overhead is per-MMA, so doubling MMAs per TG (more simdgroups)
   amortizes that fixed cost better.
2. The cooperative_tensor allocation favors having more lanes available.
3. Our 1D warp grid (16×1) has less inter-warp synchronization than a
   2D 4×4 grid would, simplifying the schedule.

If we reimplement using `simdgroup_matrix` (Sprint 2 recommendation),
we should test Apple's `(WM=4, WN=1, BQ=64)` config first.

---

## Function constants (specialization gates)

From `steel_attention_nax.h:14-19`:
```cpp
constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];
constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];
constant bool has_sinks [[function_constant(302)]];
```

**Implications**:
- 5 function constants → 32 kernel variants per (T, BQ, BK, BD, WM, WN)
  combination. Apple precompiles a metallib with all combinations.
- We pass all params as buffer arguments → fewer specializations but more
  runtime branching.
- LSE output is via function constant 304 (PR #3306, not yet landed).

---

## Layout: BHND vs BNHD

Apple's kernel uses **`[B, H, N, D]`** directly (no transpose needed):
```cpp
// steel_attention_nax.h:97-105
Q += tidl.z * params->Q_strides[0]    // batch
   + tidl.y * params->Q_strides[1]    // head
   + tidl.x * BQ * params->Q_strides[2];  // sequence

ulong kv_head_idx = int(tid.y) / params->gqa_factor;
K += tidl.z * params->K_strides[0]
   + kv_head_idx * params->K_strides[1];
```

Our V6 uses **`[B, N, H, D]`** because that's what Draw Things expects:
```cpp
// mfa_v6_nax_primitive.cpp:355-357
auto q_bnhd = mlx::core::transpose(q, std::vector<int>{0, 2, 1, 3}, s);
auto k_bnhd = mlx::core::transpose(k, std::vector<int>{0, 2, 1, 3}, s);
auto v_bnhd = mlx::core::transpose(v, std::vector<int>{0, 2, 1, 3}, s);
auto qc = mlx::core::contiguous(q_bnhd, false, s);
```

**Cost of our transpose**: For SeedVR2-large (B=1, H=20, N=111375, D=128):
each transpose moves 1 × 20 × 111375 × 128 × 2 bytes = ~570 MB. Three
transposes (Q, K, V) + materialization = ~1.7 GB of memory traffic
*before* the kernel even runs. Plus a return transpose on O.

**Total transpose overhead** for SeedVR2-large: ~2.3 GB extra traffic.
At ~400 GB/s memory bandwidth, that's ~6 ms — substantial vs the kernel's
~4700 ms total but should be measured as a discrete profiling line.

If we adopted Apple's BHND layout (Sprint 2), we'd save these transposes.
Sub-task: would require modifying the kernel's offset computation (the
`Q_buf` indexing currently walks a [N, H, D] block per batch).

---

## Supported head dimensions

From `scaled_dot_product_attention.cpp:622`:
```cpp
const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
    (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);
```

Apple's NAX full-attention path: `D ∈ {64, 80, 128}`. Notably:
- D=80 is supported (we don't have it, but no production model uses 80).
- **D=256 is NOT supported by NAX path** — falls through to `sdpa_vector`
  (decode-only, qL ≤ 8) per PR #3293.

For full attention with D=256, MLX falls back to the unfused multi-kernel
path (slower but no kernel limit). PR #3293 adds D=256 to the
NON-NAX `steel_attention` (legacy) path with a single-tile kernel, but
that's ~30% slower than unfused for short sequences.

---

## How Apple handles the K-loop

From `steel_attention_nax.h:175-195`:
```cpp
int kb_lim = params->NK;          // total K tiles
int kb_min_causal = params->NK;   // first K tile that needs causal mask

if (do_causal) {
  int q_max = (tid.x + 1) * BQ + params->qL_off;
  kb_lim = (q_max + BK - 1) / BK;
  kb_lim = min(params->NK, kb_lim);
  int q_min = tid.x * BQ + params->qL_off;
  kb_min_causal = q_min / BK;
}
...
for (int kb = 0; kb < kb_lim; kb++) {
  const int is_last_k = (kb == params->NK_aligned);
  // QK^T with explicit fragment scheduling
  // Length-tail mask only on last tile
  // Causal mask only when kb >= kb_min_causal
  ...
}
```

This is the same FlashAttention-2 structure as ours, with two
optimizations:
1. **Per-Q-tile causal range pre-computation** (`kb_lim`,
   `kb_min_causal`). Avoids checking the causal predicate every tile.
   We have `kb_start` for sliding-window via env vars but not for causal.
2. **Length-tail mask only on `is_last_k`**, otherwise no per-element
   range check.

---

## What Apple does that we don't

### 1. Manual fragment scheduling
```cpp
// steel_attention_nax.h:218-230 (S = Q @ K^T, partial)
NAXTile<T, 1, 1> Qtile;
NAXTile<T, 2, 1> Ktile;
const int Q_load_off = iq * kU * Q_strides[2] + id * kU;
Qtile.load(Q + Q_load_off, Q_strides[2]);
Ktile.load(K + K_load_off, K_strides[2]);
stile_t::NAXFrag_t::mma(
    Stile.frag_at(iq, ik),
    Stile.frag_at(iq, ik + 1),
    Qtile.frag_at(0, 0), false_type{},
    Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), true_type{});
```
Apple loads Q/K fragment-by-fragment from device memory, calls `mma()`
explicitly, and updates two `Stile` fragments per call. Direct HW control.

We have:
```cpp
// (our v6, generated MSL via NAAttentionKernel.cpp:847)
matmul_qk_op.run(mQ, mK_0, cS_0);
```
Single MPP call. The driver picks fragment count, register layout,
scheduling.

### 2. `metal::uniform<float>` for scale
```cpp
// steel_attention_nax.h:138-140
const metal::uniform<float> scale2 =
    make_uniform(params->scale) * make_uniform(1.44269504089f);
```
`metal::uniform<T>` tells the compiler the value is identical across all
threads in a SIMDgroup, enabling the compiler to use shared scalar
registers (saves vector registers). Small win, but adds up.

### 3. `metal::vec<AccumType, kRowsPT>` for online softmax state
```cpp
// steel_attention_nax.h:165-170
constexpr short kRowsPT = otile_t::kRowsPerThread;
metal::vec<AccumType, kRowsPT> max_score;
metal::vec<AccumType, kRowsPT> sum_score{0};
```
Per-thread scalar vector for max/sum, not threadgroup memory. We use
cooperative_tensor reductions which round-trip through TGP for
inter-simdgroup sum.

### 4. Sinks support
```cpp
constant bool has_sinks [[function_constant(302)]];
...
if (has_sinks) {
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
    sum_score[i] = 1;
  }
}
```
Registered-token sink (initial max = log of sink-attention bias). Used
in streaming attention / GPT-OSS-style models. We don't have this.

### 5. Causal range elision
Apple skips per-element causal mask for K-tiles fully under the diagonal
(`kb < kb_min_causal`). Our generated kernel has the equivalent guard
(`if (causal_mask_0)`) at `NAAttentionKernel.cpp:893`, so we already
match this.

---

## What we do that Apple doesn't

### 1. Morton-order grid layout
Our forward grid is 1D Morton-flattened to 2^(rb+hb), kernel decodes
internally. Apple uses natural 3D `(NQ, H, B)`.

The advantage of Morton order: adjacent threadgroups process spatially
nearby (row_block, head) tiles, improving L2 cache hit rate when the
same K/V rows are reused across heads. For B=1 (most production VSR),
this matters because head-dim alone limits parallelism.

For Apple, with `(NQ, H, B)` natural ordering, all heads of a given
row_block are dispatched together — same effective ordering on the GPU's
HW dispatcher when B=1 and grid is filled sequentially.

Net: **we both achieve cache locality through different mechanisms**.

### 2. Threadgroup memory P-tile staging
Our v2 stages P (post-softmax probability) in threadgroup memory before
P@V. Apple keeps P in registers (`metal::vec<AccumType, ...>` → mma
input). For large BQ × BK, register pressure differs.

### 3. Bypass-threadgroup-memory fallback
Our v2 has `bypassThreadgroupMemory=true` (Path A) which we found
empirically NO-GO on M5 (Axe 3). Apple has no such configuration switch
— they always work in registers.

---

## Recommendations (Sprint 2 candidates)

| Idea | Difficulty | Expected gain | Confidence |
|------|-----------|---------------|------------|
| Reimplement V6 with `simdgroup_matrix` (Apple-style) | HIGH (~2 weeks) | +5–10% | MEDIUM |
| Switch V6 to BHND layout (no host transpose) | MEDIUM (~3 days) | +1–6% (varies by N) | HIGH |
| Try Apple's tile config (BQ=64, BK=32, WM=4) | LOW (1 day) | -3% to +3% | MEDIUM |
| Use `metal::uniform<float>` for scale factor | LOW (30 min) | <1% | LOW |
| Add sinks support | MEDIUM (~2 days) | 0% (feature, not perf) | HIGH |

**Highest priority**: switching to BHND layout and the simdgroup_matrix
abstraction. The transpose-elimination alone could account for the
SeedVR2-large gap (V6/SDPA 0.82–0.96×).

# V6NAX — Apple NAX reference mapping (Phase 0a)

**Date:** 2026-05-06
**Branch:** `experiment/v6nax-nax-direct`
**Apple sources** (`~/code/mlx-source/mlx/backend/metal/kernels/steel/attn/`):
- `kernels/steel_attention_nax.h` — 482 LOC — Apple reference forward kernel
- `nax.h` — 887 LOC — `BaseNAXFrag` (with `mma`, `load`, `row_reduce`, `row_bin_op`) + `NAXTile<T, TQ, TD>` wrapper
- `transforms.h` — 70 LOC — `BlockSwizzle`, `AccumHelper`, `Transform*`
- `params.h` — 44 LOC — `AttnParams` struct
- `loader.h` — 264 LOC — block loaders (not needed for V6NAX: NAXFrag::load handles tile loads)

## Critical architectural insight

**Apple's `NAXFrag::mma` (`nax.h:393-456` and `nax.h:464-528`) uses
`mpp::tensor_ops::matmul2d` INSIDE the static method.** It uses
`metal::execution_simdgroup` (singular — equivalent to `<1>`). The
cooperative_tensors are EPHEMERAL inside `mma()`: created, populated
from the lane's fragment registers, used for one matmul, copied back
out, discarded. There is NO `<N>` cooperative_tensor at any point.

This is why V33's hybrid approach (cooperative_tensor `<1>` + bridge
to `<N>` cO_0) was structurally unsound: it tried to use MPP's
distributed cooperative_tensor across SGs, which has opaque cross-SG
distribution semantics. V6NAX doesn't have this problem because it
**never has cross-SG cooperative_tensor state**.

The "multi-SG parallelism" in V6NAX comes from a different source: each
SG processes its own slice of BQ rows independently. With BQ=64,
WM=4: each SG handles 16 rows; together they cover all 64 rows. They
share threadgroup memory only for K/V/P (in some kernels), not for
the matmul cooperative state.

## Sub-system mapping

| V6NAX sub-system | Apple file:line | Notes |
|---|---|---|
| Per-batch+head+Q-block ptr offset | `steel_attention_nax.h:102-117` | `Q += tid.z * Q_strides[0] + tid.y * Q_strides[1] + tid.x * BQ * Q_strides[2]`. BHND-ready: strides[0]=H*qL*D, strides[1]=qL*D, strides[2]=D. GQA: `kv_head_idx = tid.y / gqa_factor` for K/V. |
| Otile decl + clear | `steel_attention_nax.h:143-146` | `using otile_t = NAXTile<float, TQ, TD>;` then `Otile.clear()`. TQ=BQ/(WM*16), TD=BD/16. |
| SG-local Q row offset | `steel_attention_nax.h:149-150` | `const short tm = kU * TQ * simd_group_id; Q += tm * Q_strides[2];` |
| Softmax state init | `steel_attention_nax.h:158-166` | `metal::vec<float, kRowsPT> max_score, sum_score{0};` init max to `Limits<float>::finite_min`. kRowsPT = otile_t::kRowsPerThread = TQ * 2. |
| Stile = Q @ K^T | `steel_attention_nax.h:200-246` | nested loop `for iq, ik+=2, id`. Loads `NAXTile<T,1,1> Qtile` + `NAXTile<T,2,1> Ktile`. mma call with `transpose_b=true_type{}`. |
| Q tile load (per-fragment) | `steel_attention_nax.h:218-225`, `nax.h:60-95` (BaseNAXFrag::load) | `Qtile.load(Q + Q_load_off, Q_strides[2])` for aligned, or `Qtile.load_rows(...)` for unaligned. |
| K tile load | `steel_attention_nax.h:227-234` | Same as Q but `K_load_off = ik*kU*K_strides[2] + id*kU`. |
| QK mma | `steel_attention_nax.h:236-243`, `nax.h:393-456` | `stile_t::NAXFrag_t::mma(Stile.frag_at(iq, ik), Stile.frag_at(iq, ik+1), Qtile.frag_at(0,0), false_type, Ktile.frag_at(0,0), Ktile.frag_at(1,0), true_type)`. transpose_b=true for K^T. |
| Scale | `steel_attention_nax.h:248-252` | `Stile.elems()[ii] *= float(scale2)` where `scale2 = scale * 1.44269504089f` (log2e). |
| Online softmax — `new_max` capture | `steel_attention_nax.h:382-388` | `metal::vec<float, kRowsPT> new_max; for i: new_max[i] = max_score[i];` |
| Row max reduce | `steel_attention_nax.h:391`, `nax.h:639-650` | `Stile.template row_reduce<MaxOp>(new_max);` Internally uses simd_shuffle_xor (nax.h:363, 366). |
| ExpSub on Stile | `steel_attention_nax.h:394`, `nax.h:373-385` | `Stile.template row_bin_op<ExpSubOp>(new_max);` Replaces `cS = exp2(cS - max)` from V33. |
| Compute factor + update max_score | `steel_attention_nax.h:397-401` | `factor[i] = exp2(max_score[i] - new_max[i]); max_score[i] = new_max[i];` |
| Update sum_score (decay + accumulate) | `steel_attention_nax.h:404-409` | `sum_score[i] *= factor[i];` then `Stile.template row_reduce<SumOp>(sum_score);` |
| Apply factor to Otile (online correction) | `steel_attention_nax.h:412` | `Otile.template row_bin_op<MulOp>(factor);` Replaces V33's correction.map_iterator. |
| simdgroup_barrier(mem_none) | `steel_attention_nax.h:414` | Pre-PV-mma fence. Compiler-only fence (mem_none). |
| Otile += Stile @ Vtile | `steel_attention_nax.h:417-452` | nested `for iq, id+=2, ik`. Load `NAXTile<T,1,2> Vtile`. mma with two C fragments and two B fragments, transpose_a=false, transpose_b=false. |
| V tile load | `steel_attention_nax.h:429-440` | `Vtile.load(V + V_load_off, V_strides[2])` aligned, `Vtile.load_rows` unaligned. |
| Threadgroup barrier inside D-loop (BD==128) | `steel_attention_nax.h:421-425` | `if (id == 4) threadgroup_barrier(mem_none);` mid-loop barrier for BD=128. |
| Advance K, V pointers | `steel_attention_nax.h:455-456` | `K += BK * K_strides[2]; V += BK * V_strides[2];` |
| Final normalize | `steel_attention_nax.h:461-469` | `threadgroup_barrier(mem_none);` then `rcp[i] = 1.f / sum_score[i];` then `Otile.template row_bin_op<MulOp>(rcp);` |
| Output store | `steel_attention_nax.h:471-481` | `O += tm * O_strides[2];` then `Otile.store(O, O_strides[2])` aligned, or `Otile.store_rows(O, O_strides[2], lim_rows_q)` unaligned. |
| Function constants | `steel_attention_nax.h:14-19` | `align_Q [[fc(200)]]`, `align_K [[fc(201)]]`, `has_mask [[fc(300)]]`, `do_causal [[fc(301)]]`, `has_sinks [[fc(302)]]`. We only need `align_Q` and `align_K` for V6NAX forward non-causal. |
| Operator structs | `steel_attention_nax.h:31-71` | `MaxOp`, `SumOp`, `MulOp`, `SubOp`, `ExpSubOp`, `DivOp`. NOT in `transforms.h` despite the name suggesting otherwise. |

## NAXTile / NAXFrag size relationships

From `nax.h:531-560`:

```
NAXTile<T, TQ, TD>:
  kFragRows = 16, kFragCols = 16, kElemsPerFrag = 8 (= 16*16/32)
  kElemRows = 2, kElemCols = 4, kElemRowsJump = 8
  kRows = TQ * 16 (BQ rows handled by this tile)
  kCols = TD * 16 (BD cols handled by this tile)
  kNumFrags = TQ * TD
  kElemsPerTile = TQ * TD * 8
  kRowsPerThread = TQ * 2 (each lane owns 2 rows per fragment, TQ fragments)
```

For Apple's example with BQ=64, WM=4, BD=128, BK=64:
- TQ = BQ / (WM * 16) = 64 / (4 * 16) = 1
- TD = BD / 16 = 128 / 16 = 8
- TK = BK / 16 = 64 / 16 = 4
- otile_t = NAXTile<float, 1, 8> → 1*8 = 8 fragments, 64 elements per lane
- kRowsPerThread = 1 * 2 = 2

## V6NAX design choices vs Apple

| Aspect | Apple kernel | V6NAX plan |
|---|---|---|
| TQ | always 1 (`static_assert(TQ == 1)` line 142) | Match: BQ = WM * 16 |
| BD | template parameter, can be any multiple of 16 | Match: head_dim ∈ {64, 128} |
| BK | template parameter | Match: tunable per-D |
| WM | template, typical 4 | Match: tunable; expose via env var |
| WN | always 1 in `attention_nax` (no col partition) | Match |
| `align_Q`, `align_K` | function constants, JIT-set at compile time | Generate kernel with both = `false` (safe default — handles all shapes via load_rows/store_rows). Phase 5 optimization: emit two pipeline variants and dispatch based on `qL % BQ`, `kL % BK`. |
| `has_mask`, `has_sinks` | function constants | Generate kernel with `false` (we don't currently support these in V6 NAX). |
| `do_causal` | function constant | Generate `false` (V6NAX scope: non-causal forward only, per the mandate). |
| Causal masking inner code | lines 278-303 | omit (compile out) |
| Mask inner code | lines 306-378 | omit |
| Sinks inner code | lines 168-174 | omit |
| `qL_off` | params field | not used in non-causal — set to 0 in our params |

## V6NAX will NOT use post-generation rewriting for BHND

The existing `mfa_v6_nax_primitive.cpp` BHND rewriter does
`replace_all` on the legacy generator's emitted strings. V6NAX will
**emit BHND-compatible code directly** — no post-gen rewriting
needed. This removes a fragility class.

Apple's stride encoding (Q_strides[0]=batch_stride, [1]=head_stride,
[2]=seq_stride, [D]=1 implicit) maps directly to MLX BHND
contiguous tensors. V6NAX computes those strides at runtime in the
primitive's eval_gpu and passes them as constants.

## Function-constant strategy

For Phase 1, V6NAX emits the kernel with hard-coded `align_Q = false`,
`align_K = false`. This is the safe variant: always uses
`load_rows`/`store_rows` and per-element bounds checks. Slightly
slower than the aligned path on perfectly-aligned shapes, but
correct on every shape.

If Phase 4 bench shows V6NAX has the right perf direction, Phase 5
adds aligned/unaligned dispatch via two compiled variants keyed on
`qL % BQ == 0` and `kL % BK == 0` at dispatch time.

## Open questions resolved by this Phase 0 reading

1. **Is `MaxOp` etc. in `nax.h` or somewhere else?** → They're in
   `kernels/steel_attention_nax.h:31-71`, not `transforms.h`. V6NAX's
   generated kernel must define them (copy verbatim from Apple
   source, lines 31-71).

2. **Does `NAXTile::store` work with raw `device T*`?** → Yes, see
   `nax.h:710-723` (`store(device U* dst, const int ld)`). No
   threadgroup-staging needed for output.

3. **How does `NAXFrag::mma` handle `<1>` in cross-SG TG?** → It
   creates a fresh `mpp::tensor_ops::matmul2d` with
   `metal::execution_simdgroup` (singular = `<1>`) per call. Each
   call is fully within ONE simdgroup; no cross-SG state. Multiple
   SGs can call `mma` in parallel — each computes its own subset of
   the result independently.

4. **Where does the GQA factor live?** → `params->gqa_factor`,
   computed as `Hq / Hk`. Used inline as `tid.y / params->gqa_factor`
   for K/V head index (steel_attention_nax.h:108).

## Phase 0c — Mapping to mlx-mfa Params struct

Our `MFAV6NaxParams` struct (in `csrc/mfa_steel_fwd_v6_nax.cpp:46-57`,
which is the legacy probe; actual production uses
`mfa_v6_nax_primitive.cpp` which does NOT pass an explicit params
struct — strides are encoded in the kernel function constants).

V6NAX strategy: **add an `AttnParams`-shaped buffer** (matching
Apple's struct, including the 12 strides) computed by the primitive
in `eval_gpu()`. Pass via `[[buffer(4)]]`. This keeps the V6NAX kernel
identical to Apple's reference layout, simplifying review and
debugging.

The primitive computes:
```cpp
params.B = batch_dim;
params.H = Hq;
params.D = head_dim;
params.qL = N_q;
params.kL = N_kv;
params.gqa_factor = Hq / Hk;
params.scale = 1.0f / sqrt(head_dim);
params.NQ = (N_q + BQ - 1) / BQ;
params.NK = (N_kv + BK - 1) / BK;
params.NQ_aligned = N_q / BQ;
params.NK_aligned = N_kv / BK;
params.qL_rem = N_q % BQ ? N_q % BQ : BQ;
params.kL_rem = N_kv % BK ? N_kv % BK : BK;
params.qL_off = 0;  // non-causal
// BHND strides:
params.Q_strides[0] = (int64_t)Hq * N_q * head_dim;   // batch stride
params.Q_strides[1] = (int64_t)N_q * head_dim;        // head stride
params.Q_strides[2] = (int64_t)head_dim;              // seq stride
// K/V (note: GQA so Hk vs Hq)
params.K_strides[0] = (int64_t)Hk * N_kv * head_dim;
params.K_strides[1] = (int64_t)N_kv * head_dim;
params.K_strides[2] = (int64_t)head_dim;
// V same as K
// O same shape as Q
params.O_strides[0] = (int64_t)Hq * N_q * head_dim;
params.O_strides[1] = (int64_t)N_q * head_dim;
params.O_strides[2] = (int64_t)head_dim;
```

These match Apple's encoding 1:1. No reformulation needed.

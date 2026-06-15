# Sprint III-10 — Final Targeted Residual Sweep (pre-v2.53.0 gate)

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High + 3 parallel target agents (Workflow `w68lhdnp7`)
**HEAD:** 94c4ac4
**Type:** bounded residual pass on THREE targets only (NOT a re-sweep of D–K, which reached
structural completeness in III-9). Each target carries an independent-fp32 oracle AND a structural
stop criterion (enumerate the complete set, prove each member safe).

## Outcome: ZERO confirmed findings. FINAL PRE-RELEASE GATE MET.

| Target | Verdict | Structural-stop evidence |
|---|---|---|
| **T1** — `0×NaN`/`x+NaN` meta-pattern beyond V reads | **CLEAN** | Complete set of masked-arith device reads enumerated + each proven safe |
| **T2** — MPP `matmul2d`/`convolution2d` K alignment | **CLEAN** | Every MPP call site enumerated; K alignment guaranteed-by-gate or correct-vs-fp32 on macOS 26.6 |
| **T3a** — lifetime-class completeness | **CLEAN** | Zero `allocator::free`; every `allocator::malloc` proven output-backed or `add_temporary` |
| **T3b** — fp32-coverage nits | **CLOSED** | 4 tests (turboquant ×2, sage ×2, paged-varlen ×1) now carry independent fp32 oracles; all pass |

## Target 1 — the `0×NaN` meta-pattern is V-specific (proven, not assumed)

The III-9 direct-V trap was unique **because P@V MULTIPLIES** (0×NaN survives, no overwrite). Every
OTHER device read that feeds masked score arithmetic is protected by a different mechanism: the
K-boundary / causal / window / GNA masks **ASSIGN** `-INFINITY` (`=`, not `+=`/`*=`) to exactly the
OOB columns (`col >= kL_rem ⇔ k_pos >= S`), and the assignment is emitted **before** `Stile` is
consumed by `row_reduce<MFAMaxOp>`/softmax — annihilating any garbage.

Complete set enumerated across all three MFA_DIRECT_READS kernels (V2 single-pass + D-split, GNA, V5):
- **K direct reads** — OOB NaN → score, overwritten by the K-boundary mask ASSIGN before softmax. SAFE.
- **`attn_bias[k_pos]` add** (the prime suspect): `score += bias[OOB]` is emitted *before* the
  K-boundary mask ASSIGN, so the `-inf` overwrite annihilates the `x+NaN`. SAFE (probed clean,
  modes 1+2, vs fp32).
- **`alibi_slopes[tid.y]`** — head-indexed, never partial-tile OOB. SAFE.
- **`rotary_cos/sin`** — doubly protected: `has_rope` forces `use_direct_reads=false` (smem
  load_safe zero-pad), and any OOB cos/sin NaN lands only in padding query rows, which per-row
  softmax isolates and `store_safe` clips. SAFE (probed clean).
- **No value-gather** (top-K/paged block_table/page_table) exists in these kernels (top-K Metal
  deferred; Python ref only). SAFE.

All probes (attn_bias partial-S, in-kernel fused-RoPE partial-Q) deterministic vs fp32 under the
pool-history + concurrent-alloc trigger → logic-correct, no lifetime artifact. **0 real bugs.**

## Target 2 — MPP alignment (version-corrected; all sites gated)

**Version flag (load-bearing):** this machine is macOS **26.6** (build 25G5028f), not 26.1/26.4.
The 26.1-reported "silent-cut K to a multiple of 32 and return a wrong result" behavior **does NOT
reproduce on 26.6**: MPP now enforces a COMPILE-TIME `static_assert((k % 16) == 0)` (loud build
failure, %16 floor — not %32, not silent). Observed: static K=40 fails to compile; static
K∈{16,48,80} compile and are correct vs fp32 (rel ~4e-4).

Every production MPP call site (conv `matmul2d`, conv `convolution2d`, NAX-attention `matmul2d`,
steel-V6 `matmul2d`) has its contraction-K alignment **guaranteed-by-gate** (documented Rule-8
`throw` / dispatch gate, e.g. conv `K % K_TILE != 0` raise + caller `pad_contraction_k`; conv3d
MPP-eligibility `C%16==0 & ≥32`) OR neutralized by the `dynamic_length` workaround, AND confirmed
correct vs an independent fp32 reference at the alignment boundary. The repo's conservative %32
guards remain safe across both 26.1 and 26.6. The two non-production sites (int8 microbench,
v6_nax_probe) make no correctness claims. **0 real bugs.** (The original threat model is
version-specific to pre-26.6 and is loud-not-silent on this machine — recorded honestly.)

## Target 3 — lifetime completeness + fp32 locks

**3a:** `grep allocator::free` → **zero hits** (the two III-9 lifetime sites use `add_temporary`).
Every `allocator::malloc` classified into exactly two safe categories: (A) output-array-backed
(`outputs[N].set_data(...)`, MLX-graph-managed) or (B) `add_temporary`-managed (the 2 scratch
pairs). No scratch buffer relies on C++ scope (encode-time free). Complete-set proven; no third
site. **CLEAN.**

**3b:** the 3 III-9-flagged non-independent-validation nits closed — 4 tests gained an independent
fp32-cast oracle (`mx.fast.scaled_dot_product_attention` fp32, not auto-hook-patched / decompress-
then-fp32-SDPA for TQ): `test_turboquant` fused-K and fused-K+V, `test_sage_attention` `_check` +
GQA, `test_attention` paged-varlen. No fp32-cast divergence beyond the lossy/fp16 tolerance →
locks are real, no latent bug, edits kept. All pass.

## Validation
- Full suite **1820 passed, 2 skipped, ×2 consecutive** (with the 4 new fp32-oracle assertions).
- Each target proven by complete-set enumeration (not "didn't find one"); every probe vs
  independent fp32 (lesson #11) under the pool-history + concurrent-alloc trigger.

## FINAL PRE-RELEASE GATE: MET.
III-9 reached structural completeness for classes D–K; III-10 closed the three bounded residuals
with zero new material. Release scope (Marco-gated, v2.53.0+):
- `da737e7` — async metallib macOS-26 gate (defense; inert for the V2 bugs)
- `240b226` — split-K / flash-decode scratch lifetime fix
- `eb68af5` — V2 single-pass non-causal last-head OOB-V clamp
- `eb5b890` — GNA + V5 multi-gate OOB-V clamp
- III-10 3b test fp32 locks (this sprint; tests only, no kernel change)
- the two III-7 `quantize_model` fixes (already on master)

# GNA / Conv / TopK / Sage / Paged — Verified Per-Kernel Spec (audit B4, durable; closes Phase B)

RUNTIME-verified on M5/26.6. Each kernel vs its APPROPRIATE independent oracle (lesson #11), with its
own tolerance discipline. Locked by `tests/test_b4_family_lock.py` (9 cells) + the existing IV-D1/D2
lock. Labels: [V]erified/[D]educed.

## 1. GNA (generalized neighborhood attention) — deferred Phase-A correctness RESOLVED
**Computes:** per query at ND position p, attends keys in the neighborhood window: per dim,
`group_base=(p//stride)*stride`, `[group_base-(win-stride)//2, group_base+stride+(win-stride+1)//2)`
clamped to `[0, seq_shape)` (the documented `make_gna_mask` rule). Native Metal kernel (`mfa_gna_fwd.cpp`,
STEEL `MFAMMAFrag::mma`), forward-only.
**RESOLUTION:** vs the EXACT per-element-window fp32 oracle (manual, the documented rule): **max_abs_err
4.8e-5** across sliding (3³, 5³) + strided (2³). **GNA is CORRECT** — the Phase-A 7.3e-2 was a
block-mask reference over-approximation, NOT a bug. [V]
**Constraints:** D=128, 3D, f16/bf16; small-N is overhead-bound (Phase A, not a throughput regime).
Native first; falls back to sparse/SDPA otherwise.

## 2. conv3d-nax
**Computes:** im2col + NAX `matmul2d` conv (`mfa_conv_nax`), auto-hooked on `mx.conv_general`.
**Correctness:** eligible NAX vs fp32 `mx.conv_general` = **2.4e-4, cos 1.00000**; ineligible fallback
= 1.1e-4. [V] **Eligibility (MPP gate):** C_in/C_out %16==0 & ≥32, H/W %8==0, B=1, pad=(1,1,1),
f16/bf16 → NAX; else `mx.conv_general` (`get_hook_stats` executed/fallback, Phase A). [V]

## 3. topk attention
**Computes:** per-query top-k key selection (`mx.topk`/sort) + softmax over the k + P@V (Python ref +
SDPA; native streaming top-K kernel deferred). **Correctness:** ratio=0.25 (k=128) vs fp32 top-k =
**1.6e-3, cos 0.99981**; ratio=1.0 vs dense = 9.3e-6. [V] Own path (Phase A Δ=1.9e-6).

## 4. sage attention (int8) — quant-aware (its own discipline)
**Computes:** per-block symmetric int8 quant of Q,K (+ smooth_k) → int8 `matmul2d` QK^T (int32 accum)
→ dequant by scales → softmax → fp16 P@V.
- **(a) quant faithful:** int8 range [-127,127]; per-block round-trip **4.0e-4 ≤ step 7.9e-4** = within
  one quantization step. [V]
- **(b) principled int8 tolerance:** int8 7-bit symmetric quant over D=128 → a **cos floor ~0.997**
  (measured stable across input amplitude 0.1/1.0/3.0: cos 0.9978/0.9975/0.9971; max_abs scales with
  output magnitude). This is the inherent int8 QK^T quantization loss — NOT a bug. Locked at cos≥0.995
  (principled int8 margin, not arbitrarily loose). [V]
- **(c) int8 GEMM correct:** the cos-0.997/faithful-quant result confirms the int8 matmul2d path. [V]
**Constraints:** D∈{64,128}, f16/bf16, the int8 path is the quality/perf trade-off (not bit-exact).

## 5. paged / TQ decode + prefill
**Computes:** KV-cache gather (+ TQ dequant if quantized KV) → Apple SDPA (Phase A: decode → SDPA,
sync-floor regime). **Correctness:** kvcache decode (N_q=1, S=1024) vs fp32 gather attention =
**2.3e-6, cos 1.00000**. [V] **IV-D1/D2 eval-collapse** (v2.56.0: deferred==eager bit-identity, both
tq_v) re-confirmed — `tests/test_iv_d1_tq_append_defer.py` 3 passed. [V]

## Cross-cutting
- **Threshold audit:** no arbitrary/overflow threshold in this family (conv MPP gate = HW divisibility;
  topk ratio = user param; GNA window = documented; paged = decode shape). The open **sparse V1↔V2
  `2^31` PERF** item carries to Phase E (overflow benign per B2). [V]
- **Comment sweep:** GNA / conv / turboquant comments clean — no stale future-tense-done. No edits. [V]
- **Routing (Phase A):** GNA→native; conv→NAX-when-eligible; topk→own; sage→int8 kernel; paged→SDPA.

# Sprint II-3 entry state — Top-K Streaming Approach 5 (spec recovered + refined)

**Date**: 2026-06-12 · **Status**: Phase A COMPLETE (spec + re-validation);
Phase B (build) ready to start from this doc.

## Recovered spec (phase-3b-approach-5-decision.md) + CONFIRMATION re-check

Original Approach 5: PASS-1 = streaming per-row top-K via TGM heap over
K-tiles (no score materialization); PASS-2 = attention over selected K/V.
The Phase-I feasibility blocker was PASS-2: Apple SDPA has no indexed-K/V
mode (re-verified Sprint C against MLX 0.31.2 — unchanged), and the
mx.take materialization workaround costs ~10ms (per-row gather =
[B,H,N,K_top,D] ≈ 1 GB at audit shape).

## DESIGN REFINEMENT (II-3 Phase A finding — changes the economics)

PASS-2 does NOT need filtered-SDPA OR score re-materialization: PASS-1's
per-row top-K **indices** build the additive bias directly via scatter:

    bias = mx.put_along_axis(full((B,H,N,S), -inf), idx, 0.0, axis=-1)
    out  = mx.fast.sdpa(q, k, v, mask=bias)     # Apple NAX

Cost model at the audit shape (B=1 H=16 N=4096 S=4096 D=128 fp16, K=64):

| Step | Architecture B (current, 11.15ms) | Approach 5 refined |
|---|---|---|
| scores | 4ms materialized matmul | — (in-register tiles in PASS-1) |
| selection | 5ms bisection over scores | PASS-1 kernel: QK^T FLOPs ≈ 4ms + heap ops |
| bias build | (from scores, in bisect) | ~0.5-1ms scatter from indices |
| SDPA-bias | ~2ms | ~2ms |
| **Total** | **11.15ms** | **~7ms projected (ESTIMATE, model above) → ~1.6×** |

Kill-criterion for the build: PASS-1 measured > ~8ms makes the total
lose to Architecture B → decline with the bench.

## PASS-1 kernel design (metal-kernel-dev review GREEN, on session record)

- `mx.fast.metal_kernel` (standard MSL — NO MPP dependency, so the II-2
  MSL4/int8 blocker does NOT apply)
- Grid: (ceil(N/BQ), H, B); TG = 32×WM threads, WM=4; each SG owns
  BQ/WM = 8 query rows
- K-loop over BK=64 column tiles: Q@K^T tile in registers (fma or
  simdgroup_matrix), per-row running top-K
- Heap state in TGM: scores fp32 [BQ][K_top] + idx int32 [BQ][K_top]
  = 16 KB at BQ=32, K_top=64 (within 32KB TGP)
- Insert: SIMD-parallel min-scan over the row's 64 slots
  (simd_min + ballot), replace-if-greater — ~5 cycles/candidate
- Edge: `lim_cols` masking for the last K-tile (S % BK)
- Output: idx [B, H, N, K_top] int32
- Flag: `MFA_TOPK_STREAM_V5=1` (NOT default until benched)

## Phase C/D protocol (unchanged from the sprint prompt)

Correctness: indices-set equality vs mx.topk per row (ties at the FP16
boundary may permute WITHIN equal values — compare selected-score SETS,
matching the documented Architecture-B boundary semantics); end-to-end
output vs the current bisection path within FP floor.  Bench: vs the
REAL current path (Architecture B, 11.15ms), 3 sessions, K ∈
{32, 64, 128} × N ∈ {2048, 4096, 8192}.

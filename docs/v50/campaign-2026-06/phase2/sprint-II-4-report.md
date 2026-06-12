# Phase II — Sprint II-4 report: conv3d small-K retune

**Date**: 2026-06-12 · **Status**: **DECLINED-AS-FRAMED — premise inverted; real lever identified, measured, Marco-gated**

## Premise inversion (re-investigate-before-acting paid off again)

The Sprint-C candidate read "up3_resnet at 41% of NAX peak → tile-config
retune headroom".  The repo's own primary data resolves the 41% as the
END-TO-END figure (Phase 1.5: im2col + GEMM); the PURE GEMM at this
shape runs at **39.5 TF = 104% of the 38-TF peak** (Phase 1.1 microbench
table).  There is NO tile-retune headroom — the matmul2d config
(M=N=K_TILE=32, 1 SG) is already at/above advertised peak for K=3456.

## Empirical decomposition (this sprint, up3-class shape M=356k K=3456 N=128)

| Component | ms | share |
|---|---|---|
| End-to-end NAX conv3d (telemetry-confirmed engaged) | 20.8 | 100% |
| Pure-GEMM floor at the measured 39.5 TF | 8.0 | 38% |
| **im2col materialization + overhead** | **12.8** | **62%** |

MLX baseline at the same shape: 20.7ms (1.00× — the documented parity
point; K=3456 is where NAX's GEMM advantage is fully eaten by im2col
traffic ≈ M×K×2B ≈ 2.3 GB written + read).

## The real lever (recorded, Marco-gated)

**Fused/implicit im2col**: read x directly inside the matmul kernel
with in-kernel patch indexing, eliminating the materialized im2col
buffer.  Ceiling on K=3456 shapes: ~2.6× (20.8 → ~8 ms).  Effort: XL
(rewrites the kernel's operand loading; same class as the II-3 XL
variant).  Larger-K shapes (13824) gain less (im2col already amortized:
mid_resnet at 87% of peak end-to-end).  Marco-gated kernel-sprint
candidate with this measured decomposition as the premise.

## KD-7 gate check (per sprint scope)

fp16-only eligibility confirmed intact (`_conv3d_nax_eligible` weight
gate + A-8 legacy-path guard from Sprint A).  bf16 stays
upstream-gated; not fought here.

# Audit Phase B4 — GNA / Conv / TopK / Sage / Paged-TQ (sprint report; CLOSES Phase B)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `335f5bc`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight: `mlx-debug-forensics`,
`benchmark-measurement-correctness`, `metal-kernel-dev`. **No kernel/routing/threshold/bug change**
(comments clean). Durable spec: `b4-family-spec.md`; lock: `tests/test_b4_family_lock.py` (9 cells) +
existing IV-D1/D2 lock.

## (1) Per-kernel specs — see `b4-family-spec.md`
GNA (per-element neighborhood, native STEEL), conv3d-nax (im2col+matmul2d auto-hook), topk (top-k +
SDPA), sage (int8 matmul2d), paged/TQ (gather/dequant + SDPA) — each spec'd with constraints + dispatch.

## (2) THE GNA RESOLUTION
GNA matches the **EXACT per-element-window fp32 oracle** (the documented `make_gna_mask` rule, manual)
to **max_abs_err 4.8e-5** across sliding (3³, 5³) + strided (2³). **GNA is CORRECT** — the Phase-A
7.3e-2 was a block-mask reference over-approximation (the block mask over-counts vs the exact
per-element window), **NOT a bug**. Deferred Phase-A item closed.

## (3) Per-type correctness (each its own oracle/discipline)
| Kernel | Oracle | Result |
|---|---|---|
| conv3d-nax eligible | fp32 mx.conv_general | 2.4e-4, cos 1.00000 |
| conv3d-nax ineligible fallback | fp32 mx.conv_general | 1.1e-4 |
| topk ratio=0.25 | fp32 top-k attention | 1.6e-3, cos 0.99981 |
| topk ratio=1.0 | fp32 dense | 9.3e-6 |
| **sage int8 quant round-trip** | int8 step | **4.0e-4 ≤ step 7.9e-4 = FAITHFUL** |
| **sage int8 attention** | fp32 dense | cos **0.997** (principled int8 floor, stable across amp; locked ≥0.995) |
| paged decode | fp32 gather | 2.3e-6, cos 1.00000 |
| IV-D1/D2 deferred==eager | (existing lock) | 3 passed, bit-identical |

Sage discipline applied: faithful quant + a **principled** int8 cos floor (~0.997 from 7-bit int8 over
D=128, measured stable) — not an arbitrarily-loose bound; the int8 GEMM is correct.

## (4) Threshold audit + comment sweep
No arbitrary/overflow threshold in this family (conv MPP gate = HW divisibility; topk ratio = param;
GNA window = documented; paged = decode shape). **Open Phase-E item:** sparse V1↔V2 `2^31` PERF
validity (overflow benign per B2). Comments clean — no edits.

## (5) PHASE B COMPLETE
Every kernel in the repo now has a **verified, fp32/oracle-correctness-locked spec**:
- B1 sparse/LCSA (V2 matmul2d + V1 scalar, 13 cells)
- B2 dense STEEL (8 variants, 14 cells incl. source-threshold locks)
- B3 backward (per-gradient native/SDPA-vjp mix, 6 cells)
- B4 GNA/conv/topk/sage/paged (9 cells) — GNA correctness resolved, sage int8 quant-aware
Total per-kernel correctness locks added this Phase B: **42 cells** across 4 spec docs. No
kernel/routing/threshold/bug change anywhere in Phase B (keep-all-paths; comment-only fixes in B1).

## Disposition
The most heterogeneous family audited, the deferred GNA correctness resolved on a correct per-element
oracle (not assumed away), and **Phase B (per-kernel) CLOSED**. Suite green. No orphans. Not tagged.
**Phase C (test audit — does each existing test exercise the path it claims, now that every kernel's
ground-truth is known?) is next.**

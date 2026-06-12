# Phase II — Sprint II-3 report: Top-K Streaming Approach 5

**Date**: 2026-06-12 · **Status**: **DECLINED — built, benched, kill criterion fired**

## What was built

PASS-1 streaming top-K kernel (`mlx_mfa/topk_stream.py`, standard MSL via
`mx.fast.metal_kernel` — no MPP dependency): BQ=32 rows/TG (4 SGs × 8
rows), BK=32 K-tiles staged in TGM, per-row running top-K (fp16 scores +
int32 idx, 24-32KB TGM), SIMD-parallel min-rescan inserts with a
whole-tile skip fast-path, ragged N/S edge handling.  PASS-2 per the
Phase-A refinement: `mx.put_along_axis` scatter-bias → Apple SDPA NAX.

## Correctness (PASS-1: EXACT)

Top-K score-SET equality vs the materialized reference at every row
across 4 shape families (MHA, N≠S, D=64, ragged edges): **worst
deviation 0.0**.  End-to-end vs the bisection path: rmse 0.012 — the
documented FP16 tie-set semantics difference (bisection admits 64-69
elements at threshold ties; this kernel selects exactly K), not a bug.

## Bench (audit shape B=1 H=16 N=S=4096 D=128 fp16 K=64, 3-block median)

| Path | Total |
|---|---|
| Architecture B bisection (current default) | **11.32 ms** |
| Approach 5 (this build) | 75.24 ms (**0.15×**) |

Pre-registered kill criterion (entry-state doc): PASS-1 > 8 ms → decline.
PASS-1 measured ~70 ms.  Root cause: scalar threadgroup-memory dot
products run ~15× below matmul-grade efficiency; the 4 ms PASS-1 model
assumed simdgroup_matrix tiling.

## Why the XL variant is not pursued (recorded ceiling)

A STEEL-style simdgroup_matrix rewrite (XL, 8-12h-class) could approach
the model — but PASS-1 can never beat the ~4 ms score-matmul floor, so
the end-to-end ceiling is ~7 ms vs 11.3 ms = **hard 1.6× maximum** for
perfect execution of an XL build.  Marco-gated if ever wanted; the
measured 0.15× datapoint replaces the Phase-I estimate as the evidence.

## Artifact disposition

Kernel module KEPT in-tree (correct, isolated, never wired into
dispatch; DECLINED verdict in its docstring).  4 artifact-lock tests
keep the negative result reproducible (suite 1376 → 1380).
Architecture B remains the production top-K path.

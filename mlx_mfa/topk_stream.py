"""Approach-5 streaming top-K (campaign 2026-06 Sprint II-3) — **DECLINED**.

SPRINT II-3 VERDICT (2026-06-12, measured): the pre-registered kill
criterion fired.  PASS-1 (this kernel, naive scalar-TGM dots) measured
~70 ms at the audit shape (B=1 H=16 N=S=4096 D=128 K=64) vs the 8 ms
kill threshold; end-to-end 75.2 ms vs Architecture B's 11.3 ms = 0.15x.
The scalar dot loop is ~15x off matmul-grade efficiency; reaching the
~4 ms PASS-1 model requires the full STEEL-style simdgroup_matrix tile
pipeline (XL effort), and even perfect execution caps the end-to-end
win at ~1.6x (PASS-1 cannot beat the 4 ms score-matmul floor; total
~7 ms vs 11.3 ms).  The XL variant is recorded as Marco-gated with that
hard ceiling.  Architecture B (bisection) remains the production path.

This module is KEPT (correct, isolated, never wired into dispatch) as
the measured artifact + revival base; correctness is locked by
tests/test_phase2_ii3_topk_stream.py so the negative result stays
reproducible.

--- original design notes below ---


PASS-1: a standard-MSL `mx.fast.metal_kernel` computes per-query-row
top-K key INDICES by streaming K-tiles through threadgroup memory and
maintaining a running unsorted top-K buffer per row — the [B,H,N,S]
score tensor is NEVER materialized (Architecture B's dominant cost).

PASS-2 (the II-3 Phase-A refinement that revived this design): the
indices build the additive -inf bias via mx.put_along_axis scatter,
then Apple SDPA NAX runs unchanged — no filtered-SDPA kernel needed.

Gated behind MFA_TOPK_STREAM_V5=1 (NOT default until it beats the
Architecture-B bisection path on the bench, per Pattern #6).

Kernel geometry (per metal-kernel-dev review, on session record):
  grid TG = (ceil(N/BQ), H, B), 128 threads = 4 SGs x 32 lanes
  BQ = 32 query rows per TG (8 rows per SG)  -- amortizes K-tile loads
  BK = 32 keys per tile, staged in TGM once per TG
  per-row state in TGM: K_top fp32 scores + K_top int32 idx + min cache
TGM: Q stage 32*D*2 + K tile 32*D*2 + heap 32*K_top*8 + min 32*8
  D=128, K_top=64: 8KB + 8KB + 16KB + 256B = ~32KB (TGP limit; D=64: 24KB)
"""
from __future__ import annotations

import math
import os

import mlx.core as mx

_KERNELS: dict = {}

_SRC_TMPL = r"""
    // params: [B, H, N, S, D, K_top, n_tiles]
    const int B = params[0];
    const int H = params[1];
    const int N = params[2];
    const int S = params[3];
    constexpr int D = __D__;
    constexpr int K_TOP = __K_TOP__;
    constexpr int BQ = 32;
    constexpr int BK = 32;

    const int b  = thread_position_in_grid.z;
    const int h  = thread_position_in_grid.y;
    const int qb = thread_position_in_grid.x / 128;   // TG index along N
    const uint lid  = thread_position_in_threadgroup.x;   // 0..127
    const uint sg   = lid / 32;                            // 0..3
    const uint lane = lid % 32;

    threadgroup half  Qs[BQ * D];
    threadgroup half  Ks[BK * D];
    // fp16 heap scores: halves TGM (D=128 was 256B over the 32KB cap);
    // within the documented FP16-boundary tie semantics (Architecture B
    // has the same ambiguity class at the K-th-value boundary).
    threadgroup half heap_s[BQ * K_TOP];
    threadgroup int   heap_i[BQ * K_TOP];
    threadgroup float row_min[BQ];
    threadgroup int   row_min_pos[BQ];

    const int row0 = qb * BQ;
    const device half* Qbase = q + ((size_t)b * H + h) * (size_t)N * D;
    const device half* Kbase = k + ((size_t)b * H + h) * (size_t)S * D;

    // ---- stage Q rows (zero-pad rows beyond N) ----
    for (int i = lid; i < BQ * D; i += 128) {
        const int r = i / D;
        Qs[i] = (row0 + r < N) ? Qbase[(size_t)(row0 + r) * D + (i % D)]
                               : (half)0.0h;
    }
    // ---- init heaps ----
    for (int i = lid; i < BQ * K_TOP; i += 128) {
        heap_s[i] = (half)(-INFINITY);
        heap_i[i] = 0;
    }
    if (lid < BQ) { row_min[lid] = -INFINITY; row_min_pos[lid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int n_tiles = (S + BK - 1) / BK;
    for (int t = 0; t < n_tiles; ++t) {
        const int key0 = t * BK;
        // stage K tile (zero-pad beyond S)
        for (int i = lid; i < BK * D; i += 128) {
            const int kr = i / D;
            Ks[i] = (key0 + kr < S) ? Kbase[(size_t)(key0 + kr) * D + (i % D)]
                                    : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // each SG handles rows [sg*8, sg*8+8); lane j scores key j
        const int my_key = key0 + (int)lane;
        const bool key_ok = my_key < S;
        for (int rr = 0; rr < 8; ++rr) {
            const int r = (int)sg * 8 + rr;             // 0..31
            float acc = 0.0f;
            const threadgroup half* qrow = Qs + r * D;
            const threadgroup half* krow = Ks + lane * D;
            #pragma clang loop unroll_count(8)
            for (int d = 0; d < D; ++d) {
                acc += (float)qrow[d] * (float)krow[d];
            }
            float score = key_ok ? acc * scale[0] : -INFINITY;

            // running top-K update for row r (SG-serial over candidates)
            float rmin = row_min[r];
            // fast path: whole tile below the current min -> skip
            if (metal::simd_all(score <= rmin)) continue;
            for (uint c = 0; c < 32; ++c) {
                const float sc  = metal::simd_broadcast(score, c);
                const int   idx = key0 + (int)c;
                if (sc > rmin && idx < S) {
                    // replace current min slot (lane 0 writes)
                    if (lane == 0) {
                        heap_s[r * K_TOP + row_min_pos[r]] = (half)sc;
                        heap_i[r * K_TOP + row_min_pos[r]] = idx;
                    }
                    metal::simdgroup_barrier(mem_flags::mem_threadgroup);
                    // SIMD-parallel rescan for the new min (K_TOP/32 each)
                    float mv = INFINITY; int mp = 0;
                    for (int u = (int)lane; u < K_TOP; u += 32) {
                        const float v = (float)heap_s[r * K_TOP + u];
                        if (v < mv) { mv = v; mp = u; }
                    }
                    const float gmin = metal::simd_min(mv);
                    // lane holding the min publishes (ties: lowest lane wins)
                    const bool is_min = (mv == gmin);
                    const uint first = (uint)metal::ctz((metal::ulong)metal::simd_ballot(is_min));
                    if (lane == first) {
                        row_min[r] = gmin;
                        row_min_pos[r] = mp;
                    }
                    metal::simdgroup_barrier(mem_flags::mem_threadgroup);
                    rmin = gmin;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- write indices: out[b,h,row,:] ----
    for (int i = lid; i < BQ * K_TOP; i += 128) {
        const int r = i / K_TOP;
        const int row = row0 + r;
        if (row < N) {
            out_idx[(((size_t)b * H + h) * N + row) * K_TOP + (i % K_TOP)]
                = heap_i[i];
        }
    }
"""


def _get_kernel(D: int, K_top: int):
    key = (D, K_top)
    kern = _KERNELS.get(key)
    if kern is None:
        kern = mx.fast.metal_kernel(
            name=f"topk_stream_v5_D{D}_K{K_top}",
            input_names=["q", "k", "scale", "params"],
            output_names=["out_idx"],
            source=_SRC_TMPL.replace("__D__", str(D)).replace("__K_TOP__", str(K_top)),
            ensure_row_contiguous=True,
        )
        _KERNELS[key] = kern
    return kern


def topk_stream_indices(q: mx.array, k: mx.array, scale: float,
                        k_count: int) -> mx.array:
    """PASS-1: per-row top-k_count key indices, [B, H, N, k_count] int32."""
    B, H, N, D = q.shape
    S = k.shape[2]
    if D not in (64, 128):
        raise ValueError(f"topk_stream: D must be 64 or 128, got {D}")
    if k_count % 32 != 0 or k_count > 128:
        raise ValueError(f"topk_stream: k_count must be a multiple of 32 "
                         f"<= 128, got {k_count}")
    kern = _get_kernel(D, k_count)
    params = mx.array([B, H, N, S, D, k_count, 0], dtype=mx.int32)
    scale_arr = mx.array([scale], dtype=mx.float32)
    n_tg = (N + 31) // 32
    (idx,) = kern(
        inputs=[q, k, scale_arr, params],
        grid=(n_tg * 128, H, B),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, H, N, k_count)],
        output_dtypes=[mx.int32],
    )
    return idx


def topk_stream_attention(q: mx.array, k: mx.array, v: mx.array,
                          scale: float, k_count: int) -> mx.array:
    """Full Approach-5: PASS-1 indices -> scatter bias -> Apple SDPA NAX."""
    B, H, N, _ = q.shape
    S = k.shape[2]
    idx = topk_stream_indices(q, k, scale, k_count)
    bias = mx.full((B, H, N, S), float("-inf"), dtype=q.dtype)
    bias = mx.put_along_axis(bias, idx.astype(mx.int64) if False else idx,
                             mx.zeros(idx.shape, dtype=q.dtype), axis=-1)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale,
                                                mask=bias)

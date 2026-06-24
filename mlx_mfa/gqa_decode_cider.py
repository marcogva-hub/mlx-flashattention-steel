"""GQA-decode attention kernel ported from Mininglamp-AI/cider (Sprint II-11).

INTERNAL module — not part of the public API (absent from ``mlx_mfa.__all__``).
Use via the high-level decode/serving entry points, not by direct import.

Port of cider's v9 ``cider_sdpa_vector_2pass`` kernels (MIT License,
Copyright (c) 2026 Mininglamp contributors — https://github.com/Mininglamp-AI/cider),
adapted to mlx-mfa conventions via ``mx.fast.metal_kernel``.

Architecture (the II-5-surveyed win): ONE threadgroup column per KV head
serves ALL gqa_factor Q heads of the group (threadgroup y-dim =
gqa_factor), so each K/V chunk is read once per group instead of once
per Q head — DRAM traffic / gqa_factor.  Contiguous-chunk split-KV
(BLOCKS z-dim) with FlashInfer-style TILE=4 register tiling; pass 2
reduces the per-chunk partials.

Scope: dense fp16/bf16/fp32 decode, N_q == 1, contiguous [B, H, S, D]
K/V.  NOTE (CC-22 audit): the `MFA_GQA_DECODE_CIDER` env flag is NOT wired up
(no read site) — it is a no-op / dormant gate, listed in `_knobs.REMOVED_KNOBS`.
This module is reachable only by direct import.  See sprint-II-11-report.md
for the measured window and the promote/decline decision.

N and the K/V strides are runtime inputs (a params buffer), NOT baked
into the source — decode N grows every step and baking it would
recompile per token (the Phase-I paged-RoPE compile-churn lesson).
"""
from __future__ import annotations

import math

import mlx.core as mx

_HDR = """
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
"""

# pass 1 body — per-(kv_head, batch, block) threadgroups, TG=(32, gqa, 1).
_P1_BODY_TMPL = """
    constexpr int D = __D__;
    constexpr int BLOCKS = __BLOCKS__;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    constexpr int TILE = 4;
    typedef float U;

    uint3 tptg = threads_per_threadgroup;
    uint3 tidtg = thread_position_in_threadgroup;
    uint3 tid = threadgroup_position_in_grid;
    uint3 tpg = threadgroups_per_grid;
    uint simd_lid = thread_index_in_simdgroup;

    const int N = (int)params[0];
    const size_t k_head_stride = (size_t)params[1];
    const size_t k_seq_stride  = (size_t)params[2];
    const size_t v_head_stride = (size_t)params[3];
    const size_t v_seq_stride  = (size_t)params[4];
    const float scale = scl[0];

    thread U q_reg[qk_per_thread];
    thread U o_reg[v_per_thread] = {0};

    const int kv_head_idx = tid.x;
    const int batch_idx = tid.y;
    const int block_idx = tid.z;
    const int gqa_factor = tptg.y;
    const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
    const int num_kv_heads = tpg.x;
    const int num_q_heads = num_kv_heads * gqa_factor;
    const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
    const int o_offset = q_batch_head_idx;

    const device T* q_ptr = q + o_offset * D + simd_lid * qk_per_thread;
    const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;

    const int chunk_size = (N + BLOCKS - 1) / BLOCKS;
    const int kv_start = block_idx * chunk_size;
    const int kv_end = min(kv_start + chunk_size, N);

    const device T* k_ptr = k + kv_batch_head_idx * (int)k_head_stride
                            + kv_start * (int)k_seq_stride + simd_lid * qk_per_thread;
    const device T* v_ptr = v + kv_batch_head_idx * (int)v_head_stride
                            + kv_start * (int)v_seq_stride + simd_lid * v_per_thread;

    device T* o_ptr = partials + o_offset * BLOCKS * D + block_idx * D
                      + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
        q_reg[i] = (U)scale * (U)q_ptr[i];
    }

    U max_score = -1e38f;
    U sum_exp_score = 0;
    const int kss = (int)k_seq_stride;
    const int vss = (int)v_seq_stride;

    int pos = kv_start;
    const int tiled_end = kv_start + ((kv_end - kv_start) / TILE) * TILE;
    for (; pos < tiled_end; pos += TILE) {
        U scores[TILE];
        for (int t = 0; t < TILE; t++) {
            U score = 0;
            const device T* kt = k_ptr + t * kss;
            for (int j = 0; j < qk_per_thread; j++) score += q_reg[j] * (U)kt[j];
            scores[t] = simd_sum(score);
        }
        for (int t = 0; t < TILE; t++) {
            U new_max = max(max_score, scores[t]);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(scores[t] - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            const device T* vt = v_ptr + t * vss;
            for (int j = 0; j < v_per_thread; j++)
                o_reg[j] = o_reg[j] * factor + exp_score * (U)vt[j];
        }
        k_ptr += TILE * kss;
        v_ptr += TILE * vss;
    }
    for (; pos < kv_end; pos++) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) score += q_reg[j] * (U)k_ptr[j];
        score = simd_sum(score);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++)
            o_reg[j] = o_reg[j] * factor + exp_score * (U)v_ptr[j];
        k_ptr += kss;
        v_ptr += vss;
    }

    if (simd_lid == 0) {
        sums[o_offset * BLOCKS + block_idx] = sum_exp_score;
        maxs[o_offset * BLOCKS + block_idx] = max_score;
    }
    for (int i = 0; i < v_per_thread; i++) {
        o_ptr[i] = (T)o_reg[i];
    }
"""

# pass 2 body — per-q-head threadgroups, TG=(1024,1,1) = 32 SGs.
_P2_BODY_TMPL = """
    constexpr int D = __D__;
    constexpr int BLOCKS = __BLOCKS__;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    thread U o_reg[elem_per_thread] = {0};
    threadgroup U tg_outputs[BN * BD];

    const int head_idx = tid.x;
    const device T* p_ptr = partials + head_idx * BLOCKS * D
                            + simd_gid * D + simd_lid * elem_per_thread;
    const device float* s_ptr = sums + head_idx * BLOCKS;
    const device float* m_ptr = maxs + head_idx * BLOCKS;

    U max_score = -1e38f;
    for (int b = 0; b < BLOCKS / BN; ++b)
        max_score = max(max_score, m_ptr[simd_lid + BN * b]);
    max_score = simd_max(max_score);

    U sum_exp_score = 0;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(m_ptr[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * s_ptr[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    const device float* m_walk = m_ptr;
    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(m_walk[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++)
            o_reg[i] += factor * (U)p_ptr[i];
        m_walk += BN;
        p_ptr += BN * D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o_reg[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid]);
        o_reg[i] = sum_exp_score == 0 ? o_reg[i] : (o_reg[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device T* optr = out + head_idx * D + simd_gid * elem_per_thread;
        for (int i = 0; i < elem_per_thread; i++)
            optr[i] = (T)o_reg[i];
    }
"""

_KERNEL_CACHE: dict = {}


def _kernels(D: int, blocks: int):
    key = (D, blocks)
    got = _KERNEL_CACHE.get(key)
    if got is not None:
        return got
    p1 = mx.fast.metal_kernel(
        name=f"mfa_cider_gqa_p1_d{D}_b{blocks}",
        input_names=["q", "k", "v", "params", "scl"],
        output_names=["partials", "sums", "maxs"],
        source=_P1_BODY_TMPL.replace("__D__", str(D)).replace("__BLOCKS__", str(blocks)),
        header=_HDR,
        ensure_row_contiguous=True,
    )
    p2 = mx.fast.metal_kernel(
        name=f"mfa_cider_gqa_p2_d{D}_b{blocks}",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=_P2_BODY_TMPL.replace("__D__", str(D)).replace("__BLOCKS__", str(blocks)),
        header=_HDR,
        ensure_row_contiguous=True,
    )
    _KERNEL_CACHE[key] = (p1, p2)
    return p1, p2


def _pick_blocks(S: int) -> int:
    if S < 8192:
        return 32
    if S < 24576:
        return 64
    return 128


def gqa_decode_cider(q: mx.array, k: mx.array, v: mx.array,
                     scale: float | None = None) -> mx.array:
    """cider-ported GQA decode attention: q [B,Hq,1,D], k/v [B,Hkv,S,D]."""
    B, Hq, Nq, D = q.shape
    _, Hkv, S, _ = k.shape
    if Nq != 1:
        raise ValueError("gqa_decode_cider: decode only (N_q == 1)")
    if Hq % Hkv != 0:
        raise ValueError("gqa_decode_cider: Hq must be a multiple of Hkv")
    # M1 (CC final-cert): the P1 kernel bakes D from q and S/Hkv from k and strides
    # the V pool by them — v.shape was NEVER read, so q.D!=v.D / k.S!=v.S / k.Hkv!=v.Hkv
    # drove an OOB read of the V buffer (finite but NON-deterministic across calls =
    # uninitialized memory). Cross-check k.D==q.D and v fully matches k before dispatch.
    if k.shape[3] != D:
        raise ValueError(
            f"gqa_decode_cider: k head_dim ({k.shape[3]}) must equal q head_dim ({D}).")
    if tuple(v.shape) != (B, Hkv, S, D):
        raise ValueError(
            f"gqa_decode_cider: v shape {tuple(v.shape)} must equal k shape "
            f"{(B, Hkv, S, D)} (batch, kv-heads, kv-seq, head_dim) — the kernel reads "
            "V at K's strides and would read out of bounds otherwise.")
    # III-4 R12 FIX: the kernel splits D over BD=32 lanes (qk_per_thread =
    # D / BD, integer division) — a non-multiple of 32 (e.g. D=80) silently
    # truncates the head dimension and produces wrong output.
    if D % 32 != 0:
        raise ValueError(
            f"gqa_decode_cider: head_dim must be a multiple of 32 "
            f"(kernel tiles D over 32 SIMD lanes), got D={D}"
        )
    gqa = Hq // Hkv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    blocks = _pick_blocks(S)
    p1, p2 = _kernels(D, blocks)

    # params: N, k strides (head, seq), v strides; scale as f32 input
    params = mx.array([S, S * D, D, S * D, D], dtype=mx.int64)
    scl = mx.array([scale], dtype=mx.float32)

    partials, sums, maxs = p1(
        inputs=[q, k, v, params, scl],
        template=[("T", q.dtype)],
        grid=(Hkv * 32, B * gqa, blocks),
        threadgroup=(32, gqa, 1),
        output_shapes=[(B * Hq, blocks, D), (B * Hq, blocks), (B * Hq, blocks)],
        output_dtypes=[q.dtype, mx.float32, mx.float32],
    )
    (out,) = p2(
        inputs=[partials, sums, maxs],
        template=[("T", q.dtype)],
        grid=(B * Hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(B * Hq, D)],
        output_dtypes=[q.dtype],
    )
    return out.reshape(B, Hq, 1, D)

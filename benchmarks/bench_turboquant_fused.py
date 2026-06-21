#!/usr/bin/env python3
"""Benchmark TurboQuant Phase 2: fused kernel vs decompress-then-attend vs fp16.

Compares three paths for paged varlen attention:
  1. fp16 baseline: flash_attention_paged_varlen (full-precision K pool)
  2. Phase 1: turboquant_decompress → flash_attention_paged_varlen
  3. Phase 2: flash_attention_paged_varlen_turboquant (fused K dequant)

Reports: latency (ms), speedup vs fp16, speedup vs Phase 1.
"""
import math
import sys

import mlx.core as mx
import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).parent))  # repo review 2026-05: allow `python benchmarks/<f>.py` from repo root
from bench_utils import med, check_mfa_available

check_mfa_available()


def _build_fp16_pool(k_seqs, v_seqs, block_size):
    """Build fp16 paged pool from per-sequence KV tensors."""
    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]
    blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    pool_k = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    table = np.full((B, max_blocks), 0, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)

    blk = 0
    for b in range(B):
        k_np = np.array(k_seqs[b].astype(mx.float16))[0].transpose(1, 0, 2)
        v_np = np.array(v_seqs[b].astype(mx.float16))[0].transpose(1, 0, 2)
        S = k_np.shape[0]
        lens[b] = S
        n_blk = blocks_per_seq[b]
        for lb in range(n_blk):
            table[b, lb] = blk + lb
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            pool_k[blk + lb, :s1 - s0] = k_np[s0:s1]
            pool_v[blk + lb, :s1 - s0] = v_np[s0:s1]
        blk += n_blk

    return mx.array(pool_k), mx.array(pool_v), mx.array(table, dtype=mx.int32), mx.array(lens, dtype=mx.int32)


def _build_tq_pool(k_seqs, v_seqs, block_size, bits=3):
    """Build TQ-packed K pool + fp16 V pool."""
    from mlx_mfa.turboquant import pack_k_for_metal, _get_centroids

    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]
    packed_D = D // 2
    blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    pool_k_tq = np.zeros((total_blocks, block_size, H_kv, packed_D), dtype=np.uint8)
    pool_k_scales = np.zeros((total_blocks, block_size, H_kv), dtype=np.float32)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float16)
    table = np.full((B, max_blocks), 0, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)

    _, centroids_f32 = _get_centroids(bits)
    centroids_fp16 = centroids_f32.astype(mx.float16)

    blk = 0
    for b in range(B):
        S = k_seqs[b].shape[2]
        lens[b] = S
        k_packed, k_scales, _ = pack_k_for_metal(k_seqs[b], bits=bits)
        mx.synchronize()

        k_packed_np = np.array(k_packed)[0]
        k_scales_np = np.array(k_scales.astype(mx.float32))[0]
        v_np = np.array(v_seqs[b].astype(mx.float16))[0]

        n_blk = blocks_per_seq[b]
        for lb in range(n_blk):
            table[b, lb] = blk + lb
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            chunk = s1 - s0
            pool_k_tq[blk + lb, :chunk] = k_packed_np.transpose(1, 0, 2)[s0:s1]
            pool_k_scales[blk + lb, :chunk] = k_scales_np.transpose(1, 0)[s0:s1]
            pool_v[blk + lb, :chunk] = v_np.transpose(1, 0, 2)[s0:s1]
        blk += n_blk

    return (
        mx.array(pool_k_tq), mx.array(pool_v),
        mx.array(pool_k_scales, dtype=mx.float32), centroids_fp16,
        mx.array(table, dtype=mx.int32), mx.array(lens, dtype=mx.int32),
    )


def _pack_queries(q_seqs):
    offsets = [0]
    for q in q_seqs:
        offsets.append(offsets[-1] + int(q.shape[2]))
    q_pack = mx.concatenate(q_seqs, axis=2)
    cu = mx.array(offsets, dtype=mx.int32)
    return q_pack, cu


def bench():
    from mlx_mfa import flash_attention_paged_varlen, flash_attention_paged_varlen_turboquant
    from mlx_mfa.turboquant import turboquant_compress, turboquant_decompress, apply_rotation

    configs = [
        # (H_q, H_kv, D, q_lens, kv_lens, bits, causal, label)
        (8, 8, 64,  [1],    [512],   3, False, "decode D=64 S=512"),
        (8, 8, 128, [1],    [512],   3, False, "decode D=128 S=512"),
        (8, 8, 128, [1],    [2048],  3, False, "decode D=128 S=2048"),
        (8, 8, 64,  [32],   [512],   3, True,  "prefill D=64 Q=32 S=512"),
        (8, 8, 128, [32],   [512],   3, True,  "prefill D=128 Q=32 S=512"),
        (8, 2, 128, [1],    [2048],  3, False, "GQA 8:2 decode D=128 S=2048"),
        (8, 8, 128, [1],    [512],   2, False, "2-bit decode D=128 S=512"),
        (8, 8, 128, [1],    [512],   4, False, "4-bit decode D=128 S=512"),
        (8, 8, 128, [3,1,4],[512,256,768], 3, False, "multi-seq decode"),
    ]

    block_size = 16

    hdr = f"{'Config':42s}  {'fp16 ms':>8}  {'P1 ms':>8}  {'P2 ms':>8}  {'P2/fp16':>8}  {'P2/P1':>8}"
    print(hdr)
    print("-" * len(hdr))

    mx.random.seed(42)
    for H_q, H_kv, D, q_lens, kv_lens, bits, causal, label in configs:
        scale = 1.0 / math.sqrt(D)

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.synchronize()

        # --- fp16 baseline ---
        pool_k, pool_v, table, lens = _build_fp16_pool(k_seqs, v_seqs, block_size)
        q_pack, cu_q = _pack_queries(q_seqs)

        ms_fp16 = med(
            lambda: flash_attention_paged_varlen(
                q_pack, pool_k, pool_v, table, lens, cu_q,
                scale=scale, causal=causal, block_size=block_size,
            ),
            warmup=3, iters=10,
        )

        # --- Phase 1: decompress + attend ---
        compressed_seqs = [
            turboquant_compress(k, bits=bits, use_qjl=False, rotation="wht")
            for k in k_seqs
        ]
        mx.synchronize()

        def phase1():
            k_dec_list = [turboquant_decompress(c) for c in compressed_seqs]
            k_dec_pack = mx.concatenate(k_dec_list, axis=2) if len(k_dec_list) > 1 else k_dec_list[0]
            # Rebuild fp16 pool (simplified: single concat for single-seq, approximate for multi)
            # For timing: we measure decompress + attend together
            pool_k_dec, pool_v_dec, tab_dec, lens_dec = _build_fp16_pool(
                [k_d.astype(mx.float16) for k_d in [turboquant_decompress(c) for c in compressed_seqs]],
                v_seqs, block_size,
            )
            return flash_attention_paged_varlen(
                q_pack, pool_k_dec, pool_v_dec, tab_dec, lens_dec, cu_q,
                scale=scale, causal=causal, block_size=block_size,
            )

        ms_p1 = med(phase1, warmup=2, iters=5)

        # --- Phase 2: fused TQ kernel ---
        q_rot_seqs = [apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16) for q in q_seqs]
        mx.synchronize()
        q_rot_pack, cu_q_rot = _pack_queries(q_rot_seqs)
        pool_k_tq, pool_v_tq, k_scales, centroids, table_tq, lens_tq = _build_tq_pool(
            k_seqs, v_seqs, block_size, bits=bits,
        )

        ms_p2 = med(
            lambda: flash_attention_paged_varlen_turboquant(
                q_rot_pack, pool_k_tq, pool_v_tq, table_tq, lens_tq, cu_q_rot,
                centroids, k_scales,
                scale=scale, causal=causal, block_size=block_size, tq_bits=bits,
            ),
            warmup=3, iters=10,
        )

        ratio_fp16 = ms_p2 / ms_fp16 if ms_fp16 > 0 else 0
        ratio_p1 = ms_p2 / ms_p1 if ms_p1 > 0 else 0

        print(
            f"{label:42s}  {ms_fp16:8.3f}  {ms_p1:8.3f}  {ms_p2:8.3f}"
            f"  {ratio_fp16:8.2f}x  {ratio_p1:8.2f}x"
        )


if __name__ == "__main__":
    from _bench_guard import require_accel_or_die as _phantom_gate  # audit H7/H-09
    _phantom_gate(__file__)
    bench()

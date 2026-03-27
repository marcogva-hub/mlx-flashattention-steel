#!/usr/bin/env python3
"""bench_turboquant_full.py — Complete TurboQuant benchmark matrix.

Compares four paths for paged varlen attention:
  1. fp16 baseline: flash_attention_paged_varlen (full-precision KV pool)
  2. Phase 1: turboquant_compress → turboquant_decompress → fp16 attend
  3. Phase 2: K-only TQ fused kernel (V stays fp16)
  4. Phase 3: K+V TQ fused kernel (both quantized)

Measures: latency (ms), memory estimate (MB), quality (cosine similarity vs fp16)

Reports: table + JSON saved to devnotes/turboquant_full_bench.json
"""
import json
import math
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_utils import med, check_mfa_available

check_mfa_available()

from mlx_mfa import (
    flash_attention_paged_varlen,
    flash_attention_paged_varlen_turboquant,
)
from mlx_mfa.turboquant import (
    turboquant_compress,
    turboquant_decompress,
    pack_k_for_metal,
    pack_v_for_metal,
    apply_rotation,
    _get_centroids,
    build_tq_paged_v_pool,
)


# ---------------------------------------------------------------------------
# Pool builders
# ---------------------------------------------------------------------------

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

    return (
        mx.array(pool_k), mx.array(pool_v),
        mx.array(table, dtype=mx.int32), mx.array(lens, dtype=mx.int32),
    )


def _build_tq_pool(k_seqs, v_seqs, block_size, bits=3):
    """Build TQ-packed K pool + fp16 V pool."""
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


def _cosine_sim(a, b):
    """Cosine similarity between two flattened arrays."""
    a_np = np.array(a.astype(mx.float32)).flatten()
    b_np = np.array(b.astype(mx.float32)).flatten()
    dot = np.dot(a_np, b_np)
    na = np.linalg.norm(a_np)
    nb = np.linalg.norm(b_np)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(dot / (na * nb))


def _kv_memory_mb(B, kv_lens, H_kv, D, mode="fp16"):
    """Estimate KV cache memory in MB."""
    total_tokens = sum(kv_lens) * B if isinstance(kv_lens[0], int) else sum(kv_lens)
    if mode == "fp16":
        # K + V both fp16: 2 * total_tokens * H_kv * D * 2 bytes
        return total_tokens * H_kv * D * 2 * 2 / (1024 * 1024)
    elif mode == "k_only_tq":
        # K: packed_D uint8 + scales float32; V: fp16
        k_bytes = total_tokens * H_kv * (D // 2 + 4)  # packed + scale
        v_bytes = total_tokens * H_kv * D * 2
        return (k_bytes + v_bytes) / (1024 * 1024)
    elif mode == "kv_tq":
        # K+V both packed
        kv_bytes = 2 * total_tokens * H_kv * (D // 2 + 4)
        return kv_bytes / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench():
    configs = [
        # (q_lens, kv_lens, H_q, H_kv, D, label)
        # Llama-3-8B decode
        ([1],    [2048],  32, 8, 128, "Llama-8B 1seq 2K"),
        ([1],    [8192],  32, 8, 128, "Llama-8B 1seq 8K"),
        ([1]*4,  [2048]*4, 32, 8, 128, "Llama-8B 4seq 2K"),
        ([1]*8,  [4096]*8, 32, 8, 128, "Llama-8B 8seq 4K"),
        # Llama-3-8B prefill
        ([512],  [512],   32, 8, 128, "Llama-8B prefill 512"),
        ([2048], [2048],  32, 8, 128, "Llama-8B prefill 2K"),
        # Qwen-2.5-7B (GQA 28:4)
        ([1],    [8192],  28, 4, 128, "Qwen-7B 1seq 8K"),
        ([1]*4,  [4096]*4, 28, 4, 128, "Qwen-7B 4seq 4K"),
        # Mixed decode
        ([1]*8, [512, 1024, 2048, 4096, 512, 1024, 2048, 4096], 32, 8, 128, "Mixed 8seq hetero"),
    ]

    block_size = 16
    bits = 3
    results = []

    hdr = (
        f"{'Config':30s}  {'fp16':>8}  {'P1':>8}  {'P2(K)':>8}  "
        f"{'P3(KV)':>8}  {'P3/fp16':>8}  {'P3/P1':>8}  "
        f"{'cos(P2)':>7}  {'cos(P3)':>7}  "
        f"{'fp16 MB':>8}  {'K-TQ MB':>8}  {'KV-TQ MB':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    mx.random.seed(42)
    for q_lens, kv_lens, H_q, H_kv, D, label in configs:
        B = len(q_lens)
        scale = 1.0 / math.sqrt(D)
        causal = any(ql > 1 for ql in q_lens)

        q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(mx.float16) for ql in q_lens]
        k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(mx.float16) for kl in kv_lens]
        mx.synchronize()

        # --- fp16 baseline ---
        pool_k, pool_v, table, lens = _build_fp16_pool(k_seqs, v_seqs, block_size)
        q_pack, cu_q = _pack_queries(q_seqs)

        out_fp16 = flash_attention_paged_varlen(
            q_pack, pool_k, pool_v, table, lens, cu_q,
            scale=scale, causal=causal, block_size=block_size,
        )
        mx.synchronize()

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
            k_dec_list = [turboquant_decompress(c).astype(mx.float16) for c in compressed_seqs]
            pool_k_dec, pool_v_dec, tab_dec, lens_dec = _build_fp16_pool(
                k_dec_list, v_seqs, block_size,
            )
            return flash_attention_paged_varlen(
                q_pack, pool_k_dec, pool_v_dec, tab_dec, lens_dec, cu_q,
                scale=scale, causal=causal, block_size=block_size,
            )

        ms_p1 = med(phase1, warmup=2, iters=5)

        # --- Phase 2: K-only TQ fused ---
        q_rot_seqs = [
            apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
            for q in q_seqs
        ]
        mx.synchronize()
        q_rot_pack, cu_q_rot = _pack_queries(q_rot_seqs)
        pool_k_tq, pool_v_tq_fp16, k_scales, centroids, table_tq, lens_tq = _build_tq_pool(
            k_seqs, v_seqs, block_size, bits=bits,
        )

        out_p2 = flash_attention_paged_varlen_turboquant(
            q_rot_pack, pool_k_tq, pool_v_tq_fp16, table_tq, lens_tq, cu_q_rot,
            centroids, k_scales,
            scale=scale, causal=causal, block_size=block_size, tq_bits=bits,
        )
        mx.synchronize()

        ms_p2 = med(
            lambda: flash_attention_paged_varlen_turboquant(
                q_rot_pack, pool_k_tq, pool_v_tq_fp16, table_tq, lens_tq, cu_q_rot,
                centroids, k_scales,
                scale=scale, causal=causal, block_size=block_size, tq_bits=bits,
            ),
            warmup=3, iters=10,
        )

        # --- Phase 3: K+V TQ fused ---
        v_pool_tq_packed, v_scales, v_centroids = build_tq_paged_v_pool(
            pool_v_tq_fp16, bits=bits
        )
        mx.synchronize()

        out_p3 = flash_attention_paged_varlen_turboquant(
            q_rot_pack, pool_k_tq, pool_v_tq_fp16, table_tq, lens_tq, cu_q_rot,
            centroids, k_scales,
            scale=scale, causal=causal, block_size=block_size, tq_bits=bits,
            tq_v_enabled=True,
            v_pool_tq=v_pool_tq_packed, v_centroids=v_centroids, v_scales=v_scales,
        )
        mx.synchronize()

        ms_p3 = med(
            lambda: flash_attention_paged_varlen_turboquant(
                q_rot_pack, pool_k_tq, pool_v_tq_fp16, table_tq, lens_tq, cu_q_rot,
                centroids, k_scales,
                scale=scale, causal=causal, block_size=block_size, tq_bits=bits,
                tq_v_enabled=True,
                v_pool_tq=v_pool_tq_packed, v_centroids=v_centroids, v_scales=v_scales,
            ),
            warmup=3, iters=10,
        )

        # Quality
        cos_p2 = _cosine_sim(out_fp16, out_p2)
        cos_p3 = _cosine_sim(out_fp16, out_p3)

        # Memory estimates
        mem_fp16 = _kv_memory_mb(1, kv_lens, H_kv, D, "fp16")
        mem_k_tq = _kv_memory_mb(1, kv_lens, H_kv, D, "k_only_tq")
        mem_kv_tq = _kv_memory_mb(1, kv_lens, H_kv, D, "kv_tq")

        ratio_fp16 = ms_p3 / ms_fp16 if ms_fp16 > 0 else 0
        ratio_p1 = ms_p3 / ms_p1 if ms_p1 > 0 else 0

        print(
            f"{label:30s}  {ms_fp16:8.3f}  {ms_p1:8.3f}  {ms_p2:8.3f}  "
            f"{ms_p3:8.3f}  {ratio_fp16:8.2f}x  {ratio_p1:8.2f}x  "
            f"{cos_p2:7.4f}  {cos_p3:7.4f}  "
            f"{mem_fp16:8.2f}  {mem_k_tq:8.2f}  {mem_kv_tq:8.2f}"
        )

        results.append({
            "label": label,
            "q_lens": q_lens, "kv_lens": kv_lens,
            "H_q": H_q, "H_kv": H_kv, "D": D, "bits": bits,
            "causal": causal,
            "ms_fp16": round(ms_fp16, 3),
            "ms_p1": round(ms_p1, 3),
            "ms_p2": round(ms_p2, 3),
            "ms_p3": round(ms_p3, 3),
            "ratio_p3_fp16": round(ratio_fp16, 3),
            "ratio_p3_p1": round(ratio_p1, 3),
            "cos_p2_vs_fp16": round(cos_p2, 5),
            "cos_p3_vs_fp16": round(cos_p3, 5),
            "mem_fp16_mb": round(mem_fp16, 2),
            "mem_k_tq_mb": round(mem_k_tq, 2),
            "mem_kv_tq_mb": round(mem_kv_tq, 2),
        })

    # Save JSON
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "devnotes", "turboquant_full_bench.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    bench()

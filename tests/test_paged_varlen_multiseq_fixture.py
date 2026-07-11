"""Paged-varlen multi-sequence fixture locks.

These tests protect the benchmark fixture used to compare page-native
PagedVarlenForward with materialize-then-attend baselines.  The historical
blocked measurement used an invalid batched-Q -> packed-Q reshape and produced
cosine around 0.08; the correct packed layout concatenates per-sequence
``[1,H,Q_i,D]`` tensors along the token axis.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _ext


def _cos_np(a, b) -> float:
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def _tile_offsets(q_lens: list[int], bq: int = 32) -> mx.array:
    offsets = [0]
    for ql in q_lens:
        offsets.append(offsets[-1] + (ql + bq - 1) // bq)
    return mx.array(offsets, dtype=mx.int32)


def _build_paged_pool(k_seqs, v_seqs, block_size: int, dtype):
    B = len(k_seqs)
    H_kv = k_seqs[0].shape[1]
    D = k_seqs[0].shape[3]
    blocks_per_seq = [(int(k.shape[2]) + block_size - 1) // block_size for k in k_seqs]
    total_blocks = sum(blocks_per_seq)
    max_blocks = max(blocks_per_seq)

    pool_k = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float32)
    pool_v = np.zeros((total_blocks, block_size, H_kv, D), dtype=np.float32)
    table = np.full((B, max_blocks), -1, dtype=np.int32)
    lens = np.zeros((B,), dtype=np.int32)

    base = 0
    for b, (k_b, v_b) in enumerate(zip(k_seqs, v_seqs)):
        k_np = np.array(k_b.astype(mx.float32))[0].transpose(1, 0, 2)
        v_np = np.array(v_b.astype(mx.float32))[0].transpose(1, 0, 2)
        S = k_np.shape[0]
        lens[b] = S
        for lb in range(blocks_per_seq[b]):
            s0 = lb * block_size
            s1 = min(S, s0 + block_size)
            table[b, lb] = base + lb
            pool_k[base + lb, : s1 - s0] = k_np[s0:s1]
            pool_v[base + lb, : s1 - s0] = v_np[s0:s1]
        base += blocks_per_seq[b]

    return (
        mx.array(pool_k).astype(dtype),
        mx.array(pool_v).astype(dtype),
        mx.array(table, dtype=mx.int32),
        mx.array(lens, dtype=mx.int32),
    )


def _causal_bias(q_len: int, kv_len: int) -> mx.array:
    q_pos = max(0, kv_len - q_len) + np.arange(q_len)[:, None]
    k_pos = np.arange(kv_len)[None, :]
    bias = np.where(k_pos <= q_pos, 0.0, -np.inf).astype(np.float32)
    return mx.array(bias)


def _fp32_oracle(q_seqs, k_seqs, v_seqs, scale: float, causal: bool):
    parts = []
    for q_i, k_i, v_i in zip(q_seqs, k_seqs, v_seqs):
        mask = _causal_bias(q_i.shape[2], k_i.shape[2]) if causal else None
        parts.append(
            mx.fast.scaled_dot_product_attention(
                q_i.astype(mx.float32),
                k_i.astype(mx.float32),
                v_i.astype(mx.float32),
                scale=scale,
                mask=mask,
            )
        )
    return mx.concatenate(parts, axis=2)


def _raw_paged_varlen(q_pack, pool_k, pool_v, table, lens, cu_q, q_lens, scale, causal, block_size):
    out, _lse = _ext.mfa_paged_varlen_forward(
        q_pack,
        pool_k,
        pool_v,
        cu_q,
        _tile_offsets(q_lens),
        table,
        lens,
        scale,
        causal,
        block_size,
    )
    return out


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_raw_paged_varlen_multiseq_fixture_matches_fp32_oracle(dtype, D, causal):
    mx.random.seed(911 + D + int(causal))
    H_q, H_kv = 8, 4
    q_lens = [3, 5, 2]
    kv_lens = [27, 19, 33]
    block_size = 16
    scale = 1.0 / math.sqrt(D)

    q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(dtype) for ql in q_lens]
    k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in kv_lens]
    v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in kv_lens]
    mx.eval(*q_seqs, *k_seqs, *v_seqs)

    q_pack = mx.concatenate(q_seqs, axis=2)
    cu_q = mx.array([0, 3, 8, 10], dtype=mx.int32)
    pool_k, pool_v, table, lens = _build_paged_pool(k_seqs, v_seqs, block_size, dtype)

    got = _raw_paged_varlen(
        q_pack, pool_k, pool_v, table, lens, cu_q, q_lens, scale, causal, block_size
    )
    ref = _fp32_oracle(q_seqs, k_seqs, v_seqs, scale, causal)
    mx.eval(got, ref)

    got_np = np.array(got.astype(mx.float32))
    ref_np = np.array(ref)
    assert np.isfinite(got_np).all()
    assert _cos_np(got_np, ref_np) >= 0.999


def test_batched_q_reshape_is_not_a_valid_packed_varlen_fixture():
    mx.random.seed(977)
    H_q = H_kv = 4
    D = 64
    q_lens = [4, 4, 4]
    kv_lens = [32, 40, 48]
    block_size = 16
    dtype = mx.float16
    scale = 1.0 / math.sqrt(D)

    q_seqs = [mx.random.normal((1, H_q, ql, D)).astype(dtype) for ql in q_lens]
    k_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in kv_lens]
    v_seqs = [mx.random.normal((1, H_kv, kl, D)).astype(dtype) for kl in kv_lens]
    mx.eval(*q_seqs, *k_seqs, *v_seqs)

    correct_pack = mx.concatenate(q_seqs, axis=2)
    batched_q = mx.concatenate(q_seqs, axis=0)
    wrong_pack = batched_q.reshape(1, H_q, sum(q_lens), D)
    cu_q = mx.array([0, 4, 8, 12], dtype=mx.int32)
    pool_k, pool_v, table, lens = _build_paged_pool(k_seqs, v_seqs, block_size, dtype)
    ref = _fp32_oracle(q_seqs, k_seqs, v_seqs, scale, causal=True)

    correct = _raw_paged_varlen(
        correct_pack, pool_k, pool_v, table, lens, cu_q, q_lens, scale, True, block_size
    )
    wrong = _raw_paged_varlen(
        wrong_pack, pool_k, pool_v, table, lens, cu_q, q_lens, scale, True, block_size
    )
    mx.eval(correct, wrong, ref)

    ref_np = np.array(ref)
    assert _cos_np(np.array(correct.astype(mx.float32)), ref_np) >= 0.999
    assert _cos_np(np.array(wrong.astype(mx.float32)), ref_np) < 0.95

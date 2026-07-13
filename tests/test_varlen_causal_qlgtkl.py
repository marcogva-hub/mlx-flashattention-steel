"""Public packed-varlen correction for causal segments with q_len > k_len."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import _ext, flash_attention_varlen


def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return values


def _tile_offsets(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + math.ceil(length / 32))
    return mx.array(values, dtype=mx.int32)


def _cos(a, b):
    a_flat = a.astype(mx.float32).reshape(-1)
    b_flat = b.astype(mx.float32).reshape(-1)
    return float(
        mx.sum(a_flat * b_flat)
        / (mx.sqrt(mx.sum(a_flat * a_flat)) * mx.sqrt(mx.sum(b_flat * b_flat)))
    )


def _inputs(dtype, q_lengths, k_lengths, *, hq=4, hk=2, d=64):
    mx.random.seed(731)
    q = mx.random.normal((1, hq, sum(q_lengths), d)).astype(dtype)
    k = mx.random.normal((1, hk, sum(k_lengths), d)).astype(dtype)
    v = mx.random.normal((1, hk, sum(k_lengths), d)).astype(dtype)
    return q, k, v


def _per_segment_fp32_oracle(q, k, v, q_prefix, k_prefix, scale):
    q_np = np.array(q.astype(mx.float32))[0]
    k_np = np.array(k.astype(mx.float32))[0]
    v_np = np.array(v.astype(mx.float32))[0]
    hq, total_q, d = q_np.shape
    hk = k_np.shape[0]
    output = np.empty((hq, total_q, d), dtype=np.float32)
    gqa_factor = hq // hk
    for qs, qe, ks, ke in zip(
        q_prefix[:-1], q_prefix[1:], k_prefix[:-1], k_prefix[1:]
    ):
        q_len, k_len = qe - qs, ke - ks
        qL_off = max(0, k_len - q_len)
        for q_head in range(hq):
            scores = q_np[q_head, qs:qe] @ k_np[q_head // gqa_factor, ks:ke].T
            scores *= scale
            rows = np.arange(q_len)[:, None]
            cols = np.arange(k_len)[None, :]
            scores = np.where(cols <= rows + qL_off, scores, -1e30)
            scores -= scores.max(axis=1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=1, keepdims=True)
            output[q_head, qs:qe] = probs @ v_np[q_head // gqa_factor, ks:ke]
    return mx.array(output[None])


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_causal_q_longer_than_k_forces_per_segment_sdpa(monkeypatch, dtype):
    q_lengths = [7, 65]
    k_lengths = [3, 33]
    q_prefix = _prefix(q_lengths)
    k_prefix = _prefix(k_lengths)
    q, k, v = _inputs(dtype, q_lengths, k_lengths)
    cu_q = mx.array(q_prefix, dtype=mx.int32)
    cu_k = mx.array(k_prefix, dtype=mx.int32)
    scale = 1.0 / math.sqrt(q.shape[-1])

    def steel_must_not_run(*args, **kwargs):
        raise AssertionError("causal q_len>k_len routed to STEEL")

    monkeypatch.setattr(_ext, "mfa_attention_varlen_forward", steel_must_not_run)
    with dtrace.capture() as events:
        out = flash_attention_varlen(
            q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths),
            scale=scale, causal=True,
        )
    ref = _per_segment_fp32_oracle(q, k, v, q_prefix, k_prefix, scale)
    mx.eval(out, ref)

    assert [backend for backend, _ in events] == [
        "varlen_split_concat", "varlen_sdpa", "varlen_sdpa"
    ]
    assert _cos(out, ref) >= 0.999
    # Compare the asymmetric leading rows separately so a global cosine cannot
    # hide a contract mismatch at the qL>kL edge.
    for qs, qe, q_len, k_len in zip(
        q_prefix[:-1], q_prefix[1:], q_lengths, k_lengths
    ):
        leading = q_len - k_len
        assert leading > 0
        actual = out[:, :, qs : qs + leading]
        expected = ref[:, :, qs : qs + leading]
        tolerance = 5e-3 if dtype == mx.float16 else 1e-2
        assert float(mx.max(mx.abs(actual.astype(mx.float32) - expected))) < tolerance


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "q_lengths,k_lengths,hq,hk,d",
    [
        ([7, 65], [3, 33], 4, 2, 64),
        ([63, 64, 7], [1, 33, 3], 8, 1, 64),
        ([1, 33, 65], [1, 7, 31], 4, 2, 128),
    ],
)
def test_expert_steel_causal_q_longer_than_k_matches_public_contract(
    dtype, q_lengths, k_lengths, hq, hk, d
):
    """The raw STEEL surface must implement clamped upper-left qL>kL."""
    q_prefix = _prefix(q_lengths)
    k_prefix = _prefix(k_lengths)
    q, k, v = _inputs(
        dtype, q_lengths, k_lengths, hq=hq, hk=hk, d=d
    )
    cu_q = mx.array(q_prefix, dtype=mx.int32)
    cu_k = mx.array(k_prefix, dtype=mx.int32)
    scale = 1.0 / math.sqrt(d)
    out, lse = _ext.mfa_attention_varlen_forward(
        q, k, v, cu_q, cu_k, _tile_offsets(q_lengths), scale, True
    )
    ref = _per_segment_fp32_oracle(q, k, v, q_prefix, k_prefix, scale)
    mx.eval(out, lse, ref)

    assert _cos(out, ref) >= 0.999
    assert bool(mx.all(mx.isfinite(out)).item())
    assert bool(mx.all(mx.isfinite(lse)).item())
    for qs, qe, q_len, k_len in zip(
        q_prefix[:-1], q_prefix[1:], q_lengths, k_lengths
    ):
        leading = min(q_len, max(1, q_len - k_len))
        actual = out[:, :, qs : qs + leading]
        expected = ref[:, :, qs : qs + leading]
        tolerance = 5e-3 if dtype == mx.float16 else 1e-2
        assert float(mx.max(mx.abs(actual.astype(mx.float32) - expected))) < tolerance


@pytest.mark.parametrize(
    "causal,q_lengths,k_lengths",
    [
        (True, [32, 64], [32, 64]),
        (False, [7, 65], [3, 33]),
    ],
)
def test_valid_varlen_cases_remain_steel_byte_identical(
    causal, q_lengths, k_lengths
):
    q_prefix = _prefix(q_lengths)
    k_prefix = _prefix(k_lengths)
    q, k, v = _inputs(mx.float16, q_lengths, k_lengths)
    cu_q = mx.array(q_prefix, dtype=mx.int32)
    cu_k = mx.array(k_prefix, dtype=mx.int32)
    scale = 1.0 / math.sqrt(q.shape[-1])
    original = _ext.mfa_attention_varlen_forward
    with dtrace.capture() as events:
        public = flash_attention_varlen(
            q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths),
            scale=scale, causal=causal,
        )
        mx.eval(public)
    direct, _ = original(
        q, k, v, cu_q, cu_k, _tile_offsets(q_lengths), scale, causal
    )
    mx.eval(direct)

    assert events == [("varlen_native", "STEEL varlen _ext (D<=256 f16/bf16)")]
    assert float(mx.max(mx.abs(public.astype(mx.float32) - direct.astype(mx.float32)))) == 0.0

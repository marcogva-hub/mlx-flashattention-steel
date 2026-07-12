import math

import mlx.core as mx
import pytest

from mlx_mfa import _ext
from mlx_mfa.attention import flash_attention_varlen


def _prefix(lengths):
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _tile_offsets(lengths, bq):
    out = [0]
    for length in lengths:
        out.append(out[-1] + math.ceil(length / bq))
    return mx.array(out, dtype=mx.int32)


def _oracle(q, k, v, q_lengths, k_lengths, scale, causal):
    q_prefix = _prefix(q_lengths)
    k_prefix = _prefix(k_lengths)
    parts = []
    for index in range(len(q_lengths)):
        qs, qe = q_prefix[index : index + 2]
        ks, ke = k_prefix[index : index + 2]
        parts.append(
            mx.fast.scaled_dot_product_attention(
                q[:, :, qs:qe].astype(mx.float32),
                k[:, :, ks:ke].astype(mx.float32),
                v[:, :, ks:ke].astype(mx.float32),
                scale=scale,
                mask="causal" if causal else None,
            )
        )
    return mx.concatenate(parts, axis=2)


def _cos(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    return float(mx.sum(af * bf) / (mx.sqrt(mx.sum(af * af)) * mx.sqrt(mx.sum(bf * bf))))


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "q_lengths,k_lengths,hq,hk",
    [
        ([1, 17, 65, 129], [1, 17, 65, 129], 4, 4),
        ([3, 33, 70], [7, 65, 95], 4, 2),
    ],
)
def test_v6_nax_varlen_matches_per_segment_fp32(
    dtype, head_dim, causal, q_lengths, k_lengths, hq, hk
):
    mx.random.seed(17)
    total_q = sum(q_lengths)
    total_k = sum(k_lengths)
    q = mx.random.normal((1, hq, total_q, head_dim)).astype(dtype)
    k = mx.random.normal((1, hk, total_k, head_dim)).astype(dtype)
    v = mx.random.normal((1, hk, total_k, head_dim)).astype(dtype)
    cu_q = mx.array(_prefix(q_lengths), dtype=mx.int32)
    cu_k = mx.array(_prefix(k_lengths), dtype=mx.int32)
    bq = 32 if head_dim == 64 else 64
    scale = 1.0 / math.sqrt(head_dim)

    out, _ = _ext.v6_nax_varlen_forward(
        q, k, v, cu_q, cu_k, _tile_offsets(q_lengths, bq), scale, causal
    )
    ref = _oracle(q, k, v, q_lengths, k_lengths, scale, causal)
    mx.eval(out, ref)

    q_prefix = _prefix(q_lengths)
    segment_cos = [
        _cos(out[:, :, q_prefix[i] : q_prefix[i + 1]], ref[:, :, q_prefix[i] : q_prefix[i + 1]])
        for i in range(len(q_lengths))
    ]
    assert _cos(out, ref) >= 0.999
    assert min(segment_cos) >= 0.999
    assert not bool(mx.any(mx.isnan(out)))


def test_v6_nax_varlen_is_distinct_from_steel_and_sdpa():
    mx.random.seed(23)
    lengths = [33, 71]
    total = sum(lengths)
    d = 64
    q = mx.random.normal((1, 2, total, d)).astype(mx.float16)
    k = mx.random.normal((1, 2, total, d)).astype(mx.float16)
    v = mx.random.normal((1, 2, total, d)).astype(mx.float16)
    cu = mx.array(_prefix(lengths), dtype=mx.int32)

    nax, _ = _ext.v6_nax_varlen_forward(
        q, k, v, cu, cu, _tile_offsets(lengths, 32), 1 / math.sqrt(d), False
    )
    steel, _ = _ext.mfa_attention_varlen_forward(
        q, k, v, cu, cu, _tile_offsets(lengths, 32), 1 / math.sqrt(d), False
    )
    sdpa = _oracle(q, k, v, lengths, lengths, 1 / math.sqrt(d), False).astype(mx.float16)
    mx.eval(nax, steel, sdpa)

    # Direct symbols establish engagement; non-zero deltas corroborate that the
    # three independently compiled implementations did not alias one another.
    assert float(mx.max(mx.abs(nax.astype(mx.float32) - steel.astype(mx.float32)))) > 0
    assert float(mx.max(mx.abs(nax.astype(mx.float32) - sdpa.astype(mx.float32)))) > 0
    assert _cos(nax, sdpa) >= 0.999


def test_v6_nax_varlen_extreme_segment_count_and_length():
    lengths = [1] + [8 + (index % 5) for index in range(62)] + [8193]
    total = sum(lengths)
    d = 64
    mx.random.seed(31)
    q = mx.random.normal((1, 1, total, d)).astype(mx.float16)
    k = mx.random.normal((1, 1, total, d)).astype(mx.float16)
    v = mx.random.normal((1, 1, total, d)).astype(mx.float16)
    cu = mx.array(_prefix(lengths), dtype=mx.int32)
    out, _ = _ext.v6_nax_varlen_forward(
        q, k, v, cu, cu, _tile_offsets(lengths, 32), 1 / math.sqrt(d), False
    )
    ref = _oracle(q, k, v, lengths, lengths, 1 / math.sqrt(d), False)
    mx.eval(out, ref)
    prefix = _prefix(lengths)
    assert _cos(out, ref) >= 0.999
    assert min(
        _cos(out[:, :, prefix[i] : prefix[i + 1]], ref[:, :, prefix[i] : prefix[i + 1]])
        for i in range(len(lengths))
    ) >= 0.999


def _small_inputs(dtype=mx.float16, d=64):
    q = mx.zeros((1, 2, 8, d), dtype=dtype)
    k = mx.zeros((1, 2, 8, d), dtype=dtype)
    v = mx.zeros((1, 2, 8, d), dtype=dtype)
    return q, k, v


@pytest.mark.parametrize(
    "cu_q,cu_k,tile,match",
    [
        ([1, 8], [0, 8], [0, 1], "start at 0"),
        ([0, 5, 4, 8], [0, 4, 6, 8], [0, 1, 2, 3], "strictly increasing"),
        ([0, 4, 7], [0, 4, 8], [0, 1, 2], "packed Q/K totals"),
        ([0, 0, 8], [0, 4, 8], [0, 0, 1], "empty segments unsupported"),
        ([0, 4, 8], [0, 4, 8], [0, 2, 3], "tile_offsets must equal"),
    ],
)
def test_v6_nax_varlen_rejects_invalid_prefix_metadata(cu_q, cu_k, tile, match):
    q, k, v = _small_inputs()
    with pytest.raises(ValueError, match=match):
        out, _ = _ext.v6_nax_varlen_forward(
            q,
            k,
            v,
            mx.array(cu_q, dtype=mx.int32),
            mx.array(cu_k, dtype=mx.int32),
            mx.array(tile, dtype=mx.int32),
        )
        mx.eval(out)


def test_v6_nax_varlen_rejects_empty_or_mismatched_metadata():
    q, k, v = _small_inputs()
    cases = [
        (mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)),
        (mx.array([0, 8], dtype=mx.int32), mx.array([0, 4, 8], dtype=mx.int32), mx.array([0, 1], dtype=mx.int32)),
    ]
    for cu_q, cu_k, tile in cases:
        with pytest.raises(ValueError, match="metadata arrays"):
            _ext.v6_nax_varlen_forward(q, k, v, cu_q, cu_k, tile)


def test_v6_nax_varlen_rejects_metadata_dtype_and_unsupported_data():
    q, k, v = _small_inputs()
    good = mx.array([0, 8], dtype=mx.int32)
    with pytest.raises(ValueError, match="1-D int32"):
        _ext.v6_nax_varlen_forward(q, k, v, good.astype(mx.int64), good, mx.array([0, 1], dtype=mx.int32))
    q32, k32, v32 = _small_inputs(mx.float32)
    with pytest.raises(ValueError, match="float16/bfloat16"):
        _ext.v6_nax_varlen_forward(q32, k32, v32, good, good, mx.array([0, 1], dtype=mx.int32))
    qd, kd, vd = _small_inputs(mx.float16, 32)
    with pytest.raises(ValueError, match="D must be 64 or 128"):
        _ext.v6_nax_varlen_forward(qd, kd, vd, good, good, mx.array([0, 1], dtype=mx.int32))


@pytest.mark.parametrize("dtype,d", [(mx.float32, 64), (mx.float16, 32)])
def test_public_varlen_keeps_split_concat_for_nax_unsupported_cases(monkeypatch, dtype, d):
    q, k, v = _small_inputs(dtype, d)
    cu = mx.array([0, 8], dtype=mx.int32)

    def forbidden_nax(*args, **kwargs):
        raise AssertionError("public varlen unexpectedly selected expert NAX")

    monkeypatch.setattr(_ext, "v6_nax_varlen_forward", forbidden_nax)
    out = flash_attention_varlen(q, k, v, cu, cu, 8, 8)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1 / math.sqrt(d))
    mx.eval(out, ref)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))) == 0.0


def test_public_varlen_steel_path_is_preserved(monkeypatch):
    q, k, v = _small_inputs()
    cu = mx.array([0, 8], dtype=mx.int32)

    def forbidden_nax(*args, **kwargs):
        raise AssertionError("public varlen unexpectedly selected expert NAX")

    monkeypatch.setattr(_ext, "v6_nax_varlen_forward", forbidden_nax)
    out = flash_attention_varlen(q, k, v, cu, cu, 8, 8)
    steel, _ = _ext.mfa_attention_varlen_forward(
        q, k, v, cu, cu, mx.array([0, 1], dtype=mx.int32), 1 / math.sqrt(64), False
    )
    mx.eval(out, steel)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - steel.astype(mx.float32)))) == 0.0

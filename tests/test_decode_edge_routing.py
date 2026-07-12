"""Public-route locks for the finite M5 qL=8/qL=16 decode edges."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention


_HAS_NAX = bool(mlx_mfa.get_device_info().get("is_m5_plus", False))
_BATCH = 1
_QUERY_HEADS = 16
_STDDEV = 0.25


def _qkv(q_len: int, kv_len: int, kv_heads: int, head_dim: int, dtype):
    mx.random.seed(31_000 + q_len + kv_len + kv_heads * 17 + head_dim * 19)
    q = (mx.random.normal((_BATCH, _QUERY_HEADS, q_len, head_dim)) * _STDDEV).astype(dtype)
    k = (mx.random.normal((_BATCH, kv_heads, kv_len, head_dim)) * _STDDEV).astype(dtype)
    v = (mx.random.normal((_BATCH, kv_heads, kv_len, head_dim)) * _STDDEV).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _require_nax() -> None:
    if not _HAS_NAX:
        pytest.skip("M5/NAX-only decode-edge routing")


def _run(q, k, v, *, causal: bool, backend: str):
    return flash_attention(
        q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), causal=causal, backend=backend
    )


def _traced(q, k, v, *, causal: bool, backend: str):
    with dtrace.capture() as trace:
        out = _run(q, k, v, causal=causal, backend=backend)
        mx.eval(out)
    return out, trace


def _max_abs(left, right) -> float:
    return float(np.max(np.abs(
        np.asarray(left.astype(mx.float32)) - np.asarray(right.astype(mx.float32))
    )))


def _cosine(left, right) -> float:
    a = np.asarray(left.astype(mx.float32)).astype(np.float64, copy=False).reshape(-1)
    b = np.asarray(right.astype(mx.float32)).astype(np.float64, copy=False).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("kv_len,dtype", [(4096, mx.float16), (65536, mx.bfloat16)])
def test_public_decode_edge_routes_to_real_mfa_primitive(kv_len, dtype):
    """Measured β3 envelope: qL8/D64/GQA8/non-causal f16/bf16 only."""
    _require_nax()
    q, k, v = _qkv(8, kv_len, 2, 64, dtype)
    auto, auto_trace = _traced(q, k, v, causal=False, backend="auto")
    forced, forced_trace = _traced(q, k, v, causal=False, backend="mfa")
    sdpa, sdpa_trace = _traced(q, k, v, causal=False, backend="sdpa")
    reference = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(64),
    )
    mx.eval(reference)

    assert auto_trace[-1][0] == "mfa_primitive"
    assert forced_trace[-1][0] == "mfa_primitive"
    assert sdpa_trace[-1][0] == "sdpa"
    assert _max_abs(auto, forced) == 0.0
    assert _max_abs(auto, sdpa) > 0.0
    assert _cosine(auto, reference) >= 0.999


@pytest.mark.parametrize(
    "kv_len,kv_heads,dtype",
    [
        (16384, 4, mx.float16),  # GQA = 4 at the lower measured boundary
        (32768, 2, mx.bfloat16), # GQA = 8 in the interior
        (65536, 1, mx.float16),  # GQA = 16 at the upper measured boundary
    ],
)
def test_public_decode_ql16_edge_routes_to_real_mfa_primitive(
    kv_len, kv_heads, dtype
):
    """β3 finite qL16/D64/non-causal edge proven by consolidation."""
    _require_nax()
    q, k, v = _qkv(16, kv_len, kv_heads, 64, dtype)
    auto, auto_trace = _traced(q, k, v, causal=False, backend="auto")
    forced, forced_trace = _traced(q, k, v, causal=False, backend="mfa")
    sdpa, sdpa_trace = _traced(q, k, v, causal=False, backend="sdpa")
    reference = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(64),
    )
    mx.eval(reference)

    assert auto_trace[-1][0] == "mfa_primitive"
    assert forced_trace[-1][0] == "mfa_primitive"
    assert sdpa_trace[-1][0] == "sdpa"
    assert _max_abs(auto, forced) == 0.0
    assert _max_abs(auto, sdpa) > 0.0
    assert _cosine(auto, reference) >= 0.999


@pytest.mark.parametrize(
    "q_len,kv_len,kv_heads,head_dim,causal,dtype",
    [
        (4, 4096, 2, 64, False, mx.float16),   # qL != 8
        (16, 8192, 2, 64, False, mx.bfloat16), # qL16 below measured threshold
        (16, 16384, 8, 64, False, mx.float16), # qL16 GQA = 2
        (16, 16384, 2, 64, True, mx.bfloat16), # qL16 causal
        (8, 4096, 4, 64, False, mx.float16),   # GQA = 4
        (8, 4096, 1, 64, False, mx.bfloat16),  # GQA = 16
        (8, 4096, 2, 128, False, mx.float16),  # D = 128
        (8, 4096, 2, 64, True, mx.bfloat16),   # causal
        (8, 2048, 2, 64, False, mx.float16),   # below measured winning threshold
        (8, 65537, 2, 64, False, mx.bfloat16), # above measured upper bound
    ],
)
def test_public_decode_edge_neighbours_stay_sdpa(
    q_len, kv_len, kv_heads, head_dim, causal, dtype
):
    _require_nax()
    q, k, v = _qkv(q_len, kv_len, kv_heads, head_dim, dtype)
    auto, trace = _traced(q, k, v, causal=causal, backend="auto")
    sdpa = _run(q, k, v, causal=causal, backend="sdpa")
    mx.eval(sdpa)

    assert trace[-1][0] == "sdpa"
    assert _max_abs(auto, sdpa) == 0.0

"""Correctness and engagement locks for the expert-only D=256 NAX prototype."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _ext


def _cosine(a: mx.array, b: mx.array) -> float:
    a = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    b = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _oracle(q: mx.array, k: mx.array, v: mx.array, causal: bool) -> mx.array:
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / math.sqrt(q.shape[-1]), mask="causal" if causal else None,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("N", [256, 1024])
def test_v6_nax_d256_subtile_matches_fp32(dtype, causal, N):
    """The direct expert binding must execute a real D=256 NAX source."""
    mx.random.seed(2560 + N + int(causal) + (1 if dtype == mx.bfloat16 else 0))
    q = (mx.random.normal((1, 2, N, 256)) * 0.05).astype(dtype)
    k = (mx.random.normal((1, 2, N, 256)) * 0.05).astype(dtype)
    v = (mx.random.normal((1, 2, N, 256)) * 0.05).astype(dtype)
    out, _ = _ext.v6_nax_forward(q, k, v, causal, True)
    ref = _oracle(q, k, v, causal)
    mx.eval(out, ref)
    assert bool(mx.all(mx.isfinite(out)).item())
    assert _cosine(out, ref) >= 0.999


def test_v6_nax_d256_requires_explicit_expert_opt_in():
    """D=256 must not silently enter this prototype through the raw default."""
    q = mx.zeros((1, 1, 64, 256), dtype=mx.float16)
    with pytest.raises(RuntimeError, match="expert-only"):
        _ext.v6_nax_forward(q, q, q, False, False)

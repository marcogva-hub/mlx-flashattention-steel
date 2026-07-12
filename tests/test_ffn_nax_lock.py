"""Correctness and engagement locks for expert V6 NAX Linear/GELU."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_mfa import _ext


def _has_nax():
    try:
        return bool(_ext.device_has_neural_accelerators())
    except Exception:
        return False


def _cos(a, b):
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


@pytest.mark.skipif(not _has_nax(), reason="requires V6 NAX hardware")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("gelu", [False, True])
def test_v6_nax_linear_matches_fp32(dtype, gelu):
    mx.random.seed(901)
    x = (mx.random.normal((2, 32, 256)) * 0.05).astype(dtype)
    w = (mx.random.normal((128, 256)) * 0.02).astype(dtype)
    b = (mx.random.normal((128,)) * 0.01).astype(dtype)
    y = _ext.v6_nax_linear(x, w, b, gelu)
    ref = x.astype(mx.float32) @ w.astype(mx.float32).T + b.astype(mx.float32)
    if gelu:
        ref = nn.gelu_approx(ref)
    mx.eval(y, ref)
    assert bool(mx.all(mx.isfinite(y)).item())
    assert _cos(y, ref) >= 0.999


@pytest.mark.skipif(not _has_nax(), reason="requires V6 NAX hardware")
def test_v6_nax_linear_is_distinct_from_mlx_binary():
    mx.random.seed(902)
    x = mx.random.normal((64, 256)).astype(mx.float16)
    w = mx.random.normal((128, 256)).astype(mx.float16)
    b = mx.random.normal((128,)).astype(mx.float16)
    y = _ext.v6_nax_linear(x, w, b, True)
    ref = nn.gelu_approx(x @ w.T + b)
    mx.eval(y, ref)
    assert _cos(y, ref) >= 0.999
    assert float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))).item()) > 0.0

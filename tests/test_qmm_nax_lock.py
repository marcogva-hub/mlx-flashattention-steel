"""V6 NAX quantized-matmul expert lock tests."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _ext


def _has_nax():
    try:
        return bool(_ext.device_has_neural_accelerators())
    except Exception:
        return False


def _cos(a, b):
    af = np.array(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.array(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


@pytest.mark.skipif(not _has_nax(), reason="requires V6 NAX hardware")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("M,N,K", [(17, 96, 256), (128, 128, 512)])
def test_v6_nax_quantized_matmul_matches_fp32_oracle(dtype, bits, group_size, M, N, K):
    key = mx.random.key(1000 + bits * 100 + group_size + M + N + K)
    x = mx.random.normal((M, K), key=key).astype(dtype)
    w = mx.random.normal((N, K), key=mx.random.split(key)[0]).astype(dtype)
    w_q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)

    y = _ext.v6_nax_quantized_matmul(x, w_q, scales, biases, group_size, bits)
    w_deq = mx.dequantize(w_q, scales, biases, group_size=group_size, bits=bits).astype(mx.float32)
    ref = mx.matmul(x.astype(mx.float32), mx.transpose(w_deq))
    mlx_qmm = mx.quantized_matmul(
        x, w_q, scales=scales, biases=biases,
        group_size=group_size, bits=bits, transpose=True)
    mx.eval(y, ref, mlx_qmm)

    assert np.isfinite(np.array(y.astype(mx.float32))).all()
    assert _cos(y, ref) >= 0.999
    # Engagement guard: the direct _ext expert path is a distinct callable from
    # mx.quantized_matmul, and should agree numerically with the MLX NAX arm.
    assert _cos(y, mlx_qmm) >= 0.999


@pytest.mark.skipif(not _has_nax(), reason="requires V6 NAX hardware")
def test_v6_nax_quantized_matmul_rejects_non_covered_surface():
    x = mx.zeros((16, 128), dtype=mx.float16)
    w = mx.zeros((64, 128), dtype=mx.float16)
    w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
    with pytest.raises(ValueError, match="bits=4 and bits=8"):
        _ext.v6_nax_quantized_matmul(x, w_q, scales, biases, 64, 2)

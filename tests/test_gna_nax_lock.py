"""GNA V6 NAX expert lock tests."""

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


pytestmark = pytest.mark.skipif(not _has_nax(), reason="requires V6 NAX hardware")


def _cos_np(a, b):
    af = np.asarray(a, dtype=np.float64).reshape(-1)
    bf = np.asarray(b, dtype=np.float64).reshape(-1)
    den = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / den) if den else 1.0


def _gna_reference(q, k, v, seq_shape, window_size, stride, scale):
    dim0, dim1, dim2 = seq_shape
    win0, win1, win2 = window_size
    str0, str1, str2 = stride
    dim12 = dim1 * dim2
    N = dim0 * dim12
    D = q.shape[-1]
    B, Hq = q.shape[:2]
    Hk = k.shape[1]
    gqa = Hq // Hk

    mask = np.zeros((N, N), dtype=bool)
    for qi in range(N):
        q0, q1, q2 = qi // dim12, (qi // dim2) % dim1, qi % dim2
        g0, g1, g2 = q0 // str0, q1 // str1, q2 // str2
        lo0 = max(0, g0 * str0 - (win0 - str0) // 2)
        hi0 = min(dim0 - 1, (g0 + 1) * str0 + (win0 - str0 + 1) // 2 - 1)
        lo1 = max(0, g1 * str1 - (win1 - str1) // 2)
        hi1 = min(dim1 - 1, (g1 + 1) * str1 + (win1 - str1 + 1) // 2 - 1)
        lo2 = max(0, g2 * str2 - (win2 - str2) // 2)
        hi2 = min(dim2 - 1, (g2 + 1) * str2 + (win2 - str2 + 1) // 2 - 1)
        for ki in range(N):
            k0, k1, k2 = ki // dim12, (ki // dim2) % dim1, ki % dim2
            mask[qi, ki] = lo0 <= k0 <= hi0 and lo1 <= k1 <= hi1 and lo2 <= k2 <= hi2

    out = []
    for b in range(B):
        for h in range(Hq):
            hk = h // gqa
            qq = np.array(q[b, h].astype(mx.float32)).reshape(N, D)
            kk = np.array(k[b, hk].astype(mx.float32)).reshape(N, D)
            vv = np.array(v[b, hk].astype(mx.float32)).reshape(N, D)
            scores = qq @ kk.T * scale
            scores[~mask] = -1e9
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out.append(probs @ vv)
    return np.stack(out).reshape(B, Hq, N, D)


def _inputs(dtype, D, seed=7):
    seq_shape = (1, 4, 16)
    N = math.prod(seq_shape)
    key = mx.random.key(seed + D)
    q = mx.random.normal((1, 2, N, D), key=key).astype(dtype)
    k = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[0]).astype(dtype)
    v = mx.random.normal((1, 1, N, D), key=mx.random.split(key)[1]).astype(dtype)
    return q, k, v, seq_shape


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
def test_gna_nax_matches_fp32_oracle(dtype, D):
    q, k, v, seq_shape = _inputs(dtype, D)
    window = (1, 3, 5)
    stride = (1, 1, 2)
    scale = 1.0 / math.sqrt(D)
    out = _ext.mfa_gna_nax_forward(q, k, v, *seq_shape, *window, *stride, scale)
    mx.eval(out)
    got = np.array(out.astype(mx.float32))
    ref = _gna_reference(q, k, v, seq_shape, window, stride, scale)

    assert np.isfinite(got).all()
    assert _cos_np(got, ref) >= 0.999


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_gna_nax_d128_matches_steel_native(dtype):
    q, k, v, seq_shape = _inputs(dtype, 128, seed=31)
    window = (1, 3, 5)
    stride = (1, 1, 2)
    scale = 1.0 / math.sqrt(128)
    nax = _ext.mfa_gna_nax_forward(q, k, v, *seq_shape, *window, *stride, scale)
    steel = _ext.mfa_gna_forward(q, k, v, scale, *seq_shape, *window, *stride)
    mx.eval(nax, steel)

    assert _cos_np(np.array(nax.astype(mx.float32)), np.array(steel.astype(mx.float32))) >= 0.999
    assert _ext.mfa_gna_nax_forward is not _ext.mfa_gna_forward


def test_gna_nax_is_not_steel_fallback_for_d64():
    q, k, v, seq_shape = _inputs(mx.float16, 64, seed=51)
    scale = 1.0 / math.sqrt(64)
    out = _ext.mfa_gna_nax_forward(q, k, v, *seq_shape, 1, 3, 5, 1, 1, 2, scale)
    mx.eval(out)
    assert out.shape == q.shape
    with pytest.raises(ValueError, match="only D=128"):
        _ext.mfa_gna_forward(q, k, v, scale, *seq_shape, 1, 3, 5, 1, 1, 2)


def test_gna_nax_rejects_unsupported_head_dim():
    q = mx.zeros((1, 1, 64, 256), dtype=mx.float16)
    with pytest.raises(ValueError, match="D=64 and D=128"):
        _ext.mfa_gna_nax_forward(q, q, q, 1, 4, 16, 1, 3, 5, 1, 1, 2, 0.0625)

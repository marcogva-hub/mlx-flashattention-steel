"""Correctness and routing locks for the FlashVSR LQ Conv3D envelope."""

from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import _ext
from mlx_mfa import _auto_hooks as hooks


_M5_PLUS = bool(_ext.get_device_info().get("is_m5_plus"))
_STRIDE = (2, 1, 1)
_PAD = (0, 0, 0, 0, 0, 0)


def _inputs(dtype=mx.float16):
    mx.random.seed(711)
    x = (mx.random.normal((1, 6, 18, 18, 32)) * 0.1).astype(dtype)
    w = (mx.random.normal((64, 4, 3, 3, 32)) * 0.1).astype(dtype)
    mx.eval(x, w)
    return x, w


@pytest.mark.skipif(not _M5_PLUS, reason="M5+ MPP convolution2d required")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_lq_nax_direct_matches_conv_general(dtype):
    x, w = _inputs(dtype)
    actual = _ext.conv3d_nax_forward(
        x, w, stride=_STRIDE, padding=_PAD, dilation=(1, 1, 1), chunk_M=0
    )
    expected = hooks._ORIGINAL_CONV_GENERAL(x, w, stride=_STRIDE, padding=0)
    mx.eval(actual, expected)

    af = np.asarray(actual.astype(mx.float32)).reshape(-1)
    ef = np.asarray(expected.astype(mx.float32)).reshape(-1)
    cosine = float(np.dot(af, ef) / (np.linalg.norm(af) * np.linalg.norm(ef)))
    assert actual.shape == (1, 2, 16, 16, 64)
    assert np.isfinite(af).all()
    assert cosine >= 0.999


@pytest.mark.skipif(not _M5_PLUS, reason="M5+ MPP convolution2d required")
def test_lq_public_hook_stays_on_mlx():
    x, w = _inputs()
    real = _ext.conv3d_nax_forward
    with patch("mlx_mfa._ext.conv3d_nax_forward", wraps=real) as called:
        mlx_mfa.enable()
        try:
            actual = mx.conv_general(x, w, stride=_STRIDE, padding=0)
            mx.eval(actual)
        finally:
            mlx_mfa.disable()
    assert called.call_count == 0


def test_lq_gate_remains_default_off():
    x, w = _inputs()
    assert not hooks._conv3d_nax_eligible(w, _STRIDE, 0, 1, 1, False)
    assert not hooks._conv3d_nax_eligible(w, (1, 1, 1), 0, 1, 1, False)
    assert not hooks._conv3d_nax_eligible(w, (2, 2, 1), 0, 1, 1, False)
    assert not hooks._conv3d_mpp_eligible(x, w, _PAD)

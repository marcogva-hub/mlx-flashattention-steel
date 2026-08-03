"""Locks the opt-in SeedVR2 VAE spatial pad-and-slice envelope."""

import math
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

import mlx_mfa
from mlx_mfa import _auto_hooks as hooks
from mlx_mfa import _ext


SHAPE = (1, 5, 108, 132, 512)
WEIGHT_SHAPE = (512, 3, 3, 3, 512)


def _cosine(a, b):
    af = np.asarray(a.astype(mx.float32)).reshape(-1).astype(np.float64)
    bf = np.asarray(b.astype(mx.float32)).reshape(-1).astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf)))


# (input shape, expected sliced output). input-T = output-T + 2 (stride-1, pad_T=0).
#   108x132 family #1 (input-T {4,5}); 54x66 family #2 (input-T {3,4}, added 2.62.1).
_ENGAGED_FAMILIES = [
    ((1, 4, 108, 132, 512), (1, 2, 108, 132, 512)),
    ((1, 5, 108, 132, 512), (1, 3, 108, 132, 512)),
    ((1, 3, 54, 66, 512), (1, 1, 54, 66, 512)),   # family #2 dominant (2.62.1)
    ((1, 4, 54, 66, 512), (1, 2, 54, 66, 512)),   # family #2 boundary (2.62.1)
]


@pytest.mark.skipif(not _ext.get_device_info()["is_m5_plus"], reason="requires M5+")
@pytest.mark.parametrize("shape,out_shape", _ENGAGED_FAMILIES)
def test_seedvr2_spatial_pad_slice_is_correct_and_engaged(monkeypatch, shape, out_shape):
    monkeypatch.setenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", "1")
    mx.random.seed(20260712)
    C = shape[-1]
    x = (mx.random.normal(shape) * 0.05).astype(mx.float16)
    weight = (
        mx.random.normal((C, 3, 3, 3, C)) * (1.0 / math.sqrt(27 * C))
    ).astype(mx.float16)
    oracle = hooks._ORIGINAL_CONV_GENERAL(
        x.astype(mx.float32),
        weight.astype(mx.float32),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
    )
    mlx_mfa.reset_hook_stats()
    with patch("mlx_mfa._ext.conv3d_nax_forward", wraps=_ext.conv3d_nax_forward) as native:
        actual = mx.conv_general(
            x, weight, stride=(1, 1, 1), padding=(0, 1, 1)
        )
        mx.eval(actual, oracle)
    assert native.call_count == 1
    assert actual.shape == oracle.shape == out_shape
    assert _cosine(actual, oracle) >= 0.999
    assert np.isfinite(np.asarray(actual.astype(mx.float32))).all()
    stats = mlx_mfa.get_hook_stats()
    assert stats["executed"]["conv3d_nax_spatial_pad_slice"] == 1
    assert stats["fallback"].get("conv3d_nax_forward", 0) == 0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 108, 132, 512),   # T=3 outside the 108x132 family (input-T {4,5})
        (1, 2, 54, 66, 512),     # T=2 outside the 54x66 family (input-T {3,4})
        (1, 5, 54, 66, 512),     # T=5 outside the 54x66 family
        (1, 5, 108, 132, 256),   # wrong C_out
        (1, 3, 54, 66, 256),     # wrong channels
    ],
)
def test_seedvr2_spatial_pad_slice_rejects_unmeasured_families(monkeypatch, shape):
    monkeypatch.setenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", "1")
    x = mx.zeros(shape, dtype=mx.float16)
    weight = mx.zeros((shape[-1], 3, 3, 3, shape[-1]), dtype=mx.float16)
    result = hooks._try_conv3d_spatial_pad_and_slice(
        x, weight, (0, 0, 1, 1, 1, 1)
    )
    assert result is None


def test_seedvr2_spatial_pad_slice_is_default_off(monkeypatch):
    monkeypatch.delenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", raising=False)
    x = mx.zeros(SHAPE, dtype=mx.float16)
    weight = mx.zeros(WEIGHT_SHAPE, dtype=mx.float16)
    assert (
        hooks._try_conv3d_spatial_pad_and_slice(
            x, weight, (0, 0, 1, 1, 1, 1)
        )
        is None
    )


@pytest.mark.skipif(not _ext.get_device_info()["is_m5_plus"], reason="requires M5+")
def test_seedvr2_spatial_pad_slice_default_off_preserves_public_baseline(monkeypatch):
    """The opt-in must not perturb the public mx.conv_general fallback."""
    monkeypatch.delenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", raising=False)
    mx.random.seed(20260713)
    x = (mx.random.normal(SHAPE) * 0.05).astype(mx.float16)
    weight = (
        mx.random.normal(WEIGHT_SHAPE) * (1.0 / math.sqrt(27 * 512))
    ).astype(mx.float16)
    baseline = hooks._ORIGINAL_CONV_GENERAL(
        x, weight, stride=(1, 1, 1), padding=(0, 1, 1)
    )
    mlx_mfa.reset_hook_stats()
    with patch("mlx_mfa._ext.conv3d_nax_forward") as native:
        actual = mx.conv_general(
            x, weight, stride=(1, 1, 1), padding=(0, 1, 1)
        )
        mx.eval(actual, baseline)
    native.assert_not_called()
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(baseline.astype(mx.float32))
    )
    assert mlx_mfa.get_hook_stats()["executed"].get(
        "conv3d_nax_spatial_pad_slice", 0
    ) == 0


@pytest.mark.parametrize("shape", [(1, 5, 108, 132, 512), (1, 3, 54, 66, 512)])
def test_seedvr2_spatial_pad_slice_rejects_unmeasured_bf16(monkeypatch, shape):
    """bf16 is inert for BOTH families (fp16-only gate); locked so a dtype-broadening
    would trip CI."""
    monkeypatch.setenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", "1")
    x = mx.zeros(shape, dtype=mx.bfloat16)
    weight = mx.zeros((shape[-1], 3, 3, 3, shape[-1]), dtype=mx.bfloat16)
    assert (
        hooks._try_conv3d_spatial_pad_and_slice(
            x, weight, (0, 0, 1, 1, 1, 1)
        )
        is None
    )

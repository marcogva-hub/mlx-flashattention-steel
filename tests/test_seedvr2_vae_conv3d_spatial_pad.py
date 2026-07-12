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


@pytest.mark.skipif(not _ext.get_device_info()["is_m5_plus"], reason="requires M5+")
def test_seedvr2_spatial_pad_slice_is_correct_and_engaged(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", "1")
    mx.random.seed(20260712)
    x = (mx.random.normal(SHAPE) * 0.05).astype(mx.float16)
    weight = (
        mx.random.normal(WEIGHT_SHAPE) * (1.0 / math.sqrt(27 * 512))
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
    assert actual.shape == oracle.shape == (1, 3, 108, 132, 512)
    assert _cosine(actual, oracle) >= 0.999
    assert np.isfinite(np.asarray(actual.astype(mx.float32))).all()
    stats = mlx_mfa.get_hook_stats()
    assert stats["executed"]["conv3d_nax_spatial_pad_slice"] == 1
    assert stats["fallback"].get("conv3d_nax_forward", 0) == 0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 3, 108, 132, 512),
        (1, 3, 54, 66, 512),
        (1, 4, 54, 66, 512),
        (1, 5, 54, 66, 512),
        (1, 5, 108, 132, 256),
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


def test_seedvr2_spatial_pad_slice_rejects_unmeasured_bf16(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE", "1")
    x = mx.zeros(SHAPE, dtype=mx.bfloat16)
    weight = mx.zeros(WEIGHT_SHAPE, dtype=mx.bfloat16)
    assert (
        hooks._try_conv3d_spatial_pad_and_slice(
            x, weight, (0, 0, 1, 1, 1, 1)
        )
        is None
    )

"""bf16 1×1×1 conv3d through the NAX pointwise fast path (2.63.0; pre-existing since 2026-05).

`conv3d_nax_forward` routes 1×1×1 / stride-1 / unpadded convs to
`dispatch_pointwise_fast_path` (csrc/mfa_conv_nax.cpp), which builds the conv `matmul2d`
kernel with `metal_kernel`.  That source was fp16-only (`device half*` inputs, `(half)`
store) and the fast path never rejected bf16, so every bf16 1×1×1 conv failed to build
("Unable to build metal library from source") on EVERY MLX version — reachable through
`patch_seedvr2_vae` on bf16 models.  Found by the 2.62.3 metal_kernel matrix
(`conv_pointwise/bf16`, formerly a strict known failure).

Oracle: CPU fp32 `mx.conv_general` (independent of every mlx-mfa kernel).  Both dtype
orders are exercised on the SAME shape so a dtype-blind kernel cache key cannot serve an
fp16 library to a bf16 call (or the reverse).
"""
from __future__ import annotations

import pytest

import mlx.core as mx
import mlx.nn as nn

import mlx_mfa

pytestmark = pytest.mark.skipif(not mlx_mfa.has_nax(), reason="NAX conv kernels need M5+")

SHAPE_X, SHAPE_W = (1, 4, 16, 16, 128), (96, 1, 1, 1, 128)


def _ref(x, w):
    with mx.stream(mx.cpu):
        r = mx.conv_general(x.astype(mx.float32), w.astype(mx.float32))
        mx.eval(r)
    return r


def _rel(out, ref):
    return float(mx.abs(out.astype(mx.float32) - ref).max().item()) / float(mx.abs(ref).max().item())


@pytest.mark.parametrize("order", [("bf16", "f16"), ("f16", "bf16")], ids=["bf16_first", "f16_first"])
def test_pointwise_fast_path_both_dtypes_same_shape(order):
    from mlx_mfa import _ext
    mx.random.seed(11)
    dts = {"bf16": mx.bfloat16, "f16": mx.float16}
    for name in order:
        dt = dts[name]
        x = (mx.random.normal(SHAPE_X) * 0.5).astype(dt)
        w = (mx.random.normal(SHAPE_W) * 0.1).astype(dt)
        out = _ext.conv3d_nax_forward(x, w)
        mx.eval(out)
        assert out.dtype == dt, (name, out.dtype)
        assert bool(mx.all(mx.isfinite(out.astype(mx.float32))).item()), name
        assert _rel(out, _ref(x, w)) < 1e-2, name


def test_patch_seedvr2_vae_bf16_pointwise_matches_unpatched():
    from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv3d(128, 96, kernel_size=1)

        def __call__(self, x):
            return self.c(x)

    mx.random.seed(12)
    m = M()
    m.set_dtype(mx.bfloat16)
    x = (mx.random.normal(SHAPE_X) * 0.5).astype(mx.bfloat16)
    with mx.stream(mx.cpu):
        ref = mx.conv_general(x.astype(mx.float32), m.c.weight.astype(mx.float32)) \
            + m.c.bias.astype(mx.float32)
        mx.eval(ref)
    patch_seedvr2_vae(m)
    out = m(x)
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    assert _rel(out, ref) < 1e-2

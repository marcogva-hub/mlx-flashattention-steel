"""Phase III-4 D16 — sparse-backward mask-downsample contamination lock.

The V6NAX sparse backward kernels skip at TILE granularity only.  When the
KD-1 mask conversion OR-DOWNSAMPLED a finer mask (bt < kernel tile dim),
a coarse tile merging active+inactive source tiles computed
P = exp(s - L) for positions the FORWARD masked out (absent from L) —
measured dV RMSE 0.506 / dK max-abs 1.17 vs the token-level reference at
bt=32 D=64 with a non-uniform random mask.

Fix: `_v6nax_hybrid_eligible` requires bt >= 64 (the max tile dim in
_V6NAX_BWD_SPARSE_KERNEL_TILES) so conversions never downsample; finer
masks route to the (correct) SDPA-vjp default.

Locks (env-gated research path, M5+):
  1. bt=32 non-uniform mask + both env opt-ins: grads match the
     token-level SDPA+bias reference (i.e. the contaminating path no
     longer engages).
  2. bt=64 non-uniform mask: the native path engages (no downsample)
     and matches the reference within the fp16 floor.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import flash_attention_sparse, get_device_info

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(not _HAS_NAX, reason="M5+ NAX required")

D = 64


def _grads_and_ref(BT, N, monkeypatch, native):
    monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
    if native:
        monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
    else:
        monkeypatch.delenv("MFA_V6_BWD_SPARSE_NATIVE", raising=False)
    B, H = 1, 4
    mx.random.seed(11)
    q = mx.random.normal((B, H, N, D), dtype=mx.float16)
    k = mx.random.normal((B, H, N, D), dtype=mx.float16)
    v = mx.random.normal((B, H, N, D), dtype=mx.float16)
    nb = N // BT
    mask = (mx.random.uniform(shape=(nb, nb)) < 0.5) | mx.eye(nb, dtype=mx.bool_)
    mx.eval(q, k, v, mask)
    dO = mx.ones_like(q)
    scale = 1.0 / math.sqrt(D)
    _, g = mx.vjp(
        lambda a, b, c: flash_attention_sparse(a, b, c, mask, scale=scale),
        [q, k, v], [dO])
    mx.eval(*g)
    bias = mx.repeat(mx.repeat(
        mx.where(mask, mx.zeros((nb, nb)), mx.full((nb, nb), float("-inf"))),
        BT, axis=0), BT, axis=1)
    _, gr = mx.vjp(
        lambda a, b, c: mx.fast.scaled_dot_product_attention(
            a, b, c, scale=scale,
            mask=bias.astype(mx.float32).astype(a.dtype)),
        [q, k, v], [dO])
    mx.eval(*gr)
    return g, gr


@_skipif_no_nax
class TestD16DownsampleContaminationLock:
    @pytest.mark.parametrize("native", [True, False])
    def test_bt32_nonuniform_mask_grads_correct(self, monkeypatch, native):
        """bt=32 < kernel tile 64 -> must NOT take the native path; grads
        match the token-level reference (pre-fix: dV RMSE 0.5)."""
        g, gr = _grads_and_ref(32, 2048, monkeypatch, native)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            err = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert err < 5e-3, (
                f"{name} max_abs={err:.4f} — the D16 downsample-"
                f"contamination gate regressed (native={native})")

    @pytest.mark.parametrize("native", [True, False])
    def test_bt64_native_engages_and_correct(self, monkeypatch, native):
        """bt=64: no downsample anywhere -> native/hybrid may engage and
        must stay at the fp16 floor vs the token-level reference."""
        g, gr = _grads_and_ref(64, 4096, monkeypatch, native)
        for name, x, y in zip(("dQ", "dK", "dV"), g, gr):
            err = float(mx.max(mx.abs(
                x.astype(mx.float32) - y.astype(mx.float32))).item())
            assert err < 1e-2, f"{name} max_abs={err:.4f} (native={native})"

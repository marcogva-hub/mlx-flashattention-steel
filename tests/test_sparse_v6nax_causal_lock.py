"""Causal V6NAX sparse correctness and binary-fingerprint lock."""
from __future__ import annotations

import math

import numpy as np
import pytest

import mlx.core as mx

from mlx_mfa import flash_attention_sparse
from mlx_mfa.lcsa_nax import _bool_mask_to_float_bias


def _require_m5_ext():
    try:
        from mlx_mfa import get_device_info
        from mlx_mfa._ext import sparse_attention_forward
    except Exception:
        pytest.skip("mlx-mfa extension unavailable")
    if not bool(get_device_info().get("is_m5_plus")):
        pytest.skip("causal V6NAX sparse lock asserts M5+ kernel")
    return sparse_attention_forward


def _mask(nb: int, density: float, seed: int) -> mx.array:
    rng = np.random.default_rng(seed)
    m = rng.random((nb, nb)) < density
    m[:, 0] = True
    np.fill_diagonal(m, True)
    # Future blocks deliberately set true: causal skip/mask must suppress them.
    m[0, nb - 1] = True
    m[1, nb - 1] = True
    return mx.array(m.astype(np.bool_))


def _fp32_oracle(q, k, v, block_mask, block_tile: int, scale: float):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    scores = (qf @ kf.swapaxes(-1, -2)) * scale
    qL, kL = q.shape[2], k.shape[2]
    expanded = mx.repeat(mx.repeat(block_mask, block_tile, axis=-2), block_tile, axis=-1)
    while expanded.ndim < scores.ndim:
        expanded = mx.expand_dims(expanded, 0)
    causal = mx.arange(qL).reshape(-1, 1) >= mx.arange(kL).reshape(1, -1)
    while causal.ndim < scores.ndim:
        causal = mx.expand_dims(causal, 0)
    keep = expanded & causal
    scores = mx.where(keep, scores, mx.array(-1e30, dtype=mx.float32))
    out = mx.softmax(scores, axis=-1) @ vf
    active = mx.sum(keep.astype(mx.float32), axis=-1, keepdims=True) > 0
    return mx.where(active, out, mx.zeros_like(out))


def _sdpa_sparse_causal(q, k, v, block_mask, block_tile: int, scale: float):
    qL, kL = q.shape[2], k.shape[2]
    bias = _bool_mask_to_float_bias(block_mask, block_tile, qL, kL, q.dtype)
    q_idx = mx.arange(qL).reshape(-1, 1)
    k_idx = mx.arange(kL).reshape(1, -1)
    causal_bias = mx.where(k_idx > q_idx,
                           mx.array(-float("inf"), dtype=q.dtype),
                           mx.array(0.0, dtype=q.dtype))
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=bias + causal_bias)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("density", [0.10, 0.50])
def test_v6nax_sparse_causal_matches_fp32_and_fingerprints(dtype, D, density):
    sparse_attention_forward = _require_m5_ext()
    mx.random.seed(123 + D + int(density * 1000))
    B, H, L, BT = 1, 1, 2048, 32
    scale = 1.0 / math.sqrt(D)
    q = mx.random.normal((B, H, L, D)).astype(dtype)
    k = mx.random.normal((B, H, L, D)).astype(dtype)
    v = mx.random.normal((B, H, L, D)).astype(dtype)
    block_mask = _mask(L // BT, density, seed=D + int(density * 1000))
    mx.eval(q, k, v, block_mask)

    out_v6 = sparse_attention_forward(
        q, k, v, block_mask, BT, True, scale, "v6nax_sparse")
    out_v2 = sparse_attention_forward(
        q, k, v, block_mask, BT, True, scale, "v2")
    out_scalar = sparse_attention_forward(
        q, k, v, block_mask, BT, True, scale, "scalar_fallback")
    ref = _fp32_oracle(q, k, v, block_mask, BT, scale)
    mx.eval(out_v6, out_v2, out_scalar, ref)

    out32 = out_v6.astype(mx.float32)
    assert bool(mx.all(mx.isfinite(out32)).item()), "V6NAX causal sparse produced non-finite output"

    alias_delta = float(mx.max(mx.abs(out32 - out_v2.astype(mx.float32))).item())
    assert alias_delta == 0.0, "legacy v2 alias no longer maps to V6NAX sparse"

    scalar_delta = float(mx.max(mx.abs(out32 - out_scalar.astype(mx.float32))).item())
    assert scalar_delta > 0.0, "V6NAX sparse and scalar fallback fingerprint collapsed"

    out_np = np.array(out32).reshape(-1)
    ref_np = np.array(ref).reshape(-1)
    cos = float(np.dot(out_np, ref_np) /
                (np.linalg.norm(out_np) * np.linalg.norm(ref_np) + 1e-12))
    assert cos >= 0.999, f"causal V6NAX sparse cosine {cos:.6f} below 0.999"


def test_flash_attention_sparse_causal_n2048_delegates_to_sdpa():
    """Hardened map: the measured-loss N=2048 causal cell delegates to SDPA."""
    sparse_attention_forward = _require_m5_ext()
    mx.random.seed(777)
    B, H, L, D, BT = 1, 1, 2048, 64, 32
    scale = 1.0 / math.sqrt(D)
    q = mx.random.normal((B, H, L, D)).astype(mx.float16)
    k = mx.random.normal((B, H, L, D)).astype(mx.float16)
    v = mx.random.normal((B, H, L, D)).astype(mx.float16)
    block_mask = _mask(L // BT, 0.10, seed=777)
    mx.eval(q, k, v, block_mask)

    out_public = flash_attention_sparse(
        q, k, v, block_mask, scale=scale, causal=True)
    out_v6 = sparse_attention_forward(
        q, k, v, block_mask, BT, True, scale, "v6nax_sparse")
    out_scalar = sparse_attention_forward(
        q, k, v, block_mask, BT, True, scale, "scalar_fallback")
    out_sdpa = _sdpa_sparse_causal(q, k, v, block_mask, BT, scale)
    mx.eval(out_public, out_v6, out_scalar, out_sdpa)

    public32 = out_public.astype(mx.float32)
    sdpa_delta = float(mx.max(mx.abs(public32 - out_sdpa.astype(mx.float32))).item())
    v6_delta = float(mx.max(mx.abs(public32 - out_v6.astype(mx.float32))).item())
    scalar_delta = float(mx.max(mx.abs(public32 - out_scalar.astype(mx.float32))).item())
    assert sdpa_delta == 0.0, "N=2048 causal sparse did not delegate to SDPA"
    assert v6_delta > 0.0, "N=2048 causal sparse unexpectedly remained on V6NAX"
    assert scalar_delta > 0.0, "N=2048 causal sparse collapsed to scalar fallback"

"""BT64-to-BT32 sparse expansion correctness and routing locks."""
from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention_sparse, get_device_info
from mlx_mfa.lcsa_nax import sparse_attention_nax


_HAS_NAX = bool(get_device_info().get("is_m5_plus", False))


def _require_nax():
    if not _HAS_NAX:
        pytest.skip("BT64 expansion requires M5/NAX")


def _inputs(N=4096, D=128, H=4, seed=7):
    mx.random.seed(seed)
    q = mx.random.normal((1, H, N, D)).astype(mx.float16)
    k = mx.random.normal((1, H, N, D)).astype(mx.float16)
    v = mx.random.normal((1, H, N, D)).astype(mx.float16)
    rng = np.random.default_rng(seed)
    blocks = N // 64
    mask = np.zeros((blocks, blocks), dtype=np.bool_)
    mask[:, 0] = True
    target = max(int(np.floor(0.04 * blocks * blocks)), int(mask.sum()))
    candidates = np.flatnonzero(~mask.reshape(-1))
    rng.shuffle(candidates)
    mask.reshape(-1)[candidates[:target - int(mask.sum())]] = True
    mask64 = mx.array(mask)
    mask32 = mx.repeat(mx.repeat(mask64, 2, axis=-2), 2, axis=-1)
    mx.eval(q, k, v, mask64, mask32)
    return q, k, v, mask64, mask32


def _delta(a, b) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _cos(a, b) -> float:
    x = np.asarray(a.astype(mx.float32)).reshape(-1)
    y = np.asarray(b.astype(mx.float32)).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))


def test_bt64_public_expansion_is_bt32_byte_identical_and_engaged():
    _require_nax()
    q, k, v, mask64, mask32 = _inputs()
    scale = 1.0 / math.sqrt(q.shape[-1])
    with dtrace.capture() as trace:
        expanded = flash_attention_sparse(q, k, v, mask64, scale=scale)
        mx.eval(expanded)
    native32 = flash_attention_sparse(q, k, v, mask32, scale=scale)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    token_mask = mx.repeat(mx.repeat(mask64, 64, axis=-2), 64, axis=-1)
    scores = (qf @ kf.swapaxes(-1, -2)) * scale
    ref = mx.softmax(mx.where(token_mask, scores, mx.array(-1e30, mx.float32)), axis=-1) @ vf
    mx.eval(native32, ref)

    assert trace[-1][0] == "v6nax_sparse"
    assert _delta(expanded, native32) == 0.0
    assert _cos(expanded, ref) >= 0.999


def test_bt64_public_expansion_is_distinct_from_forced_scalar(monkeypatch):
    _require_nax()
    q, k, v, mask64, _ = _inputs(seed=11)
    scale = 1.0 / math.sqrt(q.shape[-1])
    with dtrace.capture() as trace_fast:
        fast = flash_attention_sparse(q, k, v, mask64, scale=scale)
        mx.eval(fast)
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v1")
    with dtrace.capture() as trace_scalar:
        scalar = flash_attention_sparse(q, k, v, mask64, scale=scale)
        mx.eval(scalar)

    assert trace_fast[-1][0] == "v6nax_sparse"
    assert trace_scalar[-1][0] == "scalar_fallback"
    assert _delta(fast, scalar) > 0.0


def test_bt64_outside_nax_window_keeps_scalar_direct_path():
    _require_nax()
    q, k, v, _, _ = _inputs(N=4096, D=64, H=1)
    mask64 = mx.ones((4096 // 64, 4096 // 64), dtype=mx.bool_)
    scale = 1.0 / math.sqrt(q.shape[-1])
    with dtrace.capture() as trace:
        out = sparse_attention_nax(
            q, k, v, mask64, block_tile=64, scale=scale, causal=True
        )
        mx.eval(out)
    assert trace[-1][0] == "scalar_fallback"


def test_bt32_native_path_is_unchanged():
    _require_nax()
    q, k, v, _, mask32 = _inputs(seed=13)
    scale = 1.0 / math.sqrt(q.shape[-1])
    public = flash_attention_sparse(q, k, v, mask32, scale=scale)
    direct = sparse_attention_nax(q, k, v, mask32, block_tile=32, scale=scale)
    mx.eval(public, direct)
    assert _delta(public, direct) == 0.0

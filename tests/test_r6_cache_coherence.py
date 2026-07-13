"""R6 locks for dev-source and baked-scale cache discriminators."""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_mfa import _ext


def _eval(value):
    mx.eval(value)
    mx.synchronize()


def test_mfa_steel_msl_change_invalidates_same_process_cache(monkeypatch):
    q = mx.ones((1, 1, 64, 64), dtype=mx.float16)
    monkeypatch.setenv("MFA_DISABLE_V2", "1")
    monkeypatch.delenv("MFA_STEEL_MSL", raising=False)
    _ext.shader_cache_clear()
    try:
        _eval(_ext.mfa_attention_forward(q, q, q, 0.125, False))
        initial_size = _ext.shader_cache_size()
        assert initial_size >= 1

        # Without MFA_STEEL_MSL in KernelKey, this exact-shape call reuses the
        # first pipeline and the positive-control compile failure never fires.
        monkeypatch.setenv("MFA_STEEL_MSL", "PROVE")
        with pytest.raises(Exception, match="STEEL_INJECTION_REACHED"):
            _eval(_ext.mfa_attention_forward(q, q, q, 0.125, False))
        assert _ext.shader_cache_size() == initial_size
    finally:
        _ext.shader_cache_clear()


def _masked_reference(q, k, v, mask, scale):
    expanded = mx.repeat(mx.repeat(mask, 32, -2), 32, -1)
    bias = mx.where(
        expanded,
        mx.array(0.0, dtype=mx.float32),
        mx.array(-1e30, dtype=mx.float32),
    )
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=bias,
    )


def test_sparse_baked_scale_uses_distinct_effective_cache_entries():
    n, d = 2048, 64
    mx.random.seed(20260713)
    q = mx.random.normal((1, 1, n, d)).astype(mx.float16)
    k = mx.random.normal((1, 1, n, d)).astype(mx.float16)
    v = mx.random.normal((1, 1, n, d)).astype(mx.float16)
    mask = mx.ones((n // 32, n // 32), dtype=mx.bool_)
    scales = (1.0 / math.sqrt(d), 0.5 / math.sqrt(d))

    outputs = []
    for scale in scales:
        out = _ext.sparse_attention_forward(
            q, k, v, mask, 32, False, scale, "v6nax_sparse"
        )
        ref = _masked_reference(q, k, v, mask, scale)
        _eval(out)
        _eval(ref)
        of = out.astype(mx.float32)
        cos = mx.sum(of * ref) / mx.sqrt(mx.sum(of ** 2) * mx.sum(ref ** 2))
        mx.eval(cos)
        assert float(cos.item()) >= 0.999
        outputs.append(out)

    delta = mx.max(mx.abs(
        outputs[0].astype(mx.float32) - outputs[1].astype(mx.float32)
    ))
    mx.eval(delta)
    assert float(delta.item()) > 1e-4

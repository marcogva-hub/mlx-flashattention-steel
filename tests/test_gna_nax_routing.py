"""Public GNA routing locks for the V6 NAX path."""

import math

import mlx.core as mx
import pytest

import mlx_mfa.attention as attention
from mlx_mfa import _ext


def _qkv(seq_shape, D, dtype=mx.float16):
    N = math.prod(seq_shape)
    q = mx.zeros((1, 1, N, D), dtype=dtype)
    k = mx.zeros((1, 1, N, D), dtype=dtype)
    v = mx.zeros((1, 1, N, D), dtype=dtype)
    return q, k, v


def _install_route_spies(monkeypatch):
    calls = {"nax": 0, "steel": 0, "sparse": 0}

    def fake_nax(q, k, v, *args):
        calls["nax"] += 1
        return mx.zeros_like(q)

    def fake_steel(q, k, v, *args):
        calls["steel"] += 1
        return mx.zeros_like(q)

    def fake_sparse(q, k, v, *args, **kwargs):
        calls["sparse"] += 1
        return mx.zeros_like(q)

    monkeypatch.setattr(_ext, "mfa_gna_nax_forward", fake_nax)
    monkeypatch.setattr(_ext, "mfa_gna_forward", fake_steel)
    monkeypatch.setattr(attention, "flash_attention_sparse", fake_sparse)
    return calls


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_public_gna_d128_large_3d_routes_to_nax(monkeypatch, dtype):
    calls = _install_route_spies(monkeypatch)
    seq_shape = (2, 32, 32)  # N=2048
    q, k, v = _qkv(seq_shape, 128, dtype)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (1, 7, 7), (1, 1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 1, "steel": 0, "sparse": 0}


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_public_gna_d64_large_3d_routes_to_nax(monkeypatch, dtype):
    calls = _install_route_spies(monkeypatch)
    seq_shape = (4, 32, 32)  # N=4096
    q, k, v = _qkv(seq_shape, 64, dtype)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (1, 7, 7), (1, 1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 1, "steel": 0, "sparse": 0}


def test_public_gna_d64_small_3d_stays_sparse_fallback(monkeypatch):
    calls = _install_route_spies(monkeypatch)
    seq_shape = (2, 32, 32)  # N=2048, measured losing D=64 fp16 small-N cell
    q, k, v = _qkv(seq_shape, 64)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (1, 7, 7), (1, 1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 0, "steel": 0, "sparse": 1}


def test_public_gna_d128_small_3d_preserves_steel(monkeypatch):
    calls = _install_route_spies(monkeypatch)
    seq_shape = (1, 32, 32)  # N=1024, below the D=128 NAX route threshold
    q, k, v = _qkv(seq_shape, 128)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (1, 7, 7), (1, 1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 0, "steel": 1, "sparse": 0}


def test_public_gna_non_3d_does_not_route_to_nax(monkeypatch):
    calls = _install_route_spies(monkeypatch)
    seq_shape = (64, 64)  # N=4096, but 2D is outside the measured NAX envelope
    q, k, v = _qkv(seq_shape, 128)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (7, 7), (1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 0, "steel": 0, "sparse": 1}


def test_public_gna_native_disable_blocks_nax(monkeypatch):
    calls = _install_route_spies(monkeypatch)
    monkeypatch.setenv("MFA_DISABLE_GNA_NATIVE", "1")
    seq_shape = (2, 32, 32)
    q, k, v = _qkv(seq_shape, 128)

    out = attention.flash_attention_gna(q, k, v, seq_shape, (1, 7, 7), (1, 1, 1))

    assert out.shape == q.shape
    assert calls == {"nax": 0, "steel": 0, "sparse": 1}

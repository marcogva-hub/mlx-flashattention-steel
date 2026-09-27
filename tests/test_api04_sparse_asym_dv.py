"""API-04 (code review 2026-09, P1 PUBLISHED) — asymmetric D_v on the sparse NAX route.

`flash_attention_sparse` documents D_v != D_qk as valid (served by an SDPA-class path),
but `_nax_sparse_route_viable` checked only Q/K: an in-envelope shape (B*H=12, D=128,
N=4096, density ~0.11) with D_v=64 was routed to the NAX kernel, whose C++ guard
raised "head_dim mismatch".  The existing acceptance test used N=256 (mask 64 B <
4096 B) and never reached the route.  Fix: the gate also takes V and refuses a V the
kernel cannot serve (head_dim / dtype / length) -> SDPA route.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import is_mfa_available
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa.attention import flash_attention_sparse
from mlx_mfa.lcsa_nax import sparse_attention_dispatch

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

B, H, N, D, DV = 1, 12, 4096, 128, 64


def _inputs(dv=DV):
    mx.random.seed(0)
    q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.normal((B, H, N, dv)) * 0.1).astype(mx.float16)
    nq = N // 32
    bm = (mx.random.uniform(shape=(nq, nq)) < 0.1) | mx.eye(nq, dtype=mx.bool_)
    mx.eval(q, k, v, bm)
    return q, k, v, bm


def _reference(q, k, v, bm):
    em = mx.repeat(mx.repeat(bm, 32, axis=0), 32, axis=1)
    bias = mx.where(em, 0.0, float("-inf")).astype(mx.float32)
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=D ** -0.5, mask=bias)


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def test_symmetric_control_takes_the_nax_route():
    """Control: the same shape with D_v == D_qk IS in the NAX envelope."""
    q, k, v, bm = _inputs(dv=D)
    with dt.capture() as tr:
        mx.eval(flash_attention_sparse(q, k, v, bm))
    assert tr and tr[-1][0].startswith("v6nax_sparse"), [t[0] for t in tr]


def test_flash_attention_sparse_asym_dv_in_nax_envelope():
    q, k, v, bm = _inputs()
    with dt.capture() as tr:
        o = flash_attention_sparse(q, k, v, bm)
        mx.eval(o)
    assert o.shape == (B, H, N, DV)
    assert not any(t[0].startswith("v6nax_sparse") for t in tr), [t[0] for t in tr]
    assert _err(o, _reference(q, k, v, bm)) < 1e-2


def test_sparse_attention_dispatch_asym_dv_in_nax_envelope():
    q, k, v, bm = _inputs()
    o = sparse_attention_dispatch(q, k, v, bm, block_tile=32, scale=D ** -0.5)
    mx.eval(o)
    assert o.shape == (B, H, N, DV)
    assert _err(o, _reference(q, k, v, bm)) < 1e-2


def test_opt_in_v6_hybrid_gate_asym_dv(monkeypatch):
    """Pre-merge review 2026-09-28 (N3): the opt-in V6 hybrid gate
    (MFA_ENABLE_V6_BACKWARD=1, bt >= 64) never looked at V — D_v != D reached the
    NAX sparse forward and raised "head_dim mismatch"."""
    monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
    mx.random.seed(3)
    q = (mx.random.normal((1, 2, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((1, 2, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.normal((1, 2, N, DV)) * 0.1).astype(mx.float16)
    nb = N // 64
    bm = (mx.random.uniform(shape=(nb, nb)) < 0.1) | mx.eye(nb, dtype=mx.bool_)
    o = flash_attention_sparse(q, k, v, bm)
    mx.eval(o)
    em = mx.repeat(mx.repeat(bm, 64, axis=0), 64, axis=1)
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=D ** -0.5, mask=mx.where(em, 0.0, float("-inf")).astype(mx.float32))
    assert o.shape == (1, 2, N, DV) and _err(o, ref) < 1e-2

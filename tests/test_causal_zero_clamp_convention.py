"""Decision 2026-09 (Marco, R1'/CRIT-01): ONE causal convention everywhere — NAMING.md's
canonical "bottom-right-aligned, zero-clamped" masking: key j is visible to query row i
iff  j <= i + max(0, S - N).

For N <= S it equals SDPA's bottom-right mask="causal".  For N > S it clamps to 0
(top-left, no fully masked row): a DELIBERATE divergence from
mx.fast.scaled_dot_product_attention(mask="causal").  Before 2.62.2 flash_attention's
default route followed SDPA for N > S while return_lse / backend="mfa" / varlen / paged
followed the clamp (return_lse flipped the output by ~2.7), and several backward legs
used SDPA's convention even when the forward clamped.

Locks: flash_attention == flash_attention_varlen with one segment (the canonical oracle
in the library itself) for N > S on every route; O, L and gradients vs a CPU oracle;
the sibling entries fixed with it.
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, flash_attention_paged, flash_attention_varlen, is_mfa_available
from mlx_mfa.lcsa_nax import sparse_attention_dispatch

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

_LN2 = math.log(2.0)
TOL = {mx.float16: 1e-2, mx.bfloat16: 3e-2, mx.float32: 5e-3}   # fp32: M5 GPU floor ~2e-3


def _zc_scores(q, k):
    N, S, D = q.shape[2], k.shape[2], q.shape[3]
    s = (q @ mx.swapaxes(k, -1, -2)) * (D ** -0.5)
    vis = mx.arange(S)[None, :] <= mx.arange(N)[:, None] + max(0, S - N)
    return mx.where(vis, s, float("-inf"))


def _oracle(q, k, v):
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        s = _zc_scores(q32, k32)
        o = mx.softmax(s, axis=-1) @ v32
        lse = mx.logsumexp(s, axis=-1) / _LN2
        mx.eval(o, lse)
    return o, lse


def _oracle_grads(q, k, v, g):
    with mx.stream(mx.cpu):
        def f(a, b, c):
            return ((mx.softmax(_zc_scores(a, b), axis=-1) @ c) * g).sum()
        grads = mx.grad(f, argnums=(0, 1, 2))(*(x.astype(mx.float32) for x in (q, k, v)))
        mx.eval(*grads)
    return grads


def _qkv(N, S, D, dtype, H=2, seed=0):
    mx.random.seed(seed)
    return tuple(mx.random.normal((1, H, L, D)).astype(dtype) for L in (N, S, S))


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _varlen1(q, k, v):
    N, S = q.shape[2], k.shape[2]
    return flash_attention_varlen(q, k, v, mx.array([0, N], dtype=mx.int32),
                                  mx.array([0, S], dtype=mx.int32), N, S, causal=True)


DTYPES = [mx.float16, mx.bfloat16, mx.float32]
SHAPES_N_GT_S = [(256, 128, 64), (100, 37, 128), (2048, 1024, 128)]


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16", "f32"])
@pytest.mark.parametrize("N,S,D", SHAPES_N_GT_S)
@pytest.mark.parametrize("route", ["default", "return_lse", "backend_sdpa"])
def test_flash_attention_equals_varlen_one_segment(route, N, S, D, dtype):
    if dtype == mx.float32 and D not in (64, 128):
        pytest.skip("varlen fp32 D")
    q, k, v = _qkv(N, S, D, dtype, seed=N + S)
    if route == "default":
        o = flash_attention(q, k, v, causal=True)
    elif route == "return_lse":
        o, _ = flash_attention(q, k, v, causal=True, return_lse=True)
    else:
        o = flash_attention(q, k, v, causal=True, backend="sdpa")
    ref_vl = _varlen1(q, k, v)
    ref, _ = _oracle(q, k, v)
    assert _err(o, ref_vl) < TOL[dtype], ("vs varlen", _err(o, ref_vl))
    assert _err(o, ref) < TOL[dtype], ("vs oracle", _err(o, ref))


@pytest.mark.parametrize("dtype", DTYPES, ids=["f16", "bf16", "f32"])
def test_return_lse_matches_oracle_and_default_route(dtype):
    """The CRIT-01 symptom: return_lse must not change the output any more."""
    q, k, v = _qkv(256, 128, 64, dtype, seed=3)
    o_def = flash_attention(q, k, v, causal=True)
    o_lse, lse = flash_attention(q, k, v, causal=True, return_lse=True)
    ref, ref_l = _oracle(q, k, v)
    assert _err(o_def, o_lse) < TOL[dtype]
    assert _err(lse, ref_l) < TOL[dtype]


@pytest.mark.parametrize("route", ["default", "return_lse", "backend_mfa"])
@pytest.mark.parametrize("N,S", [(256, 128), (64, 128)])     # N>S and an N<S control
def test_gradients_follow_the_same_convention(route, N, S):
    D, dtype = 64, mx.float16
    q, k, v = _qkv(N, S, D, dtype, seed=11)
    mx.random.seed(12)
    g = mx.random.normal((1, 2, N, D)).astype(mx.float32)

    def f(a, b, c):
        if route == "return_lse":
            o, _ = flash_attention(a, b, c, causal=True, return_lse=True)
        elif route == "backend_mfa":
            o = flash_attention(a, b, c, causal=True, backend="mfa")
        else:
            o = flash_attention(a, b, c, causal=True)
        return (o.astype(mx.float32) * g).sum()

    grads = mx.grad(f, argnums=(0, 1, 2))(q, k, v)
    mx.eval(*grads)
    for name, x, y in zip(("dQ", "dK", "dV"), grads, _oracle_grads(q, k, v, g)):
        assert _err(x, y) < 3e-2, (name, _err(x, y))


@pytest.mark.parametrize("kw", [dict(softcap=30.0), dict(alibi_slopes=mx.array([0.1, 0.2])),
                                dict(return_attn_weights=True)], ids=["softcap", "alibi", "weights"])
def test_feature_routes_have_no_empty_rows_for_n_gt_s(kw):
    """Zero-clamp => every row sees >= 1 key; the SDPA-style empty leading rows are gone."""
    q, k, v = _qkv(256, 128, 64, mx.float16, seed=5)
    r = flash_attention(q, k, v, causal=True, **kw)
    o = r[0] if isinstance(r, tuple) else r
    assert bool(mx.all(mx.isfinite(o.astype(mx.float32))).item())
    if "return_attn_weights" in kw:          # plain attention -> compare to the oracle
        ref, _ = _oracle(q, k, v)
        assert _err(o, ref) < 1e-2


def test_sparse_attention_dispatch_causal_q_shorter_than_k():
    """DSP-12 sibling: the dispatcher's SDPA route used top-left for EVERY shape."""
    N, S, D = 64, 128, 64
    q, k, v = _qkv(N, S, D, mx.float16, H=1, seed=7)
    bm = mx.ones((N // 32, S // 32), dtype=mx.bool_)
    o = sparse_attention_dispatch(q, k, v, bm, block_tile=32, scale=D ** -0.5, causal=True)
    ref, _ = _oracle(q, k, v)
    assert _err(o, ref) < 1e-2


def test_paged_backward_q_longer_than_kv():
    """Paged backward placed row i at kv_len - N_q + i (unclamped): its gradient disagreed
    with the forward (flash_attention, clamped) whenever N_q > kv_len."""
    B, H, N, D, BS = 1, 2, 24, 64, 16
    kv_len = 16                                   # N_q > kv_len
    mx.random.seed(9)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, kv_len, D)).astype(mx.float16)
    v = mx.random.normal((B, H, kv_len, D)).astype(mx.float16)
    # Contiguous pools on purpose: a NON-contiguous pool view gives wrong paged
    # GRADIENTS for any convention (separate finding, flagged 2026-09-28, not fixed
    # here — devnotes/remediation_2026_09/repro_paged_noncontig_bwd.py).
    kp = mx.contiguous(k[0].transpose(1, 0, 2).reshape(1, BS, H, D))
    vp = mx.contiguous(v[0].transpose(1, 0, 2).reshape(1, BS, H, D))
    bt = mx.array([[0]], dtype=mx.int32)
    sl = mx.array([kv_len], dtype=mx.int32)
    mx.random.seed(10)
    g = mx.random.normal((B, H, N, D)).astype(mx.float32)
    dq = mx.grad(lambda a: (flash_attention_paged(a, kp, vp, bt, sl, causal=True, block_size=BS)
                            .astype(mx.float32) * g).sum())(q)
    mx.eval(dq)
    ref_dq = _oracle_grads(q, k, v, g)[0]
    assert _err(dq, ref_dq) < 3e-2

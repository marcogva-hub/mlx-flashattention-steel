"""Volet H2 — raw-backward input-validation completion (CX-04).

Exported raw backward `_ext` bindings accepted malformed auxiliaries / mismatched
K-V shapes and returned finite, materially-wrong gradients. Volet C added the
common backward validator to some bindings but missed others and the K↔V mutual
shape. This locks that EVERY raw backward binding now validates aux shapes
(L/lse/o/dO/D consistent with Q), K↔V mutual shape (kv-seq, heads, head_dim), and
GQA — raising before dispatch (Rule 8). Valid gradients are unchanged.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa._ext as e

F16 = mx.float16
B, H, Hk, N, D = 1, 4, 2, 64, 64  # GQA-capable (Hq=4, Hk=2)
SC = 1.0 / math.sqrt(D)


def _r(*shape, dt=F16, seed=0):
    mx.random.seed(seed)
    a = mx.random.normal(shape).astype(dt)
    mx.eval(a)
    return a


def _valid():
    return dict(
        q=_r(B, H, N, D, seed=1), k=_r(B, Hk, N, D, seed=2), v=_r(B, Hk, N, D, seed=3),
        o=_r(B, H, N, D, seed=4), dO=_r(B, H, N, D, seed=5),
        lse=_r(B, H, N, dt=mx.float32, seed=6), dvec=_r(B, H, N, dt=mx.float32, seed=7),
        bm=mx.ones((N // 16, N // 16), dtype=mx.bool_),
    )


# Each malformed builder mutates exactly one input to be inconsistent.
_MAL = {
    # CX-04 cited: mfa_steel_backward_sparse undersized L
    "steel_sparse_bad_L": lambda d: e.mfa_steel_backward_sparse(
        d["q"], d["k"], d["v"], d["o"], _r(B, H, N // 2, dt=mx.float32), d["dO"],
        d["bm"], SC, False),
    "steel_sparse_kv_mismatch": lambda d: e.mfa_steel_backward_sparse(
        d["q"], d["k"], _r(B, Hk, N // 2, D), d["o"], d["lse"], d["dO"], d["bm"], SC, False),
    # mfa_steel_backward (dense) — already validated; re-pin
    "steel_bad_dO": lambda d: e.mfa_steel_backward(
        d["q"], d["k"], d["v"], d["o"], d["lse"], _r(B, H, N // 2, D), SC, False),
    # CX-04 cited: v6_nax_backward_query undersized lse + KV mismatch
    "v6_query_bad_lse": lambda d: e.v6_nax_backward_query(
        d["q"], d["k"], d["v"], d["o"], _r(B, H, N // 2, dt=mx.float32), d["dO"],
        d["dvec"], SC, False),
    "v6_query_kv_mismatch": lambda d: e.v6_nax_backward_query(
        d["q"], d["k"], _r(B, Hk, N // 2, D), d["o"], d["lse"], d["dO"], d["dvec"], SC, False),
    # CX-04 cited: v6_nax_backward_fused_dkdv_raw V mismatch
    "v6_fused_kv_mismatch": lambda d: e.v6_nax_backward_fused_dkdv_raw(
        d["q"], d["k"], _r(B, Hk, 1, D), d["lse"], d["dO"], d["dvec"], SC, 4, False),
    "v6_fused_bad_lse": lambda d: e.v6_nax_backward_fused_dkdv_raw(
        d["q"], d["k"], d["v"], _r(B, H, N // 2, dt=mx.float32), d["dO"], d["dvec"], SC, 4, False),
    # remaining v6 bindings — KV mismatch
    "v6_kv_kvmismatch": lambda d: e.v6_nax_backward_kv(
        d["q"], d["k"], _r(B, Hk, N // 2, D), d["o"], d["lse"], d["dO"], d["dvec"], SC, False),
    "v6_dk_kvmismatch": lambda d: e.v6_nax_backward_dk_raw(
        d["q"], d["k"], _r(B, Hk, N // 2, D), d["o"], d["lse"], d["dO"], d["dvec"], SC, 4, False),
    "v6_dv_kvmismatch": lambda d: e.v6_nax_backward_dv_raw(
        d["q"], d["k"], _r(B, Hk, N // 2, D), d["lse"], d["dO"], SC, 4, False),
    # GQA invalid (Hq not multiple of Hk) on a v6 binding
    "v6_query_bad_gqa": lambda d: e.v6_nax_backward_query(
        _r(B, 3, N, D), d["k"], d["v"], _r(B, 3, N, D), _r(B, 3, N, dt=mx.float32),
        _r(B, 3, N, D), _r(B, 3, N, dt=mx.float32), SC, False),
}


@pytest.mark.parametrize("cid", list(_MAL.keys()))
def test_raw_backward_malformed_raises(cid):
    d = _valid()
    with pytest.raises((ValueError, RuntimeError)):
        out = _MAL[cid](d)
        mx.eval(out[0] if isinstance(out, tuple) else out)


# Valid inputs must NOT raise (boundary gate adds raises only).
_VALID_CALLS = {
    "steel": lambda d: e.mfa_steel_backward(d["q"], d["k"], d["v"], d["o"], d["lse"], d["dO"], SC, False),
    "steel_sparse": lambda d: e.mfa_steel_backward_sparse(d["q"], d["k"], d["v"], d["o"], d["lse"], d["dO"], d["bm"], SC, False),
    "v6_query": lambda d: e.v6_nax_backward_query(d["q"], d["k"], d["v"], d["o"], d["lse"], d["dO"], d["dvec"], SC, False),
    "v6_kv": lambda d: e.v6_nax_backward_kv(d["q"], d["k"], d["v"], d["o"], d["lse"], d["dO"], d["dvec"], SC, False),
    "v6_dk": lambda d: e.v6_nax_backward_dk_raw(d["q"], d["k"], d["v"], d["o"], d["lse"], d["dO"], d["dvec"], SC, 4, False),
    "v6_dv": lambda d: e.v6_nax_backward_dv_raw(d["q"], d["k"], d["v"], d["lse"], d["dO"], SC, 4, False),
    "v6_fused": lambda d: e.v6_nax_backward_fused_dkdv_raw(d["q"], d["k"], d["v"], d["lse"], d["dO"], d["dvec"], SC, 4, False),
}


@pytest.mark.parametrize("cid", list(_VALID_CALLS.keys()))
def test_raw_backward_valid_runs(cid):
    d = _valid()
    out = _VALID_CALLS[cid](d)
    mx.eval(out[0] if isinstance(out, tuple) else out)
    arr = out[0] if isinstance(out, tuple) else out
    assert bool(np.isfinite(np.array(arr.astype(mx.float32))).all())

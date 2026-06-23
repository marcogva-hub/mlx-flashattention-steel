"""Volet J — pre-quantized Sage buffer validation (CX-R7-01).

Round-7 probed an OMITTED entry — `sage_attention_prequantized` / raw
`mfa_sage_forward` — and found malformed buffers ACCEPTED with finite-wrong/NaN
output instead of raising: half-length V → OOB, batch mismatch → NaN, wrong
k_int8/k_scale dtype → garbage, short k_scale → OOB. The kernel derives extents
from q/k_int8 and reads v / k_scale at K's offsets without re-checking.

Fixed both surfaces: `_assert_sage_prequant_buffers` (Python) +
`mfa_sage_forward` C++ guards. Both directions, first-hand.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as e
from mlx_mfa.quantize import quantize_per_block, sage_block_sizes

_H, _Hk, _N, _D = 8, 2, 256, 64
_SC = 1.0 / math.sqrt(_D)


def _valid():
    mx.random.seed(0)
    q = mx.random.normal((1, _H, _N, _D)).astype(mx.float16)
    k = mx.random.normal((1, _Hk, _N, _D)).astype(mx.float16)
    v = mx.random.normal((1, _Hk, _N, _D)).astype(mx.float16)
    mx.eval(q, k, v)
    _, BK = sage_block_sizes(_D)
    ki, ks = quantize_per_block(k, BK)
    ks = ks.squeeze(-1)
    mx.eval(ki, ks)
    return q, ki, ks, v


def test_prequant_valid_runs():
    q, ki, ks, v = _valid()
    o = mlx_mfa.sage_attention_prequantized(q, ki, ks, v, scale=_SC)
    mx.eval(o)
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


def test_raw_valid_runs():
    q, ki, ks, v = _valid()
    o, _ = e.mfa_sage_forward(q, ki, v, ks, _SC, False, -1, -1)
    mx.eval(o)
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


@pytest.mark.parametrize("mut", ["v_half", "v_batch", "ks_f16", "ki_f16", "ks_short", "v_dim"])
def test_prequant_malformed_raises(mut):
    q, ki, ks, v = _valid()
    kw = dict(q=q, k_int8=ki, k_scale=ks, v=v)
    if mut == "v_half":
        kw["v"] = v[:, :, : _N // 2, :]
    elif mut == "v_batch":
        kw["q"] = mx.broadcast_to(q, (2, _H, _N, _D))
    elif mut == "ks_f16":
        kw["k_scale"] = ks.astype(mx.float16)
    elif mut == "ki_f16":
        kw["k_int8"] = ki.astype(mx.float16)
    elif mut == "ks_short":
        kw["k_scale"] = ks[:, :, :-1]
    elif mut == "v_dim":
        kw["v"] = mx.random.normal((1, _Hk, _N, _D // 2)).astype(mx.float16)
    with pytest.raises((ValueError, Exception)):
        o = mlx_mfa.sage_attention_prequantized(
            kw["q"], kw["k_int8"], kw["k_scale"], kw["v"], scale=_SC)
        mx.eval(o)


@pytest.mark.parametrize("mut", ["v_half", "ks_f16", "ks_short", "v_dim"])
def test_raw_malformed_raises(mut):
    q, ki, ks, v = _valid()
    a = [q, ki, v, ks]
    if mut == "v_half":
        a[2] = v[:, :, : _N // 2, :]
    elif mut == "ks_f16":
        a[3] = ks.astype(mx.float16)
    elif mut == "ks_short":
        a[3] = ks[:, :, :-1]
    elif mut == "v_dim":
        a[2] = mx.random.normal((1, _Hk, _N, _D // 2)).astype(mx.float16)
    with pytest.raises((ValueError, Exception)):
        o, _ = e.mfa_sage_forward(a[0], a[1], a[2], a[3], _SC, False, -1, -1)
        mx.eval(o)

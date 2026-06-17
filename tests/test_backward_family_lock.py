"""Backward family gradient-correctness + per-gradient which-binary LOCK (audit B3).

Two locks for the backward family:

1. **Gradient correctness** — dQ/dK/dV vs an INDEPENDENT fp32 gradient oracle
   (`mx.vjp` of a MANUAL pure-mlx fp32 forward — NOT another kernel's gradient,
   lesson #11; the oracle agrees with SDPA-vjp to ~1e-7 on the SDPA paths, two
   independent impls, so it is trusted).

2. **Per-gradient which-binary** — the backward is a MIX of native NAX and
   SDPA-vjp. Native gradients are BYTE-DISTINCT from SDPA-vjp (Δ≠0), so each
   gradient's source is locked by byte-identity vs an in-test SDPA-vjp reference:
   Δ==0.0 ⇒ SDPA-vjp, Δ>0 ⇒ native. A reroute (e.g. D=64 default-on backward
   reverting to SDPA-vjp) flips a Δ and fails CI.

Verified map (M5/26.6): dense D=128 = all SDPA-vjp; dense D=64 causal (default-on,
N≥2048) = all native; sparse default = all SDPA-vjp; sparse opt-in
(MFA_ENABLE_V6_BACKWARD, bt≥64) = native dV + SDPA-vjp dQ/dK (hybrid); sparse
opt-in MFA_V6_BWD_SPARSE_NATIVE = all native.  M5+-gated.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, flash_attention_sparse
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="backward family lock asserts M5+ routes",
)

mx.random.seed(0)
_TOL = 5e-2


def _fp32_fwd(q, k, v, scale, causal, bias=None):
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:
        r = Hq // Hk; kf = mx.repeat(kf, r, 1); vf = mx.repeat(vf, r, 1)
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    N, S = q.shape[2], k.shape[2]
    if causal:
        s = mx.where(mx.arange(N)[:, None] >= mx.arange(S)[None, :] + (S - N), s, mx.array(-1e30))
    if bias is not None:
        s = s + bias
    return mx.softmax(s, -1) @ vf


def _vjp(fn, q, k, v, dO):
    _, g = mx.vjp(fn, (q, k, v), (dO,))
    return g  # (dQ, dK, dV)


def _delta(a, b):
    mx.eval(a, b)
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def _qkv(B, H, N, D, Hk=None):
    Hk = Hk or H
    f = lambda h: (mx.random.uniform(-1, 1, (B, h, N, D)) * 0.1).astype(mx.float16)
    q, k, v = f(H), f(Hk), f(Hk); mx.eval(q, k, v)
    return q, k, v


def _block_bias(m, N):
    NB = m.shape[-2]
    em = mx.repeat(mx.repeat(m.astype(mx.float32), N // NB, -2), N // NB, -1)
    return mx.where(em > 0, mx.array(0.0), mx.array(-1e9))[None, None].astype(mx.float16)


def _audit(q, k, v, scale, causal, expect, bias=None, smask=None):
    """expect: dict {'dQ','dK','dV'} -> 'native' | 'sdpa-vjp'. Asserts correctness + which-binary."""
    dO = (mx.random.uniform(-1, 1, q.shape) * 0.1).astype(mx.float16); mx.eval(dO)
    kfn = ((lambda a, b, c: flash_attention_sparse(a, b, c, smask, scale=scale, causal=causal))
           if smask is not None else
           (lambda a, b, c: flash_attention(a, b, c, scale=scale, causal=causal)))
    gk = _vjp(kfn, q, k, v, dO)
    go = _vjp(lambda a, b, c: _fp32_fwd(a, b, c, scale, causal, bias), q, k, v, dO)
    sdpa_mask = bias if bias is not None else ("causal" if causal else None)
    gs = _vjp(lambda a, b, c: mx.fast.scaled_dot_product_attention(a, b, c, scale=scale, mask=sdpa_mask), q, k, v, dO)
    for i, nm in enumerate(("dQ", "dK", "dV")):
        assert _delta(gk[i], go[i]) < _TOL, f"{nm} wrong vs fp32 oracle"
        wb = _delta(gk[i], gs[i])
        if expect[nm] == "sdpa-vjp":
            assert wb == 0.0, f"{nm} expected SDPA-vjp but Δ={wb} (rerouted to native? update map)"
        else:
            assert wb > 0.0, f"{nm} expected native but byte-identical to SDPA-vjp (reverted? update map)"


_ALL_SDPA = {"dQ": "sdpa-vjp", "dK": "sdpa-vjp", "dV": "sdpa-vjp"}
_ALL_NATIVE = {"dQ": "native", "dK": "native", "dV": "native"}
_HYBRID = {"dQ": "sdpa-vjp", "dK": "sdpa-vjp", "dV": "native"}


class TestBackwardWhichBinaryAndCorrectness:
    def test_dense_d128_causal_all_sdpa_vjp(self):
        q, k, v = _qkv(2, 8, 4096, 128)
        _audit(q, k, v, 1 / math.sqrt(128), True, _ALL_SDPA)

    def test_dense_d64_causal_default_on_all_native(self):
        q, k, v = _qkv(2, 8, 4096, 64)
        _audit(q, k, v, 1 / math.sqrt(64), True, _ALL_NATIVE)

    def test_dense_d64_noncausal_default_on_all_native(self):
        q, k, v = _qkv(2, 8, 4096, 64)
        _audit(q, k, v, 1 / math.sqrt(64), False, _ALL_NATIVE)

    def test_sparse_default_all_sdpa_vjp(self):
        q, k, v = _qkv(2, 8, 4096, 128); NB = 4096 // 64
        m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
        _audit(q, k, v, 1 / math.sqrt(128), False, _ALL_SDPA, bias=_block_bias(m, 4096), smask=m)

    def test_sparse_optin_hybrid_native_dV(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        q, k, v = _qkv(2, 8, 4096, 128); NB = 4096 // 64
        m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
        _audit(q, k, v, 1 / math.sqrt(128), False, _HYBRID, bias=_block_bias(m, 4096), smask=m)

    def test_sparse_optin_full_native(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        monkeypatch.setenv("MFA_V6_BWD_SPARSE_NATIVE", "1")
        q, k, v = _qkv(2, 8, 4096, 128); NB = 4096 // 64
        m = np.zeros((NB, NB), bool); m[:, :NB // 4] = True; m = mx.array(m)
        _audit(q, k, v, 1 / math.sqrt(128), False, _ALL_NATIVE, bias=_block_bias(m, 4096), smask=m)

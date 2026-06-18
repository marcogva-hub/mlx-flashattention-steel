"""v2.58.1 P1 — return_attn_weights must apply attn_bias / alibi_slopes / window_size.

Pre-fix bug (silent correctness error): `flash_attention(..., return_attn_weights=True)`
routed to `_sdpa_with_weights(q,k,v,scale,causal,softcap,dropout_p)` — dropping `attn_bias`,
`alibi_slopes`, and `window_size` entirely → plausible-but-wrong output AND weights, in the
exact path users hit when *validating* attention.

Discipline:
  - Consistency oracle: weights-path output ≈ the production (non-weights) path for the same args.
  - Lesson #11: BOTH paths ≈ an INDEPENDENT manual fp32 SDPA-with-feature (never kernel-vs-kernel
    alone); the returned WEIGHTS also match the fp32 softmax probabilities.
"""
from __future__ import annotations
import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import flash_attention


def _fp32_attn(q, k, v, scale, causal, *, attn_bias=None, alibi=None,
               window=None, softcap=0.0):
    """Independent fp32 reference: scores = scale·q@kᵀ [softcap][+bias][+alibi]
    [+window −inf][+causal −inf] → softmax → @v.  Returns (out, probs) in fp32."""
    qf = np.array(q.astype(mx.float32)); kf = np.array(k.astype(mx.float32))
    vf = np.array(v.astype(mx.float32))
    B, H, N, D = qf.shape; S = kf.shape[2]
    s = np.einsum("bhnd,bhsd->bhns", qf, kf) * scale
    if softcap > 0.0:
        s = np.tanh(s / softcap) * softcap
    if attn_bias is not None:
        s = s + np.array(attn_bias.astype(mx.float32))
    if alibi is not None:
        sl = np.array(alibi.astype(mx.float32))            # [H]
        qp = np.arange(N)[:, None]; kp = np.arange(S)[None, :]
        s = s + sl[None, :, None, None] * (kp - qp)[None, None, :, :]
    qidx = np.arange(S - N, S)[:, None]; kidx = np.arange(S)[None, :]
    if window is not None:
        wl, wr = window
        wl_eff = wl if wl >= 0 else S; wr_eff = wr if wr >= 0 else S
        in_win = (kidx >= qidx - wl_eff) & (kidx <= qidx + wr_eff)
        s = np.where(in_win[None, None], s, -np.inf)
    if causal:
        s = np.where((kidx > qidx)[None, None], -np.inf, s)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s); probs = e / e.sum(axis=-1, keepdims=True)
    out = np.einsum("bhns,bhsd->bhnd", probs, vf)
    return out, probs


def _mk(B=1, H=4, N=128, S=128, D=128, dt=mx.float32, seed=0):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.5).astype(dt)
    k = (mx.random.uniform(-1, 1, (B, H, S, D)) * 0.5).astype(dt)
    v = (mx.random.uniform(-1, 1, (B, H, S, D)) * 0.5).astype(dt)
    mx.eval(q, k, v); return q, k, v


def _maxerr(a, b):
    return float(np.abs(np.array(a.astype(mx.float32)) - np.asarray(b, dtype=np.float32)).max())


_FEATURES = ["none", "attn_bias", "alibi", "window"]


@pytest.mark.parametrize("feat", _FEATURES)
@pytest.mark.parametrize("causal", [False, True])
def test_weights_path_matches_production_and_fp32(feat, causal):
    """Consistency (weights == non-weights) AND correctness (both == fp32), per feature."""
    D = 128; N = S = 128; H = 4
    q, k, v = _mk(H=H, N=N, S=S, D=D, dt=mx.float32, seed=7)
    scale = 1.0 / math.sqrt(D)
    kw = {}; fp = {}
    if feat == "attn_bias":
        mx.random.seed(3)
        bias = (mx.random.uniform(-1, 1, (1, H, N, S)) * 0.3).astype(mx.float32)
        mx.eval(bias); kw["attn_bias"] = bias; fp["attn_bias"] = bias
    elif feat == "alibi":
        slopes = mx.array([2.0 ** (-(i + 1)) for i in range(H)], dtype=mx.float32)
        mx.eval(slopes); kw["alibi_slopes"] = slopes; fp["alibi"] = slopes
    elif feat == "window":
        kw["window_size"] = (32, 16); fp["window"] = (32, 16)

    out_w, probs = flash_attention(q, k, v, scale=scale, causal=causal,
                                   return_attn_weights=True, **kw)
    out_prod = flash_attention(q, k, v, scale=scale, causal=causal, **kw)
    ref_out, ref_probs = _fp32_attn(q, k, v, scale, causal, **fp)
    mx.eval(out_w, probs, out_prod)

    # (a) consistency: weights-path output == production (non-weights) path
    assert _maxerr(out_w, np.array(out_prod.astype(mx.float32))) < 1e-3, \
        f"[{feat} causal={causal}] weights-path output diverges from production path"
    # (b) Lesson #11: output == independent fp32
    assert _maxerr(out_w, ref_out) < 1e-3, f"[{feat} causal={causal}] output != fp32"
    # (c) the returned WEIGHTS == fp32 softmax probabilities
    assert _maxerr(probs, ref_probs) < 1e-3, f"[{feat} causal={causal}] weights != fp32 softmax"


@pytest.mark.parametrize("feat", ["none", "window"])
def test_weights_path_with_softcap_allowed(feat):
    """softcap composes with the ALLOWED features (none / window): weights path ==
    fp32 (correctness)."""
    D = 128; N = S = 128; H = 4; softcap = 30.0
    q, k, v = _mk(H=H, N=N, S=S, D=D, dt=mx.float32, seed=9)
    scale = 1.0 / math.sqrt(D)
    kw = {"softcap": softcap}; fp = {"softcap": softcap}
    if feat == "window":
        kw["window_size"] = (32, 16); fp["window"] = (32, 16)
    out_w, probs = flash_attention(q, k, v, scale=scale, causal=True,
                                   return_attn_weights=True, **kw)
    ref_out, ref_probs = _fp32_attn(q, k, v, scale, True, **fp)
    mx.eval(out_w, probs)
    assert _maxerr(out_w, ref_out) < 1e-3, f"[{feat}+softcap] output != fp32"
    assert _maxerr(probs, ref_probs) < 1e-3, f"[{feat}+softcap] weights != fp32 softmax"


@pytest.mark.parametrize("kw", [
    {"softcap": 30.0, "alibi_slopes": mx.array([0.5, 0.25, 0.125, 0.0625])},
    {"softcap": 30.0, "attn_bias": None},  # filled below
])
def test_weights_path_rejects_unsupported_softcap_combos(kw):
    """Rule 8: softcap+alibi and softcap+attn_bias have no kernel path — production
    RAISES (it does not silently drop). The weights path inherits that guard (the
    check fires before the return_attn_weights branch), so it must raise too — never
    return a silently-wrong result for these combos."""
    D = 128; N = S = 128; H = 4
    q, k, v = _mk(H=H, N=N, S=S, D=D, dt=mx.float32, seed=9)
    scale = 1.0 / math.sqrt(D)
    if "attn_bias" in kw:
        mx.random.seed(4)
        kw = dict(kw); kw["attn_bias"] = (mx.random.uniform(-1, 1, (1, H, N, S)) * 0.3).astype(mx.float32)
    with pytest.raises(ValueError):
        flash_attention(q, k, v, scale=scale, causal=True,
                        return_attn_weights=True, **kw)


def test_weights_path_fp16_each_feature():
    """fp16: looser tolerance; each feature applied (vs fp32 reference)."""
    D = 128; N = S = 128; H = 4
    q, k, v = _mk(H=H, N=N, S=S, D=D, dt=mx.float16, seed=11)
    scale = 1.0 / math.sqrt(D)
    slopes = mx.array([2.0 ** (-(i + 1)) for i in range(H)], dtype=mx.float16)
    mx.random.seed(5)
    bias = (mx.random.uniform(-1, 1, (1, H, N, S)) * 0.3).astype(mx.float16)
    mx.eval(slopes, bias)
    for kw, fp in [({"attn_bias": bias}, {"attn_bias": bias}),
                   ({"alibi_slopes": slopes}, {"alibi": slopes}),
                   ({"window_size": (32, 16)}, {"window": (32, 16)})]:
        out_w, probs = flash_attention(q, k, v, scale=scale, causal=True,
                                       return_attn_weights=True, **kw)
        ref_out, _ = _fp32_attn(q, k, v, scale, True, **fp)
        mx.eval(out_w)
        assert _maxerr(out_w, ref_out) < 2e-2, f"fp16 {list(kw)} output != fp32 (within fp16 floor)"


def test_return_attn_weights_lse_still_mutually_exclusive():
    q, k, v = _mk()
    with pytest.raises(ValueError):
        flash_attention(q, k, v, return_attn_weights=True, return_lse=True)

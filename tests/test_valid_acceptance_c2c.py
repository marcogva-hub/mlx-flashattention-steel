"""Volet C2c — valid-acceptance sweep (symmetric completion of the validation work).

C / C2 / C2b were one-directional (reject-malformed). The GNA-GQA regression
(C2 wrongly required q==k==v heads, hidden because the GQA test used the raw
_ext path) proved the *accept-valid* direction had real, unpinned holes. This
file pins that every validation check ACCEPTS the valid inputs the underlying
kernel supports — so a too-strict check can never silently hide again.

Verdict (verified first-hand, C2c): the shared `_assert_qkv_mutual_compat` helper
is GQA-aware (`q_heads % kv_heads == 0`, not `q==k==v`) and v-dim-agnostic — it
does NOT reject valid GQA or asymmetric `D_v`. No too-strict check was found; the
C2b worry did not materialize. These cells are the durable pins.

Coverage (the envelope test_oracle_envelope.py already pins dtype/D/causal
acceptance for the symmetric-head case; this file adds the GQA and asymmetric-D_v
axes that were unpinned):
  - GQA (Hq=8, Hk=Hv=2, Hq%Hk==0) accepted + oracle-correct, per entry
  - asymmetric D_v != D_qk accepted where supported (dense, sparse), and
    cleanly rejected where unsupported (sage, varlen) — pinned either way
  - f16 + bf16
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as _ext

F16, BF16 = mx.float16, mx.bfloat16
# f16: ~1e-3 floor; bf16: ~2e-2; sage is int8 (lossy) → ~2e-2.
BOUND = {F16: 5e-3, BF16: 3e-2}
SAGE_BOUND = {F16: 3e-2, BF16: 4e-2}


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _rand(shape, dt, seed):
    mx.random.seed(seed)
    a = mx.random.normal(shape).astype(dt)
    mx.eval(a)
    return a


def _gqa_oracle(q, k, v, scale, causal):
    """Independent fp64 GQA oracle: q head h attends kv head h // (Hq/Hk)."""
    q, k, v = _f64(q), _f64(k), _f64(v)
    B, Hq, N, D = q.shape
    Hk, S = k.shape[1], k.shape[2]
    g = Hq // Hk
    out = np.zeros((B, Hq, N, v.shape[3]))
    for b in range(B):
        for h in range(Hq):
            hk = h // g
            s = (q[b, h] @ k[b, hk].T) * scale
            if causal:
                i = np.arange(N)[:, None]
                j = np.arange(S)[None, :]
                s = np.where(j <= i + (S - N), s, -1e30)
            s -= s.max(1, keepdims=True)
            e = np.exp(s)
            p = e / e.sum(1, keepdims=True)
            out[b, h] = p @ v[b, hk]
    return out


def _relerr(o, ref):
    return float(np.max(np.abs(_f64(o) - ref)) / (np.max(np.abs(ref)) + 1e-12))


# ───────────────────────── GQA acceptance + correctness ──────────────────────
@pytest.mark.parametrize("dt", [F16, BF16])
def test_gqa_accepted_flash_attention(dt):
    Hq, Hk, N, D = 8, 2, 256, 64
    sc = 1.0 / math.sqrt(D)
    q, k, v = _rand((1, Hq, N, D), dt, 1), _rand((1, Hk, N, D), dt, 2), _rand((1, Hk, N, D), dt, 3)
    for causal in (False, True):
        o = mlx_mfa.flash_attention(q, k, v, scale=sc, causal=causal)
        mx.eval(o)
        assert _relerr(o, _gqa_oracle(q, k, v, sc, causal)) <= BOUND[dt]


@pytest.mark.parametrize("dt", [F16, BF16])
def test_gqa_accepted_sparse(dt):
    Hq, Hk, N, D = 8, 2, 256, 128
    sc = 1.0 / math.sqrt(D)
    q, k, v = _rand((1, Hq, N, D), dt, 4), _rand((1, Hk, N, D), dt, 5), _rand((1, Hk, N, D), dt, 6)
    bm = mlx_mfa.make_causal_block_mask(N, head_dim=D)
    o = mlx_mfa.flash_attention_sparse(q, k, v, bm, scale=sc, causal=True)
    mx.eval(o)
    assert _relerr(o, _gqa_oracle(q, k, v, sc, True)) <= BOUND[dt]


@pytest.mark.parametrize("dt", [F16, BF16])
def test_gqa_accepted_varlen(dt):
    Hq, Hk, N, D = 8, 2, 256, 64
    sc = 1.0 / math.sqrt(D)
    q, k, v = _rand((1, Hq, N, D), dt, 7), _rand((1, Hk, N, D), dt, 8), _rand((1, Hk, N, D), dt, 9)
    cu = mx.array([0, N], dtype=mx.int32)
    o = mlx_mfa.flash_attention_varlen(q, k, v, cu, cu, N, N, scale=sc, causal=False)
    mx.eval(o)
    assert _relerr(o, _gqa_oracle(q, k, v, sc, False)) <= BOUND[dt]


@pytest.mark.parametrize("dt", [F16, BF16])
def test_gqa_accepted_sage(dt):
    # sage is int8-quantized → looser bound (lossy by design).
    Hq, Hk, N, D = 8, 2, 256, 64
    sc = 1.0 / math.sqrt(D)
    q, k, v = _rand((1, Hq, N, D), dt, 10), _rand((1, Hk, N, D), dt, 11), _rand((1, Hk, N, D), dt, 12)
    o = mlx_mfa.sage_attention(q, k, v, scale=sc, causal=False)
    mx.eval(o)
    assert _relerr(o, _gqa_oracle(q, k, v, sc, False)) <= SAGE_BOUND[dt]


@pytest.mark.parametrize("dt", [F16, BF16])
def test_gqa_accepted_gna(dt):
    # GNA supports GQA (gqa_factor = Hq/Hk; see test_gna_native_gqa). The public
    # boundary must ACCEPT it (the C2 q==k==v check wrongly rejected this). Pin:
    # accepted + finite + byte-matches the raw native kernel (validated separately).
    Hq, Hk, N, D = 8, 2, 64, 128
    sc = 1.0 / math.sqrt(D)
    q, k, v = _rand((1, Hq, N, D), dt, 13), _rand((1, Hk, N, D), dt, 14), _rand((1, Hk, N, D), dt, 15)
    seq, win, st = (4, 4, 4), (2, 2, 2), (1, 1, 1)
    o = mlx_mfa.flash_attention_gna(q, k, v, seq_shape=seq, window_size=win, stride=st)
    mx.eval(o)
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())
    raw = _ext.mfa_gna_forward(q, k, v, sc, *seq, *win, *st)
    mx.eval(raw)
    # public routes to the native kernel for this envelope → byte-identical
    assert float(np.max(np.abs(np.array(o.astype(mx.float32)) -
                               np.array(raw.astype(mx.float32))))) == 0.0


# ───────────────────────── asymmetric D_v acceptance ─────────────────────────
@pytest.mark.parametrize("dt", [F16, BF16])
def test_asymmetric_dv_accepted_where_supported(dt):
    # dense + sparse support D_v != D_qk (SDPA-class path). Accepted + correct.
    N, Dqk, Dv = 256, 128, 64
    sc = 1.0 / math.sqrt(Dqk)
    q, k, v = _rand((1, 4, N, Dqk), dt, 16), _rand((1, 4, N, Dqk), dt, 17), _rand((1, 4, N, Dv), dt, 18)
    o = mlx_mfa.flash_attention(q, k, v, scale=sc, causal=False)
    mx.eval(o)
    assert _relerr(o, _gqa_oracle(q, k, v, sc, False)) <= BOUND[dt]
    bm = mlx_mfa.make_causal_block_mask(N, head_dim=Dqk)
    o2 = mlx_mfa.flash_attention_sparse(q, k, v, bm, scale=sc, causal=True)
    mx.eval(o2)
    assert _relerr(o2, _gqa_oracle(q, k, v, sc, True)) <= BOUND[dt]


def test_asymmetric_dv_rejected_where_unsupported():
    # sage + varlen kernels require D_v == D_qk; a mismatch must raise (not silent).
    # The helper is v-dim-agnostic, so this raise comes from the kernel contract.
    N, Dqk, Dv = 256, 128, 64
    sc = 1.0 / math.sqrt(Dqk)
    q, k, v = _rand((1, 4, N, Dqk), F16, 19), _rand((1, 4, N, Dqk), F16, 20), _rand((1, 4, N, Dv), F16, 21)
    with pytest.raises(Exception):
        mx.eval(mlx_mfa.sage_attention(q, k, v, scale=sc, causal=False))
    cu = mx.array([0, N], dtype=mx.int32)
    with pytest.raises(Exception):
        mx.eval(mlx_mfa.flash_attention_varlen(q, k, v, cu, cu, N, N, scale=sc))

"""Volet K2 — 4-axis hardening locks for the remaining 8 raw entries.

R2 mfa_forward_with_lse, R8 mfa_gna_forward, R9 sparse_attention_forward
(verify-only — confirmed comprehensive), R11 mfa_quantize_per_block,
R12 mfa_smooth_quantize_k, R14 conv3d_nax_forward, R16 v6_nax_backward_{query,kv}.

Defects found + fixed (verified first-hand): R2 accepted float32; R8 accepted
dtype-mismatched K/V and non-positive window/stride (→ NaN); R16 accepted
dtype-mismatched K and float16 lse/d_vec (silent-wrong gradients). Each new check
was validated against the production valid space first (the K1 over-strict lesson).
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa._ext as e


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


# ── R2 mfa_forward_with_lse ─────────────────────────────────────────────────────
def _qkv(dt=mx.float16, bq=1, hk=2, nk=256, dq=128):
    mx.random.seed(0)
    q = mx.random.normal((bq, 8, 256, dq)).astype(dt)
    k = mx.random.normal((1, hk, nk, 128)).astype(dt)
    v = mx.random.normal((1, hk, nk, 128)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


_SC = 1.0 / math.sqrt(128)


def test_r2_valid_and_correctness():
    q, k, v = _qkv()
    o, lse = e.mfa_forward_with_lse(q, k, v, _SC, False)
    mx.eval(o, lse)
    qf, kf, vf = _f64(q), _f64(k), _f64(v)
    kk, vv = np.repeat(kf, 4, 1), np.repeat(vf, 4, 1)
    s = np.einsum("bhnd,bhmd->bhnm", qf, kk) * _SC
    s -= s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True)
    ref = np.einsum("bhnm,bhmd->bhnd", p, vv)
    rel = float(np.max(np.abs(_f64(o) - ref)) / (np.max(np.abs(ref)) + 1e-9))
    assert rel < 3e-3, f"R2 relerr {rel:.3e}"


def test_r2_float32_is_valid():
    # R2 is verify-only: the dense MFAttention primitive SUPPORTS float32 (the
    # return_lse path upcasts mixed dtypes to f32). f32 must RUN, not raise.
    q, k, v = _qkv(dt=mx.float32)
    o, _l = e.mfa_forward_with_lse(q, k, v, _SC, False)
    mx.eval(o)


@pytest.mark.parametrize("mut", ["batch", "k_seq", "q_D", "dtype", "gqa"])
def test_r2_malformed_raises(mut):
    if mut == "batch":
        q, k, v = _qkv(bq=2)
    elif mut == "k_seq":
        q, k, v = _qkv(); v = v[:, :, :128, :]
    elif mut == "q_D":
        q, k, v = _qkv(dq=64)
    elif mut == "dtype":
        q, k, v = _qkv(); k = k.astype(mx.bfloat16)
    elif mut == "gqa":
        q, k, v = _qkv(hk=3)
    with pytest.raises((ValueError, Exception)):
        o, _l = e.mfa_forward_with_lse(q, k, v, _SC, False); mx.eval(o)


# ── R8 mfa_gna_forward ──────────────────────────────────────────────────────────
def _gqkv(hk=2, dtk=None, dt=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, 8, 256, 128)).astype(dt)
    k = mx.random.normal((1, hk, 256, 128)).astype(dtk or dt)
    v = mx.random.normal((1, hk, 256, 128)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def _gna(q, k, v, w0=3, s0=1):
    return e.mfa_gna_forward(q, k, v, _SC, 8, 8, 4, w0, 3, 3, s0, 1, 1)


def test_r8_valid_runs():
    mx.eval(_gna(*_gqkv()))


@pytest.mark.parametrize("mut", ["k_seq", "dtype", "win0", "stride0", "win_neg"])
def test_r8_malformed_raises(mut):
    q, k, v = _gqkv()
    kw = {}
    if mut == "k_seq":
        v = v[:, :, :128, :]
    elif mut == "dtype":
        q, k, v = _gqkv(dtk=mx.bfloat16)
    elif mut == "win0":
        kw["w0"] = 0
    elif mut == "stride0":
        kw["s0"] = 0
    elif mut == "win_neg":
        kw["w0"] = -1
    with pytest.raises((ValueError, Exception)):
        mx.eval(_gna(q, k, v, **kw))


# ── R9 sparse_attention_forward (verify-only: comprehensive) ────────────────────
def _sq(bq=1, hk=2, vseq=1024, dq=128):
    mx.random.seed(0)
    q = mx.random.normal((bq, 8, 1024, dq)).astype(mx.float16)
    k = mx.random.normal((1, hk, 1024, 128)).astype(mx.float16)
    v = mx.random.normal((1, hk, vseq, 128)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


_SMASK = None


def _smask():
    return mx.ones((8, 32, 32), dtype=mx.bool_)


def test_r9_valid_runs():
    mx.eval(e.sparse_attention_forward(*_sq(), _smask(), 32, False, _SC))


@pytest.mark.parametrize("mut", ["batch", "k_seq", "q_D"])
def test_r9_malformed_raises(mut):
    if mut == "batch":
        q, k, v = _sq(bq=2)
    elif mut == "k_seq":
        q, k, v = _sq(vseq=512)
    elif mut == "q_D":
        q, k, v = _sq(dq=64)
    with pytest.raises((ValueError, Exception)):
        mx.eval(e.sparse_attention_forward(q, k, v, _smask(), 32, False, _SC))


# ── R11 / R12 quantizers ────────────────────────────────────────────────────────
def test_r11_valid_runs():
    qi, sc = e.mfa_quantize_per_block(mx.random.normal((1, 8, 256, 128)).astype(mx.float16), 32)
    mx.eval(qi, sc)


@pytest.mark.parametrize("mut", ["block0", "notpow2", "f32"])
def test_r11_malformed_raises(mut):
    x = mx.random.normal((1, 8, 256, 128)).astype(mx.float32 if mut == "f32" else mx.float16)
    bs = {"block0": 0, "notpow2": 33, "f32": 32}[mut]
    with pytest.raises((ValueError, Exception)):
        a, b = e.mfa_quantize_per_block(x, bs); mx.eval(a)


def test_r12_valid_runs():
    a, b, c = e.mfa_smooth_quantize_k(mx.random.normal((1, 2, 256, 128)).astype(mx.float16), 32)
    mx.eval(a, b, c)


@pytest.mark.parametrize("mut", ["block0", "f32"])
def test_r12_malformed_raises(mut):
    x = mx.random.normal((1, 2, 256, 128)).astype(mx.float32 if mut == "f32" else mx.float16)
    bs = 0 if mut == "block0" else 32
    with pytest.raises((ValueError, Exception)):
        a, b, c = e.mfa_smooth_quantize_k(x, bs); mx.eval(a)


# ── R14 conv3d_nax_forward ──────────────────────────────────────────────────────
def _conv(x=None, w=None, **kw):
    if x is None:
        x = mx.random.normal((1, 8, 8, 8, 16)).astype(mx.float16)
    if w is None:
        w = mx.random.normal((32, 3, 3, 3, 16)).astype(mx.float16)
    mx.eval(x, w)
    return e.conv3d_nax_forward(x, w, **kw)


def test_r14_valid_runs():
    mx.eval(_conv())


@pytest.mark.parametrize("mut", ["cin", "dtype", "stride0", "negpad"])
def test_r14_malformed_raises(mut):
    kw = {}
    w = None
    if mut == "cin":
        w = mx.random.normal((32, 3, 3, 3, 8)).astype(mx.float16)
    elif mut == "dtype":
        w = mx.random.normal((32, 3, 3, 3, 16)).astype(mx.bfloat16)
    elif mut == "stride0":
        kw["stride"] = [0, 1, 1]
    elif mut == "negpad":
        kw["padding"] = [-1, 0, 0, 0, 0, 0]
    with pytest.raises((ValueError, Exception)):
        mx.eval(_conv(w=w, **kw))


# ── R16 v6_nax_backward_{query,kv} ──────────────────────────────────────────────
def _bwd_inputs(dt=mx.float16, Hq=4, Hk=4, N=512, D=64):
    mx.random.seed(0)
    q = mx.random.normal((1, Hq, N, D)).astype(dt)
    k = mx.random.normal((1, Hk, N, D)).astype(dt)
    v = mx.random.normal((1, Hk, N, D)).astype(dt)
    mx.eval(q, k, v)
    sc = 1.0 / math.sqrt(D)
    o, lse = e.v6_nax_forward(q, k, v, False, True, sc)
    mx.eval(o, lse)
    do = mx.random.normal((1, Hq, N, D)).astype(dt)
    dvec = mx.sum(do.astype(mx.float32) * o.astype(mx.float32), axis=-1)
    mx.eval(do, dvec)
    return q, k, v, o, lse, do, dvec, sc


def test_r16_correctness_vs_vjp_oracle():
    q, k, v, o, lse, do, dvec, sc = _bwd_inputs()
    dQ = e.v6_nax_backward_query(q, k, v, o, lse, do, dvec, sc, False)
    dK, dV = e.v6_nax_backward_kv(q, k, v, o, lse, do, dvec, sc, False)
    mx.eval(dQ, dK, dV)

    def fwd(q_, k_, v_):
        s = mx.matmul(q_, mx.swapaxes(k_, -1, -2)) * sc
        p = mx.softmax(s, axis=-1)
        return mx.matmul(p, v_)
    qf = q.astype(mx.float32); kf = k.astype(mx.float32); vf = v.astype(mx.float32)
    _, vjps = mx.vjp(fwd, (qf, kf, vf), (do.astype(mx.float32),))
    rdQ, rdK, rdV = vjps
    mx.eval(rdQ, rdK, rdV)
    for got, ref, nm in [(dQ, rdQ, "dQ"), (dK, rdK, "dK"), (dV, rdV, "dV")]:
        rel = float(np.max(np.abs(_f64(got) - _f64(ref))) / (np.max(np.abs(_f64(ref))) + 1e-9))
        assert rel < 5e-2, f"R16 {nm} relerr {rel:.3e} vs fp32 vjp oracle"


@pytest.mark.parametrize("entry", ["query", "kv"])
def test_r16_valid_runs(entry):
    q, k, v, o, lse, do, dvec, sc = _bwd_inputs()
    fn = e.v6_nax_backward_query if entry == "query" else e.v6_nax_backward_kv
    r = fn(q, k, v, o, lse, do, dvec, sc, False)
    mx.eval(r[0] if isinstance(r, tuple) else r)


@pytest.mark.parametrize("entry", ["query", "kv"])
@pytest.mark.parametrize("mut", ["dtype", "dvec_f16", "lse_f16", "k_seq"])
def test_r16_malformed_raises(entry, mut):
    q, k, v, o, lse, do, dvec, sc = _bwd_inputs()
    if mut == "dtype":
        k = k.astype(mx.bfloat16)
    elif mut == "dvec_f16":
        dvec = dvec.astype(mx.float16)
    elif mut == "lse_f16":
        lse = lse.astype(mx.float16)
    elif mut == "k_seq":
        v = v[:, :, :256, :]
    fn = e.v6_nax_backward_query if entry == "query" else e.v6_nax_backward_kv
    with pytest.raises((ValueError, Exception)):
        r = fn(q, k, v, o, lse, do, dvec, sc, False)
        mx.eval(r[0] if isinstance(r, tuple) else r)

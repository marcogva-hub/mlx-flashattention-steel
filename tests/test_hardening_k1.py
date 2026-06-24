"""Volet K1 — 4-axis hardening locks for priority groups 1–6.

Entries (K0 groups): R15 v6_nax_forward, R7 mfa_attention_varlen_forward,
R5 mfa_attention_rope_forward, R3 mfa_attention_alibi_forward,
R4 mfa_attention_bias_forward, R6 mfa_attention_sparse_forward[_with_lse],
R10 sparse_attention_forward_with_lse, R13 mfa_scatter_kv.

Every one of these read multiple buffers while missing one or more mutual
shape/dtype/count checks (verified first-hand): batch mismatch → NaN, k_seq /
k_heads / q_D / dtype mismatch → silent-wrong no-raise. Fixed via the shared
`validate_dense_qkv` C++ helper (+ per-entry residual). Both directions locked.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa._ext as e

B, Hq, Hk, N, D = 1, 8, 2, 256, 128
SC = 1.0 / math.sqrt(D)
try:
    _IS_M5 = bool(e.get_device_info().get("is_m5_plus", False))
except Exception:
    _IS_M5 = False


def _qkv(dt=mx.float16, bq=B, hk=Hk, nk=N, dq=D, dtk=None):
    mx.random.seed(0)
    q = mx.random.normal((bq, Hq, N, dq)).astype(dt)
    k = mx.random.normal((B, hk, nk, D)).astype(dtk or dt)
    v = mx.random.normal((B, hk, nk, D)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _oracle(q, k, v, sc, causal=False):
    qf, kf, vf = _f64(q), _f64(k), _f64(v)
    g = qf.shape[1] // kf.shape[1]
    kk, vv = np.repeat(kf, g, 1), np.repeat(vf, g, 1)
    s = np.einsum("bhnd,bhmd->bhnm", qf, kk) * sc
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhnm,bhmd->bhnd", p, vv)


# ── dense-family entries: (name, callable taking q,k,v) ───────────────────────
_DENSE = {
    "v6": lambda q, k, v: e.v6_nax_forward(q, k, v, False, True, SC),
    "alibi": lambda q, k, v: e.mfa_attention_alibi_forward(
        q, k, v, mx.zeros((Hq,), dtype=mx.float32), SC, False),
    "bias": lambda q, k, v: e.mfa_attention_bias_forward(
        q, k, v, mx.zeros((1, 1, 1, int(k.shape[2])), dtype=mx.float32), 1, SC, False),
    "rope": lambda q, k, v: e.mfa_attention_rope_forward(
        q, k, v, mx.ones((N, D // 2), dtype=mx.float32),
        mx.zeros((N, D // 2), dtype=mx.float32), SC, False, 0, True),
    "sparse": lambda q, k, v: e.mfa_attention_sparse_forward(
        q, k, v, mx.ones((Hq, (N + 31) // 32, (N + 31) // 32), dtype=mx.uint8), SC, False),
    "sparse_lse": lambda q, k, v: e.mfa_attention_sparse_forward_with_lse(
        q, k, v, mx.ones((Hq, (N + 31) // 32, (N + 31) // 32), dtype=mx.uint8), SC, False),
}


@pytest.mark.parametrize("name", list(_DENSE))
def test_dense_valid_runs(name):
    # SPARSE-D128-OOB (sweep iter-1): the raw STEEL V1 block-sparse forward is OOB
    # at D=128 on M5+ and is now correctly REJECTED (public path uses SDPA). These
    # cells (D=128) previously asserted the broken kernel "runs" = green-on-wrong-
    # binary. Assert the TRUE behavior: it raises on M5+. (D=64 raw sparse + the
    # public SDPA path are exercised in test_validation_matrix's sparse-D128 lock.)
    if name in ("sparse", "sparse_lse") and D == 128 and _IS_M5:
        with pytest.raises(Exception):
            o = _DENSE[name](*_qkv())
            mx.eval(o[0] if isinstance(o, tuple) else o)
        return
    o = _DENSE[name](*_qkv())
    mx.eval(o[0] if isinstance(o, tuple) else o)  # accept-valid


@pytest.mark.parametrize("name", list(_DENSE))
@pytest.mark.parametrize("mut", ["batch", "k_seq", "k_heads", "q_D", "dtype", "gqa"])
def test_dense_malformed_raises(name, mut):
    call = _DENSE[name]
    if mut == "batch":
        args = _qkv(bq=2)
    elif mut == "k_seq":
        q, k, v = _qkv(); args = (q, k, v[:, :, : N // 2, :])
    elif mut == "k_heads":
        q, k, v = _qkv(); args = (q, k[:, :1], v)
    elif mut == "q_D":
        args = _qkv(dq=64)
    elif mut == "dtype":
        args = _qkv(dtk=mx.bfloat16)
    elif mut == "gqa":
        args = _qkv(hk=3)
    with pytest.raises((ValueError, Exception)):
        o = call(*args)
        mx.eval(o[0] if isinstance(o, tuple) else o)


def test_v6_correctness_vs_oracle():
    q, k, v = _qkv()
    o, _ = e.v6_nax_forward(q, k, v, False, True, SC)
    mx.eval(o)
    rel = float(np.max(np.abs(_f64(o) - _oracle(q, k, v, SC))) /
                (np.max(np.abs(_oracle(q, k, v, SC))) + 1e-9))
    assert rel < 3e-3, f"v6 forward relerr {rel:.3e} vs fp64 oracle"


# ── R7 varlen ─────────────────────────────────────────────────────────────────
def _vqkv(dt=mx.float16, hk=Hk, dtk=None, dq=D):
    mx.random.seed(0)
    q = mx.random.normal((1, Hq, N, dq)).astype(dt)
    k = mx.random.normal((1, hk, N, D)).astype(dtk or dt)
    v = mx.random.normal((1, hk, N, D)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def _vcall(q, k, v, cu_q=None, cu_k=None, to=None):
    nt = (N + 31) // 32
    cu_q = cu_q if cu_q is not None else mx.array([0, N], dtype=mx.int32)
    cu_k = cu_k if cu_k is not None else mx.array([0, N], dtype=mx.int32)
    to = to if to is not None else mx.array([0, nt], dtype=mx.int32)
    return e.mfa_attention_varlen_forward(q, k, v, cu_q, cu_k, to, SC, True)


def test_varlen_valid_runs():
    o, _ = _vcall(*_vqkv()); mx.eval(o)


@pytest.mark.parametrize("mut", ["k_seq", "q_D", "dtype", "cu_float", "cu_int64"])
def test_varlen_malformed_raises(mut):
    q, k, v = _vqkv()
    kw = {}
    if mut == "k_seq":
        v = v[:, :, : N // 2, :]
    elif mut == "q_D":
        q, k, v = _vqkv(dq=64)
    elif mut == "dtype":
        q, k, v = _vqkv(dtk=mx.bfloat16)
    elif mut == "cu_float":
        kw["cu_q"] = mx.array([0.0, N], dtype=mx.float32)
    elif mut == "cu_int64":
        kw["cu_q"] = mx.array([0, N], dtype=mx.int64)
    with pytest.raises((ValueError, Exception)):
        o, _ = _vcall(q, k, v, **kw); mx.eval(o)


# ── R10 NAX sparse-LSE ─────────────────────────────────────────────────────────
# R10 requires mask total bytes >= 4096 (small-mask address-space floor); use
# N=1024 (NQ=NK=32 → 8*1024 bool = 8192 bytes) so the valid case clears it.
_NL = 1024


def _lqkv(bq=B, hk=Hk, dq=D, vseq=_NL):
    mx.random.seed(0)
    q = mx.random.normal((bq, Hq, _NL, dq)).astype(mx.float16)
    k = mx.random.normal((B, hk, _NL, D)).astype(mx.float16)
    v = mx.random.normal((B, hk, vseq, D)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def _lmask():
    return mx.ones((Hq, _NL // 32, _NL // 32), dtype=mx.bool_)


def test_nax_sparse_lse_valid_runs():
    q, k, v = _lqkv()
    o, _l = e.sparse_attention_forward_with_lse(q, k, v, _lmask(), 32, False, SC)
    mx.eval(o)


@pytest.mark.parametrize("mut", ["batch", "k_seq", "q_D"])
def test_nax_sparse_lse_malformed_raises(mut):
    if mut == "batch":
        q, k, v = _lqkv(bq=2)
    elif mut == "k_seq":
        q, k, v = _lqkv(vseq=_NL // 2)
    elif mut == "q_D":
        q, k, v = _lqkv(dq=64)
    with pytest.raises((ValueError, Exception)):
        o, _l = e.sparse_attention_forward_with_lse(q, k, v, _lmask(), 32, False, SC)
        mx.eval(o)


# ── R13 scatter ────────────────────────────────────────────────────────────────
def _scat(td=mx.float16):
    mx.random.seed(0)
    pool = mx.zeros((8, 16, 4, 64), dtype=mx.float16)
    tok = mx.random.normal((3, 4, 64)).astype(td)
    ids = mx.array([2, 5, 1], dtype=mx.int32)
    offs = mx.array([0, 1, 2], dtype=mx.int32)
    mx.eval(pool, tok, ids, offs)
    return e.mfa_scatter_kv(pool, tok, ids, offs)


def test_scatter_valid_runs():
    mx.eval(_scat())


def test_scatter_dtype_mismatch_raises():
    with pytest.raises((ValueError, Exception)):
        mx.eval(_scat(td=mx.float32))


# ── Determinism axis (gather entries only): rope (R5) reuses shared KV_smem
# (disables DIRECT_READS), varlen (R7) stages packed K/V. Both source-verified
# to emit the inter-iteration barrier (barrier X / start-of-iter). Lock byteΔ=0
# at multi-tile N over 20 fresh-but-identical runs (the I2 lesson: warm/repeated).
# The rest (v6/alibi/bias/sparse/scatter/nax-sparse) are N-A (no scattered gather).
def _det(mk, nruns=20):
    outs = []
    for _ in range(nruns):
        o = mk()
        oo = o[0] if isinstance(o, tuple) else o
        mx.eval(oo)
        outs.append(np.array(oo.astype(mx.float32)))
    return max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, nruns))


@pytest.mark.parametrize("Nd", [512, 1024])
def test_rope_deterministic(Nd):
    def mk():
        mx.random.seed(0)
        q = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        k = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        cos = mx.ones((Nd, D // 2), dtype=mx.float32)
        sin = mx.zeros((Nd, D // 2), dtype=mx.float32)
        mx.eval(q, k, v, cos, sin)
        return e.mfa_attention_rope_forward(q, k, v, cos, sin, SC, True, 0, True)
    assert _det(mk) == 0.0


@pytest.mark.parametrize("Nd", [512, 1024])
def test_varlen_deterministic(Nd):
    def mk():
        mx.random.seed(0)
        q = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        k = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        v = mx.random.normal((1, 4, Nd, D)).astype(mx.float16)
        cu = mx.array([0, Nd], dtype=mx.int32)
        to = mx.array([0, (Nd + 31) // 32], dtype=mx.int32)
        mx.eval(q, k, v)
        return e.mfa_attention_varlen_forward(q, k, v, cu, cu, to, SC, True)
    assert _det(mk) == 0.0

"""CC Batch — raw/function-surface defect classes C, D, E (CI-locked).

Companion to `test_raw_dtype_matrix.py` (Classes A, B). Table-driven so Codex's
next findings append rows, not files.

  Class C — output allocated from q.shape under a validator that permitted
    D_v != D_qk → wrong-shape silent-wrong. Fix: `validate_dense_qkv` now rejects
    D_v != D_qk; the dense feature family (dense/sparse/rope/alibi/bias/varlen)
    inherits it. Public wrappers keep asym D_v via SDPA fallback.
  Class D — missing input validation at entry → silent broadcast / mixed-dtype.
    Fix: paged pools must match q dtype; TQ buffers must be uint8/fp16/fp32;
    flash_attention_topk routes through the Q/K/V mutual-compat + dtype-equality.
  Class E — early-return bypassed mixed-dtype normalization → wrong output dtype.
    Fix: backend="sdpa" normalizes k/v → q.dtype before the early-return.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as _ext

assert mlx_mfa.has_nax() is True
F16, BF16, F32, U8 = mx.float16, mx.bfloat16, mx.float32, mx.uint8


# ── Class C: dense-family raw forwards reject D_v != D_qk; symmetric runs ────────
def _raw_dense(q, k, v):
    return _ext.mfa_attention_forward(q, k, v, 0.125, False)


_C_CELLS = [
    # (name, Dqk, Dv, expect)  — asym rejects, symmetric runs
    ("dense_asym_Dv", 64, 32, "reject"),
    ("dense_sym_Dv", 64, 64, "ok"),
]


@pytest.mark.parametrize("name,dqk,dv,expect", _C_CELLS, ids=[c[0] for c in _C_CELLS])
def test_class_c_dense_dv(name, dqk, dv, expect):
    q = mx.random.normal((1, 4, 8, dqk)).astype(F16)
    k = mx.random.normal((1, 4, 8, dqk)).astype(F16)
    v = mx.random.normal((1, 4, 8, dv)).astype(F16)
    mx.eval(q, k, v)
    if expect == "reject":
        with pytest.raises(ValueError, match="head_dim"):
            _raw_dense(q, k, v)
    else:
        o = _raw_dense(q, k, v)
        mx.eval(o)
        ref = mx.fast.scaled_dot_product_attention(
            q.astype(F32), k.astype(F32), v.astype(F32), scale=0.125)
        mx.eval(ref)
        assert tuple(o.shape) == tuple(ref.shape)          # shape == SDPA
        a = np.array(o.astype(F32)).reshape(-1); b = np.array(ref).reshape(-1)
        assert float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) >= 0.99


def test_class_c_public_asym_dv_still_runs():
    # public flash_attention keeps asymmetric D_v via SDPA fallback (no over-reject)
    q = mx.random.normal((1, 4, 8, 64)).astype(F16)
    k = mx.random.normal((1, 4, 8, 64)).astype(F16)
    v = mx.random.normal((1, 4, 8, 32)).astype(F16)
    mx.eval(q, k, v)
    o = mlx_mfa.flash_attention(q, k, v, backend="auto")
    mx.eval(o)
    assert tuple(o.shape) == (1, 4, 8, 32)                 # D_v shape, correct


# ── Class D: paged pool dtype equality ───────────────────────────────────────────
def _raw_paged_steel(qdt, pooldt):
    q = mx.zeros((1, 4, 8, 64), qdt)
    pk = mx.zeros((2, 16, 4, 64), pooldt); pv = mx.zeros((2, 16, 4, 64), pooldt)
    tab = mx.array([[0, 1]], mx.int32); lens = mx.array([20], mx.int32)
    mx.eval(q, pk, pv)
    return _ext.mfa_paged_steel_forward(q, pk, pv, tab, lens, 0.125, False, 16)


def test_class_d_paged_mixed_dtype_rejects():
    with pytest.raises(ValueError, match="dtype"):
        _raw_paged_steel(F16, BF16)


def test_class_d_paged_matched_dtype_runs():
    o = _raw_paged_steel(F16, F16)
    mx.eval(o[0] if isinstance(o, (tuple, list)) else o)


# ── Class D: TQ backing-buffer required dtypes ──────────────────────────────────
def _raw_tq(cent_dt=F16, scale_dt=F32, k_dt=U8, v_dt=F16):
    q = mx.zeros((1, 8, 8, 64), F16); pk = mx.zeros((1, 16, 4, 24), k_dt)
    pv = mx.zeros((1, 16, 4, 64), v_dt)
    cu = mx.array([0, 8], mx.int32); tile = mx.array([0, 1], mx.int32)
    tab = mx.array([[0]], mx.int32); lens = mx.array([8], mx.int32)
    cent = mx.zeros((8,), cent_dt); ks = mx.zeros((1, 16, 4), scale_dt)
    return _ext.mfa_paged_varlen_tq_forward(
        q, pk, pv, cu, tile, tab, lens, cent, ks, 0.1, False, 16, 3,
        False, False, None, None, None)


_TQ_BAD = [
    ("centroids_fp32", dict(cent_dt=F32)),
    ("scales_fp16", dict(scale_dt=F16)),
    ("k_pool_fp16", dict(k_dt=F16)),
    ("v_pool_bf16", dict(v_dt=BF16)),
]


@pytest.mark.parametrize("name,kw", _TQ_BAD, ids=[c[0] for c in _TQ_BAD])
def test_class_d_tq_buffer_dtype_rejects(name, kw):
    with pytest.raises(ValueError):
        _raw_tq(**kw)


# ── Class D: flash_attention_topk mutual-compat + dtype equality ─────────────────
def _topk(q, k, v):
    return mlx_mfa.flash_attention_topk(q, k, v, 0.5)


def test_class_d_topk_mixed_dtype_rejects():
    with pytest.raises(ValueError, match="dtype"):
        _topk(mx.zeros((1, 8, 16, 64), F16), mx.zeros((1, 8, 16, 64), BF16),
              mx.zeros((1, 8, 16, 64), F16))


def test_class_d_topk_malformed_heads_rejects():
    with pytest.raises(ValueError):
        _topk(mx.zeros((1, 8, 16, 64), F16), mx.zeros((1, 8, 16, 64), F16),
              mx.zeros((1, 4, 16, 64), F16))


def test_class_d_topk_valid_runs():
    o = _topk(mx.random.normal((1, 8, 16, 64)).astype(F16),
              mx.random.normal((1, 8, 16, 64)).astype(F16),
              mx.random.normal((1, 8, 16, 64)).astype(F16))
    mx.eval(o)
    assert o.dtype == F16


# ── Class E: backend="sdpa" mixed dtype returns the requested output dtype ───────
@pytest.mark.parametrize("qd,kd,vd", [(F16, BF16, F16), (BF16, F16, F16), (F16, F16, BF16)])
def test_class_e_sdpa_mixed_dtype_returns_requested(qd, kd, vd):
    q = mx.random.normal((1, 4, 8, 64)).astype(qd)
    k = mx.random.normal((1, 4, 8, 64)).astype(kd)
    v = mx.random.normal((1, 4, 8, 64)).astype(vd)
    mx.eval(q, k, v)
    o = mlx_mfa.flash_attention(q, k, v, backend="sdpa")
    mx.eval(o)
    assert o.dtype == qd, f"requested {qd}, got {o.dtype}"   # was fp32 pre-fix


def test_class_e_sdpa_same_dtype_unchanged():
    # no over-reject / no perturbation: same-dtype path byteΔ=0 vs direct SDPA
    mx.random.seed(5)
    q = mx.random.normal((1, 4, 8, 64)).astype(F16)
    k = mx.random.normal((1, 4, 8, 64)).astype(F16)
    v = mx.random.normal((1, 4, 8, 64)).astype(F16)
    mx.eval(q, k, v)
    o = mlx_mfa.flash_attention(q, k, v, backend="sdpa")
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / 8)
    mx.eval(o, ref)
    assert np.array(o.astype(F32)).tobytes() == np.array(ref.astype(F32)).tobytes()
    assert o.dtype == F16


# ── Class B: bf16 paged gather + return_lse compile & match fp32 oracle ──────────
@pytest.mark.parametrize("return_lse", [False, True])
def test_class_b_bf16_gather_compiles_and_matches(return_lse):
    H, Nq, S, D, bs = 4, 2, 320, 64, 16
    nblk = (S + bs - 1) // bs
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(11)
    q = mx.random.normal((1, H, Nq, D)).astype(BF16)
    pk = mx.random.normal((nblk, bs, H, D)).astype(BF16)
    pv = mx.random.normal((nblk, bs, H, D)).astype(BF16)
    tab = mx.array([list(range(nblk))], dtype=mx.int32)
    lens = mx.array([S], dtype=mx.int32)
    mx.eval(q, pk, pv)
    r = mlx_mfa.flash_attention_paged(q, pk, pv, tab, lens, scale=scale,
                                      causal=False, block_size=bs, return_lse=return_lse)
    o = r[0] if return_lse else r
    mx.eval(o)
    kf = np.array(pk.astype(F32)).reshape(nblk * bs, H, D)[:S].transpose(1, 0, 2)
    vf = np.array(pv.astype(F32)).reshape(nblk * bs, H, D)[:S].transpose(1, 0, 2)
    qf = np.array(q.astype(F32))[0]
    o2 = np.zeros((H, Nq, D))
    for h in range(H):
        ss = (qf[h] @ kf[h].T) * scale; ss -= ss.max(1, keepdims=True)
        e = np.exp(ss); o2[h] = (e / e.sum(1, keepdims=True)) @ vf[h]
    a = np.array(o.astype(F32)).reshape(-1); b = o2.reshape(-1)
    assert o.dtype == BF16
    assert float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) >= 0.99

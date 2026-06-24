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
    O = o[0] if isinstance(o, (tuple, list)) else o
    mx.eval(O)
    assert bool(np.isfinite(np.array(O.astype(F32))).all())   # M4: full oracle, not just "runs"


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


# ── Optional-input dimension: V-TQ buffers validated ONLY-when-present ────────────
def _raw_tq_v(vtq_dt=U8, vc_dt=F16, vs_dt=F32, bits=3):
    nb, bs, Hkv, D = 1, 16, 4, 64
    pdk = (D // 32) * 12
    q = mx.zeros((1, 8, 8, D), F16); pk = mx.zeros((nb, bs, Hkv, pdk), U8)
    pv = mx.zeros((nb, bs, Hkv, D), F16)
    cu = mx.array([0, 8], mx.int32); tile = mx.array([0, 1], mx.int32)
    tab = mx.array([[0]], mx.int32); lens = mx.array([8], mx.int32)
    cent = mx.zeros((2 ** bits,), F16); ks = mx.zeros((nb, bs, Hkv), F32)
    vtq = mx.zeros((nb, bs, Hkv, pdk), vtq_dt)
    vc = mx.zeros((2 ** bits,), vc_dt); vs = mx.zeros((nb, bs, Hkv), vs_dt)
    return _ext.mfa_paged_varlen_tq_forward(
        q, pk, pv, cu, tile, tab, lens, cent, ks, 0.1, False, bs, bits,
        True, False, vtq, vc, vs)


_VTQ_BAD = [
    ("v_pool_tq_fp16", dict(vtq_dt=F16)),
    ("v_centroids_fp32", dict(vc_dt=F32)),
    ("v_scales_bf16", dict(vs_dt=BF16)),
]


@pytest.mark.parametrize("name,kw", _VTQ_BAD, ids=[c[0] for c in _VTQ_BAD])
def test_class_d_vtq_optional_buffer_dtype_rejects(name, kw):
    # the optional/conditional-input hole: V-TQ buffers carry the same dtype
    # contract as the K-side but were unvalidated when tq_v_enabled=True.
    with pytest.raises(ValueError):
        _raw_tq_v(**kw)


def test_class_d_vtq_all_correct_runs():
    # no over-rejection: all-correct V-TQ (tq_v_enabled) still runs
    r = _raw_tq_v()
    mx.eval(r[0] if isinstance(r, (tuple, list)) else r)


def test_class_d_vtq_absent_runs():
    # conditional-presence: tq_v_enabled=False with absent V buffers must NOT choke
    nb, bs, Hkv, D, bits = 1, 16, 4, 64, 3
    pdk = (D // 32) * 12
    q = mx.zeros((1, 8, 8, D), F16); pk = mx.zeros((nb, bs, Hkv, pdk), U8)
    pv = mx.zeros((nb, bs, Hkv, D), F16)
    cu = mx.array([0, 8], mx.int32); tile = mx.array([0, 1], mx.int32)
    tab = mx.array([[0]], mx.int32); lens = mx.array([8], mx.int32)
    cent = mx.zeros((2 ** bits,), F16); ks = mx.zeros((nb, bs, Hkv), F32)
    r = _ext.mfa_paged_varlen_tq_forward(
        q, pk, pv, cu, tile, tab, lens, cent, ks, 0.1, False, bs, bits,
        False, False, None, None, None)
    mx.eval(r[0] if isinstance(r, (tuple, list)) else r)


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


# ── Feature-tensor dtype-misread class (CC feature-tensor batch) ─────────────────
# Several kernels read a host-passed SECONDARY tensor at a hardcoded dtype; a
# non-contract dtype was byte-misread = silent-wrong (alibi slopes f16 cos 0.87,
# rope cos/sin f16 cos 0.72, sparse mask int32/f32 cos=nan). Remedy = upcast to
# the kernel's contract dtype in the host. Lock: contract dtype → byteΔ=0; other
# dtypes → cos ≥ floor vs the contract reference (was a misread).
def _cos1(o, ref):
    a = np.array(o.astype(F32)).reshape(-1); b = np.array(ref.astype(F32)).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _alibi(slopes_dt):
    mx.random.seed(0)
    q = mx.random.normal((1, 4, 16, 64)).astype(F16)
    k = mx.random.normal((1, 4, 16, 64)).astype(F16)
    v = mx.random.normal((1, 4, 16, 64)).astype(F16)
    mx.eval(q, k, v)
    sv = mx.array([0.1, 0.2, 0.4, 0.8], dtype=F32)
    o = _ext.mfa_attention_alibi_forward(q, k, v, sv.astype(slopes_dt), 0.125, True)
    mx.eval(o)
    return o


def _rope(table_dt):
    mx.random.seed(0)
    q = mx.random.normal((1, 4, 16, 64)).astype(F16)
    k = mx.random.normal((1, 4, 16, 64)).astype(F16)
    v = mx.random.normal((1, 4, 16, 64)).astype(F16)
    cos = mx.cos(mx.arange(16 * 32).reshape(16, 32).astype(F32) * 0.05)
    sin = mx.sin(mx.arange(16 * 32).reshape(16, 32).astype(F32) * 0.05)
    mx.eval(q, k, v, cos, sin)
    o = _ext.mfa_attention_rope_forward(q, k, v, cos.astype(table_dt),
                                        sin.astype(table_dt), 0.125, False, 0)
    mx.eval(o)
    return o


def _sparse_mask(mask_dt):
    mx.random.seed(0)
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    nt = (N + 31) // 32
    mnp = np.ones((Hq, nt, nt), np.uint8); mnp[:, :, nt - 1] = 0
    o = _ext.mfa_attention_sparse_forward(q, k, v, mx.array(mnp).astype(mask_dt),
                                          1.0 / math.sqrt(D), False)
    mx.eval(o)
    return o


# (runner, contract_dtype, [other dtypes that were misread pre-fix])
_FEATURE_CASES = [
    ("alibi_slopes", _alibi, F32, [F16, BF16]),
    ("rope_cos_sin", _rope, F32, [F16, BF16]),
    ("sparse_block_mask", _sparse_mask, U8, [mx.int32, F32, mx.bool_]),
]


@pytest.mark.parametrize("name,run,contract,others",
                         _FEATURE_CASES, ids=[c[0] for c in _FEATURE_CASES])
def test_feature_tensor_dtype_corrected(name, run, contract, others):
    ref = run(contract)
    assert np.isfinite(np.array(ref.astype(F32))).all()
    # contract dtype is identity (byteΔ=0)
    again = run(contract)
    assert np.array(again.astype(F32)).tobytes() == np.array(ref.astype(F32)).tobytes()
    # non-contract dtypes now CORRECT (host upcast) — was a misread (cos<<1)
    for dt in others:
        o = run(dt)
        c = _cos1(o, ref)
        assert c >= 0.999, f"{name} {dt}: cos {c:.4f} < 0.999 (misread not fixed)"


def test_feature_tensor_bite_documents_contract():
    # bite: the pre-fix MISREAD magnitude — if a host upcast is reverted, the
    # non-contract dtype cosine collapses well below this. (Documents that f16
    # slopes/cos-sin and wide masks were genuinely silent-wrong, cos ~0.7–0.87,
    # and are now ~1.0; a regression would reproduce the collapse.)
    assert _cos1(_alibi(F16), _alibi(F32)) >= 0.999
    assert _cos1(_rope(F16), _rope(F32)) >= 0.999
    assert _cos1(_sparse_mask(mx.int32), _sparse_mask(U8)) >= 0.999


# ── Feature-metadata SHAPE/rank/extent/cardinality class (CC shape batch) ────────
# Companion to the dtype axis: entries validated dtype but not shape — malformed
# shape metadata → finite-wrong / non-finite. Validate-and-raise (a malformed
# shape can't be cast valid). Dual oracle: malformed → raises; valid → runs.
_SC = 1.0 / math.sqrt(64)


def _sparse(mask):
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    return _ext.mfa_attention_sparse_forward(q, k, v, mask, _SC, False)


_NT = (128 + 31) // 32  # NQ=NK for N=S=128, BQ=BK=32 (D=64, M5)
_SPARSE_BAD = [
    ("mask_1x1", lambda: mx.ones((1, 1), U8)),
    ("mask_H_1_1", lambda: mx.ones((4, 1, 1), U8)),
    ("mask_Hplus1", lambda: mx.ones((5, _NT, _NT), U8)),
    ("mask_wrong_NK", lambda: mx.ones((4, _NT, _NT - 1), U8)),
]
_SPARSE_OK = [
    ("mask_2d", lambda: mx.ones((_NT, _NT), U8)),
    ("mask_3d_H", lambda: mx.ones((4, _NT, _NT), U8)),
    ("mask_4d_BH", lambda: mx.ones((1, 4, _NT, _NT), U8)),
]


@pytest.mark.parametrize("name,mk", _SPARSE_BAD, ids=[c[0] for c in _SPARSE_BAD])
def test_sparse_mask_shape_rejects(name, mk):
    with pytest.raises(ValueError):
        _sparse(mk())


@pytest.mark.parametrize("name,mk", _SPARSE_OK, ids=[c[0] for c in _SPARSE_OK])
def test_sparse_mask_shape_accepts(name, mk):
    o = _sparse(mk()); mx.eval(o)
    assert np.isfinite(np.array(o.astype(F32))).all()


def _rope_shape(cos, sin, cache=0):
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    return _ext.mfa_attention_rope_forward(q, k, v, cos, sin, _SC, False, cache)


def _full_table():
    c = mx.cos(mx.arange(128 * 32).reshape(128, 32).astype(F32) * 0.01)
    s = mx.sin(mx.arange(128 * 32).reshape(128, 32).astype(F32) * 0.01)
    mx.eval(c, s)
    return c, s


def test_rope_rank1_rejects():
    with pytest.raises(ValueError):
        _rope_shape(mx.ones((32,), F32), mx.ones((32,), F32))


def test_rope_short_table_rejects():
    c, s = _full_table()
    with pytest.raises(ValueError):
        _rope_shape(c[:127], s[:127])          # one row short of N=128


def test_rope_full_table_accepts():
    c, s = _full_table()
    o = _rope_shape(c, s); mx.eval(o)
    assert np.isfinite(np.array(o.astype(F32))).all()


def _varlen(cu_q, cu_k, tile):
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    return _ext.mfa_attention_varlen_forward(q, k, v, cu_q, cu_k, tile, _SC, False)


_I = mx.int32
_VALID_CU = (mx.array([0, 64, 128], _I), mx.array([0, 64, 128], _I), mx.array([0, 2, 4], _I))
_VARLEN_BAD = [
    ("cu_q_rank2", lambda: _varlen(mx.array([[0], [64], [128]], _I), _VALID_CU[1], _VALID_CU[2])),
    ("cu_k_cardinality", lambda: _varlen(_VALID_CU[0], mx.array([0, 128], _I), _VALID_CU[2])),
    ("tile_cardinality", lambda: _varlen(_VALID_CU[0], _VALID_CU[1], mx.array([0, 4], _I))),
]


@pytest.mark.parametrize("name,fn", _VARLEN_BAD, ids=[c[0] for c in _VARLEN_BAD])
def test_varlen_metadata_shape_rejects(name, fn):
    with pytest.raises(ValueError):
        fn()


def test_varlen_metadata_valid_runs():
    o = _varlen(*_VALID_CU)
    mx.eval(o[0] if isinstance(o, (tuple, list)) else o)


def test_shape_bite_sparse_undersized_must_raise():
    # bite: if the sparse shape validator is reverted, (1,1) stops raising.
    with pytest.raises(ValueError):
        _sparse(mx.ones((1, 1), U8))


# ── Sibling-bypass generator + metadata-VALUE class (CC optional-input batch) ────
# A variant (LSE) bypassed the shape validator its base sibling has; and varlen
# metadata VALUES (cu_seqlens/tile_offsets) drove finite-wrong without value checks.
import os
import pathlib

_SCv = 1.0 / math.sqrt(64)
_NTv = (128 + 31) // 32


def _sparse_lse(mask):
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    return _ext.mfa_attention_sparse_forward_with_lse(q, k, v, mask, _SCv, False)


@pytest.mark.parametrize("mk", [
    lambda: mx.ones((1, 1), U8),
    lambda: mx.ones((5, _NTv, _NTv), U8),       # H+1 heads
    lambda: mx.ones((4, 1, 1), U8),
], ids=["lse_1x1", "lse_Hplus1", "lse_H_1_1"])
def test_sparse_lse_shape_rejects(mk):
    with pytest.raises(ValueError):
        _sparse_lse(mk())


@pytest.mark.parametrize("mk", [
    lambda: mx.ones((_NTv, _NTv), U8),
    lambda: mx.ones((4, _NTv, _NTv), U8),
    lambda: mx.ones((1, 4, _NTv, _NTv), U8),
], ids=["lse_2d", "lse_3d", "lse_4d"])
def test_sparse_lse_shape_accepts(mk):
    o = _sparse_lse(mk())
    mx.eval(o[0] if isinstance(o, (tuple, list)) else o)


def test_sibling_coverage_sparse_shape_validator():
    # sibling-bypass bite: BOTH sparse entries (base + LSE) must route through the
    # shape validator. If a future variant skips it, this count drops and fails.
    src = (pathlib.Path(__file__).parent.parent / "csrc" / "mfa_attention.cpp").read_text()
    n = src.count("validate_sparse_block_mask_shape(block_mask")
    assert n >= 2, f"both sparse hosts (base+LSE) must call the shape validator; found {n}"


# ── varlen metadata-VALUE cells ─────────────────────────────────────────────────
_I32 = mx.int32


def _varlen_v(cu_q, cu_k, tile):
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    return _ext.mfa_attention_varlen_forward(q, k, v, cu_q, cu_k, tile, _SCv, False)


_GOOD = (mx.array([0, 64, 128], _I32), mx.array([0, 64, 128], _I32), mx.array([0, 2, 4], _I32))
_VALUE_BAD = [
    ("cu_q_nonmonotonic", lambda: _varlen_v(mx.array([0, 128, 64], _I32), _GOOD[1], _GOOD[2])),
    ("cu_q_negative", lambda: _varlen_v(mx.array([0, -64, 128], _I32), _GOOD[1], _GOOD[2])),
    ("cu_q_sum_mismatch", lambda: _varlen_v(mx.array([0, 64, 200], _I32), _GOOD[1], _GOOD[2])),
    ("tile_inconsistent", lambda: _varlen_v(_GOOD[0], _GOOD[1], mx.array([0, 9, 4], _I32))),
]


@pytest.mark.parametrize("name,fn", _VALUE_BAD, ids=[c[0] for c in _VALUE_BAD])
def test_varlen_metadata_value_rejects(name, fn):
    with pytest.raises(ValueError):
        fn()


def test_varlen_metadata_value_valid_runs():
    o = _varlen_v(*_GOOD)
    mx.eval(o[0] if isinstance(o, (tuple, list)) else o)


def test_varlen_metadata_value_optout_skips():
    # opt-out MFA_VARLEN_TRUST_METADATA=1 skips the value sync (bad values run)
    prev = os.environ.get("MFA_VARLEN_TRUST_METADATA")
    os.environ["MFA_VARLEN_TRUST_METADATA"] = "1"
    try:
        o = _varlen_v(mx.array([0, 128, 64], _I32), _GOOD[1], _GOOD[2])
        mx.eval(o[0] if isinstance(o, (tuple, list)) else o)   # no raise under opt-out
    finally:
        if prev is None:
            os.environ.pop("MFA_VARLEN_TRUST_METADATA", None)
        else:
            os.environ["MFA_VARLEN_TRUST_METADATA"] = prev


# ── Derived-metadata: tile_offsets validated against its DERIVATION (CC batch) ────
# tile_offsets = cumulative ceil(q_len/BQ) from cu_seqlens_q (BQ=32 STEEL). The
# generic prefix-sum check accepted monotone-but-wrong values (finite-wrong cos
# 0.64-0.88); now validated against the canonical recomputation (TOTAL contract:
# only the one correct value passes). Across non-paged / paged / TQ.
def _varlen_tile(tile):
    mx.random.seed(0)   # fixed inputs so default-vs-opt-out byteΔ is meaningful
    N, Hq, D = 128, 4, 64
    q = mx.random.normal((1, Hq, N, D)).astype(F16)
    k = mx.random.normal((1, Hq, N, D)).astype(F16)
    v = mx.random.normal((1, Hq, N, D)).astype(F16)
    mx.eval(q, k, v)
    cu = mx.array([0, 64, 128], mx.int32)
    return _ext.mfa_attention_varlen_forward(q, k, v, cu, cu, tile, _SCv, False)


_CANON_NP = mx.array([0, 2, 4], mx.int32)   # ceil(64/32)=2 each
_TILE_BAD = [
    ("wrong_final", lambda: _varlen_tile(mx.array([0, 2, 3], mx.int32))),
    ("wrong_segment", lambda: _varlen_tile(mx.array([0, 1, 4], mx.int32))),
    ("non_monotone", lambda: _varlen_tile(mx.array([0, 4, 2], mx.int32))),
    ("off_by_one", lambda: _varlen_tile(mx.array([0, 2, 5], mx.int32))),
]


@pytest.mark.parametrize("name,fn", _TILE_BAD, ids=[c[0] for c in _TILE_BAD])
def test_tile_offsets_derivation_rejects(name, fn):
    with pytest.raises(ValueError):
        fn()


def test_tile_offsets_canonical_runs_and_bytedelta0():
    # formula-perturbation bite: the canonical value (recompute) MUST pass; if the
    # C++ formula were perturbed/over-strict, this valid case would start raising.
    o = _varlen_tile(_CANON_NP)
    mx.eval(o[0] if isinstance(o, (tuple, list)) else o)
    # byteΔ=0 vs opt-out (validation is pre-dispatch, valid path unperturbed)
    prev = os.environ.get("MFA_VARLEN_TRUST_METADATA")
    os.environ["MFA_VARLEN_TRUST_METADATA"] = "1"
    try:
        o2 = _varlen_tile(_CANON_NP)
        mx.eval(o2[0])
    finally:
        if prev is None:
            os.environ.pop("MFA_VARLEN_TRUST_METADATA", None)
        else:
            os.environ["MFA_VARLEN_TRUST_METADATA"] = prev
    assert (np.array(o[0].astype(F32)).tobytes()
            == np.array(o2[0].astype(F32)).tobytes())


def test_tile_offsets_derivation_optout_skips():
    prev = os.environ.get("MFA_VARLEN_TRUST_METADATA")
    os.environ["MFA_VARLEN_TRUST_METADATA"] = "1"
    try:
        o = _varlen_tile(mx.array([0, 2, 3], mx.int32))   # wrong but trusted
        mx.eval(o[0])
    finally:
        if prev is None:
            os.environ.pop("MFA_VARLEN_TRUST_METADATA", None)
        else:
            os.environ["MFA_VARLEN_TRUST_METADATA"] = prev


def _paged_varlen_tile(tile):
    Hq, Hkv, D, bs = 8, 4, 64, 16
    ql, kl = [3, 4], [27, 33]
    qs = [mx.random.normal((1, Hq, x, D)).astype(F16) for x in ql]
    ks = [mx.random.normal((1, Hkv, x, D)).astype(F16) for x in kl]
    vs = [mx.random.normal((1, Hkv, x, D)).astype(F16) for x in kl]
    mx.eval(*qs, *ks, *vs)
    qp = mx.concatenate(qs, axis=2)
    cu = mx.array([0, 3, 7], mx.int32)
    bps = [(x + bs - 1) // bs for x in kl]
    tot, mb = sum(bps), max(bps)
    pk = np.zeros((tot, bs, Hkv, D), np.float32); pv = np.zeros((tot, bs, Hkv, D), np.float32)
    tab = np.full((2, mb), -1, np.int32); lens = np.zeros((2,), np.int32); base = 0
    for b in range(2):
        kn = np.array(ks[b].astype(F32))[0].transpose(1, 0, 2)
        vn = np.array(vs[b].astype(F32))[0].transpose(1, 0, 2)
        S = kn.shape[0]; lens[b] = S
        for lb in range(bps[b]):
            tab[b, lb] = base + lb; s0, s1 = lb * bs, min(S, lb * bs + bs)
            pk[base + lb, :s1 - s0] = kn[s0:s1]; pv[base + lb, :s1 - s0] = vn[s0:s1]
        base += bps[b]
    return _ext.mfa_paged_varlen_forward(
        qp, mx.array(pk).astype(F16), mx.array(pv).astype(F16), cu, tile,
        mx.array(tab, mx.int32), mx.array(lens, mx.int32), _SCv, False, bs)


def test_paged_varlen_tile_derivation():
    _paged_varlen_tile(mx.array([0, 1, 2], mx.int32))     # canonical → runs
    for bad in ([0, 1, 3], [0, 2, 2]):                    # wrong-final / wrong-segment
        with pytest.raises(ValueError):
            _paged_varlen_tile(mx.array(bad, mx.int32))


# ── Class-method surface: A (feature-thread), B (validate-before-mutate), C (TQ geom)
import mlx_mfa as _mm
from mlx_mfa.inference import (InferenceContext, PagedInferenceContext,
                               SageInferenceContext,
                               TurboQuantPagedInferenceContext)
from mlx_mfa.runtime import create_decode_runtime


def _qf(h, n, d):
    a = mx.random.normal((1, h, n, d)).astype(F16)
    mx.eval(a)
    return a


# Class A: prefix feature params now change the output (threaded) or reject ───────
def test_classA_prefix_threads_softcap_and_causal():
    mx.random.seed(0)
    q, k, v = _qf(8, 16, 64), _qf(8, 16, 64), _qf(8, 16, 64)
    d = np.array(_mm.make_shared_prefix_cache(q, k, v)[0].astype(F32))
    sc = np.array(_mm.make_shared_prefix_cache(q, k, v, softcap=20.0)[0].astype(F32))
    nc = np.array(_mm.make_shared_prefix_cache(q, k, v, causal=False)[0].astype(F32))
    assert np.max(np.abs(sc - d)) > 1e-3      # softcap takes effect (was dropped)
    assert np.max(np.abs(nc - d)) > 1e-3      # causal takes effect


def test_classA_register_prefix_threads_softcap():
    mx.random.seed(0)
    q, k, v = _qf(8, 16, 64), _qf(8, 16, 64), _qf(8, 16, 64)
    p1 = np.array(create_decode_runtime(backend="dense", B=1, H_q=8, H_kv=8, D=64,
                 max_seq_len=128, dtype=F16).register_prefix(0, q, k, v)[0].astype(F32))
    p2 = np.array(create_decode_runtime(backend="dense", B=1, H_q=8, H_kv=8, D=64,
                 max_seq_len=128, dtype=F16).register_prefix(0, q, k, v, softcap=20.0)[0].astype(F32))
    assert np.max(np.abs(p2 - p1)) > 1e-3       # softcap takes effect (not dropped)
    # M4: ref-match — the softcap output must MATCH the direct softcap compute
    # (correct, not merely different); cosine vs make_shared_prefix_cache(softcap).
    ref = np.array(mlx_mfa.make_shared_prefix_cache(q, k, v, softcap=20.0)[0].astype(F32))
    assert np.isfinite(p2).all()
    a, b = p2.reshape(-1), ref.reshape(-1)
    assert float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) >= 0.999


# Class B: failed (malformed-Q) call is atomic — cache byteΔ=0 ────────────────────
def test_classB_step_malformed_q_atomic():
    c = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=128, dtype=F16)
    c.prefill(_qf(8, 4, 64), _qf(4, 4, 64), _qf(4, 4, 64))
    n0 = c.seqlen
    kb = np.array(c.k_cache.astype(F32)).tobytes()
    with pytest.raises(ValueError):
        c.step(_qf(3, 1, 64), _qf(4, 1, 64), _qf(4, 1, 64))   # bad GQA Q
    assert c.seqlen == n0                                      # not mutated
    assert np.array(c.k_cache.astype(F32)).tobytes() == kb     # byteΔ=0
    o = c.step(_qf(8, 1, 64), _qf(4, 1, 64), _qf(4, 1, 64))    # valid still mutates
    mx.eval(o)
    assert c.seqlen == n0 + 1


def test_classB_prefill_malformed_q_atomic():
    c = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=128, dtype=F16)
    c.prefill(_qf(8, 4, 64), _qf(4, 4, 64), _qf(4, 4, 64))    # seqlen 4
    n0 = c.seqlen
    with pytest.raises(ValueError):
        c.prefill(_qf(3, 4, 64), _qf(4, 4, 64), _qf(4, 4, 64))  # bad GQA → must not wipe to a bad state
    assert c.seqlen == n0                                       # reset+append not reached


# Class C: TQ Q-vs-pool geometry raises at class + raw ────────────────────────────
def _tq_class(Hq, Dq):
    c = TurboQuantPagedInferenceContext(num_blocks=16, block_size=16, H_kv=4, D=64,
                                        dtype=F16, tq_bits=3)
    c.prefill(_qf(Hq, 8, Dq), _qf(4, 8, Dq), _qf(4, 8, Dq), seq_id=0)


def test_classC_tq_class_geometry():
    _tq_class(8, 64)                                  # valid → runs
    with pytest.raises(ValueError):
        _tq_class(3, 64)                              # bad GQA
    with pytest.raises(ValueError):
        _tq_class(8, 128)                            # q.D != pool D


def _tq_raw(Hq, Dq):
    nb, bs, hkv, D, bits = 1, 16, 4, 64, 3
    pdk = (D // 32) * 12
    q = mx.zeros((1, Hq, 8, Dq), F16); pk = mx.zeros((nb, bs, hkv, pdk), U8)
    pv = mx.zeros((nb, bs, hkv, D), F16)
    cu = mx.array([0, 8], mx.int32); tile = mx.array([0, 1], mx.int32)
    tab = mx.array([[0]], mx.int32); lens = mx.array([8], mx.int32)
    cent = mx.zeros((2 ** bits,), F16); ks = mx.zeros((nb, bs, hkv), F32)
    return _ext.mfa_paged_varlen_tq_forward(q, pk, pv, cu, tile, tab, lens, cent, ks,
                                            0.1, False, bs, bits, False, False, None, None, None)


def test_classC_tq_raw_geometry():
    for bad in [(3, 64), (8, 128)]:
        with pytest.raises(ValueError):
            _tq_raw(*bad)

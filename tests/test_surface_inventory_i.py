"""Volet I — TQ-paged buffer-shape + raw-paged metadata-dtype locks (CX-R6-01/03).

Round-6 found two siblings the prior surface-by-surface audits missed:
  CX-R6-01 (CRITICAL): TQ-paged entries don't shape-lock v_pages / k_scales /
    packed-K width to k_pool_tq → OOB (undersized v_pages → finite-wrong; smaller
    head_dim → NaN; undersized k_scales → OOB; incompatible packed_D → garbage).
  CX-R6-03 (HIGH): raw mfa_paged_steel_forward / mfa_paged_varlen_forward accept
    int64/float block_table/seq_lens → silent int32 cast; float seq_lens HANGS.

Both directions (accept-valid + reject-malformed) per buffer/dtype, first-hand.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as e

# ── CX-R6-03: raw paged metadata dtype ────────────────────────────────────────
_H, _D, _BS, _NB = 4, 64, 16, 8


def _pool():
    mx.random.seed(0)
    a = mx.random.normal((_NB, _BS, _H, _D)).astype(mx.float16)
    mx.eval(a)
    return a


def test_raw_steel_metadata_int32_accept_and_reject():
    kp, vp = _pool(), _pool()
    q = mx.random.normal((1, _H, 1, _D)).astype(mx.float16); mx.eval(q)
    bt32 = mx.array([[2, 5, 1, 0]], dtype=mx.int32); sl32 = mx.array([48], dtype=mx.int32)
    # accept-valid
    o, _l = e.mfa_paged_steel_forward(q, kp, vp, bt32, sl32, 1 / 8, False, -1, -1, _BS)
    mx.eval(o)
    # reject int64 block_table / float seq_lens
    with pytest.raises(ValueError):
        mx.eval(e.mfa_paged_steel_forward(q, kp, vp, mx.array([[2, 5, 1, 0]], dtype=mx.int64), sl32, 1 / 8, False, -1, -1, _BS)[0])
    with pytest.raises(ValueError):
        mx.eval(e.mfa_paged_steel_forward(q, kp, vp, bt32, mx.array([48.0], dtype=mx.float32), 1 / 8, False, -1, -1, _BS)[0])


def test_raw_varlen_metadata_int32_accept_and_reject():
    kp, vp = _pool(), _pool()
    q = mx.random.normal((1, _H, 2, _D)).astype(mx.float16); mx.eval(q)
    cu = mx.array([0, 1, 2], dtype=mx.int32); to = mx.array([0, 1, 2], dtype=mx.int32)
    bt = mx.array([[2, 5], [3, 4]], dtype=mx.int32); sl = mx.array([20, 20], dtype=mx.int32)
    o, _l = e.mfa_paged_varlen_forward(q, kp, vp, cu, to, bt, sl, 1 / 8, False, _BS)
    mx.eval(o)
    with pytest.raises(ValueError):  # int64 block_table
        mx.eval(e.mfa_paged_varlen_forward(q, kp, vp, cu, to, mx.array([[2, 5], [3, 4]], dtype=mx.int64), sl, 1 / 8, False, _BS)[0])
    with pytest.raises(ValueError):  # float seq_lens_kv (was HANG)
        mx.eval(e.mfa_paged_varlen_forward(q, kp, vp, cu, to, bt, mx.array([20.0, 20.0], dtype=mx.float32), 1 / 8, False, _BS)[0])


# ── CX-R6-01: TQ paged backing-buffer shapes ──────────────────────────────────
def _tqctx():
    import sys
    sys.path.insert(0, "tests")
    import test_phase3_iii2_tq_decode as T
    from mlx_mfa.turboquant import apply_rotation
    ctx, q = T._mkctx(3)
    qr = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16); mx.eval(qr)
    return T, ctx, qr


def _tqcall(T, ctx, qr, **kw):
    cu = mx.array([0, 1], dtype=mx.int32)
    a = dict(scale=T.SCALE, causal=True, block_size=T.BS, tq_bits=3,
             tq_v_enabled=False, tq_wht_enabled=False)
    a.update(kw)
    return mlx_mfa.flash_attention_paged_varlen_turboquant(
        qr, ctx._k_pool, a.pop("v_pages", ctx._v_pool_fp16),
        ctx.get_block_table([0]), ctx.get_seq_lens([0]), cu,
        ctx._k_centroids, a.pop("k_scales", ctx._k_scales), **a)


def test_tq_paged_valid_runs():
    T, ctx, qr = _tqctx()
    o = _tqcall(T, ctx, qr); mx.eval(o)
    assert bool(np.isfinite(np.array(o.astype(mx.float32))).all())


@pytest.mark.parametrize("mutate", ["v_blocks", "v_heads", "v_headdim", "k_scales"])
def test_tq_paged_buffer_mismatch_raises(mutate):
    T, ctx, qr = _tqctx()
    vp = ctx._v_pool_fp16; nb = vp.shape[0]; ks = ctx._k_scales
    kw = {}
    if mutate == "v_blocks":
        kw["v_pages"] = vp[:nb - 2]
    elif mutate == "v_heads":
        kw["v_pages"] = mx.random.normal((nb, vp.shape[1], 1, vp.shape[3])).astype(mx.float16)
    elif mutate == "v_headdim":
        kw["v_pages"] = mx.random.normal((nb, vp.shape[1], vp.shape[2], vp.shape[3] // 2)).astype(mx.float16)
    elif mutate == "k_scales":
        kw["k_scales"] = ks[:nb - 2]
    with pytest.raises(ValueError):
        mx.eval(_tqcall(T, ctx, qr, **kw))


# ── determinism axis (identical inputs, 8 runs, byteΔ must be 0) ───────────────
@pytest.mark.parametrize("name,fn", [
    ("paged_hetero_causal", lambda: _det_paged()),
])
def test_determinism(name, fn):
    outs = [np.array(fn().astype(mx.float32)) for _ in range(8)]
    md = max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, 8))
    assert md == 0.0, f"{name}: nondeterministic, max pairwise byteΔ={md:.2e}"


def _det_paged():
    mx.random.seed(1)
    nb = 8
    kp = mx.random.normal((nb, 16, 4, 128)).astype(mx.float16)
    vp = mx.random.normal((nb, 16, 4, 128)).astype(mx.float16)
    q = mx.random.normal((2, 4, 17, 128)).astype(mx.float16)
    bt = mx.array([[0, 1, -1, -1], [2, 3, 4, 5]], dtype=mx.int32)
    sl = mx.array([31, 50], dtype=mx.int32)
    mx.eval(kp, vp, q, bt, sl)
    o = mlx_mfa.flash_attention_paged(q, kp, vp, bt, sl, scale=1 / math.sqrt(128),
                                      causal=True, block_size=16)
    mx.eval(o)
    return o

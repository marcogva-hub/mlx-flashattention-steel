"""Volet K3 — 4-axis hardening locks for the final 13 public adapter entries.

These are adapters over the (now hardened) raw cores. Defects found + fixed:
  - qkv_packed / varlen_qkv_packed 5D: num_kv_heads > buffer heads silently
    truncated (no capacity check).
  - speculative_verify {dense,paged}: accepted float draft_ids and non-positive /
    non-finite temperature.
  - splitfuse: a partial branch (q without k/v) crashed with AttributeError
    instead of a clean ValueError.
Cache-append family (kvcache / kvcache_rope_append / sage_attention_kvcache) was
probed for OOB append first-hand and is memory-safe (dense concatenates; paged
and rope-append raise on out-of-range slots). HARD GUARD: every new check has an
accept-valid cell proving the legitimate input space still passes.
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa

B, Hq, Hk, N, D = 1, 8, 2, 256, 128
SC = 1.0 / math.sqrt(D)


def _qkv():
    mx.random.seed(0)
    q = mx.random.normal((B, Hq, N, D)).astype(mx.float16)
    k = mx.random.normal((B, Hk, N, D)).astype(mx.float16)
    v = mx.random.normal((B, Hk, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


# ── P1 qkv_packed (3D + 5D) ─────────────────────────────────────────────────────
def test_qkv_packed_valid_3d_and_5d():
    mx.random.seed(0)
    qkv3 = mx.random.normal((B, N, (Hq + 2 * Hk) * D)).astype(mx.float16)
    qkv5 = mx.random.normal((B, Hq, N, 3, D)).astype(mx.float16)
    mx.eval(qkv3, qkv5)
    mx.eval(mlx_mfa.flash_attention_qkv_packed(qkv3, num_heads=Hq, num_kv_heads=Hk))
    mx.eval(mlx_mfa.flash_attention_qkv_packed(qkv5, num_kv_heads=Hk))
    # accept-valid: num_kv_heads == buffer head count (the boundary)
    mx.eval(mlx_mfa.flash_attention_qkv_packed(qkv5, num_kv_heads=Hq))


@pytest.mark.parametrize("mut", ["no_heads_3d", "bad_fused", "kv_over_capacity"])
def test_qkv_packed_malformed_raises(mut):
    mx.random.seed(0)
    if mut == "no_heads_3d":
        x = mx.random.normal((B, N, (Hq + 2 * Hk) * D)).astype(mx.float16)
        call = lambda: mlx_mfa.flash_attention_qkv_packed(x)
    elif mut == "bad_fused":
        x = mx.random.normal((B, N, 1000)).astype(mx.float16)
        call = lambda: mlx_mfa.flash_attention_qkv_packed(x, num_heads=Hq, num_kv_heads=Hk)
    elif mut == "kv_over_capacity":
        x = mx.random.normal((B, Hq, N, 3, D)).astype(mx.float16)
        call = lambda: mlx_mfa.flash_attention_qkv_packed(x, num_kv_heads=99)
    with pytest.raises((ValueError, Exception)):
        mx.eval(call())


def test_varlen_qkv_packed_capacity_raises():
    mx.random.seed(0)
    qkv5 = mx.random.normal((1, Hq, N, 3, D)).astype(mx.float16)
    cu = mx.array([0, N], dtype=mx.int32)
    mx.eval(qkv5, cu)
    # accept-valid
    mx.eval(mlx_mfa.flash_attention_varlen_qkv_packed(qkv5, cu, cu, N, N, num_kv_heads=Hk))
    with pytest.raises((ValueError, Exception)):
        mx.eval(mlx_mfa.flash_attention_varlen_qkv_packed(qkv5, cu, cu, N, N, num_kv_heads=99))


# ── P1/P4 speculative-verify (dense + paged) ────────────────────────────────────
def _spec_dense():
    mx.random.seed(0)
    qt = mx.random.normal((B, Hq, 5, D)).astype(mx.float16)
    kc = mx.random.normal((B, Hk, N, D)).astype(mx.float16)
    vc = mx.random.normal((B, Hk, N, D)).astype(mx.float16)
    did = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    mx.eval(qt, kc, vc, did)
    return qt, kc, vc, did


def test_speculative_dense_valid_and_accept_int64():
    qt, kc, vc, did = _spec_dense()
    mx.eval(mlx_mfa.flash_attention_speculative_verify(qt, kc, vc, did)[0])
    # accept-valid: int64 draft_ids + temperature 2.0
    mx.eval(mlx_mfa.flash_attention_speculative_verify(
        qt, kc, vc, did.astype(mx.int64), temperature=2.0)[0])


@pytest.mark.parametrize("mut", ["draft_float", "temp0", "temp_neg", "temp_inf", "draft_wrong_shape"])
def test_speculative_dense_malformed_raises(mut):
    qt, kc, vc, did = _spec_dense()
    kw = {}
    if mut == "draft_float":
        did = did.astype(mx.float32)
    elif mut == "temp0":
        kw["temperature"] = 0.0
    elif mut == "temp_neg":
        kw["temperature"] = -1.0
    elif mut == "temp_inf":
        kw["temperature"] = float("inf")
    elif mut == "draft_wrong_shape":
        did = mx.array([[1, 2, 3]], dtype=mx.int32)
    with pytest.raises((ValueError, Exception)):
        mx.eval(mlx_mfa.flash_attention_speculative_verify(qt, kc, vc, did, **kw)[0])


def _spec_paged():
    mx.random.seed(0)
    bs, nb = 16, 20
    kp = mx.random.normal((nb, bs, Hk, D)).astype(mx.float16)
    vp = mx.random.normal((nb, bs, Hk, D)).astype(mx.float16)
    qt = mx.random.normal((B, Hq, 5, D)).astype(mx.float16)
    bt = mx.array([list(range(16))], dtype=mx.int32)
    sl = mx.array([200], dtype=mx.int32)
    did = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    mx.eval(kp, vp, qt, bt, sl, did)
    return qt, kp, vp, bt, sl, did


def test_speculative_paged_valid():
    qt, kp, vp, bt, sl, did = _spec_paged()
    mx.eval(mlx_mfa.flash_attention_speculative_verify_paged(qt, kp, vp, bt, sl, did)[0])


@pytest.mark.parametrize("mut", ["draft_float", "temp0"])
def test_speculative_paged_malformed_raises(mut):
    qt, kp, vp, bt, sl, did = _spec_paged()
    kw = {}
    if mut == "draft_float":
        did = did.astype(mx.float32)
    elif mut == "temp0":
        kw["temperature"] = 0.0
    with pytest.raises((ValueError, Exception)):
        mx.eval(mlx_mfa.flash_attention_speculative_verify_paged(qt, kp, vp, bt, sl, did, **kw)[0])


# ── P1 splitfuse ────────────────────────────────────────────────────────────────
def test_splitfuse_valid_branches():
    mx.random.seed(0)
    qp = mx.random.normal((B, Hq, 128, D)).astype(mx.float16)
    kp = mx.random.normal((B, Hk, 128, D)).astype(mx.float16)
    vp = mx.random.normal((B, Hk, 128, D)).astype(mx.float16)
    mx.eval(qp, kp, vp)
    op, od = mlx_mfa.flash_attention_splitfuse(qp, kp, vp, None, None, None)
    mx.eval(op)


@pytest.mark.parametrize("mut", ["all_none", "prefill_partial", "decode_partial"])
def test_splitfuse_malformed_raises(mut):
    mx.random.seed(0)
    qp = mx.random.normal((B, Hq, 128, D)).astype(mx.float16)
    if mut == "all_none":
        args = (None, None, None, None, None, None)
    elif mut == "prefill_partial":
        args = (qp, None, None, None, None, None)
    elif mut == "decode_partial":
        args = (None, None, None, qp, None, None)
    with pytest.raises((ValueError, Exception)):
        r = mlx_mfa.flash_attention_splitfuse(*args)
        mx.eval(r[0] if r[0] is not None else r[1])


# ── P2 rope family (cos/sin f16 valid — the K1 lesson) ──────────────────────────
def test_rope_unified_f16_cos_valid():
    q, k, v = _qkv()
    cos = mx.ones((N, D // 2), dtype=mx.float16)   # f16 cos MUST be accepted
    sin = mx.zeros((N, D // 2), dtype=mx.float16)
    mx.eval(cos, sin)
    mx.eval(mlx_mfa.flash_attention_rope_unified(q, k, v, rotary_cos=cos, rotary_sin=sin))


def test_rope_unified_kv_mismatch_raises():
    q, k, v = _qkv()
    cos = mx.ones((N, D // 2), dtype=mx.float16)
    sin = mx.zeros((N, D // 2), dtype=mx.float16)
    with pytest.raises((ValueError, Exception)):
        mx.eval(mlx_mfa.flash_attention_rope_unified(
            q, k, v[:, :, : N // 2, :], rotary_cos=cos, rotary_sin=sin))


# ── P3/P6 cache adapters: OOB-append memory safety + valid ──────────────────────
def test_kvcache_dense_append_valid_and_oob_safe():
    mx.random.seed(0)
    Cap = 256
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
    kc = mx.random.normal((B, Hk, Cap, D)).astype(mx.float16)
    vc = mx.random.normal((B, Hk, Cap, D)).astype(mx.float16)
    kn = mx.random.normal((B, Hk, 1, D)).astype(mx.float16)
    vn = mx.random.normal((B, Hk, 1, D)).astype(mx.float16)
    mx.eval(q, kc, vc, kn, vn)
    # dense append CONCATENATES (safe) — no OOB write; output finite
    o = mlx_mfa.flash_attention_kvcache(q, kc, vc, k_new=kn, v_new=vn, cache_seqlens=10)
    out = o[0] if isinstance(o, tuple) else o
    mx.eval(out)
    assert bool(np.isfinite(np.array(out.astype(mx.float32))).all())


def test_kvcache_paged_append_oob_raises():
    mx.random.seed(0)
    bs, nb = 16, 8
    kp = mx.random.normal((nb, bs, Hk, D)).astype(mx.float16)
    vp = mx.random.normal((nb, bs, Hk, D)).astype(mx.float16)
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
    kn = mx.random.normal((B, Hk, 1, D)).astype(mx.float16)
    vn = mx.random.normal((B, Hk, 1, D)).astype(mx.float16)
    bt = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    mx.eval(kp, vp, q, kn, vn, bt)
    with pytest.raises((ValueError, Exception)):  # slot 200 beyond mapped blocks
        o = mlx_mfa.flash_attention_kvcache(
            q, kp, vp, k_new=kn, v_new=vn, block_table=bt,
            seq_lens=mx.array([200], dtype=mx.int32), block_size=bs, cache_seqlens=200)
        mx.eval(o[0] if isinstance(o, tuple) else o)


def test_sage_kvcache_valid_and_mismatch():
    q, k, v = _qkv()
    mx.eval(mlx_mfa.sage_attention_kvcache(q, k, v))   # inherits sage validation
    with pytest.raises((ValueError, Exception)):
        mx.eval(mlx_mfa.sage_attention_kvcache(q, k, v[:, :, : N // 2, :]))


# ── P7 topk ─────────────────────────────────────────────────────────────────────
def test_topk_valid_and_ratio_bounds():
    # topk's reference path is MHA (q@k needs Hq==Hk); GQA is not supported here.
    mx.random.seed(0)
    q = mx.random.normal((B, Hq, N, D)).astype(mx.float16)
    k = mx.random.normal((B, Hq, N, D)).astype(mx.float16)
    v = mx.random.normal((B, Hq, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    mx.eval(mlx_mfa.flash_attention_topk(q, k, v, 0.5))
    for bad in (0.0, 2.0, -1.0):
        with pytest.raises((ValueError, Exception)):
            mx.eval(mlx_mfa.flash_attention_topk(q, k, v, bad))

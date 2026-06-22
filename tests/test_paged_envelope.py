"""Volet H — exhaustive paged oracle + validation envelope.

The paged surface was the systematic blind spot: the kernel-math envelope's only
paged cell was a trivial B=1/Nq=1/S=48 (causal a no-op), so three CRITICALs hid
there (CX-01 per-sequence causal offset, CX-02 cu_seqlens dtype, CX-03 K/V pool
shape). This gives paged the volet-G treatment — every paged entry × cells, both
directions (valid → independent fp64 oracle; malformed → loud raise).

Independent oracle: gather each sequence's REAL keys from the pool (seq_lens[b]
tokens) and run fp64 attention with the PER-SEQUENCE causal offset
off_b = max(0, kv_len_b - Nq) — the convention CX-01 fixed. A cell that matched a
batch-global offload would FAIL the per-row oracle (that is the CX-01 bite).
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
import mlx_mfa._ext as _ext

F16, BF16 = mx.float16, mx.bfloat16
BOUND = {F16: 5e-3, BF16: 3e-2}
DTN = {F16: "f16", BF16: "bf16"}


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def _build(H, Hk, D, bs, seqs, Nq, dt, seed=0):
    """Build a paged scenario. seqs = list of per-sequence kv lengths.
    Returns q, k_pool, v_pool, block_table, seq_lens, and the page lists."""
    mx.random.seed(seed)
    B = len(seqs)
    max_blocks = max((L + bs - 1) // bs for L in seqs)
    # assign distinct contiguous pages per sequence
    nblk = sum((L + bs - 1) // bs for L in seqs) + 2
    k_pool = mx.random.normal((nblk, bs, Hk, D)).astype(dt)
    v_pool = mx.random.normal((nblk, bs, Hk, D)).astype(dt)
    q = mx.random.normal((B, H, Nq, D)).astype(dt)
    mx.eval(k_pool, v_pool, q)
    bt_rows, page_lists, p = [], [], 0
    for L in seqs:
        nb = (L + bs - 1) // bs
        pages = list(range(p, p + nb)); p += nb
        page_lists.append((pages, L))
        bt_rows.append(pages + [-1] * (max_blocks - nb))
    bt = mx.array(bt_rows, dtype=mx.int32)
    sl = mx.array(seqs, dtype=mx.int32)
    mx.eval(bt, sl)
    return q, k_pool, v_pool, bt, sl, page_lists


def _oracle(q, k_pool, v_pool, page_lists, scale, causal, Hk):
    """Per-sequence fp64 oracle; GQA via head h -> kv head h//(H/Hk)."""
    kp, vp = _f64(k_pool), _f64(v_pool)
    B, H, Nq, D = q.shape
    g = H // Hk
    out = np.zeros((B, H, Nq, D))
    qf = _f64(q)
    for b, (pages, L) in enumerate(page_lists):
        kk = np.concatenate([kp[pp] for pp in pages], 0)[:L]   # [L, Hk, D]
        vv = np.concatenate([vp[pp] for pp in pages], 0)[:L]
        for h in range(H):
            hk = h // g
            s = (qf[b, h] @ kk[:, hk, :].T) * scale            # [Nq, L]
            if causal:
                off = max(0, L - Nq)                            # PER-SEQUENCE (CX-01)
                i = np.arange(Nq)[:, None]; j = np.arange(L)[None, :]
                s = np.where(j <= i + off, s, -1e30)
            m = s.max(1, keepdims=True); m[~np.isfinite(m)] = 0.0
            e = np.exp(s - m); ss = e.sum(1, keepdims=True); ss[ss == 0] = 1.0
            out[b, h] = (e / ss) @ vv[:, hk, :]
    return out


def _relerr(o, r):
    return float(np.max(np.abs(_f64(o) - r)) / (np.max(np.abs(r)) + 1e-9))


# ── correctness cells ─────────────────────────────────────────────────────────
# (label, H, Hk, D, seqs, Nq, causal)
_CORR = []
for dt in (F16, BF16):
    for D in (64, 128):
        # homogeneous, Nq>1 causal
        _CORR.append((f"homo_D{D}_{DTN[dt]}", 4, 4, D, [48, 48], 17, True, dt))
        # heterogeneous causal, per-seq Nq<Nk (CX-01 completeness oracle)
        _CORR.append((f"hetero_NqLTNk_D{D}_{DTN[dt]}", 4, 4, D, [31, 50], 17, True, dt))
        # heterogeneous causal, per-seq Nq>Nk (queries longer than some kv)
        _CORR.append((f"hetero_NqGTNk_D{D}_{DTN[dt]}", 4, 4, D, [13, 40], 24, True, dt))
        # non-causal heterogeneous
        _CORR.append((f"hetero_nc_D{D}_{DTN[dt]}", 4, 4, D, [31, 50], 17, False, dt))
        # GQA heterogeneous causal
        _CORR.append((f"gqa_hetero_D{D}_{DTN[dt]}", 8, 2, D, [31, 50], 17, True, dt))
        # 3-way heterogeneous
        _CORR.append((f"hetero3_D{D}_{DTN[dt]}", 4, 4, D, [20, 48, 70], 9, True, dt))


@pytest.mark.parametrize("label,H,Hk,D,seqs,Nq,causal,dt", _CORR,
                         ids=[c[0] for c in _CORR])
def test_paged_correctness(label, H, Hk, D, seqs, Nq, causal, dt):
    q, kp, vp, bt, sl, pages = _build(H, Hk, D, 16, seqs, Nq, dt)
    o = mlx_mfa.flash_attention_paged(q, kp, vp, bt, sl, scale=1.0 / math.sqrt(D),
                                      causal=causal, block_size=16)
    mx.eval(o)
    ref = _oracle(q, kp, vp, pages, 1.0 / math.sqrt(D), causal, Hk)
    assert _relerr(o, ref) <= BOUND[dt], (
        f"{label}: relerr {_relerr(o, ref):.3e} > {BOUND[dt]:.0e} vs PER-SEQUENCE "
        f"fp64 oracle (CX-01: heterogeneous causal must use per-seq offset)")


def test_cx01_heterogeneous_not_batch_global():
    """CX-01 completeness oracle: heterogeneous causal must match the PER-ROW
    oracle and be far from the batch-global oracle (the pre-fix behavior)."""
    q, kp, vp, bt, sl, pages = _build(4, 4, 64, 16, [31, 50], 17, F16)
    o = mlx_mfa.flash_attention_paged(q, kp, vp, bt, sl, scale=1 / 8,
                                      causal=True, block_size=16)
    mx.eval(o)
    perrow = _oracle(q, kp, vp, pages, 1 / 8, True, 4)
    # batch-global oracle (pre-fix): every seq uses off = max(seq)-Nq
    kpn, vpn = _f64(kp), _f64(vp); qf = _f64(q); B, H, Nq, D = q.shape
    glob = np.zeros((B, H, Nq, D)); G = max(L for _, L in pages) - Nq
    for b, (pg, L) in enumerate(pages):
        kk = np.concatenate([kpn[p] for p in pg], 0)[:L]; vv = np.concatenate([vpn[p] for p in pg], 0)[:L]
        for h in range(H):
            s = (qf[b, h] @ kk[:, h, :].T) / 8; i = np.arange(Nq)[:, None]; j = np.arange(L)[None, :]
            s = np.where(j <= i + G, s, -1e30); m = s.max(1, keepdims=True); m[~np.isfinite(m)] = 0
            e = np.exp(s - m); ss = e.sum(1, keepdims=True); ss[ss == 0] = 1; glob[b, h] = (e / ss) @ vv[:, h, :]
    assert _relerr(o, perrow) < 5e-3, "must match per-row oracle"
    assert _relerr(o, glob) > 0.1, "must NOT match the batch-global (pre-fix) oracle"


# ── validation cells (malformed → raise) ──────────────────────────────────────
def _ok(H=4, D=64, bs=16, nblk=8):
    mx.random.seed(0)
    kp = mx.random.normal((nblk, bs, H, D)).astype(F16)
    vp = mx.random.normal((nblk, bs, H, D)).astype(F16)
    mx.eval(kp, vp)
    return kp, vp


_VAL = {
    # CX-02 — int64 / float cu_seqlens (paged_varlen)
    "cx02_int64_cu": lambda: mlx_mfa.flash_attention_paged_varlen(
        mx.random.normal((1, 4, 2, 64)).astype(F16), *_ok(),
        mx.array([[2, 5], [3, 4]], dtype=mx.int32), mx.array([20, 20], dtype=mx.int32),
        mx.array([0, 1, 2], dtype=mx.int64), scale=1 / 8, block_size=16),
    "cx02_float_cu": lambda: mlx_mfa.flash_attention_paged_varlen(
        mx.random.normal((1, 4, 2, 64)).astype(F16), *_ok(),
        mx.array([[2, 5], [3, 4]], dtype=mx.int32), mx.array([20, 20], dtype=mx.int32),
        mx.array([0, 1, 2], dtype=mx.float32), scale=1 / 8, block_size=16),
    # CX-03 — K/V pool shape mismatch
    "cx03_v_blocks": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), _ok()[0],
        mx.random.normal((6, 16, 4, 64)).astype(F16),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
    "cx03_v_heads": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), _ok()[0],
        mx.random.normal((8, 16, 2, 64)).astype(F16),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
    "cx03_v_headdim": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), _ok()[0],
        mx.random.normal((8, 16, 4, 32)).astype(F16),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
    "cx03_v_blocksize": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), _ok()[0],
        mx.random.normal((8, 8, 4, 64)).astype(F16),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
    # CX-03 raw steel
    "cx03_raw_steel_v": lambda: _ext.mfa_paged_steel_forward(
        mx.random.normal((1, 4, 1, 64)).astype(F16), _ok()[0],
        mx.random.normal((8, 16, 2, 64)).astype(F16),
        mx.array([[2, 5, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        1 / 8, False, -1, -1, 16),
    # batch-cardinality (carried from C2, re-pinned here)
    "seq_short": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((2, 4, 1, 64)).astype(F16), *_ok(),
        mx.array([[2, 5, 1, 0], [3, 4, 0, 1]], dtype=mx.int32),
        mx.array([48], dtype=mx.int32), scale=1 / 8, block_size=16),
    # float block_table metadata
    "float_meta": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), *_ok(),
        mx.array([[2, 5, 1, 0]], dtype=mx.float32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
    # OOB page index
    "oob_page": lambda: mlx_mfa.flash_attention_paged(
        mx.random.normal((1, 4, 1, 64)).astype(F16), *_ok(),
        mx.array([[2, 99, 1, 0]], dtype=mx.int32), mx.array([48], dtype=mx.int32),
        scale=1 / 8, block_size=16),
}


@pytest.mark.parametrize("cid", list(_VAL.keys()))
def test_paged_validation_raises(cid):
    with pytest.raises(ValueError):
        out = _VAL[cid]()
        mx.eval(out[0] if isinstance(out, tuple) else out)

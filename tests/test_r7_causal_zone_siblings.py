"""R7 (CRIT-03) + sibling audit (code review 2026-09, P0 PUBLISHED) — causal mask zone.

The per-element causal mask must run on EVERY K-tile a Q-tile's diagonal crosses:
from (qb*BQ + qL_off)/BK on.  The old "last few K-tiles" heuristic
``kb >= kb_lim - (BQ+BK-1)/BK`` masks only the tail, which is exact only when
qL_off % BK == 0; otherwise rows attend up to BK-1 FUTURE keys.  The RC-A fix
(mfa_causal_mask_zone_gate) reached V1/V2 only.  The remediation audit found the
heuristic in 5 more live kernels; 4 were silently wrong on published entries:
  * Sage (R7)                         sage_attention / sage_attention_kvcache
  * STEEL V3 (new, err 1.57)          flash_attention(backend='mfa'); auto on M1-M4
  * paged-varlen fused (new, 0.1-0.3) flash_attention_paged_varlen, heterogeneous q_lens
  * paged-varlen TQ (new, leak <=7.8) flash_attention_paged_varlen_turboquant
  * STEEL backward: qL_off == 0 there -> heuristic was exact (helper now, bit-identical)

Primary check = FUTURE-KEY INVARIANCE (exact, oracle-free, immune to int8/TQ
quantization noise): row r's output must be bit-identical when V is changed only at
keys j > r + qL_off (their softmax weight is exactly 0 via -inf).  V is fp16 on every
path here (Sage/TQ quantize Q/K only), so the perturbation touches nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import (
    flash_attention,
    flash_attention_paged_varlen,
    is_mfa_available,
    sage_attention,
)
from mlx_mfa.quantize import sage_block_sizes

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

BQ_STEEL = 32  # Q tile of the STEEL-family kernels exercised here (D=64/128)


def _oracle(q, k, v):
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        N, S, D = q.shape[2], k.shape[2], q.shape[3]
        s = (q32 @ mx.swapaxes(k32, -1, -2)) * (D ** -0.5)
        s = s + mx.where(mx.arange(S)[None, :] <= mx.arange(N)[:, None] + (S - N), 0.0, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v32
        mx.eval(o)
    return o


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _qkv(N, S, D, H=2, seed=0):
    mx.random.seed(seed)
    return (mx.random.normal((1, H, N, D)).astype(mx.float16),
            mx.random.normal((1, H, S, D)).astype(mx.float16),
            mx.random.normal((1, H, S, D)).astype(mx.float16))


def _invariance_rows(N):
    """First/last rows of the first two Q tiles.  The leak shows on the FIRST row(s) of
    a Q tile whose diagonal straddles two K tiles (e.g. Sage D=64, BK=64, qL_off%BK=30:
    only row r=BQ leaks) — probing mid-tile rows alone can miss it."""
    B = BQ_STEEL
    return sorted(r for r in {0, B - 1, B, B + 1, 2 * B - 1, 2 * B} if r < N)


def _assert_future_key_invariant(fn, v, N, S):
    """fn(v) -> [1,H,N,D].  For each probe row r, perturb V at keys j > r+qL_off only."""
    off = S - N
    base = fn(v)
    mx.eval(base)
    for r in _invariance_rows(N):
        first_future = off + r + 1
        if first_future >= S:
            continue
        bump = mx.concatenate([mx.zeros_like(v[:, :, :first_future]),
                               mx.full(v[:, :, first_future:].shape, 100.0, dtype=v.dtype)], axis=2)
        out = fn(v + bump)
        mx.eval(out)
        d = _err(out[:, :, : r + 1], base[:, :, : r + 1])
        assert d == 0.0, f"rows 0..{r} changed by {d} when only keys > {off + r} (future) changed"


# ── Sage (R7) ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rem", [0, 8, 30])
@pytest.mark.parametrize("D", [64, 128])
def test_sage_causal_no_future_keys(D, rem):
    BK = sage_block_sizes(D)[1]
    N = 100
    S = N + 2 * BK + rem          # qL_off % BK == rem
    q, k, v = _qkv(N, S, D, seed=D + rem)
    _assert_future_key_invariant(lambda vv: sage_attention(q, k, vv, causal=True), v, N, S)


@pytest.mark.parametrize("rem", [0, 8, 30])
@pytest.mark.parametrize("D", [64, 128])
def test_sage_causal_matches_oracle_per_row(D, rem):
    """Oracle check at the int8 floor (aligned floor measured ~4e-3; the leak was 0.05-0.3)."""
    BK = sage_block_sizes(D)[1]
    N = 100
    S = N + 2 * BK + rem
    q, k, v = _qkv(N, S, D, seed=100 + D + rem)
    out = sage_attention(q, k, v, causal=True)
    assert _err(out, _oracle(q, k, v)) < 2e-2


# ── STEEL V3 (new sibling) ───────────────────────────────────────────────────
@pytest.mark.parametrize("off", [4, 8, 30, 32])   # 32 = aligned control
def test_v3_forced_mfa_causal_n_lt_s(off):
    """V3 shape (causal D=64 N>=4096 B*H>=4); backend='mfa' reaches it on M5."""
    N, D = 4096, 64
    q, k, v = _qkv(N, N + off, D, H=4, seed=off)
    out = flash_attention(q, k, v, causal=True, backend="mfa")
    assert _err(out, _oracle(q, k, v)) < 1e-2


def test_v3_forced_mfa_no_future_keys():
    N, D, off = 4096, 64, 4
    q, k, v = _qkv(N, N + off, D, H=4, seed=7)
    _assert_future_key_invariant(
        lambda vv: flash_attention(q, k, vv, causal=True, backend="mfa"), v, N, N + off)


# ── paged-varlen fused kernel (new sibling) ──────────────────────────────────
def _paged(seqs, D, H=2, BS=16):
    """seqs: list of (q, k, v) [1,H,*,D]. Returns packed q, pools, block table, lens, cu."""
    kp, vp, bt, nb = [], [], [], 0
    maxb = max((k.shape[2] + BS - 1) // BS for _, k, _ in seqs)
    for _, k, v in seqs:
        kl = k.shape[2]
        n = (kl + BS - 1) // BS
        pad = n * BS - kl
        kp.append(mx.pad(k[0].transpose(1, 0, 2), [(0, pad), (0, 0), (0, 0)]).reshape(n, BS, H, D))
        vp.append(mx.pad(v[0].transpose(1, 0, 2), [(0, pad), (0, 0), (0, 0)]).reshape(n, BS, H, D))
        bt.append(list(range(nb, nb + n)) + [0] * (maxb - n))
        nb += n
    cu = [0]
    for q, _, _ in seqs:
        cu.append(cu[-1] + q.shape[2])
    return (mx.concatenate([q for q, _, _ in seqs], axis=2), mx.concatenate(kp, 0),
            mx.concatenate(vp, 0), mx.array(bt, dtype=mx.int32),
            mx.array([k.shape[2] for _, k, _ in seqs], dtype=mx.int32), mx.array(cu, dtype=mx.int32), cu)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("qls,kls", [([40, 8], [100, 77]), ([33, 17], [200, 50]),
                                     ([64, 32], [128, 96]), ([70, 5], [101, 300])])
def test_paged_varlen_heterogeneous_causal_matches_oracle(D, qls, kls):
    seqs = [_qkv(ql, kl, D, seed=ql * 3 + kl) for ql, kl in zip(qls, kls)]
    q, kp, vp, bt, sl, cu_arr, cu = _paged(seqs, D)
    out = flash_attention_paged_varlen(q, kp, vp, bt, sl, cu_arr, causal=True, block_size=16)
    for i, (qi, ki, vi) in enumerate(seqs):
        e = _err(out[:, :, cu[i]:cu[i + 1]], _oracle(qi, ki, vi))
        assert e < 1e-2, f"seq {i} (q={qls[i]}, kv={kls[i]}, qL_off%32={(kls[i]-qls[i]) % 32}): {e}"


# ── paged-varlen TurboQuant fused kernel (new sibling) ───────────────────────
def test_paged_varlen_turboquant_no_future_keys():
    """Sweep qL_off = 512 - q_len across all residues; row 0 must ignore future-key V."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_phase3_iii2_tq_decode as T
    import mlx_mfa
    from mlx_mfa.turboquant import apply_rotation

    ctx, _ = T._mkctx(3)
    S = T.S0
    bt = ctx.get_block_table([0])
    sl = ctx.get_seq_lens([0])
    btn = np.array(bt)[0]
    vbase = np.array(ctx._v_pool_fp16.astype(mx.float32))
    leaks = []
    for ql in range(17, 101, 3):
        mx.random.seed(ql)
        q = apply_rotation(mx.random.normal((1, T.Hq, ql, T.D)).astype(mx.float32), "wht").astype(mx.float16)
        cu = mx.array([0, ql], dtype=mx.int32)
        off = S - ql

        def run(vpool):
            return mlx_mfa.flash_attention_paged_varlen_turboquant(
                q, ctx._k_pool, vpool, bt, sl, cu, ctx._k_centroids, ctx._k_scales,
                scale=T.SCALE, causal=True, block_size=T.BS, tq_bits=3,
                tq_v_enabled=False, tq_wht_enabled=False)

        vn = vbase.copy()
        for p in range(off + 1, S):
            vn[btn[p // T.BS], p % T.BS] += 100.0
        o1, o2 = run(ctx._v_pool_fp16), run(mx.array(vn).astype(mx.float16))
        mx.eval(o1, o2)
        d = _err(o1[:, :, :1], o2[:, :, :1])
        if d != 0.0:
            leaks.append((ql, off, off % 32, d))
    assert not leaks, f"row 0 attended future keys: {leaks}"

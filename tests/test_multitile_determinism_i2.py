"""Volet I2 — multi-tile determinism across ALL shared-buffer forward kernels.

Volet S found mfa_sage_forward raced on the shared KV_smem buffer (no barrier
between iter kb's P@V-read and kb+1's K-load), manifest only at N>=512 (>=2 K-tiles).
Volet I's determinism axis sampled N=256 (single-tile) and missed it. This locks the
axis PROPERLY: every kernel that aliases Ks==Vs==KV_smem must be byte-deterministic
over identical inputs at MULTI-tile sizes (N>=512), with fresh-but-identical arrays
each call (defeats the MLX graph cache — the other reason volet I missed it).

Shared-buffer kernels audited (static barrier present + runtime here): dense STEEL
V1/V2 (backend="mfa"), sparse (V2), varlen, GNA, paged STEEL, paged TQ. Sage is
locked separately in test_sage_determinism_s.py. v3 / v6_nax use separate K/V smem
(no reuse → not in scope).
"""
import math
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa

_NRUNS = 20


def _det(mk):
    outs = []
    for _ in range(_NRUNS):
        o = mk()
        oo = o[0] if isinstance(o, tuple) else o
        mx.eval(oo)
        outs.append(np.array(oo.astype(mx.float32)))
    return max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, _NRUNS))


def _qkv(H, N, D, Hk, dt):
    mx.random.seed(0)  # identical VALUES, fresh OBJECTS each call (cache-defeat)
    q = mx.random.normal((1, H, N, D)).astype(dt)
    k = mx.random.normal((1, Hk, N, D)).astype(dt)
    v = mx.random.normal((1, Hk, N, D)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


_HEADS = [(4, 4), (8, 2)]          # MHA, GQA
_DT = [mx.float16, mx.bfloat16]
_N = [512, 1024]                   # multi-tile (the race threshold)


@pytest.mark.parametrize("N", _N)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("Hq,Hk", _HEADS)
@pytest.mark.parametrize("dt", _DT)
def test_dense_mfa_deterministic(N, D, Hq, Hk, dt):
    assert _det(lambda: mlx_mfa.flash_attention(
        *_qkv(Hq, N, D, Hk, dt), scale=1 / math.sqrt(D), causal=True, backend="mfa")) == 0.0


@pytest.mark.parametrize("N", _N)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("Hq,Hk", _HEADS)
@pytest.mark.parametrize("dt", _DT)
def test_sparse_deterministic(N, D, Hq, Hk, dt):
    bm = mlx_mfa.make_causal_block_mask(N, head_dim=D)
    assert _det(lambda: mlx_mfa.flash_attention_sparse(
        *_qkv(Hq, N, D, Hk, dt), bm, causal=True)) == 0.0


@pytest.mark.parametrize("N", _N)
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dt", _DT)
def test_varlen_deterministic(N, D, dt):
    cu = mx.array([0, N], dtype=mx.int32)
    assert _det(lambda: mlx_mfa.flash_attention_varlen(
        *_qkv(4, N, D, 4, dt), cu, cu, N, N, scale=1 / math.sqrt(D), causal=True)) == 0.0


@pytest.mark.parametrize("seq,N", [((8, 8, 8), 512), ((10, 10, 10), 1000)])
@pytest.mark.parametrize("Hq,Hk", _HEADS)
@pytest.mark.parametrize("dt", _DT)
def test_gna_deterministic(seq, N, Hq, Hk, dt):
    assert _det(lambda: mlx_mfa.flash_attention_gna(
        *_qkv(Hq, N, 128, Hk, dt), seq_shape=seq, window_size=(3, 3, 3), stride=(1, 1, 1))) == 0.0


@pytest.mark.parametrize("S", [512, 1024])
@pytest.mark.parametrize("D", [64, 128, 256])   # volet J: D=256 added (CX-J-02 was D-agnostic)
@pytest.mark.parametrize("dt", _DT)
def test_paged_steel_deterministic(S, D, dt):
    # CX-J-02: the paged STEEL K-gather reused KV_smem with NO inter-iteration
    # barrier → nondeterministic at multi-tile S. The volet-I2 runtime probe got
    # lucky (race is intermittent, pytest-context-triggered); the fix added the
    # start-of-loop barrier. These cells reliably flake without it.
    def mk():
        mx.random.seed(0)
        bs = 16
        nb = (S + bs - 1) // bs + 2
        kp = mx.random.normal((nb, bs, 4, D)).astype(dt)
        vp = mx.random.normal((nb, bs, 4, D)).astype(dt)
        q = mx.random.normal((1, 4, 8, D)).astype(dt)
        bt = mx.array([list(range((S + bs - 1) // bs))], dtype=mx.int32)
        sl = mx.array([S], dtype=mx.int32)
        mx.eval(kp, vp, q, bt, sl)
        return mlx_mfa.flash_attention_paged(q, kp, vp, bt, sl, scale=1 / math.sqrt(D),
                                             causal=True, block_size=bs)
    assert _det(mk) == 0.0


@pytest.mark.parametrize("S", [512, 1024])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("dt", _DT)
def test_paged_varlen_deterministic(S, D, dt):
    # volet J (CX-R7-03): the inventory claimed paged-varlen determinism was locked
    # but no cell ran it. Source-confirmed the inter-iteration barrier IS present;
    # this makes the claim executable.
    def mk():
        mx.random.seed(0)
        bs = 16
        nbk = (S + bs - 1) // bs
        kp = mx.random.normal((nbk + 2, bs, 4, D)).astype(dt)
        vp = mx.random.normal((nbk + 2, bs, 4, D)).astype(dt)
        q = mx.random.normal((1, 4, 8, D)).astype(dt)
        bt = mx.array([list(range(nbk))], dtype=mx.int32)
        sl = mx.array([S], dtype=mx.int32)
        cu = mx.array([0, 8], dtype=mx.int32)
        mx.eval(kp, vp, q, bt, sl, cu)
        return mlx_mfa.flash_attention_paged_varlen(
            q, kp, vp, bt, sl, cu, scale=1 / math.sqrt(D), causal=True, block_size=bs)
    assert _det(mk) == 0.0


def test_paged_tq_deterministic():
    # volet J (CX-R7-03): paged-TQ determinism claim made executable.
    # Source-confirmed the start-of-loop barrier is present (paged-TQ K-gather).
    import sys
    sys.path.insert(0, "tests")
    import test_phase3_iii2_tq_decode as T
    from mlx_mfa.turboquant import apply_rotation

    def mk():
        ctx, q = T._mkctx(3)
        qr = apply_rotation(q.astype(mx.float32), "wht").astype(mx.float16)
        cu = mx.array([0, 1], dtype=mx.int32)
        mx.eval(qr)
        return mlx_mfa.flash_attention_paged_varlen_turboquant(
            qr, ctx._k_pool, ctx._v_pool_fp16, ctx.get_block_table([0]),
            ctx.get_seq_lens([0]), cu, ctx._k_centroids, ctx._k_scales,
            scale=T.SCALE, causal=True, block_size=T.BS, tq_bits=3,
            tq_v_enabled=False, tq_wht_enabled=False)
    assert _det(mk) == 0.0

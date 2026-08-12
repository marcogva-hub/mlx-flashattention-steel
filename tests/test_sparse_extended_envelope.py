"""Volet A Phase 1 — sparse extended-envelope tests (spec_blocksparse_v1.md).

Three axes (gold parity · engagement · edge cases) + explicit LOCKS for the §4
binary gates. Skips cleanly off M5+/NAX (the extended path is M5+-only per §1).
"""
import os
import math
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import attention as A
from mlx_mfa.attention import flash_attention_sparse

try:
    from mlx_mfa import _ext  # noqa: F401
    _HAS_EXT = True
except Exception:
    _HAS_EXT = False

BT = 32
_M5 = _HAS_EXT and bool(getattr(A, "_get_is_m5_plus_cached", lambda: False)())
m5only = pytest.mark.skipif(not _M5, reason="extended sparse path is M5+/NAX only (spec §1)")


# ----------------------------------------------------------------------- helpers
def _qkv(B, H, N, D, dt=mx.float16, seed=0):
    mx.random.seed(seed)
    f = lambda: (mx.random.normal((B, H, N, D)) * 0.1).astype(dt)
    q, k, v = f(), f(), f()
    mx.eval(q, k, v)
    return q, k, v


def _block_mask(nq, nk, density, seed=0):
    mx.random.seed(seed)
    m = (mx.random.uniform(shape=(nq, nk)) < density)
    m = (m | m.T) | (mx.eye(nq, nk, dtype=mx.float32) > 0.5)
    return m.astype(mx.bool_)


def _gold(q, k, v, block_mask, scale, causal, N, S):
    """fp32 SDPA with the block mask expanded to an element bias, over ORIGINAL N/S."""
    em = mx.repeat(mx.repeat(block_mask, BT, axis=0), BT, axis=1)[:N, :S]
    if causal:
        idx_q = mx.arange(N)[:, None]
        idx_k = mx.arange(S)[None, :]
        em = em & (idx_k <= idx_q)
    bias = mx.where(em, mx.array(0.0, mx.float32), mx.array(float("-inf"), mx.float32))
    o = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=bias)
    mx.eval(o)
    return np.asarray(o).ravel().astype(np.float64)


def _cos(a_arr, b_flat):
    a = np.asarray(a_arr.astype(mx.float32)).ravel().astype(np.float64)
    return float(np.dot(a, b_flat) / (np.linalg.norm(a) * np.linalg.norm(b_flat) + 1e-30))


# =============================================================== auto_pad (§2)
@m5only
def test_autopad_aligned_byte_identical():
    """Gate 4a / §2(a): N already aligned + auto_pad=True == auto_pad=False, byte-identical."""
    B, H, N, D = 1, 12, 4096, 128           # N % 32 == 0
    q, k, v = _qkv(B, H, N, D)
    nq = N // BT
    bm = _block_mask(nq, nq, 0.20)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    o_false = flash_attention_sparse(q, k, v, bm, scale=scale, auto_pad=False)
    o_true = flash_attention_sparse(q, k, v, bm, scale=scale, auto_pad=True)
    mx.eval(o_false, o_true)
    delta = float(np.abs(np.asarray(o_false.astype(mx.float32))
                         - np.asarray(o_true.astype(mx.float32))).max())
    assert delta == 0.0, f"auto_pad no-op on aligned N must be byte-identical; maxabs={delta}"


@m5only
@pytest.mark.parametrize("causal", [False, True])
def test_autopad_nonaligned_cos_gold(causal):
    """Gate 4b / §2(b): non-aligned N via auto_pad → cos gold vs SDPA+element-mask ≥ 0.999."""
    B, H, N, D = 1, 12, 4100, 128           # 4100 % 32 = 4  → pad to 4128
    q, k, v = _qkv(B, H, N, D)
    nq = (N + BT - 1) // BT                  # ceil → 129
    bm = _block_mask(nq, nq, 0.20)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    o = flash_attention_sparse(q, k, v, bm, scale=scale, causal=causal, auto_pad=True)
    mx.eval(o)
    assert o.shape == (B, H, N, D), f"output must be sliced to original N; got {o.shape}"
    gold = _gold(q, k, v, bm, scale, causal, N, N)
    cos = _cos(o, gold)
    assert cos >= 0.999, f"auto_pad(N={N},causal={causal}) cos vs gold = {cos:.6f} < 0.999"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))

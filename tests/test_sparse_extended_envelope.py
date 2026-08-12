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


# ============================================ MFA_SPARSE_NAX_EXTENDED (§3)
from mlx_mfa.lcsa_nax import _nax_sparse_route_viable, _sparse_extended_enabled


def _shape_arr(B, H, N, D, dt=mx.float16):
    return mx.zeros((B, H, N, D), dtype=dt)


def test_extended_gate_bypasses_policy_not_capacity(monkeypatch):
    """Gate 3 (unit): extended bypasses POLICY bounds (B·H/N) but NEVER capacity."""
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED", raising=False)
    # Out-of-policy shape (B·H=40, tiny N): default gate rejects.
    q = _shape_arr(1, 40, 256, 128); k = q
    assert _nax_sparse_route_viable(q, k, 32, 0.5) is False
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    assert _sparse_extended_enabled() is True
    assert _nax_sparse_route_viable(q, k, 32, 0.5) is True          # policy bypassed
    # Capacity constraints are NEVER bypassed, even with the env on:
    assert _nax_sparse_route_viable(_shape_arr(1, 4, 256, 256), _shape_arr(1, 4, 256, 256), 32, 0.1) is False  # D=256
    assert _nax_sparse_route_viable(_shape_arr(1, 4, 4096, 128), _shape_arr(1, 4, 2048, 128), 32, 0.1) is False  # qL≠kL
    assert _nax_sparse_route_viable(q, k, 64, 0.1) is False          # block_tile≠32
    assert _nax_sparse_route_viable(_shape_arr(1, 4, 256, 128, mx.float32),
                                    _shape_arr(1, 4, 256, 128, mx.float32), 32, 0.1) is False  # fp32


@m5only
def test_extended_route_correct_bh40(monkeypatch):
    """B·H=40 (outside default gate) routes correct under the opt-in (cos vs gold)."""
    B, H, N, D = 1, 40, 4096, 128
    q, k, v = _qkv(B, H, N, D)
    nq = N // BT
    bm = _block_mask(nq, nq, 0.20)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    o = flash_attention_sparse(q, k, v, bm, scale=scale)
    mx.eval(o)
    cos = _cos(o, _gold(q, k, v, bm, scale, False, N, N))
    assert cos >= 0.999, f"extended B·H40 cos vs gold = {cos:.6f}"


@m5only
def test_offpath_byte_identical(monkeypatch):
    """Gate 3 (lock): the opt-in never perturbs an IN-envelope shape's output.

    Density is kept low enough that the SYMMETRIZED mask stays under the 0.30
    ceiling, so env-off already routes NAX; env-on must then be byte-identical.
    """
    B, H, N, D = 1, 12, 4096, 128          # in the default β3 gate
    q, k, v = _qkv(B, H, N, D)
    bm = _block_mask(N // BT, N // BT, 0.05)
    mx.eval(bm)
    _actual = float(mx.mean(bm.astype(mx.float32)).item())
    assert _actual <= 0.30, f"test premise: in-gate density, got {_actual:.3f}"
    scale = 1.0 / math.sqrt(D)
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED", raising=False)
    o_off = flash_attention_sparse(q, k, v, bm, scale=scale)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    o_on = flash_attention_sparse(q, k, v, bm, scale=scale)
    mx.eval(o_off, o_on)
    delta = float(np.abs(np.asarray(o_off.astype(mx.float32))
                         - np.asarray(o_on.astype(mx.float32))).max())
    assert delta == 0.0, f"opt-in must not perturb in-envelope routing; maxabs={delta}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))

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
from tests.sparse_gates import assert_row_gates

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
    # unit variance: at 0.1 scale attention is near-uniform and wrong outputs pass (review B)
    f = lambda: mx.random.normal((B, H, N, D)).astype(dt)
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
    with mx.stream(mx.cpu):                    # fp32 oracle on CPU (M5 GPU fp32 ~2e-3)
        o = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
            scale=scale, mask=bias)
        mx.eval(o)
    return o


# Review 2026-09 (DSP-14 / TST-04, remediation B1): every correctness gate below was a
# GLOBAL cosine >= 0.999 — blind to the U1 row scaling (x0.026 on tail rows, cos still
# >= 0.999).  They are per-row magnitude gates now (tests/sparse_gates.py); the cosine
# stays as a complement.  The historical "7/7 PASS" of Volet A is void (at-scale re-proof:
# benchmarks/blocksparse_reproof_b4.py).
_G16 = dict(max_abs=1e-2, norm_tol=5e-3, cos_min=0.999)
_GBF16 = dict(max_abs=3e-2, norm_tol=2e-2, cos_min=0.999)


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
def test_autopad_nonaligned_gold(monkeypatch, causal):
    """Gate 4b / §2(b): non-aligned N via auto_pad vs fp32 SDPA+element-mask, per-row gates.
    2.64: the DEFAULT law routes the non-causal cell to the padded kv_valid_len kernel
    (B*H12, d 0.20 <= 0.30); causal stays on the SDPA fallback by default (causal policy
    unchanged) and the padded kernel is then locked directly (it serves the B1 rescue)."""
    B, H, N, D = 1, 12, 4100, 128           # 4100 % 32 = 4  → pad to 4128
    q, k, v = _qkv(B, H, N, D)
    nq = (N + BT - 1) // BT                  # ceil → 129
    bm = _block_mask(nq, nq, 0.10)           # symmetrised by the helper -> ~0.2
    mx.eval(bm)
    from mlx_mfa.lcsa_nax import mask_density
    assert mask_density(bm) <= 0.30, "premise: inside the kept 0.30 density ceiling"
    scale = 1.0 / math.sqrt(D)
    from mlx_mfa import _dispatch_trace as dt
    with dt.capture() as cap:
        o = flash_attention_sparse(q, k, v, bm, scale=scale, causal=causal, auto_pad=True)
        mx.eval(o)
    gold = _gold(q, k, v, bm, scale, causal, N, N)
    assert o.shape == (B, H, N, D), f"output must be sliced to original N; got {o.shape}"
    assert_row_gates(o, gold, **_G16, label=f"auto_pad(N={N},causal={causal})")
    if not causal:
        assert any("kv_valid" in r[1] for r in cap), [r[:2] for r in cap]
    else:
        ok = A._make_sparse_nax_padded_vjp(scale, True)(q, k, v, bm)
        assert_row_gates(ok, gold, **_G16, label="padded kernel, causal")


# ============================================ MFA_SPARSE_NAX_EXTENDED (§3)
from mlx_mfa.lcsa_nax import _nax_sparse_route_viable, _sparse_extended_enabled


def _shape_arr(B, H, N, D, dt=mx.float16):
    return mx.zeros((B, H, N, D), dtype=dt)


def test_extended_gate_bypasses_policy_not_capacity(monkeypatch):
    """Gate 3 (unit), 2.64: the extended opt-in is a no-op; the default law bounds POLICY
    (N >= 2048 here) and capacity is never bypassed.  Non-causal qL != kL is now inside
    capacity (B5); causal qL != kL is not (U2)."""
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED", raising=False)
    q = _shape_arr(1, 40, 256, 128); k = q
    assert _nax_sparse_route_viable(q, k, 32, 0.1) is False          # N < MIN_N
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    assert _sparse_extended_enabled() is False                       # no-op
    assert _nax_sparse_route_viable(q, k, 32, 0.1) is False
    assert _nax_sparse_route_viable(_shape_arr(1, 4, 256, 256), _shape_arr(1, 4, 256, 256), 32, 0.1) is False  # D=256
    assert _nax_sparse_route_viable(_shape_arr(1, 8, 4096, 128), _shape_arr(1, 8, 2048, 128), 32, 0.1) is True   # B5
    assert _nax_sparse_route_viable(_shape_arr(1, 8, 4096, 128), _shape_arr(1, 8, 2048, 128), 32, 0.1,
                                    causal=True) is False            # U2
    assert _nax_sparse_route_viable(_shape_arr(1, 12, 4096, 128), _shape_arr(1, 12, 4096, 128), 64, 0.1) is False  # tile
    assert _nax_sparse_route_viable(_shape_arr(1, 4, 256, 128, mx.float32),
                                    _shape_arr(1, 4, 256, 128, mx.float32), 32, 0.1) is False  # fp32


@m5only
def test_extended_route_correct_bh40(monkeypatch):
    """B·H=40 (outside default gate) routes correct under the opt-in (per-row gates)."""
    B, H, N, D = 1, 40, 4096, 128
    q, k, v = _qkv(B, H, N, D)
    nq = N // BT
    bm = _block_mask(nq, nq, 0.20)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    o = flash_attention_sparse(q, k, v, bm, scale=scale)
    mx.eval(o)
    assert_row_gates(o, _gold(q, k, v, bm, scale, False, N, N), **_G16, label="extended B·H40")


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


@m5only
def test_refusal_pre_m5_extended(monkeypatch):
    """2.64: the knob no longer refuses pre-M5 (it is a no-op); the documented pre-M5
    refusal survives where it is an API contract — sla_attention(extended=True)."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    monkeypatch.setattr(A, "_get_is_m5_plus_cached", lambda: False)
    from mlx_mfa import sla
    q, k, v = _qkv(1, 4, 4096, 128)
    with pytest.raises(RuntimeError, match="requires M5"):
        sla.sla_attention(q, k, v, topk_ratio=0.1, extended=True)


@m5only
def test_refusal_D256_extended(monkeypatch):
    """2.64: D=256 is no longer refused by the (retired) opt-in; the default path serves it
    on the SDPA route (V6NAX sparse is D in {64, 128})."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    q, k, v = _qkv(1, 4, 4096, 256)
    mx.random.seed(7)
    bm = (mx.random.uniform(shape=(4096 // BT, 4096 // 16)) < 0.05) | mx.eye(4096 // BT, 4096 // 16, dtype=mx.bool_)
    o = flash_attention_sparse(q, k, v, bm)
    mx.eval(o)
    assert bool(mx.all(mx.isfinite(o)).item())


@m5only
def test_refusal_bt_not_32_extended(monkeypatch):
    """2.64: a BT=16 mask is no longer refused by the (retired) opt-in — it takes the
    default route (the tile is not NAX-viable -> SDPA fallback)."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    N, D = 4096, 128
    q, k, v = _qkv(1, 4, N, D)
    bm16 = _block_mask(N // 16, N // 16, 0.05)          # BT=16 granularity
    o = flash_attention_sparse(q, k, v, bm16)
    mx.eval(o)
    assert o.shape == (1, 4, N, D)


def test_refusal_fp32_entry(monkeypatch):
    """fp32 is refused at entry for ALL sparse calls (spec §1 relies on this)."""
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED", raising=False)
    q, k, v = _qkv(1, 4, 256, 128, dt=mx.float32)
    bm = _block_mask(256 // BT, 256 // BT, 0.1)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        flash_attention_sparse(q, k, v, bm)


@m5only
def test_refusals_are_extended_only(monkeypatch):
    """Lock: the BT!=32 refusal fires ONLY under the opt-in; off-path routes,
    no raise from the extended guard."""
    N, D = 4096, 128
    q, k, v = _qkv(1, 4, N, D)
    bm16 = _block_mask(N // 16, N // 16, 0.05)          # BT=16 granularity
    monkeypatch.delenv("MFA_SPARSE_NAX_EXTENDED", raising=False)
    o = flash_attention_sparse(q, k, v, bm16)           # must NOT raise
    mx.eval(o)
    assert o.shape == (1, 4, N, D)


@m5only
def test_d_dense_cutoff_routes_dense(monkeypatch):
    """2.64 B6: d >= cutoff -> the V6NAX kernel when it can serve the call (differs from the
    dense fallback, oracle-correct); cutoff raised above d -> the 0.30 density ceiling
    applies -> the dense (bool) fallback, byte-identical to it."""
    B, H, N, D = 1, 40, 4096, 128
    q, k, v = _qkv(B, H, N, D)
    bm = _block_mask(N // BT, N // BT, 0.9)          # symmetrized → ~0.99 ≥ 0.85
    mx.eval(bm)
    dens = float(mx.mean(bm.astype(mx.float32)).item())
    assert dens >= 0.85, f"test premise: near-dense mask, got {dens:.3f}"
    scale = 1.0 / math.sqrt(D)
    o_dense = A._sparse_fallback_sdpa_perhead(q, k, v, bm, scale, False)
    o_cut = flash_attention_sparse(q, k, v, bm, scale=scale)
    mx.eval(o_cut, o_dense)
    d_nax = float(np.abs(np.asarray(o_cut.astype(mx.float32))
                         - np.asarray(o_dense.astype(mx.float32))).max())
    assert 0.0 < d_nax < 1e-2, f"d>=cutoff must take the V6NAX kernel (B6); maxabs vs dense={d_nax}"
    monkeypatch.setenv("MFA_SPARSE_D_DENSE_CUTOFF", "1.01")
    o_ceil = flash_attention_sparse(q, k, v, bm, scale=scale)
    mx.eval(o_ceil)
    d_dense = float(np.abs(np.asarray(o_ceil.astype(mx.float32))
                           - np.asarray(o_dense.astype(mx.float32))).max())
    assert d_dense == 0.0, f"above the ceiling, below the cutoff -> dense fallback; got {d_dense}"


# ============================================ axis 1 — gold parity sample (gate 1)
@m5only
@pytest.mark.parametrize("D,dtype,causal,N", [
    (128, mx.float16, False, 4096),
    (128, mx.float16, True, 4096),
    (64, mx.float16, False, 4096),
    (128, mx.bfloat16, False, 4096),
    (64, mx.bfloat16, True, 4096),
    (128, mx.float16, False, 4100),          # non-aligned → auto_pad
])
def test_gold_parity_sample(monkeypatch, D, dtype, causal, N):
    """Sampled v1 matrix (D × dtype × causal × N aligned/non-aligned) vs fp32 gold."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    B, H = 1, 8
    q, k, v = _qkv(B, H, N, D, dt=dtype)
    nq = (N + BT - 1) // BT
    bm = _block_mask(nq, nq, 0.10)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    o = flash_attention_sparse(q, k, v, bm, scale=scale, causal=causal, auto_pad=True)
    mx.eval(o)
    assert_row_gates(o, _gold(q, k, v, bm, scale, causal, N, N),
                     **(_G16 if dtype == mx.float16 else _GBF16),
                     label=f"D{D} {dtype} causal={causal} N={N}")


# ============================================ axis 2 — engagement probe (gate 2)
@m5only
def test_engagement_v6nax_vs_scalar():
    """The kernel that serves the extended route is genuinely v6nax_sparse
    (byteΔ vs the scalar_fallback binary > 0), and it is correct vs gold."""
    from mlx_mfa import _ext
    B, H, N, D = 1, 8, 4096, 128
    q, k, v = _qkv(B, H, N, D)
    bm = _block_mask(N // BT, N // BT, 0.10)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    o_nax = _ext.sparse_attention_forward(q, k, v, bm, BT, False, scale, "v6nax_sparse", False, 0)
    o_sca = _ext.sparse_attention_forward(q, k, v, bm, BT, False, scale, "scalar_fallback", False, 0)
    mx.eval(o_nax, o_sca)
    delta = float(np.abs(np.asarray(o_nax.astype(mx.float32))
                         - np.asarray(o_sca.astype(mx.float32))).max())
    assert delta > 0.0, "v6nax must be a distinct binary from scalar_fallback (engagement)"
    assert_row_gates(o_nax, _gold(q, k, v, bm, scale, False, N, N), **_G16, label="v6nax raw")


# ============================================ axis 3 — edge cases
@m5only
def test_edge_all_false_mask(monkeypatch):
    """All-False mask → exact zero output (kernel v2.34.0 contract)."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    B, H, N, D = 1, 8, 4096, 128
    q, k, v = _qkv(B, H, N, D)
    bm = mx.zeros((N // BT, N // BT), dtype=mx.bool_)
    mx.eval(bm)
    o = flash_attention_sparse(q, k, v, bm, scale=1.0 / math.sqrt(D))
    mx.eval(o)
    m = float(np.abs(np.asarray(o.astype(mx.float32))).max())
    assert m == 0.0, f"all-False mask must give zero output; max={m}"


@m5only
def test_edge_all_active_mask(monkeypatch):
    """All-True mask via the NAX path == dense attention."""
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    monkeypatch.setenv("MFA_SPARSE_D_DENSE_CUTOFF", "1.01")   # force NAX even at d=1.0
    B, H, N, D = 1, 8, 2048, 128
    q, k, v = _qkv(B, H, N, D)
    bm = mx.ones((N // BT, N // BT), dtype=mx.bool_)
    mx.eval(bm)
    scale = 1.0 / math.sqrt(D)
    o = flash_attention_sparse(q, k, v, bm, scale=scale)
    with mx.stream(mx.cpu):
        ref = mx.fast.scaled_dot_product_attention(
            q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=scale)
        mx.eval(ref)
    assert_row_gates(o, ref, **_G16, label="all-active mask == dense attention")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))

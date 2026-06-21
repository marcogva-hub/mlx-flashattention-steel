"""Sparse/LCSA family correctness LOCK (audit Phase B1, 2026-06-17).

Validates the two sparse forward kernels against an INDEPENDENT manual fp32
oracle (NOT SDPA, NOT another mlx-mfa kernel — strict lesson #11), across all
edges, and locks them so a future kernel change that breaks any edge fails CI.

Which-binary (env-toggle fingerprinted, see phase-B1 report):
  - V2 matmul2d (`sparse_kernel_source_v2`, BaseNAXFrag::mma cooperative-tensor)
    is `decide_auto_version`'s default for the V2-capable head_dims D in {64,128}
    (audit Phase F retired the old qL*kL*D >= 2.147e9 work-product gate — Phase E
    measured V1 is never fastest). The fast win.
  - V1 scalar (`sparse_kernel_source`) is now only the genuine fallback (a head_dim
    V2 can't handle, e.g. D=256) — kept correct, never the default. ~40x slower.
Forced here via MFA_LCSA_KERNEL_VERSION to test each generator deterministically.

M5+-gated (the routes under test are M5-specific).
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa import flash_attention_sparse
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(),
    reason="sparse-family correctness lock asserts M5+ kernels",
)

mx.random.seed(0)
_TOL = 3e-2  # fp16 sparse vs fp32 oracle

# T2-1 (audit de-vacuity, 2026-06-21): these locks ran ONLY at 0.1 input scale —
# the toy regime that suppressed the II-6 fused-dKdV corruption class.  Every
# correctness cell now runs at BOTH 0.1 (kept) AND realistic unit scale
# (std≈1.0, normal), validated vs the SAME independent fp32 oracle.  Toy keeps
# the original ABSOLUTE bound; unit uses a scale-invariant RELATIVE bound (fp16
# sparse-attention rel-err is ≲1e-2).  A unit-scale failure is a BUG-DISCOVERY
# signal — investigate which-binary; do NOT loosen it without confirming the
# kernel matches an independent correct oracle.  All sparse cells here are N==S
# (band masks built from N//32), so the (S-N) causal-mask sign-flip latent in
# sibling locks is dormant — the `i >= j` formula below is correct for N==S.
_REL_TOL = 5e-2
_MAG = {"mode": "toy"}


def _gen(shape):
    if _MAG["mode"] == "unit":
        return mx.random.normal(shape).astype(mx.float16)         # std ≈ 1.0
    return (mx.random.uniform(-1, 1, shape) * 0.1).astype(mx.float16)


def _fp32_oracle(q, k, v, bm, scale, causal=False):
    """Independent manual fp32 attention (not SDPA, not the kernel)."""
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:  # GQA: repeat KV heads
        r = Hq // Hk
        kf = mx.repeat(kf, r, axis=1); vf = mx.repeat(vf, r, axis=1)
    s = (qf @ kf.swapaxes(-1, -2)) * scale
    N, S = q.shape[2], k.shape[2]
    NQ, NK = bm.shape[-2], bm.shape[-1]
    em = mx.repeat(mx.repeat(bm.astype(mx.float32), N // NQ, axis=-2), S // NK, axis=-1)
    while em.ndim < 4:
        em = em[None]
    s = mx.where(em > 0, s, mx.array(-1e30, mx.float32))
    if causal:
        cm = (mx.arange(N)[:, None] >= mx.arange(S)[None, :]).astype(mx.float32)
        s = mx.where(cm > 0, s, mx.array(-1e30, mx.float32))
    return mx.softmax(s, axis=-1) @ vf, em


def _qkv(B, H, N, D, Hk=None):
    Hk = Hk or H
    f = lambda h: _gen((B, h, N, D))
    q, k, v = f(H), f(Hk), f(Hk); mx.eval(q, k, v)
    return q, k, v


def _assert_correct(q, k, v, bm, scale, causal=False):
    o = flash_attention_sparse(q, k, v, bm, scale=scale, causal=causal)
    ref, em = _fp32_oracle(q, k, v, bm, scale, causal)
    active = (mx.sum(em, axis=-1) > 0).astype(mx.float32)[..., None]  # mask all-empty rows (oracle NaN)
    ref = mx.where(mx.isnan(ref), mx.array(0.0), ref)
    o32 = o.astype(mx.float32)
    assert bool(mx.all(mx.isfinite(o32)).item()), "non-finite output"
    d = float(mx.max(mx.abs((o32 - ref) * active)).item())
    if _MAG["mode"] == "unit":
        denom = float(mx.max(mx.abs(ref * active)).item()) + 1e-6
        rel = d / denom
        assert rel < _REL_TOL, f"unit-scale rel_err {rel:.3e} exceeds {_REL_TOL} (abs={d:.3e})"
    else:
        assert d < _TOL, f"toy-scale max_abs_err {d} exceeds {_TOL}"


def _band(NB, d):
    m = np.zeros((NB, NB), bool); m[:, :max(1, round(d * NB))] = True
    return mx.array(m)


# ── V2 matmul2d (forced; D=128 N=4096 → work ≥ 2.147e9) ──────────────────────
@pytest.fixture
def _force_v2(monkeypatch):
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v2")


@pytest.mark.usefixtures("_force_v2")
class TestV2Matmul2dCorrectness:
    B, H, N, D = 2, 8, 4096, 128
    SC = 1 / math.sqrt(128)

    # Run each cell below at BOTH input regimes (T2-1).
    @pytest.fixture(autouse=True, params=["toy", "unit"])
    def _regime(self, request):
        _MAG["mode"] = request.param
        yield
        _MAG["mode"] = "toy"

    def _qkv(self):
        return _qkv(self.B, self.H, self.N, self.D)

    def test_banded(self):
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 0.25), self.SC)

    def test_scattered(self):
        q, k, v = self._qkv()
        NB = self.N // 32; m = np.zeros((NB, NB), bool); m[:, ::4] = True
        _assert_correct(q, k, v, mx.array(m), self.SC)

    def test_density_full(self):
        # CC-17 (audit) — ROUTING-not-kernel cell: density=1.0 is ≥ the 0.78
        # ceiling, so this routes to SDPA (byteΔ=0), the intended dense route
        # (locked by test_fingerprint_discipline dense-symmetric→SDPA).  It
        # asserts the dense route's correctness vs the fp32 oracle; the sparse V2
        # kernel itself is engaged (byteΔ>0) by test_banded/_scattered/_density_min.
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 1.0), self.SC)

    def test_density_min(self):
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 1 / (self.N // 32)), self.SC)

    def test_all_masked_query_block(self):
        q, k, v = self._qkv()
        NB = self.N // 32; m = np.zeros((NB, NB), bool); m[1:, :8] = True  # row 0 fully empty
        _assert_correct(q, k, v, mx.array(m), self.SC)

    def test_causal(self):
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 0.5), self.SC, causal=True)

    def test_gqa(self):
        q, k, v = _qkv(self.B, self.H, self.N, self.D, Hk=2)
        _assert_correct(q, k, v, _band(self.N // 32, 0.25), self.SC)

    def test_mask_ndim3(self):
        q, k, v = self._qkv()
        NB = self.N // 32
        _assert_correct(q, k, v, mx.broadcast_to(_band(NB, 0.25)[None], (self.H, NB, NB)), self.SC)

    def test_mask_ndim4(self):
        q, k, v = self._qkv()
        NB = self.N // 32
        _assert_correct(q, k, v, mx.broadcast_to(_band(NB, 0.25)[None, None], (self.B, self.H, NB, NB)), self.SC)


# ── V1 scalar (forced; D=128 N=2048 → below threshold) ───────────────────────
@pytest.fixture
def _force_v1(monkeypatch):
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v1")


@pytest.mark.usefixtures("_force_v1")
class TestV1ScalarCorrectness:
    B, H, N, D = 2, 8, 2048, 128
    SC = 1 / math.sqrt(128)

    # Run each cell below at BOTH input regimes (T2-1).
    @pytest.fixture(autouse=True, params=["toy", "unit"])
    def _regime(self, request):
        _MAG["mode"] = request.param
        yield
        _MAG["mode"] = "toy"

    def _qkv(self):
        return _qkv(self.B, self.H, self.N, self.D)

    def test_banded(self):
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 0.25), self.SC)

    def test_scattered(self):
        q, k, v = self._qkv()
        NB = self.N // 32; m = np.zeros((NB, NB), bool); m[:, ::4] = True
        _assert_correct(q, k, v, mx.array(m), self.SC)

    def test_causal(self):
        q, k, v = self._qkv(); _assert_correct(q, k, v, _band(self.N // 32, 0.5), self.SC, causal=True)

    def test_all_masked_query_block(self):
        q, k, v = self._qkv()
        NB = self.N // 32; m = np.zeros((NB, NB), bool); m[1:, :8] = True
        _assert_correct(q, k, v, mx.array(m), self.SC)

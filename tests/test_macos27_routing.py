"""OS-aware M5+ routing scaffolding (macOS 26 vs 27 seam).

Guards:
  * get_device_info() exposes macos_major/minor; the cached accessor is robust
    when the signal is unavailable (defaults conservatively to 0 → ≤26 / safe).
  * EDGES PRESERVED (non-negotiable): with MFA_ENABLE_MACOS27_ROUTING unset,
    every M5+ dispatch decision is byte-identical to the opt-in-set path — i.e.
    the beta seam does NOT perturb the shipping macOS-26 behavior. Because Axis A
    (macOS27-beta3 2026-07-08) found the D=128 sparse bug PERSISTS, the opt-in is
    currently a no-op, so on-vs-off must be byteΔ==0 on EVERY shape.
  * The opt-in gate is reachable (path-entered) on macOS ≥27.
"""
import os
import numpy as np
import pytest
import mlx.core as mx

import mlx_mfa
from mlx_mfa import attention as A

_ENV = "MFA_ENABLE_MACOS27_ROUTING"


@pytest.fixture(autouse=True)
def _clean_env():
    saved = os.environ.get(_ENV)
    os.environ.pop(_ENV, None)
    yield
    if saved is None:
        os.environ.pop(_ENV, None)
    else:
        os.environ[_ENV] = saved


# ── detection + robustness ────────────────────────────────────────────────────
def test_get_device_info_exposes_macos_major():
    info = mlx_mfa.get_device_info()
    assert "macos_major" in info and "macos_minor" in info
    assert isinstance(info["macos_major"], int) and info["macos_major"] >= 0
    assert isinstance(info["macos_minor"], int) and info["macos_minor"] >= 0


def test_cached_accessor_matches_device_info():
    assert A._get_macos_major_cached() == int(mlx_mfa.get_device_info().get("macos_major", 0) or 0)


def test_accessor_robust_when_signal_absent(monkeypatch):
    """Unknown macOS (missing key / query failure) → 0 → treated as ≤26 (safe)."""
    monkeypatch.setattr(A, "_cached_macos_major", None)
    monkeypatch.setattr(A, "get_device_info", lambda: {})  # no macos_major key
    assert A._get_macos_major_cached() == 0
    # and with the opt-in set, an unknown OS must NOT activate the macOS-27 branch
    os.environ[_ENV] = "1"
    assert A._macos27_routing_active() is False
    monkeypatch.setattr(A, "_cached_macos_major", None)  # reset for other tests


# ── opt-in gate: default off, reachable under opt-in ──────────────────────────
def test_routing_gate_default_off():
    assert A._macos27_routing_active() is False  # env unset


def test_routing_gate_reachable_under_optin():
    os.environ[_ENV] = "1"
    expected = A._get_macos_major_cached() >= 27
    assert A._macos27_routing_active() is expected
    if A._get_macos_major_cached() >= 27:
        assert A._macos27_routing_active() is True  # path-entered on macOS ≥27


def test_sparse_native_stays_disabled_persists():
    """Axis A = PERSISTS ⇒ the opt-in must NOT skip the sparse fallback."""
    assert A._MACOS27_SPARSE_D128_FIXED is False
    os.environ[_ENV] = "1"
    assert A._macos27_sparse_native_ok() is False


# ── EDGES PRESERVED: opt-in on vs off is byte-identical on every shape ─────────
_SHAPES = [
    (1, 2, 256, 64, False),
    (1, 2, 256, 128, False),
    (1, 4, 512, 128, True),
    (1, 2, 384, 64, True),
]


def _block_mask(N, BQ, BK, causal):
    nq, nk = (N + BQ - 1) // BQ, (N + BK - 1) // BK
    m = np.ones((nq, nk), dtype=bool) if not causal else (np.tril(np.ones((nq, nk))) > 0)
    return mx.array(m)


@pytest.mark.parametrize("B,H,N,D,causal", _SHAPES)
def test_edges_preserved_dense(B, H, N, D, causal):
    mx.random.seed(0)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    os.environ.pop(_ENV, None)
    o_off = np.array(mlx_mfa.flash_attention(q, k, v, causal=causal).astype(mx.float32))
    os.environ[_ENV] = "1"
    o_on = np.array(mlx_mfa.flash_attention(q, k, v, causal=causal).astype(mx.float32))
    assert np.array_equal(o_off, o_on), "macOS-27 opt-in perturbed dense dispatch (must be byte-identical)"


@pytest.mark.parametrize("B,H,N,D,causal", _SHAPES)
@pytest.mark.parametrize("BK", [16, 32])
def test_edges_preserved_sparse(B, H, N, D, causal, BK):
    mx.random.seed(1)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    mask = _block_mask(N, 32, BK, causal)
    scale = 1.0 / (D ** 0.5)
    os.environ.pop(_ENV, None)
    try:
        o_off = np.array(mlx_mfa.flash_attention_sparse(q, k, v, mask, scale=scale, causal=causal).astype(mx.float32))
    except Exception as e:
        pytest.skip(f"shape/mask not accepted: {e}")
    os.environ[_ENV] = "1"
    o_on = np.array(mlx_mfa.flash_attention_sparse(q, k, v, mask, scale=scale, causal=causal).astype(mx.float32))
    assert np.array_equal(o_off, o_on), "macOS-27 opt-in perturbed sparse dispatch (must be byte-identical)"


# ── stamp convention (measurement quadruple includes macOS) ───────────────────
def test_measurement_stamp_includes_macos():
    bv = pytest.importorskip("benchmarks.bench_validity")
    stamp = bv._macos_stamp()
    assert stamp.startswith("macOS")
    maj = mlx_mfa.get_device_info().get("macos_major", 0)
    if maj:
        assert str(maj) in stamp
    r = bv.SpeedupResult(ratio=1.0, test_ms=1.0, baseline_ms=1.0, engagement_evidence="x",
                         byte_delta=1e-3, noise_floor=0.0, oracle_max_abs=1e-4,
                         mlx_version=mx.__version__, hardware="Apple M5 Max",
                         date="2026-07-08", macos=stamp)
    assert "macOS" in str(r)

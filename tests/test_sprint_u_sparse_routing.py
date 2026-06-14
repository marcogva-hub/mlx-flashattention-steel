"""Sprint U Section B — flash_attention_sparse M5+ auto-routing validation.

Three-axis validation per CLAUDE_V6_NAX.md §3.5:
  1. Output sanity: M5+ auto-routed output is correct vs reference
  2. Path entered: sparse_attention_dispatch IS called on M5+ branch
  3. Edges preserved:
     - MFA_DISABLE_AUTO_HOOKS=1 restores pre-Sprint-U fallback
     - M1-M4 path unchanged (no regression)
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from unittest.mock import patch as _mock_patch

import numpy as np
import pytest

import mlx.core as mx


try:
    from mlx_mfa import flash_attention_sparse
    from mlx_mfa.attention import get_device_info
    _HAS_EXT = True
except (ImportError, RuntimeError):
    _HAS_EXT = False

# Skip suite if not M5+ (the auto-routing only takes effect on M5+)
_M5_PLUS = False
if _HAS_EXT:
    try:
        _M5_PLUS = bool(get_device_info().get("is_m5_plus"))
    except Exception:
        _M5_PLUS = False

pytestmark = [
    pytest.mark.skipif(not _HAS_EXT, reason="mlx_mfa not installed"),
    pytest.mark.skipif(not _M5_PLUS, reason="Sprint U auto-routing is M5+ only"),
]


@contextmanager
def env_var(name, value):
    """Set env var temporarily."""
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def _make_inputs(B=1, H=4, qL=4096, kL=4096, D=128, seed=0, mag=1.0):
    # III-4 F7: unit-scale inputs (normal, std 1.0; was uniform*0.1).
    # The II-6 lesson: 0.1-scale fixtures dilute localized kernel
    # corruption below the rmse gates.
    mx.random.seed(seed)
    Q = (mx.random.normal((B, H, qL, D)) * mag).astype(mx.float16)
    K = (mx.random.normal((B, H, kL, D)) * mag).astype(mx.float16)
    V = (mx.random.normal((B, H, kL, D)) * mag).astype(mx.float16)
    mx.async_eval(Q, K, V); mx.synchronize()
    return Q, K, V


def _bool_mask_2d(NQ, NK, density, seed):
    rng = np.random.default_rng(seed)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    # Guarantee no all-False rows
    for q in range(NQ):
        if not bm[q].any():
            bm[q, q % NK] = True
    return mx.array(bm)


# ----- AXIS 1: output sanity -----

def test_axis1_m5plus_auto_routed_output_correct():
    """Output sanity: auto-routed M5+ output matches reference SDPA+bias."""
    B, H, qL, kL, D, BT = 1, 4, 4096, 4096, 128, 16
    Q, K, V = _make_inputs(B, H, qL, kL, D, seed=10)
    mask = _bool_mask_2d(qL // BT, kL // BT, density=0.1, seed=11)

    # Auto-routed (Sprint U default)
    O_auto = flash_attention_sparse(Q, K, V, mask)
    mx.async_eval(O_auto); mx.synchronize()

    # Reference: SDPA with dense -inf bias
    full = np.repeat(np.repeat(np.array(mask), BT, axis=0), BT, axis=1)
    bias = np.where(full, 0.0, -np.inf).astype(np.float16)
    bias_mx = mx.array(bias)
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0 / math.sqrt(D), mask=bias_mx)
    mx.async_eval(O_ref); mx.synchronize()

    err = np.abs(np.array(O_auto.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 5e-3, f"M5+ auto-routed RMSE {rmse} too high vs SDPA+bias"
    assert not np.isnan(np.array(O_auto.astype(mx.float32))).any()


# ----- AXIS 2: path entered -----

def test_axis2_m5plus_calls_sparse_attention_dispatch():
    """Path entered: confirm sparse_attention_dispatch is invoked on M5+."""
    B, H, qL, kL, D, BT = 1, 4, 4096, 4096, 128, 16
    Q, K, V = _make_inputs(B, H, qL, kL, D, seed=20)
    mask = _bool_mask_2d(qL // BT, kL // BT, density=0.1, seed=21)

    # Capture _real BEFORE patching so we don't pick up the mock as side_effect
    from mlx_mfa.lcsa_nax import sparse_attention_dispatch as _real

    with _mock_patch(
        "mlx_mfa.lcsa_nax.sparse_attention_dispatch", side_effect=_real
    ) as m:
        _ = flash_attention_sparse(Q, K, V, mask)
        assert m.called, "sparse_attention_dispatch not called on M5+ path"


# ----- AXIS 3: edges preserved -----

def test_axis3_env_disable_restores_pre_sprint_u_fallback():
    """MFA_DISABLE_AUTO_HOOKS=1 routes via _sparse_fallback_sdpa_perhead.

    Uses STEEL-shape mask (BQ=32, BK=16 for D=128) which was the v2.35.0
    accepted format. Symmetric BT masks weren't supported pre-Sprint-U.
    """
    B, H, qL, kL, D = 1, 4, 4096, 4096, 128
    BQ, BK = 32, 16  # STEEL D=128 config
    Q, K, V = _make_inputs(B, H, qL, kL, D, seed=30)
    mask = _bool_mask_2d(qL // BQ, kL // BK, density=0.1, seed=31)

    # Capture _real BEFORE patching
    from mlx_mfa.attention import _sparse_fallback_sdpa_perhead as _real

    with env_var("MFA_DISABLE_AUTO_HOOKS", "1"):
        with _mock_patch(
            "mlx_mfa.attention._sparse_fallback_sdpa_perhead", side_effect=_real
        ) as m:
            _ = flash_attention_sparse(Q, K, V, mask)
            assert m.called, (
                "MFA_DISABLE_AUTO_HOOKS=1 should route to "
                "_sparse_fallback_sdpa_perhead (pre-Sprint-U behavior)"
            )


def test_axis3_low_density_routes_to_nax():
    """At low density (< DEFAULT_DENSITY_THRESHOLD), dispatcher routes to
    sparse_attention_nax (NAX-aware path) rather than SDPA fallback.

    Verified by checking the dispatcher's routing decision via the density
    parameter on a very-sparse mask.
    """
    B, H, qL, kL, D, BT = 1, 4, 4096, 4096, 128, 16
    Q, K, V = _make_inputs(B, H, qL, kL, D, seed=40)
    mask = _bool_mask_2d(qL // BT, kL // BT, density=0.005, seed=41)

    from mlx_mfa.lcsa_nax import sparse_attention_nax, DEFAULT_DENSITY_THRESHOLD
    density_actual = float(mx.mean(mask.astype(mx.float32)))
    assert density_actual < DEFAULT_DENSITY_THRESHOLD, \
        f"Test invariant: density {density_actual} should be < threshold"

    # sparse_attention_nax is already imported above (captured before patch)
    with _mock_patch(
        "mlx_mfa.lcsa_nax.sparse_attention_nax", side_effect=sparse_attention_nax
    ) as m_nax:
        _ = flash_attention_sparse(Q, K, V, mask)
        assert m_nax.called, (
            f"At density {density_actual} < {DEFAULT_DENSITY_THRESHOLD}, "
            "dispatcher should route to sparse_attention_nax (NAX path)"
        )


def test_adversarial_magnitude_finite():
    """III-4 F7: adversarial-magnitude (std 8) inputs must stay finite
    through the auto-routed flash_attention_sparse path."""
    B, H, qL, kL, D, BT = 1, 4, 4096, 4096, 128, 16
    Q, K, V = _make_inputs(B, H, qL, kL, D, seed=50, mag=8.0)
    mask = _bool_mask_2d(qL // BT, kL // BT, density=0.1, seed=51)
    O = flash_attention_sparse(Q, K, V, mask)
    mx.async_eval(O); mx.synchronize()
    assert np.isfinite(np.array(O.astype(mx.float32))).all(), \
        "non-finite auto-routed sparse output at std-8 inputs"

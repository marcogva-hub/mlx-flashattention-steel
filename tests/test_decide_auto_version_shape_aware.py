"""Three-axis tests for decide_auto_version() routing.

Updated by audit Phase F (2026-06-18): the v2.36.1 `qL*kL*D >= 2^31`
work-product threshold is RETIRED. Phase E measured the scalar fallback is
never fastest (V6NAX sparse 19-59x faster than scalar; 1.5-3.9x faster than
SDPA at low density), so routing is now by V6NAX-sparse capability (head_dim):
D in {64,128} -> "v2" (legacy public alias for V6NAX sparse; the C++ falls
v2->v1 internally when V6NAX sparse is ineligible: causal / block_tile!=32),
any other D -> "v1" (legacy public alias for scalar fallback).

Per CLAUDE_V6_NAX.md §3.5 three-axis rule:
  Axis 1 (output sanity)   - V6NAX/scalar routed dispatch produces correct output
  Axis 2 (path entered)    - decide_auto_version returns expected version
  Axis 3 (edges preserved) - env overrides preserved, D-fallback preserved
"""
from __future__ import annotations
import math
import os

import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa.lcsa_nax import (
    decide_auto_version,
    sparse_attention_nax,
    _bool_mask_to_float_bias,
)


_AE = getattr(mx, "async_" + "eval")


def _materialize(*arrays):
    _AE(*arrays)
    mx.synchronize()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with no MFA_LCSA_KERNEL_VERSION env."""
    monkeypatch.delenv("MFA_LCSA_KERNEL_VERSION", raising=False)
    yield


def _make_inputs(B, Hq, Hk, qL, kL, D, density, BT, seed):
    mx.random.seed(seed)
    Q = (mx.random.uniform(-1.0, 1.0, (B, Hq, qL, D)) * 0.1).astype(mx.float16)
    K = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    V = (mx.random.uniform(-1.0, 1.0, (B, Hk, kL, D)) * 0.1).astype(mx.float16)
    NQ, NK = qL // BT, kL // BT
    rng = np.random.default_rng(seed + 1)
    bm = (rng.random((NQ, NK)) < density).astype(np.bool_)
    for q in range(NQ):
        if not bm[q].any():
            bm[q, min(q, NK - 1)] = True
    mask = mx.array(bm)
    bias = _bool_mask_to_float_bias(mask, BT, qL, kL, mx.float16)
    _materialize(Q, K, V, mask, bias)
    return Q, K, V, mask, bias


# ---------------------------------------------------------------------------
# Axis 2: path entered (decide_auto_version returns expected version)
# ---------------------------------------------------------------------------

def test_decide_auto_version_large_seq16k_returns_v2():
    """Axis 2: large D=128 shape routes to V6NAX sparse via "v2" alias."""
    assert decide_auto_version(0.10, 16384, 16384, 128) == "v2"


def test_decide_auto_version_mid_seq8k_returns_v2():
    """Axis 2: mid D=128 shape routes to V6NAX sparse via "v2" alias."""
    assert decide_auto_version(0.10, 8192, 8192, 128) == "v2"


def test_decide_auto_version_smallest_tested_returns_v2():
    """Axis 2: D=128 N=4096 routes to V6NAX sparse via "v2" alias."""
    assert decide_auto_version(0.10, 4096, 4096, 128) == "v2"


def test_decide_auto_version_small_d128_now_routes_v2():
    """Axis 2 (Phase F): D=128 N=2048 — formerly < the 2^31 threshold and
    routed to slow scalar fallback (Phase E: 19.5x loss). Now routes to V6NAX."""
    assert decide_auto_version(0.10, 2048, 2048, 128) == "v2"


def test_decide_auto_version_d64_now_routes_v2():
    """Axis 2 (Phase F): D=64 N=4096 — work_product=1.07e9 always sat below
    the retired 2^31 threshold, routing every D=64 sparse call to slow scalar
    fallback (Phase E: 9x loss). Now routes to V6NAX sparse."""
    assert decide_auto_version(0.10, 4096, 4096, 64) == "v2"
    # Small D=64 too — V6NAX sparse is capable at every D=64 shape.
    assert decide_auto_version(0.10, 512, 512, 64) == "v2"


def test_decide_auto_version_non_v2_head_dim_falls_to_v1():
    """Axis 2: unsupported head_dim routes to scalar fallback via "v1" alias."""
    assert decide_auto_version(0.10, 4096, 4096, 256) == "v1"


# ---------------------------------------------------------------------------
# Axis 3: edges preserved (env overrides, density passthrough)
# ---------------------------------------------------------------------------

def test_env_v1_override_forces_v1_even_for_large_shape(monkeypatch):
    """Axis 3: MFA_LCSA_KERNEL_VERSION=v1 wins unconditionally."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v1")
    assert decide_auto_version(0.10, 16384, 16384, 128) == "v1"


def test_env_v2_override_forces_v2_even_for_tiny_shape(monkeypatch):
    """Axis 3: MFA_LCSA_KERNEL_VERSION=v2 wins unconditionally,
    even for shapes below the old default threshold."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v2")
    assert decide_auto_version(0.10, 128, 128, 64) == "v2"


def test_env_garbage_value_falls_through_to_shape_aware(monkeypatch):
    """Axis 3: unrecognised env value (anything besides v1/v2) falls
    through to the shape-aware default."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "garbage")
    assert decide_auto_version(0.10, 16384, 16384, 128) == "v2"
    # Phase F: small D=128 now also routes V6NAX sparse (capability, not work-product).
    assert decide_auto_version(0.10, 1024, 1024, 128) == "v2"
    # A non-V6NAX head_dim still falls to scalar under the shape-aware default.
    assert decide_auto_version(0.10, 1024, 1024, 256) == "v1"


def test_env_empty_string_falls_through_to_shape_aware(monkeypatch):
    """Axis 3: empty env value behaves like unset."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "")
    assert decide_auto_version(0.10, 16384, 16384, 128) == "v2"


def test_env_uppercase_v2_normalised(monkeypatch):
    """Axis 3: env value comparison is case-insensitive (per .lower())."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "V2")
    # Uppercase env should still force V6NAX sparse via the legacy alias.
    assert decide_auto_version(0.10, 128, 128, 64) == "v2"


def test_env_canonical_v6nax_sparse_alias_maps_to_v2(monkeypatch):
    """Axis 3: new canonical env names map to legacy public return aliases."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v6nax_sparse")
    assert decide_auto_version(0.10, 128, 128, 64) == "v2"


def test_env_canonical_scalar_alias_maps_to_v1(monkeypatch):
    """Axis 3: scalar_fallback env name remains compatible with the public API."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "scalar_fallback")
    assert decide_auto_version(0.10, 16384, 16384, 128) == "v1"


# ---------------------------------------------------------------------------
# Axis 1: output sanity (V6NAX-routed dispatch produces correct output)
# ---------------------------------------------------------------------------

def test_v2_default_shape_produces_correct_output():
    """Axis 1: large shape auto-routes to V6NAX and matches SDPA+bias reference."""
    Q, K, V, mask, bias = _make_inputs(
        B=1, Hq=4, Hk=4, qL=4096, kL=4096, D=128,
        density=0.10, BT=32, seed=12345)
    # No env override - shape-aware default kicks in (-> v2 alias -> V6NAX sparse)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=32)
    _materialize(O)
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0 / math.sqrt(128), mask=bias)
    _materialize(O_ref)
    err = np.abs(np.array(O.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 1e-3, f"RMSE too high: {rmse:.4e}"


def test_v1_override_shape_produces_correct_output(monkeypatch):
    """Axis 1: env-forced scalar fallback on a large shape remains correct."""
    monkeypatch.setenv("MFA_LCSA_KERNEL_VERSION", "v1")
    Q, K, V, mask, bias = _make_inputs(
        B=1, Hq=4, Hk=4, qL=4096, kL=4096, D=128,
        density=0.10, BT=32, seed=54321)
    O = sparse_attention_nax(Q, K, V, mask, block_tile=32)
    _materialize(O)
    O_ref = mx.fast.scaled_dot_product_attention(
        Q, K, V, scale=1.0 / math.sqrt(128), mask=bias)
    _materialize(O_ref)
    err = np.abs(np.array(O.astype(mx.float32)) -
                 np.array(O_ref.astype(mx.float32)))
    rmse = float(np.sqrt((err ** 2).mean()))
    assert rmse < 1e-3, f"RMSE too high: {rmse:.4e}"

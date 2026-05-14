"""v2.50 Prompt 5f Phase E — KD-5 disposition test.

MFA_FORCE_NATIVE_BWD=1 routes through legacy STEEL backward kernels
(KD-5 zeroed-blocks bug at D=128 N>=2048).  V34 backward NAX-direct
is the production path; STEEL backward is research-only.  This env
var is deprecated as of v2.50.0 and will be removed in v2.51+.

Test verifies the DeprecationWarning fires on MFA_FORCE_NATIVE_BWD=1.
"""
from __future__ import annotations

import warnings

import mlx.core as mx
import pytest

from mlx_mfa.dispatch_policy import should_use_native_backward


def test_force_native_bwd_emits_deprecation_warning(monkeypatch):
    monkeypatch.setenv("MFA_FORCE_NATIVE_BWD", "1")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", DeprecationWarning)
        # Eligible shape: D=64, causal, fp16
        result = should_use_native_backward(
            head_dim=64, seq_len=4096, causal=True, dtype=mx.float16)
        assert result is True  # supported shape returns True under force=1

    deprecation_warns = [
        w for w in w_list if issubclass(w.category, DeprecationWarning)
        and "MFA_FORCE_NATIVE_BWD" in str(w.message)
    ]
    assert len(deprecation_warns) >= 1, (
        f"Expected DeprecationWarning for MFA_FORCE_NATIVE_BWD=1; "
        f"got warnings: {[str(w.message) for w in w_list]}")


def test_force_native_bwd_zero_does_not_warn(monkeypatch):
    """MFA_FORCE_NATIVE_BWD=0 (explicit opt-OUT) should NOT emit deprecation."""
    monkeypatch.setenv("MFA_FORCE_NATIVE_BWD", "0")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", DeprecationWarning)
        result = should_use_native_backward(
            head_dim=64, seq_len=4096, causal=True, dtype=mx.float16)
        assert result is False  # explicit opt-out

    deprecation_warns = [
        w for w in w_list if issubclass(w.category, DeprecationWarning)
        and "MFA_FORCE_NATIVE_BWD" in str(w.message)
    ]
    assert len(deprecation_warns) == 0, (
        f"MFA_FORCE_NATIVE_BWD=0 should not warn; got {len(deprecation_warns)}")


def test_no_force_env_does_not_warn(monkeypatch):
    """Default path (no env var) should NOT emit deprecation."""
    monkeypatch.delenv("MFA_FORCE_NATIVE_BWD", raising=False)
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", DeprecationWarning)
        should_use_native_backward(
            head_dim=64, seq_len=4096, causal=True, dtype=mx.float16)

    deprecation_warns = [
        w for w in w_list if issubclass(w.category, DeprecationWarning)
        and "MFA_FORCE_NATIVE_BWD" in str(w.message)
    ]
    assert len(deprecation_warns) == 0

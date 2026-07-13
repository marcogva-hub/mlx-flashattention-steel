"""Uniform 0/1 semantics for public MFA boolean environment knobs."""

from __future__ import annotations

import warnings

import pytest

from mlx_mfa import _knobs


_PUBLIC_SAMPLES = (
    "MFA_ENABLE_VARLEN_NAX",
    "MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE",
    "MFA_ENABLE_V6_BACKWARD",
    "MFA_V6_BWD_SPARSE_NATIVE",
    "MFA_DISABLE_GNA_NATIVE",
)


@pytest.mark.parametrize("name", _PUBLIC_SAMPLES)
def test_boolean_knob_triplet(name, monkeypatch):
    monkeypatch.delenv(name, raising=False)
    assert _knobs.get_bool_env(name) is False
    monkeypatch.setenv(name, "1")
    assert _knobs.get_bool_env(name) is True
    monkeypatch.setenv(name, "0")
    assert _knobs.get_bool_env(name) is False
    for invalid in ("2", "abc", ""):
        monkeypatch.setenv(name, invalid)
        with pytest.raises(ValueError, match="must be '0' or '1'"):
            _knobs.get_bool_env(name)


def test_validate_env_reports_invalid_known_boolean(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_VARLEN_NAX", "abc")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        invalid = _knobs.validate_env(strict=True)
    assert "MFA_ENABLE_VARLEN_NAX" in invalid
    assert any("expected exactly '0' or '1'" in str(w.message) for w in caught)


def test_validate_env_reports_invalid_deprecated_boolean_alias(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_V34_BACKWARD", "abc")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        invalid = _knobs.validate_env(strict=True)
    assert "MFA_ENABLE_V34_BACKWARD" in invalid
    assert any("expected exactly '0' or '1'" in str(w.message) for w in caught)


def test_gna_pr1_invalid_value_is_rejected_by_expert(monkeypatch):
    import mlx.core as mx
    from mlx_mfa import _ext

    if not bool(_ext.get_device_info().get("is_m5_plus")):
        pytest.skip("M5+ NAX required")
    q = mx.zeros((1, 1, 64, 64), dtype=mx.float16)
    monkeypatch.setenv("MFA_GNA_NAX_PRECOMPUTE_RANGE", "2")
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        _ext.mfa_gna_nax_forward(
            q, q, q, 1, 8, 8, 1, 3, 3, 1, 1, 1, 64**-0.5
        )

"""Locks for V6 packed-varlen source transformation contracts."""

import pytest

from mlx_mfa import _ext


@pytest.mark.skipif(
    not hasattr(_ext, "v6_varlen_source_guard_probe"),
    reason="MFA_BUILD_PROBES extension required for source-transform probe",
)
def test_varlen_source_guard_rejects_ambiguous_marker():
    """A duplicated marker must fail loudly instead of replacing both sites."""
    with pytest.raises(RuntimeError, match=r"exactly one occurrence.*got 2"):
        _ext.v6_varlen_source_guard_probe("MARKER\nMARKER\n", "MARKER")


@pytest.mark.skipif(
    not hasattr(_ext, "v6_varlen_source_guard_probe"),
    reason="MFA_BUILD_PROBES extension required for source-transform probe",
)
def test_varlen_source_guard_accepts_unique_marker():
    assert (
        _ext.v6_varlen_source_guard_probe("prefix MARKER suffix", "MARKER")
        == "prefix __MFA_VARLEN_MARKER_REPLACED__ suffix"
    )

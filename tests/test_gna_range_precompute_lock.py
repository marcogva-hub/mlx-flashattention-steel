"""Locks for the GNA NAX per-Q-tile window-range precompute."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _ext


_M5_PLUS = bool(_ext.get_device_info().get("is_m5_plus"))


@pytest.mark.skipif(not _M5_PLUS, reason="M5+ NAX required")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("window", [(1, 7, 7), (3, 11, 11)])
def test_range_precompute_is_byte_identical(monkeypatch, dtype, window):
    seq_shape = (4, 8, 8)
    n = int(np.prod(seq_shape))
    mx.random.seed(715 + window[0])
    q = mx.random.normal((1, 2, n, 128)).astype(dtype)
    k = mx.random.normal((1, 2, n, 128)).astype(dtype)
    v = mx.random.normal((1, 2, n, 128)).astype(dtype)
    scale = 128**-0.5

    monkeypatch.delenv("MFA_GNA_NAX_PRECOMPUTE_RANGE", raising=False)
    baseline = _ext.mfa_gna_nax_forward(
        q, k, v, *seq_shape, *window, 1, 1, 1, scale
    )
    mx.eval(baseline)
    monkeypatch.setenv("MFA_GNA_NAX_PRECOMPUTE_RANGE", "1")
    candidate = _ext.mfa_gna_nax_forward(
        q, k, v, *seq_shape, *window, 1, 1, 1, scale
    )
    mx.eval(candidate)

    np.testing.assert_array_equal(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(baseline.astype(mx.float32)),
    )

"""Correctness and conservation locks for the GNA residency probes."""

import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import _ext


_M5_PLUS = bool(_ext.get_device_info().get("is_m5_plus"))


@pytest.mark.skipif(not _M5_PLUS, reason="M5+ NAX required")
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("D", [64, 128])
def test_gna_swizzle_is_byte_identical_to_default(monkeypatch, dtype, D):
    seq_shape = (1, 8, 8)
    window = (1, 3, 3)
    stride = (1, 1, 1)
    n = int(np.prod(seq_shape))
    scale = D ** -0.5
    key = mx.random.key(8110 + D)
    q = mx.random.normal((1, 1, n, D), key=key).astype(dtype)
    k = mx.random.normal((1, 1, n, D), key=mx.random.split(key)[0]).astype(dtype)
    v = mx.random.normal((1, 1, n, D), key=mx.random.split(key)[1]).astype(dtype)

    monkeypatch.delenv("MFA_GNA_NAX_SWIZZLE_LOG", raising=False)
    baseline = _ext.mfa_gna_nax_forward(
        q, k, v, *seq_shape, *window, *stride, scale
    )
    mx.eval(baseline)

    monkeypatch.setenv("MFA_GNA_NAX_SWIZZLE_LOG", "1")
    candidate = _ext.mfa_gna_nax_forward(
        q, k, v, *seq_shape, *window, *stride, scale
    )
    mx.eval(candidate)

    np.testing.assert_array_equal(
        np.asarray(candidate.astype(mx.float32)),
        np.asarray(baseline.astype(mx.float32)),
    )

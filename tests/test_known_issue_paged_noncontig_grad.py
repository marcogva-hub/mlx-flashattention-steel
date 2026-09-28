"""KNOWN ISSUE (found 2026-09-28 during the 2.62.2 remediation; fix planned for the next
release) — `flash_attention_paged` with NON-contiguous page pools (e.g. a transposed view)
returns a correct forward but WRONG gradients (dQ max|Δ| ≈ 3.1 causal / 1.45 non-causal vs
the dense path); `mx.contiguous(pool)` gives the right gradient.  Suspected [DEDUCED]: the
backward's page gather reads the raw buffer without a contiguity check.

strict xfail: this test documents the bug and FAILS the suite (XPASS) the moment it is
fixed, so the marker cannot outlive the bug.  The contiguous control must pass today.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, flash_attention_paged, is_mfa_available

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

B, H, N, D, BS, KV = 1, 2, 24, 64, 16, 16


def _inputs():
    mx.random.seed(9)
    q, k, v = (mx.random.normal((B, H, L, D)).astype(mx.float16) for L in (N, KV, KV))
    mx.random.seed(10)
    g = mx.random.normal((B, H, N, D)).astype(mx.float32)
    return q, k, v, g


def _dq_err(causal: bool, contiguous: bool) -> float:
    q, k, v, g = _inputs()
    kp, vp = (t[0].transpose(1, 0, 2).reshape(1, BS, H, D) for t in (k, v))   # non-contiguous views
    if contiguous:
        kp, vp = mx.contiguous(kp), mx.contiguous(vp)
    bt, sl = mx.array([[0]], dtype=mx.int32), mx.array([KV], dtype=mx.int32)
    dp = mx.grad(lambda a: (flash_attention_paged(a, kp, vp, bt, sl, causal=causal, block_size=BS)
                            .astype(mx.float32) * g).sum())(q)
    df = mx.grad(lambda a: (flash_attention(a, k, v, causal=causal).astype(mx.float32) * g).sum())(q)
    mx.eval(dp, df)
    return float(mx.max(mx.abs(dp.astype(mx.float32) - df.astype(mx.float32))))


@pytest.mark.parametrize("causal", [False, True])
def test_contiguous_pools_control(causal):
    assert _dq_err(causal, contiguous=True) < 1e-2


@pytest.mark.xfail(strict=True, reason="KNOWN ISSUE (next release): paged backward with "
                                       "non-contiguous page pools returns wrong gradients")
@pytest.mark.parametrize("causal", [False, True])
def test_noncontiguous_pools_gradient(causal):
    assert _dq_err(causal, contiguous=False) < 1e-2

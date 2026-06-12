"""Sprint II-3 artifact lock: the DECLINED streaming top-K kernel stays
correct (exact top-K score-set vs materialized reference) so the measured
negative result remains reproducible.  See mlx_mfa/topk_stream.py verdict."""
import math
import mlx.core as mx
import numpy as np
import pytest
from mlx_mfa import get_device_info
from mlx_mfa.topk_stream import topk_stream_indices

_HAS_NAX = bool(get_device_info().get("is_m5_plus", False))


@pytest.mark.skipif(not _HAS_NAX, reason="M5+ kernel")
@pytest.mark.parametrize("B,H,N,S,D,K", [
    (1, 2, 256, 256, 128, 64),
    (1, 2, 128, 320, 128, 64),   # N != S
    (2, 3, 256, 512, 64, 32),    # D=64
    (1, 2, 250, 250, 128, 64),   # ragged edges
])
def test_topk_stream_exact_score_sets(B, H, N, S, D, K):
    mx.random.seed(31)
    q = (mx.random.normal((B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((B, H, S, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k)
    s = 1.0 / math.sqrt(D)
    idx = topk_stream_indices(q, k, s, K)
    mx.eval(idx)
    ref = np.asarray(((q @ k.swapaxes(-1, -2)) * s).astype(mx.float32))
    got = np.asarray(idx)
    worst = 0.0
    for b in range(B):
        for h in range(H):
            for n in range(N):
                sel = np.sort(ref[b, h, n][got[b, h, n]])[::-1]
                top = np.sort(ref[b, h, n])[::-1][:K]
                worst = max(worst, float(np.abs(sel - top).max()))
    assert worst < 2e-3, f"top-K score-set diverged: {worst}"

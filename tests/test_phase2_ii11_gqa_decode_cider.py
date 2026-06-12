"""Sprint II-11 — cider-ported GQA decode kernel: correctness locks.

Expert API only (no auto dispatch — see sprint-II-11-report.md for the
declined promotion and the measured 1.06-1.17x @ GQA>=8/S=32K window)."""
import math

import mlx.core as mx
import pytest

from mlx_mfa import get_device_info

_HAS_M5 = bool(get_device_info().get("is_m5_plus", False))
pytestmark = pytest.mark.skipif(not _HAS_M5, reason="decode bench target is M5+")


@pytest.mark.parametrize("Hq,Hkv,S,dt", [
    (32, 8, 4096, mx.float16),
    (32, 4, 1024, mx.float16),
    (32, 32, 2048, mx.float16),   # MHA degenerate
    (32, 1, 2048, mx.float16),    # MQA
    (16, 2, 16384, mx.bfloat16),
])
def test_matches_sdpa(Hq, Hkv, S, dt):
    from mlx_mfa.gqa_decode_cider import gqa_decode_cider
    mx.random.seed(3)
    D = 128
    q = mx.random.normal((1, Hq, 1, D)).astype(dt)
    k = mx.random.normal((1, Hkv, S, D)).astype(dt)
    v = mx.random.normal((1, Hkv, S, D)).astype(dt)
    mx.eval(q, k, v)
    s = 1.0 / math.sqrt(D)
    o = gqa_decode_cider(q, k, v, s)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=s)
    mx.eval(o, ref)
    err = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))).item())
    assert err < 5e-3, f"cider decode diverged: {err}"


def test_rejects_prefill():
    from mlx_mfa.gqa_decode_cider import gqa_decode_cider
    q = mx.zeros((1, 8, 4, 128), dtype=mx.float16)
    k = mx.zeros((1, 8, 64, 128), dtype=mx.float16)
    with pytest.raises(ValueError, match="decode only"):
        gqa_decode_cider(q, k, k)

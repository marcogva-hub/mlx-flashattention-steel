"""attn_bias backward (pre-merge review 2026-09-28, P0 published; fixed in 2.62.2).

The native `attn_bias` route (bias modes 1/2) returned the raw `mfa_attention_bias_forward`
primitive: no custom vjp, so `mx.grad` used `MFAttention::vjp` (STEEL backward), which
never receives the bias and masks top-left — NaN or wrong gradients even for an all-zero
bias.  It was the only public route returning an unwrapped `MFAttention` output (every
other creator — ALiBi, rope, `_make_mfa_custom*` — sits inside an `mx.custom_function`;
the other primitives implement no vjp and raise loudly).

Fix (ALiBi pattern): keep the native forward, differentiate the SDPA route the bias
modes 0/3 already use.  The bias itself may be trainable (relative-position biases), so
its gradient is returned too, not zeros.

Locks: dQ/dK/dV AND d_bias vs a CPU fp32 oracle (canonical causal convention, N == S and
N < S); the forward still runs the native kernel (which-binary).
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, is_mfa_available
from mlx_mfa import _dispatch_trace as dt

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")


def _oracle_grads(q, k, v, bias, g, scale, causal):
    with mx.stream(mx.cpu):
        N, S = q.shape[2], k.shape[2]

        def f(a, b, c, bb):
            s = (a @ mx.swapaxes(b, -1, -2)) * scale + bb
            if causal:
                vis = mx.arange(S)[None, :] <= mx.arange(N)[:, None] + max(0, S - N)
                s = mx.where(vis, s, float("-inf"))
            return ((mx.softmax(s, axis=-1) @ c) * g).sum()
        grads = mx.grad(f, argnums=(0, 1, 2, 3))(
            *(x.astype(mx.float32) for x in (q, k, v, bias)))
        mx.eval(*grads)
    return grads


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("N,S", [(256, 256), (128, 384)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", [1, 2])
def test_attn_bias_gradients_match_oracle(mode, causal, N, S, D):
    B, H = 1, 2
    mx.random.seed(N + S + D + mode)
    q = (mx.random.normal((B, H, N, D)) * 0.5).astype(mx.float16)
    k = (mx.random.normal((B, H, S, D)) * 0.5).astype(mx.float16)
    v = (mx.random.normal((B, H, S, D)) * 0.5).astype(mx.float16)
    bshape = (1, 1, 1, S) if mode == 1 else (B, H, 1, S)
    bias = mx.random.normal(bshape).astype(mx.float32)
    g = mx.random.normal((B, H, N, D)).astype(mx.float32)
    scale = D ** -0.5
    with dt.capture() as cap:
        grads = mx.grad(lambda a, b, c, bb: (flash_attention(
            a, b, c, scale=scale, causal=causal, attn_bias=bb).astype(mx.float32) * g).sum(),
            argnums=(0, 1, 2, 3))(q, k, v, bias)
        mx.eval(*grads)
    assert any(r[0] == "mfa_bias_native" for r in cap), [r[0] for r in cap]
    for name, x, y in zip(("dQ", "dK", "dV", "dbias"), grads,
                          _oracle_grads(q, k, v, bias, g, scale, causal)):
        assert x.shape == y.shape, (name, x.shape, y.shape)
        rel = float(mx.max(mx.abs(x.astype(mx.float32) - y))) / max(float(mx.max(mx.abs(y))), 1e-6)
        assert rel < 2e-2, (name, rel)


def test_zero_bias_gradients_equal_unbiased():
    mx.random.seed(5)
    q, k, v = ((mx.random.normal((1, 2, 256, 64)) * 0.5).astype(mx.float16) for _ in range(3))
    zero = mx.zeros((1, 1, 1, 256), dtype=mx.float32)
    for causal in (False, True):
        gb = mx.grad(lambda a: flash_attention(a, k, v, causal=causal, attn_bias=zero)
                     .astype(mx.float32).sum())(q)
        gn = mx.grad(lambda a: flash_attention(a, k, v, causal=causal)
                     .astype(mx.float32).sum())(q)
        mx.eval(gb, gn)
        assert bool(mx.all(mx.isfinite(gb)).item())
        assert float(mx.max(mx.abs(gb.astype(mx.float32) - gn.astype(mx.float32)))) < 1e-2

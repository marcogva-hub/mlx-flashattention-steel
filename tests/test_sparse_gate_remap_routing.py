"""Public which-binary locks for the hardened sparse β3 routing map."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import _ext, flash_attention_sparse
from mlx_mfa.attention import _get_is_m5_plus_cached
from mlx_mfa.lcsa_nax import _bool_mask_to_float_bias, _nax_sparse_route_viable


pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(), reason="sparse gate lock asserts M5+ routes"
)


class _FakeArray:
    def __init__(self, B, H, N, D, dtype):
        self.shape = (B, H, N, D)
        self.dtype = dtype


@pytest.mark.parametrize(
    "causal,dtype,B,H,N,D,density,expected",
    [
        # Non-causal fp16: exact measured B·H values and N-entry thresholds.
        (False, mx.float16, 1, 1, 2048, 64, 0.05, False),
        (False, mx.float16, 1, 1, 8192, 64, 0.30, True),
        (False, mx.float16, 1, 2, 8192, 64, 0.05, False),
        (False, mx.float16, 1, 12, 4096, 128, 0.30, True),
        (False, mx.float16, 1, 12, 6144, 128, 0.30, True),
        (False, mx.float16, 1, 12, 4096, 64, 0.25, True),
        (False, mx.float16, 1, 12, 4096, 64, 0.2501, False),
        (False, mx.float16, 1, 4, 4096, 128, 0.05, True),
        (False, mx.float16, 1, 4, 4096, 128, 0.0501, False),
        # bf16 follows only its one measured non-causal region.
        (False, mx.bfloat16, 1, 12, 4096, 128, 0.30, True),
        (False, mx.bfloat16, 1, 1, 8192, 128, 0.05, False),
        # Causal cells are exact; no N/B·H interpolation.
        (True, mx.float16, 1, 4, 4096, 128, 0.10, True),
        (True, mx.float16, 1, 12, 4096, 128, 0.30, True),
        (True, mx.float16, 1, 12, 8192, 64, 0.30, True),
        (True, mx.float16, 1, 12, 8192, 128, 0.30, True),
        (True, mx.float16, 1, 4, 8192, 128, 0.10, False),
        (True, mx.bfloat16, 1, 4, 4096, 128, 0.10, True),
        (True, mx.bfloat16, 1, 12, 4096, 128, 0.10, False),
    ],
)
def test_pure_gate_exact_measured_envelope(
    causal, dtype, B, H, N, D, density, expected
):
    q = _FakeArray(B, H, N, D, dtype)
    k = _FakeArray(B, H, N, D, dtype)
    assert _nax_sparse_route_viable(q, k, 32, density, causal=causal) is expected


def _mask(N: int, density: float, causal: bool) -> mx.array:
    blocks = N // 32
    mask = np.eye(blocks, dtype=np.bool_) if causal else np.zeros(
        (blocks, blocks), dtype=np.bool_
    )
    if not causal:
        mask[:, 0] = True
    target = max(int(np.floor(density * blocks * blocks)), int(mask.sum()))
    candidates = np.flatnonzero(~mask.reshape(-1))
    mask.reshape(-1)[candidates[: target - int(mask.sum())]] = True
    return mx.array(mask)


def _terminal(events):
    terminal = [event for event in events if not event[1].startswith("[reentrant]")]
    assert terminal, events
    return terminal[-1][0]


def _sdpa(q, k, v, mask, scale, causal):
    bias = _bool_mask_to_float_bias(mask, 32, q.shape[2], k.shape[2], q.dtype)
    if causal:
        qi = mx.arange(q.shape[2]).reshape(-1, 1)
        ki = mx.arange(k.shape[2]).reshape(1, -1)
        bias = bias + mx.where(
            ki > qi,
            mx.array(-float("inf"), dtype=q.dtype),
            mx.array(0.0, dtype=q.dtype),
        )
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=bias)


def _cosine(a, b):
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    value = mx.sum(af * bf) / mx.sqrt(mx.sum(af * af) * mx.sum(bf * bf))
    mx.eval(value)
    return float(value.item())


@pytest.mark.parametrize(
    "dtype,H,N,D,density,causal,expected",
    [
        (mx.float16, 1, 2048, 64, 0.05, False, "sdpa"),
        (mx.float16, 1, 8192, 64, 0.05, False, "v6nax_sparse"),
        (mx.float16, 12, 4096, 128, 0.30, False, "v6nax_sparse"),
        (mx.float16, 12, 4096, 64, 0.25, False, "v6nax_sparse"),
        (mx.float16, 4, 4096, 128, 0.05, False, "v6nax_sparse"),
        (mx.float16, 4, 4096, 128, 0.10, True, "v6nax_sparse"),
        (mx.float16, 4, 4096, 64, 0.10, True, "sdpa"),
        (mx.float16, 12, 8192, 64, 0.30, True, "v6nax_sparse"),
        (mx.float16, 12, 8192, 128, 0.30, True, "v6nax_sparse"),
        (mx.bfloat16, 12, 4096, 128, 0.15, False, "v6nax_sparse"),
        (mx.bfloat16, 1, 4096, 128, 0.15, False, "sdpa"),
        (mx.bfloat16, 4, 4096, 128, 0.10, True, "v6nax_sparse"),
        (mx.bfloat16, 4, 4096, 64, 0.10, True, "sdpa"),
    ],
)
def test_public_gate_boundaries_are_real_binaries(
    dtype, H, N, D, density, causal, expected
):
    mx.random.seed(20260714 + H + D + int(causal))
    q = mx.random.normal((1, H, N, D)).astype(dtype)
    k = mx.random.normal((1, H, N, D)).astype(dtype)
    v = mx.random.normal((1, H, N, D)).astype(dtype)
    mask = _mask(N, density, causal)
    scale = 1.0 / math.sqrt(D)

    with dtrace.capture() as events:
        public = flash_attention_sparse(q, k, v, mask, scale=scale, causal=causal)
        mx.eval(public)
    assert _terminal(events) == expected

    if expected == "v6nax_sparse":
        direct = _ext.sparse_attention_forward(
            q, k, v, mask, 32, causal, scale, "v6nax_sparse"
        )
        mx.eval(direct)
        delta = mx.max(mx.abs(public.astype(mx.float32) - direct.astype(mx.float32)))
        mx.eval(delta)
        assert float(delta.item()) == 0.0
    else:
        reference = _sdpa(q, k, v, mask, scale, causal)
        mx.eval(reference)
        assert _cosine(public, reference) >= 0.999

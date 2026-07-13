"""Regression lock for list-valued public sliding-window arguments."""

import mlx.core as mx
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import flash_attention, is_mfa_available


def _terminal(trace):
    return [entry for entry in trace if not entry[1].startswith("[reentrant]")][-1]


@pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")
def test_window_size_list_matches_tuple_and_routes_identically():
    mx.random.seed(817)
    q = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)
    k = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)
    v = mx.random.normal((1, 1, 64, 64)).astype(mx.float16)

    with dtrace.capture() as tuple_trace:
        tuple_out = flash_attention(q, k, v, window_size=(0, 256))
        mx.eval(tuple_out)
    with dtrace.capture() as list_trace:
        list_out = flash_attention(q, k, v, window_size=[0, 256])
        mx.eval(list_out)

    assert _terminal(list_trace) == _terminal(tuple_trace)
    assert float(mx.max(mx.abs(
        list_out.astype(mx.float32) - tuple_out.astype(mx.float32)
    )).item()) == 0.0

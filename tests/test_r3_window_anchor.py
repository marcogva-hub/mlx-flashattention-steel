"""R3 / MSL-03 (code review 2026-09, P1 PUBLISHED) — sliding-window anchor must not
depend on dtype.

Decision (remediation brief): the window anchor is the one of the f16/bf16 native
kernels, the production path: query row i sits at position q_off + i with
    q_off = S - N   if causal and N < S,   else 0
(csrc/mfa_attention.cpp qL_off; the windowed backward oracle already used it —
III-4 pass-2 B1).  The fp32 masked-SDPA window fallback and _sdpa_with_weights anchored
at S-N UNCONDITIONALLY: non-causal N<S windows differed by dtype (1.44), and causal N>S
fp32 rows were fully masked -> NaN.  With a window, causality follows the same anchor
(key j visible iff j <= q_off + i), as in the kernel and the backward oracle.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention, is_mfa_available

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

# (N, S, causal, (wl, wr)) — every row sees >= 1 key under the native anchor.
CELLS = [
    (64, 512, False, (32, 32)),     # non-causal N<S: fp32 used the S-N anchor (1.44)
    (300, 100, False, (250, -1)),   # non-causal N>S
    (100, 300, True, (16, 16)),     # causal N<S (anchor S-N in both conventions)
    (512, 256, True, (300, -1)),    # causal N>S: fp32 rows 0..255 used to be all-NaN
    (256, 256, True, (32, -1)),
]
# Degenerate on purpose: rows 320..511 see NO key even under the native anchor
# (native f16 returns NaN there, SDPA semantics) — both dtypes must agree row-for-row.
DEGENERATE = (512, 256, True, (64, -1))


def _oracle(q, k, v, causal, window):
    wl, wr = window
    with mx.stream(mx.cpu):
        q32, k32, v32 = (x.astype(mx.float32) for x in (q, k, v))
        N, S, D = q.shape[2], k.shape[2], q.shape[3]
        q_off = (S - N) if (causal and N < S) else 0
        qi = mx.arange(N)[:, None] + q_off
        kj = mx.arange(S)[None, :]
        vis = mx.ones((N, S), dtype=mx.bool_)
        if wl >= 0:
            vis = vis & (kj >= qi - wl)
        if wr >= 0:
            vis = vis & (kj <= qi + wr)
        if causal:
            vis = vis & (kj <= qi)
        s = (q32 @ mx.swapaxes(k32, -1, -2)) * (D ** -0.5)
        s = mx.where(vis, s, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v32
        mx.eval(o)
    return o


def _qkv(N, S, D=64, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    return tuple(mx.random.normal((1, 2, L, D)).astype(dtype) for L in (N, S, S))


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _finite(x):
    return bool(mx.all(mx.isfinite(x.astype(mx.float32))).item())


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32], ids=["f16", "f32"])
@pytest.mark.parametrize("N,S,causal,window", CELLS)
def test_window_matches_native_anchor(N, S, causal, window, dtype):
    q, k, v = _qkv(N, S, dtype=dtype, seed=N + S)
    out = flash_attention(q, k, v, causal=causal, window_size=window)
    assert _finite(out)
    assert _err(out, _oracle(q, k, v, causal, window)) < (1e-2 if dtype == mx.float16 else 5e-3)


@pytest.mark.parametrize("N,S,causal,window", CELLS)
def test_window_same_math_across_dtypes(N, S, causal, window):
    q, k, v = _qkv(N, S, dtype=mx.float32, seed=7)
    o32 = flash_attention(q, k, v, causal=causal, window_size=window)
    o16 = flash_attention(*(x.astype(mx.float16) for x in (q, k, v)),
                          causal=causal, window_size=window)
    assert _err(o32, o16) < 1e-2


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32], ids=["f16", "f32"])
@pytest.mark.parametrize("N,S,causal,window", CELLS)
def test_attn_weights_path_matches_production(N, S, causal, window, dtype):
    """v2.58.1 contract: return_attn_weights output == the production output."""
    q, k, v = _qkv(N, S, dtype=dtype, seed=3)
    o_prod = flash_attention(q, k, v, causal=causal, window_size=window)
    o_w, w = flash_attention(q, k, v, causal=causal, window_size=window,
                             return_attn_weights=True)
    assert _finite(o_w) and _finite(w)
    assert _err(o_w, o_prod) < 1e-2


def test_degenerate_window_nan_rows_identical_across_dtypes():
    N, S, causal, window = DEGENERATE
    q, k, v = _qkv(N, S, dtype=mx.float32, seed=5)
    o32 = flash_attention(q, k, v, causal=causal, window_size=window)
    o16 = flash_attention(*(x.astype(mx.float16) for x in (q, k, v)),
                          causal=causal, window_size=window)
    nan32 = mx.any(mx.isnan(o32), axis=-1)
    nan16 = mx.any(mx.isnan(o16.astype(mx.float32)), axis=-1)
    assert mx.array_equal(nan32, nan16).item()
    ref = _oracle(q, k, v, causal, window)             # NaN exactly on the empty rows
    assert mx.array_equal(nan32, mx.any(mx.isnan(ref), axis=-1)).item()
    ok = ~nan32[..., None]
    assert float(mx.max(mx.abs(mx.where(ok, o32 - ref, 0.0)))) < 5e-3

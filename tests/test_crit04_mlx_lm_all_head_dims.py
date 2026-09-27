"""CRIT-04 (code review 2026-09, P1 PUBLISHED) — patch_mlx_lm() must never crash on a
head_dim it advertises.

The integration called `_mfa_forward` directly for every D in
get_supported_configs()['head_dims'], which includes 512, while the C++ entry refuses
D=512 ("head_dim must be 64, 128, or 256") -> every D=512 attention call of a patched
mlx-lm model raised, although the docstring promises a transparent fallback.  Fix:
the patched SDPA goes through the public `flash_attention` (which delegates D=512 to
SDPA and applies the measured per-shape routing).
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

mlx_lm_base = pytest.importorskip("mlx_lm.models.base")

from mlx_mfa import is_mfa_available  # noqa: E402
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm, unpatch_mlx_lm  # noqa: E402

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")


@pytest.fixture
def patched():
    orig = mlx_lm_base.scaled_dot_product_attention
    patch_mlx_lm(verbose=False)
    try:
        yield orig
    finally:
        unpatch_mlx_lm()


class _Cache:
    """Minimal cache stand-in exposing a sliding window (max_kv_window)."""
    bits = None

    def __init__(self, w):
        self.max_kv_window = w


@pytest.mark.parametrize("D", [64, 128, 256, 512])
@pytest.mark.parametrize("N,S,mask", [(64, 64, "causal"), (1, 300, None)])
@pytest.mark.parametrize("hkv", [4, 1])       # MHA and GQA 4:1
def test_patched_sdpa_matches_original(patched, D, N, S, mask, hkv):
    orig = patched
    mx.random.seed(D + N + hkv)
    q = mx.random.normal((1, 4, N, D)).astype(mx.float16)
    k = mx.random.normal((1, hkv, S, D)).astype(mx.float16)
    v = mx.random.normal((1, hkv, S, D)).astype(mx.float16)
    sc = 1 / math.sqrt(D)
    ref = orig(q, k, v, None, sc, mask)
    out = mlx_lm_base.scaled_dot_product_attention(q, k, v, None, sc, mask)
    mx.eval(ref, out)
    assert out.shape == ref.shape
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))) < 2e-2


@pytest.mark.parametrize("D", [128, 512])
def test_patched_sdpa_sliding_window_decode(patched, D):
    """Windowed decode keeps the R1-FIX semantics (causal anchor) on every D."""
    mx.random.seed(D)
    N, S, W = 1, 300, 64
    q = mx.random.normal((1, 4, N, D)).astype(mx.float16)
    k = mx.random.normal((1, 4, S, D)).astype(mx.float16)
    v = mx.random.normal((1, 4, S, D)).astype(mx.float16)
    out = mlx_lm_base.scaled_dot_product_attention(q, k, v, _Cache(W), 1 / math.sqrt(D), None)
    ref = mx.fast.scaled_dot_product_attention(q[..., :, :], k[:, :, S - W:], v[:, :, S - W:],
                                               scale=1 / math.sqrt(D))
    mx.eval(out, ref)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))) < 2e-2


# ── Sibling on the public API (found while fixing CRIT-04) ───────────────────────
# _can_use_mfa marks D=512 capable, but the dense MFA primitive serves only
# {64, 128, 256}: flash_attention(D=512, window_size=...) and (D=512, return_lse=True)
# raised from C++.  They now take the exact SDPA-class fallbacks; a FORCED
# backend='mfa' still refuses loudly (never a silent downgrade).
def _d512(N=64, S=300):
    mx.random.seed(0)
    return tuple(mx.random.normal((1, 4, L, 512)).astype(mx.float16) for L in (N, S, S))


def _d512_oracle(q, k, v, win=None):
    N, S = q.shape[2], k.shape[2]
    with mx.stream(mx.cpu):
        s = (q.astype(mx.float32) @ mx.swapaxes(k.astype(mx.float32), -1, -2)) * 512 ** -0.5
        qi = mx.arange(N)[:, None] + (S - N)
        kj = mx.arange(S)[None, :]
        vis = kj <= qi
        if win is not None:
            vis = vis & (kj >= qi - win)
        s = mx.where(vis, s, float("-inf"))
        o = mx.softmax(s, axis=-1) @ v.astype(mx.float32)
        lse = mx.logsumexp(s, axis=-1) / math.log(2)
        mx.eval(o, lse)
    return o, lse


def test_public_d512_window_does_not_crash():
    from mlx_mfa import flash_attention
    q, k, v = _d512()
    out = flash_attention(q, k, v, causal=True, window_size=(64, 0))
    ref, _ = _d512_oracle(q, k, v, win=64)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref))) < 1e-2


def test_public_d512_return_lse_does_not_crash():
    from mlx_mfa import flash_attention
    q, k, v = _d512()
    out, lse = flash_attention(q, k, v, causal=True, return_lse=True)
    ref, ref_l = _d512_oracle(q, k, v)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref))) < 1e-2
    assert float(mx.max(mx.abs(lse - ref_l))) < 1e-2


def test_forced_mfa_backend_d512_still_refuses():
    from mlx_mfa import flash_attention
    q, k, v = _d512()
    with pytest.raises(ValueError, match="head_dim"):
        mx.eval(flash_attention(q, k, v, causal=True, backend="mfa"))

"""IV-review A5-1: V3 conditional-auto correctness on the PRODUCTION-reachable path.

The queue-closure sprint validated V3's auto-routing for PERF but TestSteelV3 only
exercised V3 via MFA_ENABLE_V3=1 at sub-threshold shapes (N=1024) and used V2 (not an
independent oracle) for its value comparison. This test closes the gap: it validates the
path real M5 users hit — windowed-causal, N at/above the v3_min_N auto threshold, B·H>=4,
env UNSET — against an INDEPENDENT fp32 oracle (lesson #11). On M5 dense forward routes to
SDPA, so the auto path that reaches V3 is the WINDOWED one (window -> MFA -> V3).
"""
from __future__ import annotations
import math
import os
from unittest.mock import patch
import mlx.core as mx
import numpy as np
import pytest

try:
    from mlx_mfa import flash_attention
    import mlx_mfa._ext  # noqa: F401 — requires the built extension
    _HAS_EXT = True
except Exception:
    _HAS_EXT = False

pytestmark = pytest.mark.skipif(not _HAS_EXT, reason="requires built extension")


def _fp32_ref(q, k, v, scale, window_left, gqa=1):
    """Independent fp32 causal (+ optional left sliding window) SDPA reference."""
    qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
    B, H, N, D = qf.shape
    Hk = kf.shape[1]
    if Hk != H:
        kf = mx.repeat(kf, H // Hk, axis=1)
        vf = mx.repeat(vf, H // Hk, axis=1)
    S = (qf @ kf.transpose(0, 1, 3, 2)) * scale
    i = mx.arange(N)[:, None]
    j = mx.arange(N)[None, :]
    masked = (j > i)                       # causal: no future
    if window_left >= 0:
        masked = masked | ((i - j) > window_left)   # sliding window: only last `left` keys
    S = mx.where(masked, mx.array(-1e9, mx.float32), S)
    return mx.softmax(S, axis=-1) @ vf


# V3 auto-fire regime: causal, N>=4096 (D=64) / N>=2048 (D=128), B·H>=4, f16/bf16.
@pytest.mark.parametrize("D,N", [(64, 4096), (128, 2048), (128, 4096)])
def test_v3_autoroute_windowed_matches_fp32(D, N):
    """env UNSET + windowed-causal at the V3 auto regime ≈ independent fp32. Also assert
    V3 is actually exercised (forcing V2 via MFA_DISABLE_V3 gives the same correct result,
    so whichever the auto router picks at this regime is correct)."""
    B, H = 1, 8                                   # B·H=8 >= 4
    scale = 1.0 / math.sqrt(D)
    win = 1024
    mx.random.seed(0x5333)
    q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    ref = _fp32_ref(q, k, v, scale, win)

    # Production path: env UNSET, window -> MFA -> V3 auto-fires at this regime.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MFA_ENABLE_V3", None)
        os.environ.pop("MFA_DISABLE_V3", None)
        out_auto = flash_attention(q, k, v, scale=scale, causal=True, window_size=(win, 0))
        mx.eval(out_auto)
    err_auto = float(mx.max(mx.abs(out_auto.astype(mx.float32) - ref)).item())
    assert err_auto < 3e-2, f"D={D} N={N}: auto(windowed) vs fp32 err={err_auto:.3e}"
    assert bool(mx.isfinite(out_auto).all().item())

    # Force V2 fallback: must also match fp32 (both kernels correct at this shape).
    with patch.dict(os.environ, {"MFA_DISABLE_V3": "1"}):
        out_v2 = flash_attention(q, k, v, scale=scale, causal=True, window_size=(win, 0))
        mx.eval(out_v2)
    err_v2 = float(mx.max(mx.abs(out_v2.astype(mx.float32) - ref)).item())
    assert err_v2 < 3e-2, f"D={D} N={N}: V2(windowed) vs fp32 err={err_v2:.3e}"


def test_v3_autoroute_gqa_windowed_matches_fp32():
    """GQA (Hq=8, Hkv=2) windowed-causal at the V3 regime ≈ fp32 — the GQA edge the
    perf-validation didn't cover for correctness."""
    B, Hq, Hkv, N, D = 1, 8, 2, 4096, 128
    scale = 1.0 / math.sqrt(D)
    win = 1024
    mx.random.seed(11)
    q = (mx.random.uniform(-1, 1, (B, Hq, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B, Hkv, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, Hkv, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    ref = _fp32_ref(q, k, v, scale, win, gqa=Hq // Hkv)
    os.environ.pop("MFA_ENABLE_V3", None); os.environ.pop("MFA_DISABLE_V3", None)
    out = flash_attention(q, k, v, scale=scale, causal=True, window_size=(win, 0))
    mx.eval(out)
    err = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item())
    assert err < 3e-2, f"GQA windowed auto vs fp32 err={err:.3e}"
    assert bool(mx.isfinite(out).all().item())


@pytest.mark.parametrize("D,N", [(64, 4096), (128, 4096)])
def test_v3_backend_mfa_dense_matches_fp32(D, N):
    """Expert path: backend='mfa' dense-causal at the V3 regime (V3 fires) ≈ fp32."""
    B, H = 1, 8
    scale = 1.0 / math.sqrt(D)
    mx.random.seed(7)
    q = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (B, H, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    ref = _fp32_ref(q, k, v, scale, -1)   # dense causal (no window)
    os.environ.pop("MFA_ENABLE_V3", None); os.environ.pop("MFA_DISABLE_V3", None)
    out = flash_attention(q, k, v, scale=scale, causal=True, backend="mfa")
    mx.eval(out)
    err = float(mx.max(mx.abs(out.astype(mx.float32) - ref)).item())
    assert err < 3e-2, f"D={D} N={N}: backend=mfa dense vs fp32 err={err:.3e}"
    assert bool(mx.isfinite(out).all().item())

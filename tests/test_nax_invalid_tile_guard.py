"""V6 NAX invalid-tile guard (research/nax-autotune-m5, Item B).

Before the fix, an invalid env-set tile triple (e.g. MFA_V6_NAX_BQ=32 with the default
WM=4 → 32 % (4*16) != 0) set `use_v6nax = false` in the tile guards — but F-3 removed the
`use_v6nax=false` (simdgroup) dispatch branch, so the non-NAX source reached
`v6nax_compile("attention")` → an UNCATCHABLE Metal pipeline abort
(`function attention cannot be used to build a pipeline state`).

Rule 8: the guards now `throw std::runtime_error` for an invalid env tile triple — a clean,
catchable error raised BEFORE any pipeline build. This test asserts each invalid triple raises
(no abort) and that a valid triple still runs correctly.
"""
from __future__ import annotations
import os, math
import numpy as np
import mlx.core as mx
import pytest

try:
    from mlx_mfa._ext import v6_nax_forward
    _HAVE = True
except Exception:
    _HAVE = False
try:
    from mlx_mfa import get_device_info
    _IS_M5 = bool(get_device_info().get("is_m5_plus"))
except Exception:
    _IS_M5 = False

pytestmark = pytest.mark.skipif(not (_HAVE and _IS_M5),
                                reason="V6 NAX invalid-tile guard: M5+ NAX + extension required")

_TILE_ENVS = ("MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM")


def _clear():
    for k in _TILE_ENVS:
        os.environ.pop(k, None)


def _run(D=128, N=2048):
    q = (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q)
    O, _ = v6_nax_forward(q, q, q, False, True)
    mx.eval(O)
    return O


@pytest.mark.parametrize("env,val,why", [
    ("MFA_V6_NAX_BQ", "32", "BQ=32 with default WM=4 → 32 % (4*16) != 0"),
    ("MFA_V6_NAX_WM", "3", "WM=3 with default BQ=64 → 64 % (3*16) != 0"),
    ("MFA_V6_NAX_BK", "48", "BK=48 → 48 % 32 != 0 (paired-MMA)"),
])
def test_invalid_tile_raises_cleanly(env, val, why):
    """An invalid env tile triple raises a catchable error (NOT a Metal abort)."""
    _clear()
    os.environ[env] = val
    try:
        with pytest.raises((RuntimeError, ValueError)):
            _run()
    finally:
        _clear()


def test_valid_default_tile_runs_correct():
    """The default (no env override) tile triple runs and matches fp32."""
    _clear()
    D, N = 128, 2048
    mx.random.seed(7)
    q = (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (1, 8, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    O, _ = v6_nax_forward(q, k, v, False, True)
    mx.eval(O)
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    ref = mx.softmax((qf @ kf.transpose(0, 1, 3, 2)) * (1 / math.sqrt(D)), axis=-1) @ vf
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref)).max())
    assert err < 2e-2, f"valid-default tile wrong vs fp32: err={err:.3e}"


def test_explicit_valid_tile_runs():
    """An explicitly-set VALID triple (BQ=64 WM=4 BK=32) runs (recovery after the raises)."""
    _clear()
    os.environ["MFA_V6_NAX_BQ"] = "64"
    os.environ["MFA_V6_NAX_WM"] = "4"
    os.environ["MFA_V6_NAX_BK"] = "32"
    try:
        O = _run()
        assert bool(np.isfinite(np.array(O.astype(mx.float32))).all())
    finally:
        _clear()

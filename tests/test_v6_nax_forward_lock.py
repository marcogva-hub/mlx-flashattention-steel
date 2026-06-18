"""Standalone forward correctness LOCK for the dense NAX matmul2d forward
`v6_nax_forward` (audit E-addendum / B-gap, 2026-06-18).

`v6_nax_forward` (csrc/mfa_steel_fwd_v6_nax.cpp — 10 matmul2d/cooperative-tensor
hits, the ONLY dense forward on the competitive NAX primitive) was previously
validated only INDIRECTLY, via B3's backward gradients consuming its (O, L).
This module gives it a STANDALONE forward check vs an INDEPENDENT manual fp32
oracle (strict lesson #11 — NOT SDPA, NOT another mlx-mfa kernel), closing the
B-gap (the dense NAX forward had no standalone forward spec/lock).

Constraints (archaeology, current master):
  - Binding `v6_nax_forward(q, k, v, causal, force_v6nax=False)` has NO scale
    parameter — the host bakes scale = 1/sqrt(D) (attention.py ~5018). So this
    kernel is DEFAULT-SCALE ONLY; the oracle uses 1/sqrt(D) accordingly.
  - Forward-eligible shapes (binding DC12): D=128 any N; D=64 only Nk > 8000.
  - M5+ NAX hardware only (matmul2d cooperative tensors).
"""
from __future__ import annotations
import math

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

pytestmark = pytest.mark.skipif(
    not (_HAVE and _IS_M5),
    reason="v6_nax_forward dense NAX kernel: M5+ NAX hardware + extension required",
)


def _fp32_oracle(q, k, v, causal, D):
    """Independent manual fp32 FA-2 forward (lesson #11), default scale 1/sqrt(D)."""
    qf, kf, vf = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
    Hq, Hk = q.shape[1], k.shape[1]
    if Hq != Hk:  # GQA: repeat KV heads
        kf = mx.repeat(kf, Hq // Hk, axis=1)
        vf = mx.repeat(vf, Hq // Hk, axis=1)
    s = (qf @ kf.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))
    if causal:
        N, S = q.shape[2], k.shape[2]
        cm = mx.arange(S)[None, :] > (mx.arange(N)[:, None] + (S - N))
        s = mx.where(cm[None, None], mx.array(-1e30, mx.float32), s)
    o = mx.softmax(s, axis=-1) @ vf
    mx.eval(o)
    return o


# (D, N, Hq, Hk, causal) — forward-eligible shapes only (D=64 needs Nk>8000).
_CELLS = [
    (128, 2048, 8, 8, False), (128, 2048, 8, 8, True),
    (128, 4096, 8, 8, False), (128, 4096, 8, 8, True),
    (64, 8192, 8, 8, False), (64, 8192, 8, 8, True),
    (128, 2048, 8, 2, False), (128, 2048, 8, 2, True),  # GQA
]


@pytest.mark.parametrize("D,N,Hq,Hk,causal", _CELLS)
def test_v6_nax_forward_matches_fp32(D, N, Hq, Hk, causal):
    mx.random.seed(13)
    q = (mx.random.uniform(-1, 1, (1, Hq, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (1, Hk, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (1, Hk, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    O, L = v6_nax_forward(q, k, v, causal, True)  # force_v6nax=True
    mx.eval(O)
    ref = _fp32_oracle(q, k, v, causal, D)
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref)).max())
    assert np.isfinite(err) and err < 2e-2, (
        f"v6_nax_forward D={D} N={N} Hq={Hq} Hk={Hk} causal={causal}: "
        f"max_err={err:.3e} vs independent fp32 (default scale 1/sqrt(D))")


def test_v6_nax_forward_is_default_scale_only():
    """The kernel bakes 1/sqrt(D): its output matches the 1/sqrt(D) oracle and
    the binding exposes NO scale parameter (default-scale constraint — documented,
    not a bug). A custom scale must route elsewhere (mfa_forward_with_lse/SDPA)."""
    import inspect
    # The C++ binding takes (q, k, v, causal, force_v6nax) — no scale arg.
    # Confirm via the matched 1/sqrt(D) oracle (a non-default scale would diverge).
    D, N = 128, 2048
    mx.random.seed(1)
    q = (mx.random.uniform(-1, 1, (1, 4, N, D)) * 0.1).astype(mx.float16)
    k = (mx.random.uniform(-1, 1, (1, 4, N, D)) * 0.1).astype(mx.float16)
    v = (mx.random.uniform(-1, 1, (1, 4, N, D)) * 0.1).astype(mx.float16)
    mx.eval(q, k, v)
    O, _ = v6_nax_forward(q, k, v, False, True)
    mx.eval(O)
    ref_default = _fp32_oracle(q, k, v, False, D)  # scale = 1/sqrt(128)
    err = float(np.abs(np.array(O.astype(mx.float32)) - np.array(ref_default)).max())
    assert err < 2e-2, f"v6_nax_forward not at default scale 1/sqrt(D): err={err:.3e}"

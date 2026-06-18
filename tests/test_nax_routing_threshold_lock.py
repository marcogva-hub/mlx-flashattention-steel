"""Dense D=128 NAX routing-threshold lock (research/nax-routing-threshold-m5, M5 Max, 2026-06-18).

Tier-2 #1: the dense D=128 forward auto-routes to the NAX matmul2d kernel (F-2), but at small
N Apple's SDPA is faster — a localized regression.  Measured crossover (3-session §AA.4, absolute
ms): N<2048 SDPA robustly wins (N=512: 16-36%; N=1024: 3-17%); N>=2048 parity-to-NAX-win.  The
crossover is governed by N (sequence length) alone (equal N*B*H → opposite winners), so the gate
is `q.shape[2] >= _V6_DENSE_MIN_N_DEFAULT` (=2048).

These locks assert the BINARY that runs (Lesson #14 — fingerprint, not flaky ms): byteΔ vs the
forced-SDPA path is 0.0 when SDPA runs and ~1e-6 when the NAX kernel runs.  A drift that reroutes
small-N back to NAX (re-introducing the regression) or large-N to SDPA (losing the win) FAILS here.
keep-all-paths: `MFA_V6_DENSE_MIN_N=0` forces NAX at all N (the pre-threshold path) and is locked.
"""
from __future__ import annotations
import os
import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa.attention import _get_has_nax_cached, _V6_DENSE_MIN_N_DEFAULT

pytestmark = pytest.mark.skipif(
    not _get_has_nax_cached(),
    reason="dense D=128 NAX routing-threshold lock requires the M5+ NAX kernel")


def _routed_kernel(N, dtype=mx.float16, B=1, H=8, causal=False):
    """Return 'NAX' or 'SDPA' by fingerprinting auto vs the forced-SDPA path
    (byteΔ 0.0 ⇒ auto IS SDPA; ~1e-6 ⇒ auto is the NAX kernel)."""
    mx.random.seed(0)
    q = (mx.random.normal((B, H, N, 128)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, H, N, 128)) * 0.1).astype(dtype)
    v = (mx.random.normal((B, H, N, 128)) * 0.1).astype(dtype)
    mx.eval(q, k, v)
    os.environ.pop("MFA_DISABLE_V6_DENSE", None)
    a = mlx_mfa.flash_attention(q, k, v, causal=causal)
    os.environ["MFA_DISABLE_V6_DENSE"] = "1"
    s = mlx_mfa.flash_attention(q, k, v, causal=causal)
    os.environ.pop("MFA_DISABLE_V6_DENSE", None)
    mx.eval(a, s)
    d = float(np.abs(np.array(a.astype(mx.float32)) - np.array(s.astype(mx.float32))).max())
    return "NAX" if d > 1e-7 else "SDPA", (a, q, k, v)


def test_threshold_constant_is_2048():
    """Drift guard: the pinned crossover value.  Changing it is a perf decision, not incidental."""
    assert _V6_DENSE_MIN_N_DEFAULT == 2048


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [512, 1024])
def test_small_N_routes_sdpa(N, dtype):
    """N<2048: regression zone → must route SDPA (NOT the NAX kernel)."""
    os.environ.pop("MFA_V6_DENSE_MIN_N", None)
    route, _ = _routed_kernel(N, dtype)
    assert route == "SDPA", f"N={N} {dtype} routed {route}; expected SDPA (small-N regression zone)"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [2048, 4096])
def test_large_N_routes_nax(N, dtype):
    """N>=2048: NAX is parity-to-win → must keep routing the NAX kernel (win preserved)."""
    os.environ.pop("MFA_V6_DENSE_MIN_N", None)
    route, _ = _routed_kernel(N, dtype)
    assert route == "NAX", f"N={N} {dtype} routed {route}; expected NAX (win zone)"


def test_force_env_keeps_nax_reachable_below_threshold():
    """keep-all-paths: MFA_V6_DENSE_MIN_N=0 forces NAX at all N (pre-threshold path stays reachable)."""
    os.environ["MFA_V6_DENSE_MIN_N"] = "0"
    try:
        route, _ = _routed_kernel(1024)
        assert route == "NAX", "MFA_V6_DENSE_MIN_N=0 did not force NAX at N=1024 (keep-all-paths broken)"
    finally:
        os.environ.pop("MFA_V6_DENSE_MIN_N", None)


@pytest.mark.parametrize("N", [1024, 2048])
def test_correct_across_boundary_vs_fp32(N):
    """Both routed paths are correct attention: routed output within the fp16 floor of an fp32 ref."""
    os.environ.pop("MFA_V6_DENSE_MIN_N", None)
    _, (o, q, k, v) = _routed_kernel(N)
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=1.0 / np.sqrt(128))
    mx.eval(o, ref)
    err = float(np.abs(np.array(o.astype(mx.float32)) - np.array(ref)).max())
    assert err < 1e-2, f"N={N} routed output wrong vs fp32 (Δ={err:.2e})"

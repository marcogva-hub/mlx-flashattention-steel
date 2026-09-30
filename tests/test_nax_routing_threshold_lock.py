"""Dense D=128 NAX routing-threshold lock (research/nax-routing-threshold-m5, M5 Max, 2026-06-18;
re-scoped 2.64 D1, 2026-10-01).

2.64 D1: dense D=128 `auto` DELEGATES to SDPA by default (production evidence: the NAX route ran
on 10/10 production D=128 shapes, 5-11 % slower on 5/10 — devnotes/production_shapes_2026-10.md).
The 2.63 NAX route stays reachable behind the explicit knob `MFA_ENABLE_V6_DENSE=1`, and the
measured crossover `_V6_DENSE_MIN_N_DEFAULT` (=2048: N<2048 SDPA robustly wins) is now the
threshold OF THAT KNOB.  Measured tile-table rows (dispatch_policy.DENSE_TILE_TABLE) are locked by
tests/test_264_dense_delegation.py; the shapes here (B*H=8) have no table row.

These locks assert the BINARY that runs (Lesson #14 — fingerprint, not flaky ms): byteΔ vs the
forced-SDPA path is 0.0 when SDPA runs and ~1e-6 when the NAX kernel runs.
keep-all-paths: `MFA_ENABLE_V6_DENSE=1 MFA_V6_DENSE_MIN_N=0` forces NAX at all N and is locked.
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


@pytest.fixture(autouse=True)
def _clean_dense_env():
    for k in ("MFA_V6_DENSE_MIN_N", "MFA_ENABLE_V6_DENSE", "MFA_DISABLE_V6_DENSE"):
        os.environ.pop(k, None)
    yield
    for k in ("MFA_V6_DENSE_MIN_N", "MFA_ENABLE_V6_DENSE", "MFA_DISABLE_V6_DENSE"):
        os.environ.pop(k, None)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [512, 1024, 2048, 4096])
def test_default_routes_sdpa(N, dtype):
    """2.64 D1: without the knob, D=128 dense auto is SDPA at every N (no table row at B*H=8)."""
    route, _ = _routed_kernel(N, dtype)
    assert route == "SDPA", f"N={N} {dtype} routed {route}; expected SDPA (2.64 default delegation)"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [512, 1024])
def test_knob_small_N_routes_sdpa(N, dtype):
    """MFA_ENABLE_V6_DENSE=1, N<2048: the measured regression zone stays SDPA under the knob."""
    os.environ["MFA_ENABLE_V6_DENSE"] = "1"
    route, _ = _routed_kernel(N, dtype)
    assert route == "SDPA", f"N={N} {dtype} routed {route}; expected SDPA below the knob threshold"


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("N", [2048, 4096])
def test_knob_large_N_routes_nax(N, dtype):
    """MFA_ENABLE_V6_DENSE=1, N>=2048: the explicit NAX dense route (the 2.63 behaviour)."""
    os.environ["MFA_ENABLE_V6_DENSE"] = "1"
    route, _ = _routed_kernel(N, dtype)
    assert route == "NAX", f"N={N} {dtype} routed {route}; expected NAX under MFA_ENABLE_V6_DENSE=1"


def test_force_env_keeps_nax_reachable_below_threshold():
    """keep-all-paths: MFA_ENABLE_V6_DENSE=1 + MFA_V6_DENSE_MIN_N=0 forces NAX at all N."""
    os.environ["MFA_ENABLE_V6_DENSE"] = "1"
    os.environ["MFA_V6_DENSE_MIN_N"] = "0"
    route, _ = _routed_kernel(1024)
    assert route == "NAX", "MFA_ENABLE_V6_DENSE=1 MFA_V6_DENSE_MIN_N=0 did not force NAX at N=1024"


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

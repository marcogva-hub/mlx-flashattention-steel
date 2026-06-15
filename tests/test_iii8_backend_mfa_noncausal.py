"""III-8 lock — forced backend="mfa" non-causal correctness vs fp32.

⚠ CORRECTION (III-9): the async-metallib root cause described below was a
WRONG TURN. The actual `backend="mfa"` non-causal divergence was a split-K
scratch-buffer LIFETIME bug (premature free of lazy-executed pO/pL), fixed in
csrc/mfa_attention.cpp via enc.add_temporary — see
docs/v50/campaign-2026-06/phase3/backend-mfa-noncausal-divergence.md
§ Resolution (III-9), and the real lock in tests/test_iii9_splitk_lifetime.py.
This file (forced SINGLE-PASS) remains valid coverage of the single-pass
non-causal path (which was never the bug), but its rationale below is historical.


Root cause (docs/v50/campaign-2026-06/phase3/backend-mfa-noncausal-divergence.md,
RESOLVED in v2.52.2): the shipped `async_v2.metallib` uses
`simdgroup_async_copy` (hardware DMA), which Apple removed from the AIR
runtime in macOS 26. On macOS 26+ the precompiled async metallib was loaded
FIRST (bypassing the correct JIT source) and its broken DMA loaded only
~(qb+1)*BQ keys → wrong non-causal output (it attended only the first
(qb+1)*BQ keys per Q-tile). Causal coincidentally survived because its mask
zeroes the unloaded keys anyway, and the default dispatch routes non-causal
dense → SDPA, so it never surfaced in production — only via the expert
`backend="mfa"` path, which no test exercised non-causally (the v1.4.0
coverage illusion that let it live since the kernel's origin).

Fix: `shader_cache.mm` skips the async metallib fast path on macOS 26+; all
V2 dispatch then uses the correct JIT path (`generate_steel_v2_source`).

This file locks the previously-untested forced-`backend="mfa"` non-causal
path against an INDEPENDENT fp32 reference (Apple SDPA at fp32; the
auto-hooks do NOT patch mx.fast.sdpa — lesson #11), so this class cannot
silently regress again.
"""
from __future__ import annotations

import pytest
import mlx.core as mx

from mlx_mfa import flash_attention


@pytest.fixture(autouse=True)
def _force_single_pass(monkeypatch):
    # The async-metallib bug (this fix) is in the V2 SINGLE-PASS path
    # (SteelForwardV2 — the only key the async metallib served). Force
    # single-pass so these tests exercise exactly the path the fix corrects.
    # (A SEPARATE, pre-existing V2 split-K non-causal partial-N bug exists for
    # non-aligned/under-occupied N — see backend-mfa-noncausal-divergence.md
    # "split-K partial-N" — tracked separately; not this fix's scope.)
    monkeypatch.setenv("MFA_FORCE_SPLITK", "0")


def _ref_fp32(q, k, v, scale, causal):
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=("causal" if causal else None))


def _mae(o, ref):
    return float(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).mean().item())


_DIMS = [64, 128]
_DTYPES = [mx.float16, mx.bfloat16]
# N spanning single-tile, multi-tile, partial-tile (not a multiple of BQ=32
# or BK), and the occupancy bands that route to single-pass vs split-K.
_NS = [32, 64, 127, 128, 256, 1000, 4096]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", _DIMS)
@pytest.mark.parametrize("N", _NS)
def test_backend_mfa_matches_fp32(N, D, causal, dtype):
    """Forced backend="mfa" matches fp32 within the low-precision floor for
    BOTH causal and non-causal — the regime the async-metallib bug broke
    (non-causal attended only (qb+1)*BQ keys)."""
    B, H = 1, 4  # B*H=4 → occupied grid → V2 single-pass (the broken path)
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(3)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    o = flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa")
    ref = _ref_fp32(q, k, v, scale, causal)
    mx.eval(o, ref)
    assert not bool(mx.any(mx.isnan(o)).item()), f"NaN N={N} D={D} {dtype}"
    bound = 0.02 if dtype == mx.float16 else 0.05
    assert _mae(o, ref) < bound, (
        f"backend=mfa N={N} D={D} causal={causal} {dtype}: "
        f"MAE {_mae(o, ref):.4f} >= {bound}")


def test_forced_single_pass_noncausal_attends_all_keys():
    """The exact bug signature: with Q=0 (uniform attention) and V[j,0]=j,
    O[i,0] must equal mean(0..N-1) = (N-1)/2 — i.e. ALL keys attended. The
    async-metallib bug gave mean(0..(qb+1)*BQ-1) (only the first (qb+1)*BQ
    keys). Forced single-pass (the broken path) at every Q-tile."""
    import numpy as np
    # _force_single_pass autouse fixture sets MFA_FORCE_SPLITK=0.
    B, H, N, D = 1, 4, 128, 128  # 4 Q-tiles (BQ=32)
    k = mx.random.normal((B, H, N, D)).astype(mx.float16)
    q0 = mx.zeros((B, H, N, D)).astype(mx.float16)
    vn = np.zeros((B, H, N, D), np.float32)
    for j in range(N):
        vn[0, 0, j, 0] = float(j)
    v = mx.array(vn).astype(mx.float16)
    o = flash_attention(q0, k, v, scale=1.0, causal=False, backend="mfa")
    mx.eval(o)
    expected = (N - 1) / 2.0  # mean(0..N-1) = 63.5
    for qb in range(N // 32):
        got = float(o[0, 0, qb * 32, 0].item())
        assert abs(got - expected) < 1.0, (
            f"Q-tile qb={qb} attended truncated key set: O={got:.2f} "
            f"expected {expected:.2f} (mean of ALL keys). The async-metallib "
            f"bug gave mean(0..(qb+1)*32-1).")

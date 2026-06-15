"""III-9 lock — V2 split-K scratch-buffer lifetime correctness.

Root cause (docs/v50/campaign-2026-06/phase3/backend-mfa-noncausal-divergence.md):
the V2 split-K path allocated `pO`/`pL` scratch via `allocator::malloc` and
freed them at ENCODE time (`allocator::free`), assuming "Metal retains them
until the command buffer completes."  That assumption is false: MLX executes
lazily, so the free returned the pooled memory while the Phase-1/Phase-2
kernels were still pending.  A concurrent allocation (e.g. the reference SDPA,
or any downstream layer) could then reuse the memory and corrupt the
not-yet-executed reduce read — producing nondeterministic garbage
(MAE 12-36, output saturating to fp16 512) on the upper query rows.

Fix: wrap the scratch in `array`s and register them as command-encoder
temporaries (`enc.add_temporary`), so MLX frees them only AFTER the command
buffer completes.  Same fix applied to the flash-decode two-phase path (same
latent pattern).

The bug was masked in the original suite because it required (a) the split-K
path to engage (`backend="mfa"`, under-occupied grid, splittable N) AND (b) a
concurrent allocation between the call and `mx.eval`.  No prior test combined
both — the v1.4.0 split-K coverage exercised it in isolation or warm.

These tests force split-K (`MFA_FORCE_SPLITK=1`) and interleave a large
concurrent allocation before `mx.eval`, validating against an INDEPENDENT
fp32 reference (Apple SDPA at fp32 — the auto-hooks do NOT patch it,
lesson #11).
"""
from __future__ import annotations

import pytest
import mlx.core as mx


@pytest.fixture(autouse=True)
def _force_splitk(monkeypatch):
    # Force the V2 split-K two-phase path (Phase 1 partial + Phase 2 reduce),
    # which is where the premature-free lifetime bug lived.
    monkeypatch.setenv("MFA_FORCE_SPLITK", "1")


def _ref_fp32(q, k, v, scale, causal):
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=("causal" if causal else None))


from mlx_mfa import flash_attention

# N values chosen so split-K engages (NK_total >= 4 → splittable): need S large
# enough relative to BK.  Include block-aligned and partial N.
_NS = [256, 257, 384, 512, 1000]
_DIMS = [64, 128]


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", _DIMS)
@pytest.mark.parametrize("N", _NS)
def test_splitk_with_concurrent_alloc_matches_fp32(N, D, causal, dtype):
    """The exact bug trigger: split-K output is computed, then a LARGE
    concurrent allocation (reference SDPA + extra scratch) is made BEFORE
    mx.eval — under the lifetime bug the freed pO/pL pool memory was reused
    and corrupted the pending reduce.  With the add_temporary fix the output
    is reliably correct."""
    B, H = 1, 4  # small B*H → under-occupied grid → split-K engages
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(7)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    o = flash_attention(q, k, v, scale=scale, causal=causal, backend="mfa")
    # Concurrent allocations encoded BEFORE eval — the reuse window the bug
    # depended on.
    ref = _ref_fp32(q, k, v, scale, causal)
    junk = mx.random.normal((B, H, N, D)).astype(mx.float32) * 3.0
    mx.eval(o, ref, junk)
    assert not bool(mx.any(mx.isnan(o)).item()), f"NaN N={N} D={D} {dtype}"
    bound = 0.03 if dtype == mx.float16 else 0.06
    mae = float(mx.abs(o.astype(mx.float32) - ref).mean().item())
    assert mae < bound, (
        f"split-K N={N} D={D} causal={causal} {dtype}: MAE {mae:.4f} >= {bound} "
        f"(scratch-lifetime regression — reduce read corrupted by reused pool)")

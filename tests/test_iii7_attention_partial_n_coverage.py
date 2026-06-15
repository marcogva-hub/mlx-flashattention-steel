"""III-7 Class C gap #1 — partial-N attention at fp16/bf16.

The conv3d bug hid because a low-precision kernel with an internal tile
tail was tested only in a regime that could not expose the tail
(lesson #10). Attention has the same structure: the STEEL/V34 forward
tiles the KV/query length by a block (BK/BQ) and masks the partial final
tile (`load_safe` + `kL_rem` / `-inf` score mask). But the existing
partial-N assertion (`test_attention.py::test_seq_len_not_multiple_of_block`)
runs at **fp32 only** — exactly the precision where a tail-accumulation
bug would NOT show. This file locks the low-precision partial-N regime
across the head-dim classes, validated against an INDEPENDENT fp32
reference (Apple `mx.fast.scaled_dot_product_attention`, which the
auto-hooks do NOT patch — lesson #11), never against another mlx-mfa path.

The III-7 sweep probed these regimes and found them CLEAN; this test
makes that passing-by-construction so it cannot silently regress.
"""
from __future__ import annotations

import pytest
import mlx.core as mx

from mlx_mfa import flash_attention

# N values that are NOT multiples of the common block sizes (16/32/64/128),
# spanning small (single-tile) to multi-tile with a partial final tile.
_PARTIAL_N = [5, 17, 33, 65, 129, 257, 1000, 4097]
_DIMS = [64, 128, 256]
_DTYPES = [mx.float16, mx.bfloat16]


def _ref_fp32(q, k, v, scale, causal):
    return mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=scale, mask=("causal" if causal else None))


def _mae(o, ref):
    return float(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32)).mean().item())


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("D", _DIMS)
@pytest.mark.parametrize("N", _PARTIAL_N)
def test_partial_n_matches_fp32(N, D, causal, dtype):
    """flash_attention (default backend="auto") at a non-block-multiple N
    matches fp32 SDPA within the low-precision floor — i.e. the partial
    final tile is masked correctly, at fp16/bf16 (not just fp32)."""
    B, H = 1, 4
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(3)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, N, D)).astype(dtype)
    v = mx.random.normal((B, H, N, D)).astype(dtype)
    o = flash_attention(q, k, v, scale=scale, causal=causal)
    ref = _ref_fp32(q, k, v, scale, causal)
    mx.eval(o, ref)
    assert not bool(mx.any(mx.isnan(o)).item()), f"NaN at N={N} D={D} {dtype}"
    mae = _mae(o, ref)
    # Low-precision floor with headroom; the conv3d-class tail bug produced
    # MAE ~0.1+ (≫ this bound), so a tail regression here would fail loudly.
    bound = 0.02 if dtype == mx.float16 else 0.05
    assert mae < bound, f"N={N} D={D} causal={causal} {dtype}: MAE {mae:.4f}"


@pytest.mark.parametrize("D", [64, 128])
def test_partial_n_gqa_fp16(D):
    """GQA + partial-N + fp16 — the multi-factor combination the single
    shape class never exercised."""
    B, H, H_kv, N = 1, 8, 2, 257
    scale = 1.0 / (D ** 0.5)
    mx.random.seed(5)
    q = mx.random.normal((B, H, N, D)).astype(mx.float16)
    k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
    o = flash_attention(q, k, v, scale=scale, causal=True)
    k_exp = mx.repeat(k, H // H_kv, axis=1)
    v_exp = mx.repeat(v, H // H_kv, axis=1)
    ref = _ref_fp32(q, k_exp, v_exp, scale, causal=True)
    mx.eval(o, ref)
    assert not bool(mx.any(mx.isnan(o)).item())
    assert _mae(o, ref) < 0.02, f"GQA partial-N D={D}: MAE {_mae(o, ref):.4f}"

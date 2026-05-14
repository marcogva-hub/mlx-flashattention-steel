"""v2.50 Prompt 5f Phase D — KD-4 topk_ratio validation tests.

The pre-fix mechanism: `flash_attention_topk(topk_ratio=0)` silently
coerced to `k_count = max(1, 0) = 1` instead of failing loudly per
CLAUDE.md Rule 8.  Phase D (with Prompt 5e Phase 1 fix verified) ensures
out-of-range topk_ratio raises ValueError.

Resolves KD-4 in docs/v50/known-debt-v2.50.md.
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention_topk


def _mk_qkv(B=1, H=4, qL=256, D=64, dtype=mx.float16, seed=42):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    return q, k, v


class TestTopkRatioValidation:
    """Phase D — KD-4: out-of-range topk_ratio must raise ValueError."""

    def test_topk_ratio_zero_raises(self):
        q, k, v = _mk_qkv()
        with pytest.raises(ValueError, match="topk_ratio"):
            flash_attention_topk(q, k, v, topk_ratio=0.0)

    def test_topk_ratio_negative_raises(self):
        q, k, v = _mk_qkv()
        with pytest.raises(ValueError, match="topk_ratio"):
            flash_attention_topk(q, k, v, topk_ratio=-0.5)

    def test_topk_ratio_greater_than_one_raises(self):
        q, k, v = _mk_qkv()
        with pytest.raises(ValueError, match="topk_ratio"):
            flash_attention_topk(q, k, v, topk_ratio=1.5)

    def test_topk_ratio_very_small_positive_succeeds(self):
        q, k, v = _mk_qkv()
        # Tiny positive should succeed (k_count clamped to >= 1)
        out = flash_attention_topk(q, k, v, topk_ratio=1e-3)
        mx.eval(out); mx.synchronize()
        assert out.shape == q.shape

    def test_topk_ratio_exactly_one_succeeds(self):
        q, k, v = _mk_qkv()
        out = flash_attention_topk(q, k, v, topk_ratio=1.0)
        mx.eval(out); mx.synchronize()
        assert out.shape == q.shape

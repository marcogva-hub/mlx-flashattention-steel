"""tests/test_inference_context.py -- Tests for InferenceContext (Track LC)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import InferenceContext, flash_attention


def _rand(shape, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    return mx.random.normal(shape).astype(dtype)


def _to_np(a):
    return np.array(a.astype(mx.float32))


class TestInferenceContextConstruct:
    def test_default_attrs(self):
        ctx = InferenceContext(B=1, H_kv=4, D=64)
        assert ctx.B == 1
        assert ctx.H_kv == 4
        assert ctx.D == 64
        assert ctx.max_seq_len == 8192
        assert ctx.dtype == mx.float16
        assert ctx.stream is None
        assert ctx.seqlen == 0
        assert ctx.k_cache is None
        assert ctx.v_cache is None

    def test_custom_attrs(self):
        ctx = InferenceContext(B=2, H_kv=2, D=128, max_seq_len=512, dtype=mx.bfloat16)
        assert ctx.B == 2
        assert ctx.dtype == mx.bfloat16
        assert ctx.max_seq_len == 512

    def test_repr(self):
        ctx = InferenceContext(B=1, H_kv=4, D=64, max_seq_len=256)
        r = repr(ctx)
        assert "InferenceContext" in r
        assert "seqlen=0" in r
        assert "D=64" in r


class TestInferenceContextPrefill:
    def test_prefill_seqlen_updated(self):
        ctx = InferenceContext(B=1, H_kv=4, D=64)
        N = 32
        q = _rand((1, 4, N, 64))
        k = _rand((1, 4, N, 64), seed=1)
        v = _rand((1, 4, N, 64), seed=2)
        ctx.prefill(q, k, v, scale=1.0 / math.sqrt(64))
        assert ctx.seqlen == N

    def test_prefill_cache_shape(self):
        ctx = InferenceContext(B=1, H_kv=4, D=64)
        N = 16
        q, k, v = (_rand((1, 4, N, 64), seed=s) for s in range(3))
        ctx.prefill(q, k, v, scale=0.125)
        mx.eval(ctx.k_cache, ctx.v_cache)
        assert ctx.k_cache.shape == (1, 4, N, 64)
        assert ctx.v_cache.shape == (1, 4, N, 64)

    def test_prefill_output_matches_flash_attention(self):
        B, H, N, D = 1, 4, 64, 64
        scale = 1.0 / math.sqrt(D)
        q = _rand((B, H, N, D), seed=10)
        k = _rand((B, H, N, D), seed=11)
        v = _rand((B, H, N, D), seed=12)

        ctx = InferenceContext(B=B, H_kv=H, D=D)
        out_ctx = ctx.prefill(q, k, v, scale=scale, causal=True)
        out_ref = flash_attention(q, k, v, scale=scale, causal=True)
        mx.eval(out_ctx, out_ref)

        np.testing.assert_allclose(
            _to_np(out_ctx), _to_np(out_ref), atol=1e-3,
            err_msg="prefill output != flash_attention reference"
        )

    def test_prefill_exceeds_max_seq_len_raises(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64, max_seq_len=16)
        q, k, v = (_rand((1, 2, 32, 64), seed=s) for s in range(3))
        with pytest.raises(ValueError, match="max_seq_len"):
            ctx.prefill(q, k, v)

    def test_prefill_resets_previous_cache(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64)
        q, k, v = (_rand((1, 2, 16, 64), seed=s) for s in range(3))
        ctx.prefill(q, k, v)
        assert ctx.seqlen == 16

        q2, k2, v2 = (_rand((1, 2, 32, 64), seed=s + 10) for s in range(3))
        ctx.prefill(q2, k2, v2)
        assert ctx.seqlen == 32
        mx.eval(ctx.k_cache)
        assert ctx.k_cache.shape[2] == 32


class TestInferenceContextStep:
    def test_step_grows_cache(self):
        B, H, N, D = 1, 4, 32, 64
        ctx = InferenceContext(B=B, H_kv=H, D=D)
        q, k, v = (_rand((B, H, N, D), seed=s) for s in range(3))
        ctx.prefill(q, k, v)
        assert ctx.seqlen == N

        q1, k1, v1 = (_rand((B, H, 1, D), seed=s + 5) for s in range(3))
        ctx.step(q1, k1, v1)
        assert ctx.seqlen == N + 1
        mx.eval(ctx.k_cache)
        assert ctx.k_cache.shape[2] == N + 1

    def test_multiple_steps_accumulate(self):
        B, H, N, D = 1, 2, 16, 64
        ctx = InferenceContext(B=B, H_kv=H, D=D)
        q, k, v = (_rand((B, H, N, D), seed=s) for s in range(3))
        ctx.prefill(q, k, v)
        for i in range(5):
            q1, k1, v1 = (_rand((B, H, 1, D), seed=i + 10 + s) for s in range(3))
            ctx.step(q1, k1, v1)
        assert ctx.seqlen == N + 5

    def test_step_output_matches_manual_concat_ref(self):
        from mlx_mfa import flash_attention_kvcache
        B, H, N, D = 1, 4, 32, 64
        scale = 1.0 / math.sqrt(D)

        q_pre = _rand((B, H, N, D), seed=0)
        k_pre = _rand((B, H, N, D), seed=1)
        v_pre = _rand((B, H, N, D), seed=2)
        q_dec = _rand((B, H, 1, D), seed=3)
        k_dec = _rand((B, H, 1, D), seed=4)
        v_dec = _rand((B, H, 1, D), seed=5)

        ctx = InferenceContext(B=B, H_kv=H, D=D)
        ctx.prefill(q_pre, k_pre, v_pre, scale=scale)
        out_ctx = ctx.step(q_dec, k_dec, v_dec, scale=scale)

        k_full = mx.concatenate([k_pre.astype(mx.float16), k_dec.astype(mx.float16)], axis=2)
        v_full = mx.concatenate([v_pre.astype(mx.float16), v_dec.astype(mx.float16)], axis=2)
        out_ref = flash_attention_kvcache(q_dec, k_full, v_full, scale=scale, causal=True)

        mx.eval(out_ctx, out_ref)
        np.testing.assert_allclose(
            _to_np(out_ctx), _to_np(out_ref), atol=1e-3,
            err_msg="step output != manual concat reference"
        )

    def test_step_overflow_raises(self):
        B, H, N, D = 1, 2, 8, 64
        ctx = InferenceContext(B=B, H_kv=H, D=D, max_seq_len=10)
        q, k, v = (_rand((B, H, N, D), seed=s) for s in range(3))
        ctx.prefill(q, k, v)   # seqlen = 8
        q1, k1, v1 = (_rand((B, H, 1, D), seed=s + 5) for s in range(3))
        ctx.step(q1, k1, v1)   # seqlen = 9
        q2, k2, v2 = (_rand((B, H, 2, D), seed=s + 8) for s in range(3))
        with pytest.raises(ValueError, match="max_seq_len"):
            ctx.step(q2, k2, v2)

    def test_step_without_prefill_cold_start(self):
        B, H, D = 1, 4, 64
        scale = 1.0 / math.sqrt(D)
        ctx = InferenceContext(B=B, H_kv=H, D=D)
        q, k, v = (_rand((B, H, 1, D), seed=s) for s in range(3))
        out = ctx.step(q, k, v, scale=scale)
        mx.eval(out)
        assert out.shape == (B, H, 1, D)
        assert ctx.seqlen == 1

    def test_step_output_shape(self):
        B, H, D = 2, 8, 128
        ctx = InferenceContext(B=B, H_kv=H, D=D)
        q, k, v = (_rand((B, H, 4, D), seed=s) for s in range(3))
        ctx.prefill(q, k, v)
        q1, k1, v1 = (_rand((B, H, 1, D), seed=s + 10) for s in range(3))
        out = ctx.step(q1, k1, v1)
        mx.eval(out)
        assert out.shape == (B, H, 1, D), f"unexpected shape {out.shape}"


class TestInferenceContextReset:
    def test_reset_clears_state(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64)
        q, k, v = (_rand((1, 2, 16, 64), seed=s) for s in range(3))
        ctx.prefill(q, k, v)
        assert ctx.seqlen == 16
        ctx.reset()
        assert ctx.seqlen == 0
        assert ctx.k_cache is None
        assert ctx.v_cache is None

    def test_reset_returns_self(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64)
        result = ctx.reset()
        assert result is ctx

    def test_reuse_after_reset(self):
        B, H, N, D = 1, 4, 16, 64
        scale = 0.125
        ctx = InferenceContext(B=B, H_kv=H, D=D)
        q, k, v = (_rand((B, H, N, D), seed=s) for s in range(3))
        ctx.prefill(q, k, v, scale=scale)
        ctx.reset()
        q2, k2, v2 = (_rand((B, H, N, D), seed=s + 10) for s in range(3))
        out = ctx.prefill(q2, k2, v2, scale=scale)
        mx.eval(out)
        assert ctx.seqlen == N
        assert out.shape == (B, H, N, D)


class TestInferenceContextManager:
    def test_context_manager_resets_on_exit(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64)
        q, k, v = (_rand((1, 2, 16, 64), seed=s) for s in range(3))
        with ctx:
            ctx.prefill(q, k, v)
            assert ctx.seqlen == 16
        assert ctx.seqlen == 0
        assert ctx.k_cache is None

    def test_context_manager_returns_self(self):
        ctx = InferenceContext(B=1, H_kv=2, D=64)
        with ctx as c:
            assert c is ctx


class TestInferenceContextGQA:
    def test_gqa_prefill_step(self):
        """GQA: H_q=8, H_kv=2. prefill + step shapes are correct."""
        B, H_q, H_kv, N, D = 1, 8, 2, 32, 64
        scale = 1.0 / math.sqrt(D)

        ctx = InferenceContext(B=B, H_kv=H_kv, D=D)
        q = _rand((B, H_q, N, D), seed=0)
        k = _rand((B, H_kv, N, D), seed=1)
        v = _rand((B, H_kv, N, D), seed=2)

        out_pre = ctx.prefill(q, k, v, scale=scale)
        mx.eval(out_pre)
        assert out_pre.shape == (B, H_q, N, D)
        assert ctx.k_cache.shape == (B, H_kv, N, D)

        q1 = _rand((B, H_q, 1, D), seed=3)
        k1 = _rand((B, H_kv, 1, D), seed=4)
        v1 = _rand((B, H_kv, 1, D), seed=5)
        out_step = ctx.step(q1, k1, v1, scale=scale)
        mx.eval(out_step)
        assert out_step.shape == (B, H_q, 1, D)
        assert ctx.k_cache.shape == (B, H_kv, N + 1, D)

    def test_import_from_package(self):
        """InferenceContext is importable from the top-level package."""
        from mlx_mfa import InferenceContext as IC
        assert IC is InferenceContext

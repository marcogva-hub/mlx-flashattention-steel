"""Regression tests for DiT/UNet dispatch routing.

Verifies that cross-attention and non-causal self-attention shapes are
routed correctly after the dispatch audit (2026-04-06).
"""

import mlx.core as mx
import pytest

from mlx_mfa import flash_attention
from mlx_mfa.dispatch_policy import should_use_mfa


class TestDiTNonCausalDispatch:
    """Verify dispatch for non-causal DiT/UNet self-attention shapes."""

    def test_dit_large_n_routes_mfa(self):
        """CogVideoX-class shape (N=70K, D=128) should route to MFA."""
        assert should_use_mfa(128, 70000, causal=False, is_m3_plus=False,
                              dtype=mx.float16)

    def test_unet_d64_n4096_routes_mfa(self):
        """UNet D=64 N=4096 is above the 2048 threshold."""
        assert should_use_mfa(64, 4096, causal=False, is_m3_plus=False,
                              dtype=mx.float16)

    def test_unet_d128_n4096_routes_mfa(self):
        """UNet D=128 N=4096 is above the 2048 threshold."""
        assert should_use_mfa(128, 4096, causal=False, is_m3_plus=False,
                              dtype=mx.float16)

    def test_small_n_routes_sdpa(self):
        """N=1024 D=64 non-causal is below the 2048 threshold."""
        assert not should_use_mfa(64, 1024, causal=False, is_m3_plus=False,
                                  dtype=mx.float16)


class TestCrossAttnSmallNkvDispatch:
    """Verify cross-attention routing with small N_kv."""

    def test_small_nkv_large_nq_routes_sdpa(self):
        """N_kv=77 N_q=70200 should route to SDPA (few K-tiles)."""
        assert not should_use_mfa(128, 70200, causal=False, is_m3_plus=False,
                                  dtype=mx.float16, kv_seq_len=77)

    def test_small_nkv_512_large_nq_routes_sdpa(self):
        """N_kv=512 N_q=100000 should route to SDPA."""
        assert not should_use_mfa(128, 100000, causal=False, is_m3_plus=False,
                                  dtype=mx.float16, kv_seq_len=512)

    def test_small_nkv_small_nq_not_affected(self):
        """N_kv=77 N_q=4096 is below N_q>8192 threshold -- unaffected."""
        assert should_use_mfa(128, 4096, causal=False, is_m3_plus=False,
                              dtype=mx.float16, kv_seq_len=77)

    def test_self_attn_none_kv_seq_len(self):
        """Self-attention (kv_seq_len=None) behaves as before."""
        assert should_use_mfa(128, 70200, causal=False, is_m3_plus=False,
                              dtype=mx.float16, kv_seq_len=None)


class TestCrossAttnLargeNkvDispatch:
    """Verify cross-attention routing when N_kv >> N_q."""

    def test_large_nkv_small_nq_routes_mfa(self):
        """LTX-2 audio->video (N_q=2000, N_kv=14000) should route to MFA."""
        assert should_use_mfa(64, 2000, causal=False, is_m3_plus=False,
                              dtype=mx.float16, kv_seq_len=14000)

    def test_large_nkv_large_nq_uses_standard_dispatch(self):
        """N_q=8192 N_kv=14000 falls through to standard thresholds."""
        assert should_use_mfa(64, 8192, causal=False, is_m3_plus=False,
                              dtype=mx.float16, kv_seq_len=14000)


class TestCrossAttnFunctional:
    """Functional tests: cross-attention produces correct output shapes."""

    def test_cross_attn_small_nkv(self):
        """Cross-attention with N_kv=77 runs and produces correct shape."""
        q = mx.random.normal((1, 8, 4096, 128), dtype=mx.float16)
        k = mx.random.normal((1, 8, 77, 128), dtype=mx.float16)
        v = mx.random.normal((1, 8, 77, 128), dtype=mx.float16)
        # Force evaluation of inputs then compute
        mx.synchronize()
        out = flash_attention(q, k, v)
        mx.synchronize()
        assert out.shape == (1, 8, 4096, 128)

    def test_unet_d64_noncausal(self):
        """UNet D=64 non-causal shape runs correctly."""
        q = mx.random.normal((1, 8, 4096, 64), dtype=mx.float16)
        k = mx.random.normal((1, 8, 4096, 64), dtype=mx.float16)
        v = mx.random.normal((1, 8, 4096, 64), dtype=mx.float16)
        mx.synchronize()
        out = flash_attention(q, k, v)
        mx.synchronize()
        assert out.shape == (1, 8, 4096, 64)

    def test_cross_attn_large_nkv_small_nq(self):
        """Cross-attention N_q < N_kv runs and produces correct shape."""
        q = mx.random.normal((1, 8, 2000, 64), dtype=mx.float16)
        k = mx.random.normal((1, 8, 8000, 64), dtype=mx.float16)
        v = mx.random.normal((1, 8, 8000, 64), dtype=mx.float16)
        mx.synchronize()
        out = flash_attention(q, k, v)
        mx.synchronize()
        assert out.shape == (1, 8, 2000, 64)


class TestLLMCausalNotRegressed:
    """Verify LLM causal dispatch thresholds are unaffected."""

    def test_causal_d128_n2048(self):
        assert should_use_mfa(128, 2048, causal=True, is_m3_plus=False,
                              dtype=mx.float16)

    def test_causal_d128_n8192(self):
        assert should_use_mfa(128, 8192, causal=True, is_m3_plus=False,
                              dtype=mx.float16)

    def test_causal_d64_n1024(self):
        assert should_use_mfa(64, 1024, causal=True, is_m3_plus=False,
                              dtype=mx.float16)

    def test_causal_small_routes_sdpa(self):
        """N=512 D=128 causal is below threshold (2048)."""
        assert not should_use_mfa(128, 512, causal=True, is_m3_plus=False,
                                  dtype=mx.float16)

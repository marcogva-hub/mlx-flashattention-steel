"""Tests for mlx-lm monkey-patch integration.

Requires mlx-lm to be installed (``pip install mlx-lm``).
These tests do NOT download any model — they verify the integration
plumbing using synthetic inputs.

Run with::

    pytest tests/test_mlx_lm_integration.py -v

Or include in the full suite::

    pytest tests/ -v
"""

from __future__ import annotations

import math
import inspect

import mlx.core as mx
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Optional import guards
# ---------------------------------------------------------------------------

try:
    import mlx_lm  # noqa: F401
    _mlx_lm_available = True
except ImportError:
    _mlx_lm_available = False

requires_mlx_lm = pytest.mark.skipif(
    not _mlx_lm_available,
    reason="mlx-lm not installed (pip install mlx-lm)",
)

try:
    from mlx_mfa.attention import _ext_available
    _ext_ok = _ext_available()
except Exception:
    _ext_ok = False

requires_ext = pytest.mark.skipif(not _ext_ok, reason="C++ extension not available")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_qkv(B=1, H_q=8, H_kv=8, N=64, D=128, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, H_q, N, D)).astype(dtype)
    k = mx.random.normal((B, H_kv, N, D)).astype(dtype)
    v = mx.random.normal((B, H_kv, N, D)).astype(dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Patch / unpatch tests
# ---------------------------------------------------------------------------

@requires_mlx_lm
class TestPatchUnpatch:
    """Verify the patch is applied and removed cleanly."""

    def test_patch_returns_true_when_ext_available(self):
        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm, unpatch_mlx_lm
        import mlx_lm.models.base as base_module
        original = base_module.scaled_dot_product_attention
        try:
            result = patch_mlx_lm()
            if _ext_ok:
                assert result is True
                assert base_module.scaled_dot_product_attention is not original
            else:
                assert result is False  # extension unavailable
        finally:
            unpatch_mlx_lm()

    def test_unpatch_restores_original(self):
        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm, unpatch_mlx_lm
        import mlx_lm.models.base as base_module
        original = base_module.scaled_dot_product_attention
        patch_mlx_lm()
        unpatch_mlx_lm()
        assert base_module.scaled_dot_product_attention is original

    def test_patch_is_idempotent(self):
        from mlx_mfa.integrations.mlx_lm import (
            is_patched, patch_mlx_lm, unpatch_mlx_lm,
        )
        import mlx_lm.models.base as base_module
        try:
            patch_mlx_lm()
            patched_fn = base_module.scaled_dot_product_attention
            patch_mlx_lm()  # second call must be a no-op
            # Function pointer must not change after second patch call
            assert base_module.scaled_dot_product_attention is patched_fn
        finally:
            unpatch_mlx_lm()

    def test_is_patched_state(self):
        from mlx_mfa.integrations.mlx_lm import is_patched, patch_mlx_lm, unpatch_mlx_lm
        assert not is_patched()
        patch_mlx_lm()
        if _ext_ok:
            assert is_patched()
        unpatch_mlx_lm()
        assert not is_patched()

    def test_unpatch_safe_without_patch(self):
        from mlx_mfa.integrations.mlx_lm import unpatch_mlx_lm
        # Must not raise
        unpatch_mlx_lm()
        unpatch_mlx_lm()


# ---------------------------------------------------------------------------
# Signature compatibility tests
# ---------------------------------------------------------------------------

@requires_mlx_lm
class TestSignatureCompatibility:
    """_steel_sdpa must accept every argument that mlx_lm SDPA accepts."""

    def test_signature_matches_mlx_lm(self):
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa
        from mlx_lm.models.base import scaled_dot_product_attention as orig

        steel_sig = inspect.signature(_steel_sdpa)
        orig_sig = inspect.signature(orig)

        steel_params = set(steel_sig.parameters)
        orig_params = set(orig_sig.parameters)

        # All params in the original must be accepted by the STEEL wrapper
        assert orig_params <= steel_params, (
            f"_steel_sdpa is missing params: {orig_params - steel_params}"
        )

    def test_steel_sdpa_callable_with_causal_string(self):
        """_steel_sdpa must handle mask='causal' without error."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm, unpatch_mlx_lm
        q, k, v = _make_qkv()
        patch_mlx_lm()
        try:
            out = _steel_sdpa(q, k, v, cache=None, scale=1/math.sqrt(128), mask="causal")
            mx.eval(out)
            assert list(out.shape) == [1, 8, 64, 128]
        finally:
            unpatch_mlx_lm()

    def test_steel_sdpa_callable_with_none_mask(self):
        """_steel_sdpa must handle mask=None (decode step)."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm, unpatch_mlx_lm
        q, k, v = _make_qkv(N=1)  # single-token decode
        patch_mlx_lm()
        try:
            out = _steel_sdpa(q, k, v, cache=None, scale=1/math.sqrt(128), mask=None)
            mx.eval(out)
            assert list(out.shape) == [1, 8, 1, 128]
        finally:
            unpatch_mlx_lm()


# ---------------------------------------------------------------------------
# Numerical correctness tests
# ---------------------------------------------------------------------------

@requires_mlx_lm
@requires_ext
class TestNumericalCorrectness:
    """Patched SDPA must produce numerically close output to original."""

    def test_causal_string_matches_original(self):
        """STEEL with mask='causal' must match original SDPA with mask='causal'."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm, unpatch_mlx_lm
        from mlx_lm.models.base import scaled_dot_product_attention as orig_sdpa
        q, k, v = _make_qkv(N=128, D=128)
        scale = 1.0 / math.sqrt(128)

        # Get original result before patching
        out_orig = orig_sdpa(q, k, v, cache=None, scale=scale, mask="causal")
        mx.eval(out_orig)

        patch_mlx_lm()
        try:
            out_steel = _steel_sdpa(q, k, v, cache=None, scale=scale, mask="causal")
            mx.eval(out_steel)
        finally:
            unpatch_mlx_lm()

        np.testing.assert_allclose(
            np.array(out_orig.astype(mx.float32)),
            np.array(out_steel.astype(mx.float32)),
            atol=1e-2,
            err_msg="STEEL causal != original SDPA causal"
        )

    def test_gqa_causal_string(self):
        """STEEL handles GQA (H_q=8, H_kv=2) with mask='causal'."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm, unpatch_mlx_lm
        from mlx_lm.models.base import scaled_dot_product_attention as orig_sdpa
        q, k, v = _make_qkv(H_q=8, H_kv=2, N=128, D=128)
        scale = 1.0 / math.sqrt(128)

        out_orig = orig_sdpa(q, k, v, cache=None, scale=scale, mask="causal")
        mx.eval(out_orig)

        patch_mlx_lm()
        try:
            out_steel = _steel_sdpa(q, k, v, cache=None, scale=scale, mask="causal")
            mx.eval(out_steel)
        finally:
            unpatch_mlx_lm()

        assert list(out_steel.shape) == [1, 8, 128, 128]
        np.testing.assert_allclose(
            np.array(out_orig.astype(mx.float32)),
            np.array(out_steel.astype(mx.float32)),
            atol=1e-2,
            err_msg="GQA STEEL causal != original"
        )

    def test_array_mask_falls_back_to_original(self):
        """A non-string mask (bool array) must produce same result as original."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm, unpatch_mlx_lm
        from mlx_lm.models.base import (
            create_causal_mask,
            scaled_dot_product_attention as orig_sdpa,
        )
        q, k, v = _make_qkv(N=64, D=128)
        scale = 1.0 / math.sqrt(128)
        bool_mask = create_causal_mask(64)

        out_orig = orig_sdpa(q, k, v, cache=None, scale=scale, mask=bool_mask)
        mx.eval(out_orig)

        patch_mlx_lm()
        try:
            out_steel = _steel_sdpa(q, k, v, cache=None, scale=scale, mask=bool_mask)
            mx.eval(out_steel)
        finally:
            unpatch_mlx_lm()

        # Must be identical (both use original SDPA path)
        np.testing.assert_array_equal(
            np.array(out_orig),
            np.array(out_steel),
            err_msg="Array mask fallback must be bit-identical to original"
        )


# ---------------------------------------------------------------------------
# Track K — Quantized KV Cache tests
# ---------------------------------------------------------------------------

def _make_quantized_kv(arr, group_size=64, bits=4):
    """Quantize a [B, H, N, D] float array → (q_data, scales, biases) tuple."""
    # mx.quantize expects [..., D] layout and quantizes the last dimension.
    return mx.quantize(arr, group_size=group_size, bits=bits)


@requires_mlx_lm
@requires_ext
class TestQuantizedKVCache:
    """Verify that _steel_sdpa dequantizes quantized K/V and uses STEEL.

    mlx-lm passes keys/values as (quantized_data, scales, biases) tuples
    when cache.bits is set.  After Track K, _steel_sdpa dequantizes them
    so STEEL can run instead of falling back.
    """

    def _mock_cache(self, bits=4, group_size=64):
        """Create a lightweight mock with the same attrs as QuantizedKVCache."""
        class _FakeQuantCache:
            pass
        c = _FakeQuantCache()
        c.bits = bits
        c.group_size = group_size
        return c

    def test_steel_activates_with_4bit_cache(self):
        """_steel_sdpa must NOT fall back to _original_sdpa for 4-bit cache."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, _original_sdpa, patch_mlx_lm
        patch_mlx_lm()

        B, H, N, D = 1, 8, 64, 128
        q, k, v = _make_qkv(B=B, H_q=H, H_kv=H, N=N, D=D)

        q_k = _make_quantized_kv(k)
        q_v = _make_quantized_kv(v)
        cache = self._mock_cache(bits=4)

        # Call _steel_sdpa; it must not raise and return a valid array.
        out = _steel_sdpa(q, q_k, q_v, cache, scale=1.0 / D**0.5, mask="causal")
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert out.dtype == q.dtype

    def test_output_close_to_dequantized_reference(self):
        """STEEL output with quantized K/V is close to reference with dequantized K/V."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm
        from mlx_mfa import flash_attention
        patch_mlx_lm()

        B, H, N, D = 1, 8, 64, 128
        group_size, bits = 64, 4
        q, k, v = _make_qkv(B=B, H_q=H, H_kv=H, N=N, D=D, seed=7)
        scale = 1.0 / D**0.5

        q_k = _make_quantized_kv(k, group_size, bits)
        q_v = _make_quantized_kv(v, group_size, bits)
        cache = self._mock_cache(bits=bits, group_size=group_size)

        # Our path: dequantize → STEEL
        out_steel = _steel_sdpa(q, q_k, q_v, cache, scale=scale, mask="causal")
        mx.eval(out_steel)

        # Reference: explicit dequantize → standard flash_attention
        k_deq = mx.dequantize(*q_k, group_size=group_size, bits=bits, dtype=q.dtype)
        v_deq = mx.dequantize(*q_v, group_size=group_size, bits=bits, dtype=q.dtype)
        out_ref = flash_attention(q, k_deq, v_deq, scale=scale, causal=True)
        mx.eval(out_ref)

        err = float(mx.max(mx.abs(out_steel.astype(mx.float32) - out_ref.astype(mx.float32))))
        assert err < 1e-3, f"Quantized KV output differs: max_err={err:.6f}"

    def test_causal_with_quantized_kv(self):
        """Causal attention with 4-bit K/V: output is non-zero and finite."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm
        patch_mlx_lm()

        B, H, N, D = 1, 4, 128, 128
        q, k, v = _make_qkv(B=B, H_q=H, H_kv=H, N=N, D=D, seed=42)
        scale = 1.0 / D**0.5

        q_k = _make_quantized_kv(k)
        q_v = _make_quantized_kv(v)
        cache = self._mock_cache()

        out = _steel_sdpa(q, q_k, q_v, cache, scale=scale, mask="causal")
        mx.eval(out)
        assert out.shape == (B, H, N, D)
        assert mx.all(mx.isfinite(out)).item()
        assert not mx.all(out == 0).item()

    def test_gqa_with_quantized_kv(self):
        """GQA (8 Q heads, 2 KV heads) + 4-bit cache: correct shapes and finite."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, patch_mlx_lm
        patch_mlx_lm()

        B, H_q, H_kv, N, D = 1, 8, 2, 64, 128
        mx.random.seed(11)
        q = mx.random.normal((B, H_q, N, D)).astype(mx.float16)
        k = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        v = mx.random.normal((B, H_kv, N, D)).astype(mx.float16)
        scale = 1.0 / D**0.5

        q_k = _make_quantized_kv(k)
        q_v = _make_quantized_kv(v)
        cache = self._mock_cache()

        out = _steel_sdpa(q, q_k, q_v, cache, scale=scale, mask="causal")
        mx.eval(out)
        assert out.shape == (B, H_q, N, D)
        assert mx.all(mx.isfinite(out)).item()

    def test_sinks_still_falls_back_with_quantized(self):
        """sinks != None must always fall back, even with quantized cache."""
        from mlx_mfa.integrations.mlx_lm import _steel_sdpa, _original_sdpa, patch_mlx_lm
        patch_mlx_lm()

        B, H, N, D = 1, 8, 64, 128
        q, k, v = _make_qkv(B=B, H_q=H, H_kv=H, N=N, D=D)
        q_k = _make_quantized_kv(k)
        q_v = _make_quantized_kv(v)
        cache = self._mock_cache()
        dummy_sinks = mx.zeros((1,), dtype=mx.float16)

        # mlx-lm itself raises for sinks + quantized cache.
        # Our code must also fall back (and let _original_sdpa raise or handle).
        # We verify our code does NOT attempt to dequantize when sinks is set.
        called_original = False

        def mock_original(*args, **kwargs):
            nonlocal called_original
            called_original = True
            return mx.zeros((B, H, N, D), dtype=q.dtype)

        import mlx_mfa.integrations.mlx_lm as mod
        old = mod._original_sdpa
        mod._original_sdpa = mock_original
        try:
            _steel_sdpa(q, q_k, q_v, cache, scale=0.1, mask=None, sinks=dummy_sinks)
        finally:
            mod._original_sdpa = old

        assert called_original, "_steel_sdpa must fall back to _original_sdpa when sinks is set"

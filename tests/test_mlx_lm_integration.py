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

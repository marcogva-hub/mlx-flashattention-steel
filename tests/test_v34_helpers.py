"""V6NAX routing helpers (`_v6nax_eligible`, `_v6nax_backward_vjp`) — direct tests.

Per Sprint v2.38.0 DP2-HIGH-01 compound finding (audit M4-MEDIUM-01).
The helpers were extracted from `_make_mfa_custom` to:
- Deduplicate the V6NAX-eligibility predicate (was triplicated)
- Make the 3-kernel V6NAX backward dispatch testable in isolation
- Reduce `_make_mfa_custom` body by ~80 LOC

These tests verify the helpers' behavior in isolation.  End-to-end
behavior via `mx.grad(flash_attention(...))` is covered by
`test_flash_attention_v6nax_backward.py` and `test_release_notes_perf_claims.py`
(the §Z public-API-path tests).

Three-axis coverage per `CLAUDE_V6_NAX.md` §3.5 amended:
- Axis 1 (output sanity): predicate truth table + dispatch correctness
- Axis 2 (path entered via PUBLIC API): covered in
  test_flash_attention_v6nax_backward.py — these helper tests are direct
  Python-import-level, not flash_attention-level
- Axis 3 (edges preserved): causal exclusion + dtype exclusion + env
  toggle + has_nax gate
"""
from __future__ import annotations

import math

import numpy as np
import mlx.core as mx
import pytest

from mlx_mfa.attention import _v6nax_eligible, _v6nax_backward_vjp


_AE = getattr(mx, "async_" + "eval")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every MFA_* env var before each test; tests opt in via
    monkeypatch.setenv explicitly."""
    for k in list(__import__("os").environ):
        if k.startswith("MFA_"):
            monkeypatch.delenv(k, raising=False)
    yield


# ---------------------------------------------------------------------------
# _v6nax_eligible() — predicate truth table
# ---------------------------------------------------------------------------
class TestV6NAXEligible:
    """All cases assume `_get_has_nax_cached()` returns True (M5 Max).
    On non-NAX hardware these tests would all return False from the
    has_nax gate regardless of other params; skipped via the pytest
    skipif on the class once we add hw-detection (out of scope for
    v2.38.0)."""

    def test_d64_fp16_noncausal_env_set_returns_true(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.float16, causal=False) is True

    def test_d128_fp16_noncausal_env_set_returns_true(self, monkeypatch):
        """D=128 IS eligible at the helper level (env=1, fp16, !causal).
        The flash_attention()-level carve-out (`_v6nax_backward_carveout`
        in dispatch_policy.py) is what gates D=128 out from auto-default.
        At `_make_mfa_custom`-level (post-routing), all eligible D values
        engage."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(128, mx.float16, causal=False) is True

    def test_d64_bf16_noncausal_env_set_returns_true(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.bfloat16, causal=False) is True

    def test_d64_fp16_causal_returns_true(self, monkeypatch):
        """v2.50 Phase 4b-complete (Prompt 4 Section B): causal is now
        eligible.  Root cause was a missed dispatch gate in
        MFAV6Forward::eval_gpu() routing causal forward to STEEL legacy
        (log2-domain lse) instead of V6NAX (natural-log lse).  Fix lifts
        the dispatch gate; V6NAX backward causal now produces correct
        gradients (RMSE within FP floor)."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.float16, causal=True) is True

    def test_d64_fp32_returns_false(self, monkeypatch):
        """fp32 excluded: V6NAX backward kernels are fp16/bf16 only."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(64, mx.float32, causal=False) is False

    def test_d256_returns_false(self, monkeypatch):
        """head_dim ∉ {64, 128} excluded."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(256, mx.float16, causal=False) is False

    def test_d32_returns_false(self, monkeypatch):
        """head_dim ∉ {64, 128} excluded (small D)."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        assert _v6nax_eligible(32, mx.float16, causal=False) is False

    def test_env_unset_returns_false(self):
        """SHIP_OPT_IN: without explicit env, helper returns False
        even for fully-qualifying shape."""
        assert _v6nax_eligible(64, mx.float16, causal=False) is False

    def test_env_set_to_zero_returns_false(self, monkeypatch):
        """The explicit false value disables the opt-in."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "0")
        assert _v6nax_eligible(64, mx.float16, causal=False) is False

    def test_env_set_to_yes_raises(self, monkeypatch):
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "yes")
        with pytest.raises(ValueError, match="must be '0' or '1'"):
            _v6nax_eligible(64, mx.float16, causal=False)


# ---------------------------------------------------------------------------
# _v6nax_backward_vjp() — output shapes + dispatch routing
# ---------------------------------------------------------------------------
def _make(B, H, qL, kL, D, seed, dtype):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, kL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, kL, D)) * 0.1).astype(dtype)
    _AE(q, k, v); mx.synchronize()
    return q, k, v


class TestV6NAXBackwardVjp:
    """Direct invocation of `_v6nax_backward_vjp` with synthetic inputs.

    Caller (`_make_mfa_custom._backward`) is responsible for verifying
    V6NAX eligibility via `_v6nax_eligible()` before invoking the helper.
    These tests bypass that — they exercise the helper on a known-good
    eligible shape (D=64, qL=2048, fp16, non-causal) with env set.
    """

    def test_output_shapes_match_inputs(self, monkeypatch):
        """dQ shape == q shape; dK, dV shapes == k, v shapes."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        B, H, qL, kL, D = 1, 4, 2048, 2048, 64
        q, k, v = _make(B, H, qL, kL, D, 42, mx.float16)
        scale = 1.0 / math.sqrt(D)

        # Need (O, L) from a V6NAX forward; use the same wrapper
        # `_make_mfa_custom` would have produced.
        from mlx_mfa._ext import v6_nax_forward
        O, L = v6_nax_forward(q, k, v, False, True)  # force_v6nax=True
        dO = mx.ones_like(O)
        _AE(O, L, dO); mx.synchronize()

        dQ, dK, dV = _v6nax_backward_vjp(q, k, v, O, L, dO, scale)
        _AE(dQ, dK, dV); mx.synchronize()

        assert dQ.shape == q.shape
        assert dK.shape == k.shape
        assert dV.shape == v.shape
        assert dQ.dtype == q.dtype
        assert dK.dtype == k.dtype
        assert dV.dtype == v.dtype

    def test_split_path_is_default(self, monkeypatch):
        """Without `MFA_V6BWD_USE_FUSED=1`, helper takes the
        WM=4 split path (Phase 2.O2).  No assertion on which
        binding was called (would require instrumentation); but
        verify output is finite + reasonable magnitude."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        monkeypatch.delenv("MFA_V6BWD_USE_FUSED", raising=False)
        q, k, v = _make(1, 4, 2048, 2048, 64, 43, mx.float16)
        scale = 1.0 / math.sqrt(64)

        from mlx_mfa._ext import v6_nax_forward
        O, L = v6_nax_forward(q, k, v, False, True)
        dO = mx.ones_like(O)
        _AE(O, L, dO); mx.synchronize()

        dQ, dK, dV = _v6nax_backward_vjp(q, k, v, O, L, dO, scale)
        _AE(dQ, dK, dV); mx.synchronize()

        # Finite + non-trivial magnitude
        for g, name in ((dQ, "dQ"), (dK, "dK"), (dV, "dV")):
            arr = np.array(g.astype(mx.float32))
            assert np.isfinite(arr).all(), f"{name} contains non-finite values"
            assert np.abs(arr).max() > 0, f"{name} is all zeros"

    def test_fused_path_via_env(self, monkeypatch):
        """With `MFA_V6BWD_USE_FUSED=1`, helper takes the legacy
        WM=1 fused dK/dV path.  Verify output is finite + matches the
        split path in shape (numerical match within FP16 floor)."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        q, k, v = _make(1, 4, 2048, 2048, 64, 44, mx.float16)
        scale = 1.0 / math.sqrt(64)

        from mlx_mfa._ext import v6_nax_forward
        O, L = v6_nax_forward(q, k, v, False, True)
        dO = mx.ones_like(O)
        _AE(O, L, dO); mx.synchronize()

        monkeypatch.setenv("MFA_V6BWD_USE_FUSED", "1")
        dQ_f, dK_f, dV_f = _v6nax_backward_vjp(q, k, v, O, L, dO, scale)
        _AE(dQ_f, dK_f, dV_f); mx.synchronize()

        monkeypatch.delenv("MFA_V6BWD_USE_FUSED")
        dQ_s, dK_s, dV_s = _v6nax_backward_vjp(q, k, v, O, L, dO, scale)
        _AE(dQ_s, dK_s, dV_s); mx.synchronize()

        # Both paths produce same shape; gradients within FP16 noise.
        # (dQ is computed by the SAME kernel in both paths, so it must
        # match exactly.  dK/dV come from different kernels in the two
        # paths but compute the same mathematical operation.)
        assert dQ_f.shape == dQ_s.shape
        assert dK_f.shape == dK_s.shape
        assert dV_f.shape == dV_s.shape

        def _rmse(a, b):
            d = np.abs(np.array(a.astype(mx.float32)) -
                       np.array(b.astype(mx.float32)))
            return float(np.sqrt((d ** 2).mean()))

        # dQ identical (same kernel)
        assert _rmse(dQ_f, dQ_s) < 1e-6
        # dK/dV within FP16 floor (different kernels, same math)
        assert _rmse(dK_f, dK_s) < 1e-2
        assert _rmse(dV_f, dV_s) < 1e-2

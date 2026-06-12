"""Phase II-13 — Pattern #8 hardening: telemetry-backed hook coverage.

Two exhibits of Pattern #8 (silent hook fallback / hooks not patching
what users call): the v2.50.1 KD-6 silent-block and II-7's nn.Conv3d
gap (nn.Conv3d calls mx.conv3d; only mx.conv_general was patched —
zero NAX engagement for every standard mlx.nn model).  These tests
make the class structurally detected:

1. ENGAGEMENT: each user-facing entry point that SHOULD reach an
   mlx-mfa optimization is exercised the way a user would, and hook
   telemetry must show executed > 0 for it.
2. COMPLETENESS: install_hooks() must patch the registered underlying
   ops (the markers prove the swap happened).

If you add an acceleration with a new mx.* surface: register it in
_EXPECTED_PATCHED and add an engagement test, or the gap class
returns.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_mfa import _auto_hooks, get_device_info

_HAS_M5 = bool(get_device_info().get("is_m5_plus", False))

# Underlying ops install_hooks MUST patch (the structural chokepoints).
_EXPECTED_PATCHED = ["conv_general", "conv3d"]


@pytest.fixture(autouse=True)
def _hooks():
    _auto_hooks.install_hooks()
    yield


class TestCompleteness:
    def test_install_hooks_patches_registered_ops(self):
        for sym in _EXPECTED_PATCHED:
            fn = getattr(mx, sym)
            assert getattr(fn, "__mlx_mfa_hook__", False), (
                f"mx.{sym} is NOT patched by install_hooks() — "
                f"Pattern #8 coverage gap (the nn.Conv3d class)"
            )


@pytest.mark.skipif(not _HAS_M5, reason="NAX conv engagement requires M5+")
class TestEngagement:
    """The tests that would have caught the nn.Conv3d gap."""

    def _reset_stats(self):
        stats = _auto_hooks._HOOK_EXECUTION_STATS
        stats["executed"].clear()
        stats["fallback"].clear()
        stats["fallback_reasons"].clear()

    def test_nn_conv3d_engages_nax(self):
        self._reset_stats()
        conv = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        conv.weight = (mx.random.normal(conv.weight.shape) * 0.05).astype(mx.float16)
        if "bias" in conv:
            conv.bias = conv.bias.astype(mx.float16)
        x = mx.random.normal((1, 4, 16, 16, 32), dtype=mx.float16)
        mx.eval(conv(x))
        stats = _auto_hooks.get_hook_stats()
        assert stats["executed"].get("conv3d_nax_forward", 0) > 0, (
            f"nn.Conv3d did NOT engage the NAX conv hook "
            f"(stats={stats}) — the II-7 coverage gap regressed"
        )

    def test_mx_conv3d_direct_engages_nax(self):
        self._reset_stats()
        x = mx.random.normal((1, 4, 16, 16, 32), dtype=mx.float16)
        w = mx.random.normal((32, 3, 3, 3, 32), dtype=mx.float16) * 0.05
        mx.eval(mx.conv3d(x, w, stride=(1, 1, 1), padding=(1, 1, 1)))
        stats = _auto_hooks.get_hook_stats()
        assert stats["executed"].get("conv3d_nax_forward", 0) > 0

    def test_mx_conv_general_engages_nax(self):
        self._reset_stats()
        x = mx.random.normal((1, 4, 16, 16, 32), dtype=mx.float16)
        w = mx.random.normal((32, 3, 3, 3, 32), dtype=mx.float16) * 0.05
        mx.eval(mx.conv_general(x, w, stride=1, padding=1))
        stats = _auto_hooks.get_hook_stats()
        assert stats["executed"].get("conv3d_nax_forward", 0) > 0

    def test_ineligible_path_counts_fallback_not_silence(self):
        """Pattern #8's other half: ineligible calls must COUNT as
        fallback (visible), never vanish from telemetry."""
        self._reset_stats()
        x = mx.random.normal((1, 4, 16, 16, 32), dtype=mx.float16)
        w = mx.random.normal((32, 2, 2, 2, 32), dtype=mx.float16)  # k=2: ineligible
        mx.eval(mx.conv3d(x, w))
        stats = _auto_hooks.get_hook_stats()
        assert stats["fallback"].get("conv3d_nax_forward", 0) > 0, (
            "ineligible conv3d call left no telemetry trace — silent "
            "fallback (Pattern #8)"
        )

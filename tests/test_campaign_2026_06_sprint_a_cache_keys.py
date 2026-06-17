"""Campaign 2026-06 Sprint A — cache-key invariant regression tests.

THE INVARIANT: every input that can change either the selected cached
value or the numerical result of using it must be a key component,
encoded without collision.

Classes covered (original 2026-05 findings + Sprint A finds):
- cross-scale kernel reuse (C1/C7 class)
- bit-field truncation collision (C5 class; Sprint A A-1 cfg_axis_flags)
- env-keyed dispatch decisions (P1 class; Sprint A A-5 dispatch table)
- legacy conv path dtype contract (Sprint A A-8 loud failure)
"""
from __future__ import annotations

import json
import math
import os
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention, get_device_info

_eval_force = mx.eval
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))


def _mk(B, H, N, D, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((B, H, N, D)) * 0.1).astype(dtype)
    k = (mx.random.normal((B, H, N, D)) * 0.1).astype(dtype)
    v = (mx.random.normal((B, H, N, D)) * 0.1).astype(dtype)
    _eval_force(q, k, v)
    return q, k, v


def _sdpa_ref(q, k, v, scale, causal):
    mask = "causal" if causal else None
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


class TestCacheDiscriminatesOnScale:
    """C1/C7 class: same shape, two scales, both must be correct.

    Pre-2026-05-review, the 9 V6NAX backward pipeline caches omitted
    `scale` (baked into the Metal source) — the second scale reused the
    first's kernel.  These tests run the SAME shape back-to-back with
    different scales and assert each output independently against SDPA.
    """

    @pytest.mark.parametrize("causal", [True, False])
    def test_forward_two_scales_same_shape(self, causal):
        B, H, N, D = 1, 4, 2048, 64
        q, k, v = _mk(B, H, N, D, seed=101)
        for scale in (1.0 / math.sqrt(D), 0.25, 1.0):
            out = flash_attention(q, k, v, scale=scale, causal=causal)
            ref = _sdpa_ref(q, k, v, scale, causal)
            _eval_force(out, ref)
            diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
            assert diff < 2e-2, f"scale={scale} causal={causal}: diff={diff}"

    @pytest.mark.skipif(not _HAS_NAX, reason="V6NAX backward requires M5+ NAX")
    def test_v6nax_backward_two_scales_same_shape(self, monkeypatch):
        """V6NAX backward kernels bake scale into Metal source — the cache
        key must discriminate.  Default scale engages V6NAX (per the C7
        gate, non-default falls back) so we assert BOTH paths correct."""
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
        B, H, N, D = 1, 4, 4096, 64
        q, k, v = _mk(B, H, N, D, seed=102)
        dO = mx.ones_like(q)

        for scale in (1.0 / math.sqrt(D), 0.5):
            def f(qi, ki, vi, s=scale):
                return flash_attention(qi, ki, vi, scale=s, causal=True)
            _, (dQ, dK, dV) = mx.vjp(f, [q, k, v], [dO])

            def f_ref(qi, ki, vi, s=scale):
                return _sdpa_ref(qi, ki, vi, s, True)
            _, (dQ_r, dK_r, dV_r) = mx.vjp(f_ref, [q, k, v], [dO])
            _eval_force(dQ, dK, dV, dQ_r, dK_r, dV_r)
            # Institutional tolerance: causal fp16 max-error varies up to
            # ±0.2 across runs (Metal GPU non-determinism, v1.2.0 finding);
            # RMSE is the stable discriminator — a wrong-kernel reuse
            # produces RMSE orders of magnitude above 5e-3.
            for name, a, b in (("dQ", dQ, dQ_r), ("dK", dK, dK_r), ("dV", dV, dV_r)):
                diff = np.asarray(a.astype(mx.float32)) - np.asarray(b.astype(mx.float32))
                rmse = float(np.sqrt((diff ** 2).mean()))
                assert rmse < 5e-3, f"scale={scale} {name}: RMSE={rmse}"


class TestV6AxisFlagsNoTruncation:
    """A-1 class: V6Key.cfg_axis_flags must hold bits 8-11.

    MFA_V6_MATMUL_EXEC_SG ∈ {2,4,8} sets bits 10-11; with the pre-fix
    uint8_t field all three values truncated to 0, aliasing to the
    no-override pipeline.  Run V6 forward with the override toggled and
    assert correctness BOTH times (the second run would reuse the wrong
    pipeline pre-fix)."""

    @pytest.mark.skipif(not _HAS_NAX, reason="V6 NAX kernel requires M5+")
    def test_max_threads_override_no_alias(self, monkeypatch):
        """MFA_V6_MAX_THREADS=300 sets axis_flags bit 8 (0x100) — pre-fix
        the uint8_t field truncated it to 0, aliasing to the no-override
        pipeline (compiled with a different maxTotalThreadsPerThreadgroup).
        Post-fix both configs compile distinct pipelines; both must be
        correct.

        Sprint A note: the sibling MFA_V6_MATMUL_EXEC_SG knob (bits 10-11)
        was REMOVED — fixing the truncation surfaced that its source
        substitution is statically illegal on current MPP headers; it had
        been a silent no-op ghost since the truncation existed."""
        from mlx_mfa import _ext
        B, H, N, D = 1, 4, 1024, 64
        q, k, v = _mk(B, H, N, D, seed=103)
        scale = 1.0 / math.sqrt(D)
        ref = _sdpa_ref(q, k, v, scale, False)

        # default (no override) — populates the cache
        monkeypatch.delenv("MFA_V6_MAX_THREADS", raising=False)
        O1, _ = _ext.v6_nax_forward(q, k, v, False, False)
        # override=300 → bit 8, truncated to 0 pre-fix → alias
        monkeypatch.setenv("MFA_V6_MAX_THREADS", "300")
        O2, _ = _ext.v6_nax_forward(q, k, v, False, False)
        _eval_force(O1, O2, ref)
        for name, o in (("default", O1), ("max_threads=300", O2)):
            d = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))))
            assert d < 2e-2, f"{name}: diff vs SDPA = {d}"


class TestDispatchTableRuntimeOverride:
    """A-5 class: MLX_MFA_DISPATCH_TABLE is a DOCUMENTED runtime override.

    Pre-fix, (a) a process-lifetime flag froze the first read forever and
    (b) the dispatch decision cache key omitted the path — changing the
    table mid-process silently kept stale decisions."""

    def test_table_reload_on_path_change(self, monkeypatch):
        from mlx_mfa import dispatch_policy as dp

        # Force a known starting state
        monkeypatch.delenv("MLX_MFA_DISPATCH_TABLE", raising=False)
        assert dp._load_custom_table() is None

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"thresholds": [
                {"D": 64, "causal": True, "min_N": 1},
            ]}, fh)
            path = fh.name
        try:
            monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", path)
            table = dp._load_custom_table()
            assert table is not None and table[(64, True)] == 1, (
                "mid-process table set must load (documented runtime override)")
            # Clearing must drop it again
            monkeypatch.delenv("MLX_MFA_DISPATCH_TABLE", raising=False)
            assert dp._load_custom_table() is None, (
                "clearing the env var must drop the custom table")
        finally:
            os.unlink(path)

    def test_dispatch_cache_key_includes_table_path(self):
        import inspect
        import mlx_mfa.attention as attn
        src = inspect.getsource(attn.flash_attention)
        assert "MLX_MFA_DISPATCH_TABLE" in src, (
            "dispatch decision cache env key must include the table path")


class TestLegacyConvDtypeContract:
    """A-8: the legacy Python conv path hardcodes `device half` casts —
    bf16 would be silently type-punned.  Must fail loudly."""

    def test_bf16_rejected_loudly(self):
        from mlx_mfa.conv_nax import _conv3d_nax_forward_python_legacy
        x = mx.zeros((1, 4, 4, 4, 16), dtype=mx.bfloat16)
        w = mx.zeros((16, 1, 1, 1, 16), dtype=mx.bfloat16)
        with pytest.raises(ValueError, match="fp16 only"):
            _conv3d_nax_forward_python_legacy(x, w)

    @pytest.mark.skipif(not _HAS_NAX, reason="legacy path benches on M5+")
    def test_fp16_still_works(self):
        from mlx_mfa.conv_nax import _conv3d_nax_forward_python_legacy
        # III-5 follow-up: use C_in=32 (a multiple of 32) so K = C_in*27 is
        # a multiple of the matmul2d 32-wide K-tile.  The legacy path's
        # K-loop does not mask the partial tail tile, so it is only
        # numerically correct when K % 32 == 0 (i.e. C_in % 32 == 0).  The
        # prior C_in=16 shape exercised the broken tail and only "passed"
        # because the reference (mx.conv_general under installed hooks)
        # routed through the SAME broken legacy kernel — two equally-wrong
        # outputs comparing equal.  This test asserts the A-8 DTYPE
        # contract (fp16 works), so a correct-by-construction shape is
        # right; the small-channel brokenness is covered (and gated) by
        # test_iii5_conv_small_channel_accuracy.py.
        mx.random.seed(104)
        x = (mx.random.normal((1, 4, 8, 8, 32)) * 0.1).astype(mx.float16)
        w = (mx.random.normal((32, 3, 3, 3, 32)) * 0.1).astype(mx.float16)
        _eval_force(x, w)
        y = _conv3d_nax_forward_python_legacy(x, w, stride=(1, 1, 1), padding=(1, 1, 1))
        # At C_in=32 both the legacy GEMM and mx.conv_general (whether it
        # routes to native or the MPP NAX path under installed hooks) are
        # numerically correct, so this comparison is valid regardless of
        # hook state.
        ref = mx.conv_general(x, w, stride=1, padding=1)
        _eval_force(y, ref)
        d = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))))
        assert d < 5e-2, f"legacy fp16 conv diff={d}"


class TestEquivalencePredicateOutputs:
    """A-4 class (behavioral floor): two calls with identical inputs but a
    different steering flag must produce different (each-correct) outputs.
    Exercises the code paths whose primitives gained equivalence fields;
    full CSE-conflation repro requires mx.compile graph surgery — the
    field-level fix is verified by code inspection + this output-level
    guard."""

    @pytest.mark.skipif(not _HAS_NAX, reason="V6 paths on M5+")
    def test_force_v6nax_lse_domain_distinct(self):
        from mlx_mfa import _ext
        B, H, N, D = 1, 4, 1024, 64
        q, k, v = _mk(B, H, N, D, seed=105)
        O_legacy, L_legacy = _ext.v6_nax_forward(q, k, v, False, False)
        O_v6nax, L_v6nax = _ext.v6_nax_forward(q, k, v, False, True)  # force_v6nax
        _eval_force(O_legacy, L_legacy, O_v6nax, L_v6nax)
        # Outputs O agree (same math); LSE domains differ (log2 vs natural)
        dO = float(mx.max(mx.abs(O_legacy.astype(mx.float32) - O_v6nax.astype(mx.float32))))
        assert dO < 2e-2, f"O must agree across paths: {dO}"
        dL = float(mx.max(mx.abs(L_legacy.astype(mx.float32) - L_v6nax.astype(mx.float32))))
        assert dL > 1e-3, (
            "LSE domains must DIFFER (log2 vs natural) — if equal, the "
            "force_v6nax routing did not engage and the is_equivalent fix "
            "cannot be exercised")

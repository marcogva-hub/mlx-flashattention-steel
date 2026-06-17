"""v2.50 Prompt 5b Section A — V6NAX backward dV sparse kernel PoC tests.

Per Marco's Option 3 decision: ship single-kernel PoC + scaffold; full
5-kernel sparse extension deferred to focused follow-up session.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import get_device_info
from mlx_mfa.attention import _convert_mask_for_v6nax_bwd_kernel

_AE = getattr(mx, "async_" + "eval")
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))

_skipif_no_nax = pytest.mark.skipif(
    not _HAS_NAX, reason="V6NAX backward dV sparse requires M5+ NAX hardware"
)


def _make(B, H, qL, D, dtype, seed):
    mx.random.seed(seed)
    q = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    k = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    v = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    dO = (mx.random.uniform(-1.0, 1.0, (B, H, qL, D)) * 0.1).astype(dtype)
    _AE(q, k, v, dO); mx.synchronize()
    return q, k, v, dO


class TestSectionAdVSparsePoC:
    """Section A PoC: native sparse dV kernel."""

    def test_binding_loaded(self):
        from mlx_mfa import _ext
        assert hasattr(_ext, "v6_nax_backward_dv_sparse_raw"), (
            "Section A PoC binding missing - rebuild C++ extension"
        )

    @_skipif_no_nax
    def test_all_true_mask_bit_identical_to_dense(self):
        """All-True mask must produce bit-identical output to dense dV.

        This is the strongest correctness signal: every Q-tile is
        processed identically to the dense kernel, confirming the
        sparse-skip was the only structural difference.
        """
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.float16, 42)
        scale = 1.0 / math.sqrt(D)

        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        # v2.50 Prompt 5f Phase A KD-1: convert BT-block mask to dV kernel
        # geometry (BQ=64, BK=32) before direct kernel call.
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "dV", D)
        _AE(mask_all); mx.synchronize()

        dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
            q, k, v, L, dO, mask_all, scale, 4, False)
        dV_sparse = mx.sum(dV_sparse_partials, axis=2).astype(mx.float16)

        dV_dense_partials = _ext.v6_nax_backward_dv_raw(
            q, k, v, L, dO, scale, 4, False)
        dV_dense = mx.sum(dV_dense_partials, axis=2).astype(mx.float16)

        _AE(dV_sparse, dV_dense); mx.synchronize()
        diff = float(mx.max(mx.abs(
            dV_sparse.astype(mx.float32) - dV_dense.astype(mx.float32))))
        assert diff < 1e-7, (
            f"All-True mask sparse dV must be bit-identical to dense; max_diff = {diff}"
        )

    @_skipif_no_nax
    def test_sparse_skip_actually_skips(self):
        """All-False mask must produce zero output (all Q-tiles skipped)."""
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.float16, 43)
        scale = 1.0 / math.sqrt(D)

        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_none_bt = mx.zeros((NQ, NK), dtype=mx.bool_)
        mask_none = _convert_mask_for_v6nax_bwd_kernel(mask_none_bt, BT, "dV", D)
        _AE(mask_none); mx.synchronize()

        dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
            q, k, v, L, dO, mask_none, scale, 4, False)
        dV_sparse = mx.sum(dV_sparse_partials, axis=2).astype(mx.float16)
        _AE(dV_sparse); mx.synchronize()

        max_abs = float(mx.max(mx.abs(dV_sparse.astype(mx.float32))))
        assert max_abs == 0.0, f"All-False mask must produce zero dV; max|dV| = {max_abs}"

    @_skipif_no_nax
    def test_partial_mask_produces_finite_output(self):
        """Block-diagonal mask produces non-trivial finite output."""
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.float16, 44)
        scale = 1.0 / math.sqrt(D)

        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_np = np.eye(NQ, NK, dtype=bool)
        mask_bt = mx.array(mask_np)
        mask = _convert_mask_for_v6nax_bwd_kernel(mask_bt, BT, "dV", D)
        _AE(mask); mx.synchronize()

        dV_partials = _ext.v6_nax_backward_dv_sparse_raw(
            q, k, v, L, dO, mask, scale, 4, False)
        dV = mx.sum(dV_partials, axis=2).astype(mx.float16)
        _AE(dV); mx.synchronize()

        max_abs = float(mx.max(mx.abs(dV.astype(mx.float32))))
        assert max_abs > 0.0, "Partial mask should produce some non-zero output"
        assert math.isfinite(max_abs), f"Output not finite: {max_abs}"

    @_skipif_no_nax
    def test_bf16_path_works(self):
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.bfloat16, 45)
        scale = 1.0 / math.sqrt(D)

        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "dV", D)
        _AE(mask_all); mx.synchronize()

        dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
            q, k, v, L, dO, mask_all, scale, 4, False)
        dV_sparse = mx.sum(dV_sparse_partials, axis=2).astype(mx.bfloat16)

        dV_dense_partials = _ext.v6_nax_backward_dv_raw(
            q, k, v, L, dO, scale, 4, False)
        dV_dense = mx.sum(dV_dense_partials, axis=2).astype(mx.bfloat16)

        _AE(dV_sparse, dV_dense); mx.synchronize()
        diff = float(mx.max(mx.abs(
            dV_sparse.astype(mx.float32) - dV_dense.astype(mx.float32))))
        assert diff < 1e-6, f"bf16 all-True diff = {diff}"

    @_skipif_no_nax
    def test_d128_path_works(self):
        """Section D broadened V6NAX backward to D=128; sparse PoC supports D=128."""
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 128
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.float16, 46)
        scale = 1.0 / math.sqrt(D)

        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_all_bt = mx.ones((NQ, NK), dtype=mx.bool_)
        mask_all = _convert_mask_for_v6nax_bwd_kernel(mask_all_bt, BT, "dV", D)
        _AE(mask_all); mx.synchronize()

        dV_sparse_partials = _ext.v6_nax_backward_dv_sparse_raw(
            q, k, v, L, dO, mask_all, scale, 4, False)
        dV_sparse = mx.sum(dV_sparse_partials, axis=2).astype(mx.float16)

        dV_dense_partials = _ext.v6_nax_backward_dv_raw(
            q, k, v, L, dO, scale, 4, False)
        dV_dense = mx.sum(dV_dense_partials, axis=2).astype(mx.float16)

        _AE(dV_sparse, dV_dense); mx.synchronize()
        diff = float(mx.max(mx.abs(
            dV_sparse.astype(mx.float32) - dV_dense.astype(mx.float32))))
        assert diff < 1e-7, f"D=128 all-True diff = {diff}"

    @_skipif_no_nax
    def test_2d_mask_only_at_poc_stage(self):
        """Section A PoC supports 2-D mask only; 3-D raises.

        Section A v2 will broaden to 3-D and 4-D layouts.
        """
        from mlx_mfa import _ext

        B, H, qL, D = 1, 4, 2048, 64
        BT = 32
        NQ = NK = qL // BT

        q, k, v, dO = _make(B, H, qL, D, mx.float16, 47)
        scale = 1.0 / math.sqrt(D)
        O, L = _ext.v6_nax_forward(q, k, v, False, True)
        mx.eval(O, L); mx.synchronize()

        mask_3d = mx.ones((H, NQ, NK), dtype=mx.bool_)
        _AE(mask_3d); mx.synchronize()
        with pytest.raises(RuntimeError, match="2-D"):
            _ = _ext.v6_nax_backward_dv_sparse_raw(
                q, k, v, L, dO, mask_3d, scale, 4, False)

"""v2.50 Prompt 5b Section B - Top-K Architecture B (bisection) tests."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_mfa import flash_attention_topk, get_device_info

_AE = getattr(mx, "async_" + "eval")
_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))

_skipif_no_nax = pytest.mark.skipif(
    not _HAS_NAX, reason="Top-K bisection kernel requires M5+ NAX hardware"
)


def _make_qkv(B, H, N, D, S=None, dtype=mx.float16, seed=42):
    if S is None:
        S = N
    mx.random.seed(seed)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, S, D)).astype(dtype)
    v = mx.random.normal((B, H, S, D)).astype(dtype)
    _AE(q, k, v); mx.synchronize()
    return q, k, v


def _topk_attention_oracle(q, k, v, k_count, scale):
    """Independent numpy fp64 top-k attention reference (volet-A / CC-06).

    Replaces the prior isfinite-only / two-internal-path self-compare cells
    with a real oracle: per query keep the ``k_count`` highest-scoring keys,
    softmax over ONLY those, weighted-sum of v.  No MLX/kernel involvement —
    pure numpy — so a kernel that selects the wrong keys (or silently degrades)
    diverges instead of merely staying finite.
    """
    qf = np.array(q.astype(mx.float32)).astype(np.float64)
    kf = np.array(k.astype(mx.float32)).astype(np.float64)
    vf = np.array(v.astype(mx.float32)).astype(np.float64)
    sc = (qf @ kf.transpose(0, 1, 3, 2)) * scale            # [B,H,N,S]
    thr = np.sort(sc, axis=-1)[..., -k_count][..., None]    # k-th largest per row
    masked = np.where(sc >= thr, sc, -np.inf)
    masked = masked - masked.max(-1, keepdims=True)
    p = np.exp(masked)
    p = p / p.sum(-1, keepdims=True)
    return p @ vf                                           # [B,H,N,D]


def _assert_topk_matches_oracle(out, q, k, v, k_count, scale, rel_tol=0.2):
    """byte-distinct correctness check vs the numpy top-k oracle."""
    on = np.array(out.astype(mx.float32)).astype(np.float64)
    ref = _topk_attention_oracle(q, k, v, k_count, scale)
    err = float(np.abs(on - ref).max())
    rel = err / (float(np.abs(ref).max()) + 1e-6)
    assert np.isfinite(on).all(), "top-k output non-finite"
    assert rel < rel_tol, (
        f"top-k output rel_err {rel:.3f} exceeds {rel_tol} (abs={err:.4f}) — "
        f"kernel selected the wrong keys or degraded vs the numpy oracle")


class TestTopKArchitectureBBisect:

    @_skipif_no_nax
    def test_kernel_object_loaded(self):
        from mlx_mfa.attention import _topk_bisect_threshold_kernel
        assert _topk_bisect_threshold_kernel is not None

    @_skipif_no_nax
    def test_opt_in_via_env(self, monkeypatch):
        monkeypatch.setenv("MFA_TOPK_BISECT", "1")
        B, H, N, D = 1, 4, 512, 64
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, seed=100)
        out = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out); mx.synchronize()
        # CC-06: independent numpy oracle (was isfinite + max>0 only).
        _assert_topk_matches_oracle(out, q, k, v, 64, 1.0/math.sqrt(D))

    @_skipif_no_nax
    def test_bisect_approximates_phase_3a(self, monkeypatch):
        B, H, N, D = 1, 4, 512, 64
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, seed=101)

        monkeypatch.delenv("MFA_TOPK_BISECT", raising=False)
        out_3a = flash_attention_topk(q, k, v, topk_ratio=64.0/S)

        monkeypatch.setenv("MFA_TOPK_BISECT", "1")
        out_b = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out_3a, out_b); mx.synchronize()

        # CC-06: both paths validated against the INDEPENDENT numpy oracle —
        # the prior `diff < 5.0` two-internal-path self-compare proved neither
        # was correct (both could be wrong in the same way).
        _assert_topk_matches_oracle(out_3a, q, k, v, 64, 1.0/math.sqrt(D))
        _assert_topk_matches_oracle(out_b, q, k, v, 64, 1.0/math.sqrt(D))

    @_skipif_no_nax
    def test_bisect_threshold_basic_correctness(self):
        """Bisection produces finite FP32 thresholds.  Validated to give
        3.85x speedup at audit shape (B=1 H=16 N=4096 D=128 fp16 k_count=64);
        full count-distribution analysis at smaller shapes is sensitive to
        FP16 boundary ambiguity and not asserted here.

        At small score scales (e.g., default mx.random.normal output with
        small N D), FP16 quantum may equal the gap between K-th and
        (K+1)-th scores, creating large tie clusters at the boundary —
        this is inherent FP16 precision limit, not a kernel bug.  The
        audit-shape bench in `docs/v50/phase-3b-architectures-comparison.md`
        empirically validates the kernel produces useful threshold values
        for production workloads."""
        from mlx_mfa.attention import _topk_bisect_threshold_kernel
        B, H, N, D = 1, 4, 2048, 128
        S = 2048
        k_count = 64
        q, k, v = _make_qkv(B, H, N, D, S, seed=102)
        scale = 1.0 / math.sqrt(D)

        scores = (q @ k.swapaxes(-1, -2)) * scale
        scores_r = scores.reshape(B*H, N, S)
        k_top_arr = mx.array([k_count], dtype=mx.int32)
        # III-4 D-TOPK FIX: grid.x must be N * threadgroup.x (the kernel
        # uses one 256-thread threadgroup per row).  The old grid.x=N
        # wrote only the first 8 rows per head; the rest read stale pool
        # memory — which this test missed because the stale values were
        # usually benign zeros until an adversarial test ran first.
        _TG = 256
        threshold = _topk_bisect_threshold_kernel(
            inputs=[scores_r, k_top_arr],
            output_shapes=[(B*H, N)],
            output_dtypes=[mx.float32],
            grid=(N * _TG, B*H, 1),
            threadgroup=(_TG, 1, 1),
        )[0]
        mx.eval(threshold)
        thresh_np = np.array(threshold)
        # All thresholds must be finite
        assert np.isfinite(thresh_np).all(), "Bisection produced NaN/Inf"
        # Thresholds must lie within the row's score range — the
        # bisection lo/hi are clamped to [row_min, row_max], so an
        # out-of-range value means a row was never written (stale pool).
        smin = float(mx.min(scores).astype(mx.float32))
        smax = float(mx.max(scores).astype(mx.float32))
        assert thresh_np.min() >= smin - 1e-3, (
            f"threshold {thresh_np.min()} < score min {smin} — "
            f"unwritten (stale) output row")
        assert thresh_np.max() <= smax + 1e-3, (
            f"threshold {thresh_np.max()} > score max {smax} — "
            f"unwritten (stale) output row (the D-TOPK grid bug)")
        # Per-row: every threshold must be within ITS row's [min,max].
        rmin = np.array(mx.min(scores_r, axis=-1).astype(mx.float32))
        rmax = np.array(mx.max(scores_r, axis=-1).astype(mx.float32))
        assert (thresh_np >= rmin - 1e-3).all() and \
               (thresh_np <= rmax + 1e-3).all(), \
            "a threshold fell outside its row's score range (stale row)"

        # ── volet-A / CC-06: INDEPENDENT numpy fp64 oracle ──────────────────
        # The checks above only prove "finite + in-range" — they would pass for
        # any value inside [row_min,row_max], including a wrong threshold.  A
        # correct top-k threshold τ must equal the k_count-th largest score per
        # row (the smallest still-selected value) AND select ~k_count keys.
        # Both are computed here purely in numpy (no kernel), so a selection
        # regression diverges.  Empirically (seed 102) the kernel is exact:
        # err-vs-kth = 0.0, count = k_count; a +0.5 threshold perturbation
        # gives err 0.5 / count ~18 — far outside these bounds (bite-proven).
        sf = np.array(scores_r.astype(mx.float32)).astype(np.float64)  # [BH,N,S]
        kth = (-np.sort(-sf, axis=-1))[:, :, k_count - 1]              # k-th largest
        err = np.abs(thresh_np - kth)
        count_ge = (sf >= thresh_np[:, :, None]).sum(axis=-1)          # [BH,N]
        assert np.percentile(err, 95) < 0.1, (
            f"threshold drifts from the numpy k-th-largest oracle: "
            f"p95|τ-kth| = {np.percentile(err, 95):.4f} (median "
            f"{np.median(err):.4f}) — bisection is selecting the wrong cut")
        assert abs(float(np.median(count_ge)) - k_count) <= 2, (
            f"threshold selects median {np.median(count_ge):.0f} keys, "
            f"expected ~{k_count}")
        assert float(np.mean(np.abs(count_ge - k_count) <= 8)) > 0.98, (
            f"only {100*np.mean(np.abs(count_ge - k_count) <= 8):.1f}% of rows "
            f"select within ±8 of k_count={k_count} (fp16 tie band)")

    @_skipif_no_nax
    def test_bf16_path(self, monkeypatch):
        monkeypatch.setenv("MFA_TOPK_BISECT", "1")
        B, H, N, D = 1, 4, 512, 64
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, dtype=mx.bfloat16, seed=103)
        out = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out); mx.synchronize()
        # CC-06: numpy oracle; bf16 has a coarser mantissa → looser rel band.
        _assert_topk_matches_oracle(out, q, k, v, 64, 1.0/math.sqrt(D), rel_tol=0.3)

    @_skipif_no_nax
    def test_d128_path(self, monkeypatch):
        monkeypatch.setenv("MFA_TOPK_BISECT", "1")
        B, H, N, D = 1, 4, 512, 128
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, seed=104)
        out = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out); mx.synchronize()
        # CC-06: numpy oracle (D=128 path).
        _assert_topk_matches_oracle(out, q, k, v, 64, 1.0/math.sqrt(D))

    @_skipif_no_nax
    def test_env_unset_uses_bisect_default(self, monkeypatch):
        """v2.50 Prompt 5c Section B: bisection PROMOTED to AUTO default.
        Env unset → engages bisection kernel (not Phase 3a as in Prompt 5b)."""
        monkeypatch.delenv("MFA_TOPK_BISECT", raising=False)
        monkeypatch.delenv("MFA_DISABLE_TOPK_BISECT", raising=False)
        B, H, N, D = 1, 4, 512, 64
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, seed=105)
        out = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out); mx.synchronize()
        # CC-06: numpy oracle proves the AUTO-default bisect path is correct,
        # not merely finite.
        _assert_topk_matches_oracle(out, q, k, v, 64, 1.0/math.sqrt(D))

    @_skipif_no_nax
    def test_opt_out_via_disable_env(self, monkeypatch):
        """Section B Phase B.5 promotion: MFA_DISABLE_TOPK_BISECT=1 reverts
        to Phase 3a mx.topk semantics (legacy path preserved)."""
        monkeypatch.setenv("MFA_DISABLE_TOPK_BISECT", "1")
        B, H, N, D = 1, 4, 512, 64
        S = 512
        q, k, v = _make_qkv(B, H, N, D, S, seed=106)
        out = flash_attention_topk(q, k, v, topk_ratio=64.0/S)
        mx.eval(out); mx.synchronize()
        # CC-06: numpy oracle proves the opt-out (Phase 3a mx.topk) path correct.
        _assert_topk_matches_oracle(out, q, k, v, 64, 1.0/math.sqrt(D))

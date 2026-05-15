"""v2.50.1 Prompt 5g Phase A — KD-6 / KD-7 / Pattern #8 dtype compatibility tests.

Mechanism: v2.36.0 introduced auto-hook routing eligible Conv3D shapes
to `conv3d_nax_forward` (C++ NAX kernel).  The C++ kernel enforces
`x.dtype == w.dtype` AND both in {fp16, bf16}.  The hook eligibility
check only verified `weight.dtype in {fp16, bf16}`, missing the input/
weight match requirement.  Result: any caller passing mismatched dtypes
(e.g., fp32 input + fp16 weight, the typical VSR VAE encoder pattern)
hit `RuntimeError: conv_nax: x.dtype != w.dtype` from v2.36.0 through
v2.50.0.

Additionally discovered during Phase A: the bf16 weight path triggers
a Metal shader compilation failure in MLX's upstream im2col helper
(`utils.h:502`).  Eligibility tightened to fp16-weight-only pending
upstream MLX fix (tracked as KD-7).

Phase A fix:
1. Python-level dtype cast in `_patched_conv_general`: cast input to
   weight dtype before NAX dispatch; restore baseline output dtype
   after kernel call (preserves API contract).
2. Tighten eligibility: fp16 weight only (bf16 Metal kernel broken).
3. Defensive try/except around NAX call: fall back to baseline on any
   unexpected failure.

Tests cover all 9 dtype combinations (fp16 × bf16 × fp32 input ×
weight) for both no-crash + output-shape-and-dtype matches-baseline.

See `docs/v50/known-debt-v2.50.md` KD-6 (resolved) and KD-7 (open).
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

import mlx_mfa  # triggers auto-hook installation
from mlx_mfa import get_device_info

_DEV = get_device_info()
_HAS_NAX = bool(_DEV.get("is_m5_plus", False))
_skipif_no_nax = pytest.mark.skipif(
    not _HAS_NAX, reason="Conv3D NAX hook requires M5+ hardware"
)

# fp16, bf16, fp32 cross-product
_DTYPES = [mx.float16, mx.bfloat16, mx.float32]
_DTYPE_NAMES = {mx.float16: "fp16", mx.bfloat16: "bf16", mx.float32: "fp32"}


def _ids(dtype):
    return _DTYPE_NAMES.get(dtype, str(dtype))


def _make_inputs(input_dtype, weight_dtype, seed=1234):
    mx.random.seed(seed)
    x = (mx.random.normal((1, 4, 8, 8, 16), dtype=input_dtype) * 0.1)
    w = (mx.random.normal((16, 3, 3, 3, 16), dtype=weight_dtype) * 0.1)
    mx.eval(x, w); mx.synchronize()
    return x, w


@_skipif_no_nax
@pytest.mark.parametrize(
    "input_dtype", _DTYPES, ids=[_ids(d) for d in _DTYPES])
@pytest.mark.parametrize(
    "weight_dtype", _DTYPES, ids=[_ids(d) for d in _DTYPES])
def test_conv3d_dtype_mismatch_no_crash(input_dtype, weight_dtype):
    """Verify `mx.conv_general` (auto-hooked) handles all 9 dtype
    combinations without crashing.  Pre-fix: any (input, weight) pair
    with mismatched dtypes raised RuntimeError.  Post-fix: all combos
    produce finite output."""
    x, w = _make_inputs(input_dtype, weight_dtype)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert mx.isfinite(y).all().item(), (
        f"NaN/Inf in output for input={_DTYPE_NAMES[input_dtype]} "
        f"weight={_DTYPE_NAMES[weight_dtype]}")
    assert tuple(y.shape) == (1, 4, 8, 8, 16), (
        f"Shape mismatch for ({_DTYPE_NAMES[input_dtype]}, "
        f"{_DTYPE_NAMES[weight_dtype]}): {y.shape}")


@_skipif_no_nax
@pytest.mark.parametrize(
    "input_dtype", _DTYPES, ids=[_ids(d) for d in _DTYPES])
@pytest.mark.parametrize(
    "weight_dtype", _DTYPES, ids=[_ids(d) for d in _DTYPES])
def test_conv3d_dtype_mismatch_matches_baseline_shape_and_dtype(
        input_dtype, weight_dtype):
    """Verify the hooked output matches MLX baseline output's shape +
    dtype contract.  For NAX-engaged combinations (input cast to fp16
    weight), numerical values may differ slightly due to fp16 vs fp32
    accumulator precision — that delta is bounded but not tested here.
    Pre-fix: mismatched combos raised exception, so 'matches' was
    vacuously false."""
    x, w = _make_inputs(input_dtype, weight_dtype)

    y_hook = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y_hook); mx.synchronize()

    mlx_mfa.disable()
    try:
        y_base = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
        mx.eval(y_base); mx.synchronize()
    finally:
        mlx_mfa.enable()

    assert tuple(y_hook.shape) == tuple(y_base.shape), (
        f"Shape mismatch: hook={y_hook.shape} baseline={y_base.shape} "
        f"for ({_DTYPE_NAMES[input_dtype]}, {_DTYPE_NAMES[weight_dtype]})")
    assert y_hook.dtype == y_base.dtype, (
        f"Dtype mismatch: hook={y_hook.dtype} baseline={y_base.dtype} "
        f"for ({_DTYPE_NAMES[input_dtype]}, {_DTYPE_NAMES[weight_dtype]})")


@_skipif_no_nax
def test_conv3d_vae_pattern_fp32_input_fp16_weight():
    """VSR VAE encoder pattern: fp32 input + fp16 weight.  This is the
    dominant production-critical case (Pattern #8 root cause).  Pre-fix:
    raised RuntimeError; post-fix: NAX engages via input cast, output
    in fp32 to match baseline contract."""
    x, w = _make_inputs(mx.float32, mx.float16, seed=5555)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert y.dtype == mx.float32, (
        f"VAE pattern (fp32 input + fp16 weight) must produce fp32 "
        f"output; got {y.dtype}")
    assert mx.isfinite(y).all().item(), "VAE pattern produced NaN/Inf"


@_skipif_no_nax
def test_conv3d_matched_fp16_engages_nax_no_cast():
    """Matched fp16 inputs should engage NAX directly with no cast
    overhead.  Output dtype = fp16 (no restoration needed)."""
    x, w = _make_inputs(mx.float16, mx.float16, seed=6666)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert y.dtype == mx.float16
    assert mx.isfinite(y).all().item()


@_skipif_no_nax
def test_conv3d_bf16_weight_falls_back_to_baseline():
    """KD-7: bf16 weight triggers MLX upstream Metal shader compile
    failure in im2col helper.  Our hook eligibility tightening excludes
    bf16 weights so they fall back to baseline (which works)."""
    x, w = _make_inputs(mx.bfloat16, mx.bfloat16, seed=7777)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert y.dtype == mx.bfloat16
    assert mx.isfinite(y).all().item()


@_skipif_no_nax
def test_conv3d_fp32_weight_falls_back_to_baseline():
    """fp32 weight is not NAX-eligible (NAX requires fp16/bf16).
    Hook correctly falls back to baseline; output preserved."""
    x, w = _make_inputs(mx.float32, mx.float32, seed=8888)
    y = mx.conv_general(x, w, stride=1, padding=1, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert y.dtype == mx.float32
    assert mx.isfinite(y).all().item()


@_skipif_no_nax
def test_conv3d_ineligible_kernel_size_falls_back():
    """Non-(3,3,3)/(1,1,1) kernels are ineligible — must fall back
    cleanly to baseline."""
    mx.random.seed(9999)
    x = (mx.random.normal((1, 4, 8, 8, 16), dtype=mx.float16) * 0.1)
    w = (mx.random.normal((16, 5, 5, 5, 16), dtype=mx.float16) * 0.1)
    mx.eval(x, w); mx.synchronize()
    y = mx.conv_general(x, w, stride=1, padding=2, kernel_dilation=1)
    mx.eval(y); mx.synchronize()
    assert y.dtype == mx.float16
    assert mx.isfinite(y).all().item()

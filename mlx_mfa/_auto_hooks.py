"""Sprint U / v2.36.0 — auto-hook installation for transparent optimization routing.

Activated at import time via mlx_mfa/__init__.py unless
MFA_DISABLE_AUTO_HOOKS=1 is set in the environment.

Currently hooks:
- `mx.conv_general` → routes eligible Conv3D shapes through
  `mlx_mfa.conv_nax.conv3d_nax_forward` on M5+ Apple Silicon.

Eligibility for Conv3D NAX (mirrors seedvr2_vae patcher's check):
- weight is 5-D (Conv3D): shape (C_out, K_T, K_H, K_W, C_in)
- kernel_size ∈ {(3,3,3), (1,1,1)}
- dtype ∈ {float16, bfloat16}
- stride == (1, 1, 1)
- dilation == (1, 1, 1)
- groups == 1
- flip == False
- M5+ hardware

Ineligible shapes pass through to the original `mx.conv_general` unchanged.

Escape hatch:
- `MFA_DISABLE_AUTO_HOOKS=1` env var prevents auto-install at import.
- `mlx_mfa.disable()` programmatic API uninstalls hooks.
- `mlx_mfa.enable()` re-installs hooks (idempotent).
- `mlx_mfa.hooks_status()` introspection.

See docs/RELEASE_PHILOSOPHY.md for the auto-default principle.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

import mlx.core as mx


_HOOKS_INSTALLED = False
_ORIGINAL_CONV_GENERAL: Optional[Callable] = None
_INSTALL_LOG: list[str] = []

# Eligible Conv3D kernel shapes per seedvr2_vae patcher convention.
_ELIGIBLE_KERNEL_SIZES = {(3, 3, 3), (1, 1, 1)}


def _is_m5_plus() -> bool:
    """Cached M5+ detection via get_device_info()."""
    try:
        from mlx_mfa.attention import get_device_info
        return bool(get_device_info().get("is_m5_plus"))
    except Exception:
        return False


def _normalize_int_or_tuple(v, n_dims):
    """Convert int or short-tuple to length-n_dims tuple."""
    if isinstance(v, int):
        return (v,) * n_dims
    if isinstance(v, (list, tuple)):
        if len(v) == n_dims:
            return tuple(v)
        if len(v) == 1:
            return (v[0],) * n_dims
    return None  # unrecognized form → ineligible


def _conv3d_nax_eligible(weight, stride, padding, kernel_dilation, groups, flip) -> bool:
    """Return True if a conv_general call is Conv3D NAX-eligible."""
    if not hasattr(weight, "shape") or len(weight.shape) != 5:
        return False
    # v2.50.1 Prompt 5g Phase A — KD-7: bf16 weight path is broken at
    # the MLX upstream Metal shader im2col helper (utils.h:502 —
    # half vs bfloat16_t type mismatch).  Fails at graph-evaluation time
    # with "Unable to build metal library from source".  Tightened
    # eligibility to fp16 only until upstream MLX fix lands (see KD-7
    # in docs/v50/known-debt-v2.50.md).
    if weight.dtype != mx.float16:
        return False
    # weight shape: (C_out, K_T, K_H, K_W, C_in)
    K_T, K_H, K_W = weight.shape[1], weight.shape[2], weight.shape[3]
    if (K_T, K_H, K_W) not in _ELIGIBLE_KERNEL_SIZES:
        return False
    if groups != 1 or flip:
        return False
    # stride must be (1,1,1)
    s = _normalize_int_or_tuple(stride, 3)
    if s is None or s != (1, 1, 1):
        return False
    d = _normalize_int_or_tuple(kernel_dilation, 3)
    if d is None or d != (1, 1, 1):
        return False
    return True


def _normalize_padding_to_6tuple(padding):
    """Normalize MLX conv_general padding to a 6-tuple
    (T_l, T_r, H_l, H_r, W_l, W_r) suitable for conv3d_nax_forward.
    Returns None if unrecognized.
    """
    if isinstance(padding, int):
        return (padding,) * 6
    if isinstance(padding, (list, tuple)):
        if len(padding) == 3:
            # symmetric per-dim padding
            return (padding[0], padding[0],
                    padding[1], padding[1],
                    padding[2], padding[2])
        if len(padding) == 6:
            return tuple(padding)
        if len(padding) == 1:
            return (padding[0],) * 6
    return None


def _patched_conv_general(input, weight, stride=1, padding=0,
                          kernel_dilation=1, input_dilation=1,
                          groups=1, flip=False, **kwargs):
    """Auto-route eligible Conv3D shapes to conv3d_nax_forward on M5+.

    Ineligible shapes pass through to original mx.conv_general unchanged.
    """
    # Fast path: original call when not eligible OR not M5+ OR input_dilation != 1
    in_dil = _normalize_int_or_tuple(input_dilation, 3) if hasattr(weight, "shape") and len(weight.shape) == 5 else None
    if (not _is_m5_plus()
        or not _conv3d_nax_eligible(weight, stride, padding,
                                     kernel_dilation, groups, flip)
        or (in_dil is not None and in_dil != (1, 1, 1))):
        return _ORIGINAL_CONV_GENERAL(
            input, weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # Normalize padding into 6-tuple for conv3d_nax_forward
    pad_6tuple = _normalize_padding_to_6tuple(padding)
    if pad_6tuple is None or any(p < 0 for p in pad_6tuple):
        return _ORIGINAL_CONV_GENERAL(
            input, weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # Route to conv3d_nax_forward (returns the same array shape as conv_general).
    try:
        from mlx_mfa._ext import conv3d_nax_forward
    except ImportError:
        return _ORIGINAL_CONV_GENERAL(
            input, weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # v2.50.1 Prompt 5g Phase A — KD-6 / Pattern #8 dtype cast fix.
    #
    # The NAX C++ kernel requires `x.dtype == w.dtype` AND both must be
    # fp16/bf16 (see csrc/mfa_conv_nax.cpp:295).  But MLX baseline
    # `mx.conv_general` accepts mismatched dtypes via automatic promotion.
    # VSR VAE encoders pass fp32 input + fp16 weights — this raised
    # `RuntimeError: conv_nax: x.dtype != w.dtype` on every call from
    # v2.36.0 (auto-hooks introduction) through v2.50.0, with user
    # pipelines silently absorbing the exception via downstream try/except
    # wrappers (Pattern #8 mechanism — see audit-framing-inversions.md).
    #
    # Fix: cast input to weight dtype before NAX dispatch; restore the
    # baseline output dtype after the kernel call.  Weight dtype is
    # guaranteed to be fp16 or bf16 by `_conv3d_nax_eligible`.
    orig_input_dtype = input.dtype
    if input.dtype != weight.dtype:
        input = input.astype(weight.dtype)
    try:
        result = conv3d_nax_forward(
            input, weight,
            stride=(1, 1, 1),
            padding=pad_6tuple,
            dilation=(1, 1, 1),
            chunk_M=0,
        )
    except Exception:
        # Defensive fallback: if NAX dispatch fails for any reason
        # (unexpected runtime error, hardware feature missing, etc.),
        # revert to MLX baseline rather than propagating.  Telemetry
        # in Phase C records this as a fallback event.
        return _ORIGINAL_CONV_GENERAL(
            input.astype(orig_input_dtype), weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # Restore baseline output dtype.  MLX baseline `mx.conv_general` with
    # mismatched input/weight dtypes promotes to the higher-precision
    # type.  Common case: fp32 input + fp16 weight → baseline output fp32.
    # NAX produced fp16 output (weight dtype) which we cast up to
    # preserve the API contract.
    if orig_input_dtype != weight.dtype and result.dtype != orig_input_dtype:
        # Cast back only when the original input was higher-precision
        # than weight (typical: fp32 input → fp16/bf16 weight → fp32 out).
        # For weight-higher-than-input mismatches (rare), MLX baseline
        # would promote upward too, so cast to the broader type.
        if orig_input_dtype == mx.float32:
            result = result.astype(mx.float32)
        elif (orig_input_dtype, weight.dtype) in {
            (mx.float16, mx.bfloat16), (mx.bfloat16, mx.float16)
        }:
            # MLX promotes fp16+bf16 → fp32.  Match that contract.
            result = result.astype(mx.float32)
        else:
            result = result.astype(orig_input_dtype)
    return result


def install_hooks() -> bool:
    """Install auto-hooks. Idempotent. Returns True if hooks were newly installed."""
    global _HOOKS_INSTALLED, _ORIGINAL_CONV_GENERAL

    if _HOOKS_INSTALLED:
        return False

    if os.environ.get("MFA_DISABLE_AUTO_HOOKS") == "1":
        _INSTALL_LOG.append("Auto-hooks DISABLED via MFA_DISABLE_AUTO_HOOKS=1")
        return False

    # Detect : already hooked by another library?
    if hasattr(mx.conv_general, "__mlx_mfa_hook__"):
        _INSTALL_LOG.append("mx.conv_general already hooked, skipping")
        return False

    _ORIGINAL_CONV_GENERAL = mx.conv_general
    _patched_conv_general.__mlx_mfa_hook__ = True  # marker
    # Note: assigning to mx.conv_general — this is a module-level attribute
    # swap. Python's "from mx import conv_general" callers see the patched
    # version on subsequent imports; existing references to the original
    # remain valid (e.g., test bypass via _ORIGINAL_CONV_GENERAL).
    mx.conv_general = _patched_conv_general

    _HOOKS_INSTALLED = True
    _INSTALL_LOG.append(
        f"mlx_mfa auto-hooks installed: "
        f"mx.conv_general -> conv3d_nax_forward for eligible Conv3D shapes "
        f"(M5+={_is_m5_plus()})"
    )
    return True


def uninstall_hooks() -> bool:
    """Restore original behavior. Idempotent."""
    global _HOOKS_INSTALLED, _ORIGINAL_CONV_GENERAL

    if not _HOOKS_INSTALLED:
        return False

    if _ORIGINAL_CONV_GENERAL is not None:
        mx.conv_general = _ORIGINAL_CONV_GENERAL
        _ORIGINAL_CONV_GENERAL = None

    _HOOKS_INSTALLED = False
    _INSTALL_LOG.append("mlx_mfa auto-hooks uninstalled")
    return True


def hooks_status() -> dict:
    """Debug introspection."""
    return {
        "installed": _HOOKS_INSTALLED,
        "log": list(_INSTALL_LOG),
        "m5_plus": _is_m5_plus(),
        "auto_hooks_disabled_env": (
            os.environ.get("MFA_DISABLE_AUTO_HOOKS") == "1"
        ),
    }

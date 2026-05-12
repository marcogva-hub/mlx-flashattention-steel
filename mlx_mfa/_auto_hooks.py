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
    if weight.dtype not in (mx.float16, mx.bfloat16):
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
    return conv3d_nax_forward(
        input, weight,
        stride=(1, 1, 1),
        padding=pad_6tuple,
        dilation=(1, 1, 1),
        chunk_M=0,
    )


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

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
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional

import mlx.core as mx

from ._knobs import get_bool_env


_HOOKS_INSTALLED = False
_ORIGINAL_CONV_GENERAL: Optional[Callable] = None
_ORIGINAL_CONV3D: Optional[Callable] = None
_INSTALL_LOG: list[str] = []

# Eligible Conv3D kernel shapes per seedvr2_vae patcher convention.
# NOTE (v2.58.1 P5): (1, 1, 1) is RECOGNIZED here but is only MPP-accelerated under the
# full MPP gate (C_in/C_out %16==0 & >=32, H/W %8==0, B=1, pad=(1,1,1)); a (1,1,1) conv
# that misses the gate is NOT silently wrong — it is a TRACKED fallback to the original
# mx.conv_general (see get_hook_stats() `fallback`/`fallback_reasons`), numerically correct.
_ELIGIBLE_KERNEL_SIZES = {(3, 3, 3), (1, 1, 1)}

# ---------------------------------------------------------------------------
# v2.50.1 Prompt 5g Phase C — Hook telemetry infrastructure (Pattern #8 prevention).
# ---------------------------------------------------------------------------
# The auto-hooked Conv3D NAX path silently fell back to MLX baseline from
# v2.36.0 through v2.50.0 due to a dtype-mismatch contract gap (KD-6).  The
# bug was invisible because no telemetry surfaced fallback events.
#
# Telemetry modes (controlled via `MLX_MFA_HOOK_TELEMETRY` env var):
#   - "off"     : zero overhead; no counters maintained.  Use for max perf.
#   - "summary" : default; per-hook executed/fallback counters maintained
#                 in process memory.  Read via `mlx_mfa.get_hook_stats()`.
#                 Overhead: a single dict increment per call.
#   - "verbose" : developer mode; in addition to summary counters, emit a
#                 `UserWarning` on every fallback event.  Useful when
#                 actively debugging why an optimization isn't engaging.

_HOOK_TELEMETRY_MODE = os.environ.get("MLX_MFA_HOOK_TELEMETRY", "summary").lower()
if _HOOK_TELEMETRY_MODE not in ("off", "summary", "verbose"):
    # L-03 FIX (audit, 2026-06-21): an invalid value previously became "summary"
    # SILENTLY — a typo'd mode (e.g. "verbos") gave no feedback.  Warn loudly
    # (Rule 8), then fall back.  Not a hard raise: this runs at `import mlx_mfa`,
    # so a bad observability knob must not break the import.
    warnings.warn(
        f"[mlx-mfa] invalid MLX_MFA_HOOK_TELEMETRY={_HOOK_TELEMETRY_MODE!r} — "
        f"expected one of 'off'/'summary'/'verbose'; falling back to 'summary'.",
        RuntimeWarning, stacklevel=2)
    _HOOK_TELEMETRY_MODE = "summary"

_HOOK_EXECUTION_STATS: dict = {
    "executed": defaultdict(int),
    "fallback": defaultdict(int),
    # Capped at 10 reasons per hook to bound memory (defensive against
    # high-frequency unique-reason fallbacks).
    "fallback_reasons": defaultdict(list),
}

# M4/H-06 FIX (audit, 2026-06-21): one-time loud-on-UNEXPECTED guard.  A hook
# that falls back due to an UNEXPECTED kernel error (not a known unsupported
# shape) must warn ONCE regardless of telemetry mode — the prior code only
# surfaced it under `verbose`, so a kernel regression hid silently under the
# default `summary`.  Keyed by hook name so each hook warns at most once.
_HOOK_WARNED_UNEXPECTED: set = set()


def _record_hook_execution(hook_name: str) -> None:
    """Increment executed counter for `hook_name`.  No-op in 'off' mode."""
    if _HOOK_TELEMETRY_MODE == "off":
        return
    _HOOK_EXECUTION_STATS["executed"][hook_name] += 1


def _record_hook_fallback(hook_name: str, reason: str) -> None:
    """Increment fallback counter for `hook_name` and record up to 10
    distinct reason strings.  In 'verbose' mode, emit a UserWarning.
    No-op in 'off' mode."""
    if _HOOK_TELEMETRY_MODE == "off":
        return
    _HOOK_EXECUTION_STATS["fallback"][hook_name] += 1
    reasons = _HOOK_EXECUTION_STATS["fallback_reasons"][hook_name]
    if len(reasons) < 10 and reason not in reasons:
        reasons.append(reason)
    if _HOOK_TELEMETRY_MODE == "verbose":
        warnings.warn(
            f"mlx-mfa hook '{hook_name}' fell back to MLX baseline: {reason}. "
            f"NAX optimization NOT applied for this call.",
            UserWarning,
            stacklevel=3,
        )


def get_hook_stats() -> dict:
    """Return a snapshot of hook execution statistics.

    Useful for verifying that NAX optimization paths are actually
    engaged for your workload.  If `fallback[hook_name] > 0`, the
    NAX path is not being used for some calls — `fallback_reasons[hook_name]`
    surfaces up to 10 distinct reasons (capped to bound memory).

    Telemetry mode (`mode` key) is controlled via the
    `MLX_MFA_HOOK_TELEMETRY` environment variable (off / summary [default]
    / verbose).
    """
    return {
        "executed": dict(_HOOK_EXECUTION_STATS["executed"]),
        "fallback": dict(_HOOK_EXECUTION_STATS["fallback"]),
        "fallback_reasons": {
            k: list(v)
            for k, v in _HOOK_EXECUTION_STATS["fallback_reasons"].items()
        },
        "mode": _HOOK_TELEMETRY_MODE,
    }


def reset_hook_stats() -> None:
    """Reset all hook telemetry counters to zero.  Useful for scoping
    measurements around a specific code block."""
    _HOOK_EXECUTION_STATS["executed"].clear()
    _HOOK_EXECUTION_STATS["fallback"].clear()
    _HOOK_EXECUTION_STATS["fallback_reasons"].clear()


# Repo review 2026-05: module-level cache.  The docstring claimed "Cached"
# but the function called get_device_info() (Metal API queries) on EVERY
# mx.conv_general invocation — twice on the fallback branch.  Hardware
# cannot change at runtime; cache the result once.
_M5_PLUS_CACHE: "bool | None" = None


def _is_m5_plus() -> bool:
    """Cached M5+ detection via get_device_info() (computed once per process)."""
    global _M5_PLUS_CACHE
    if _M5_PLUS_CACHE is None:
        try:
            from mlx_mfa.attention import get_device_info
            _M5_PLUS_CACHE = bool(get_device_info().get("is_m5_plus"))
        except Exception:
            _M5_PLUS_CACHE = False
    return _M5_PLUS_CACHE


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
    # with "Unable to build metal library from source".
    #
    # Sprint III-1 (KD-7 lift): bf16 is now admissible — but ONLY via the
    # MPP convolution2d branch, which performs no im2col at all.  The
    # wrapper enforces the MPP sub-gate via _conv3d_mpp_eligible() for
    # BOTH dtypes; shapes outside it fall back to the original op (the
    # legacy im2col path is unsafe — bf16 fails to compile upstream, fp16
    # silently corrupts on partial K-tiles; III-5 follow-up).
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


def _conv3d_mpp_eligible(input, weight, pad_6tuple) -> bool:
    """Mirror the C++ conv3d_nax_forward MPP convolution2d gate
    (csrc/mfa_conv_nax.cpp:492-505): the ONLY numerically-correct NAX
    conv path is the MPP convolution2d branch.  Applies to BOTH fp16 and
    bf16.

    Two reasons a shape outside this envelope must fall back to the
    native op rather than reach conv3d_nax_forward's legacy fallback:

    1. bf16 (Sprint III-1, KD-7): the legacy im2col helper fails at
       graph-eval time ("Unable to build metal library" — upstream MLX
       utils.h:502 half-vs-bfloat16_t mismatch).
    2. fp16 (III-5 follow-up): the legacy im2col+matmul2d path is
       NUMERICALLY BROKEN for every shape that reaches it.  Its
       matmul2d K-loop reads partial K-tiles past K_FULL with no tail
       mask; K = C_in*27 is a multiple of the 32-wide K-tile iff
       C_in % 32 == 0, and every such C_in already satisfies the MPP
       gate (C_in % 16 == 0 and >= 32) and takes the MPP path.  So the
       legacy path is reached only when K % 32 != 0 -> garbage
       accumulation (C_in=16 -> ~0.11 MAE/RMS; C_in=31 -> NaN).  This
       was invisible because every conv test used the MPP envelope
       (single-shape-class gap; III-4 lesson #10).

    k=3x3x3 / stride / dilation / groups / flip are already enforced by
    _conv3d_nax_eligible.
    """
    import os as _os
    if get_bool_env("MFA_DISABLE_CONV3D_MPP"):
        return False  # MPP off -> bf16 would hit the broken legacy path
    if not hasattr(input, "shape") or len(input.shape) != 5:
        return False
    B, _T, H, W, C_in = input.shape
    C_out = weight.shape[0]
    # k must be exactly 3x3x3 for MPP (the eligible-kernel set may admit
    # other sizes for the legacy path).
    if (weight.shape[1], weight.shape[2], weight.shape[3]) != (3, 3, 3):
        return False
    # Per-axis "same"-style pad, symmetric WITHIN each axis. Temporal pad 1
    # ("same") OR 0 (causal — the VAE concats the causal frames upstream then
    # calls the conv with pad_T=0; the MPP kt time-loop handles it, T_out=T-2).
    # Spatial H/W: pad 1 only. Mirrors the C++ gate (mfa_conv_nax.cpp). Truly
    # asymmetric (T_left!=T_right) or H/W!=1 stays ineligible → native fallback.
    if B != 1 or pad_6tuple not in ((1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 1)):
        return False
    if H % 8 != 0 or W % 8 != 0:
        return False
    if C_in % 16 != 0 or C_in < 32 or C_out % 16 != 0 or C_out < 32:
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


# Pad-and-slice cost gate: skip (→ mx.conv_general) when padding the channels to
# the MPP envelope inflates the work beyond this multiplier.  Default admits the
# empirically-validated win envelope (measured wins up to pad_ratio 10.67 on M5:
# RGB in/out convs 3↔128).  Env-tunable via MFA_CONV3D_PAD_RATIO_MAX.
_CONV3D_PAD_RATIO_MAX_DEFAULT = 12.0

# SeedVR2 VAE spatial-tail probe (macOS 27 beta, M5 Max, MLX 0.31.2).
# Keep the envelope exact: only the measured dominant fallback family is
# admitted. 54x66 loses or is at parity at VAE-unit scope despite a micro win;
# stride-2 and channel-tail families also remain on MLX.
_CONV3D_SPATIAL_PAD_FAMILIES = {
    (4, 108, 132, 512, 512),
    (5, 108, 132, 512, 512),
}


def _roundup(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _restore_conv_out_dtype(result, orig_input_dtype, weight_dtype):
    """Match mx.conv_general's dtype-promotion contract (mirrors the native-path
    restore in _patched_conv_general): fp32-in/fp16-w → fp32-out, fp16+bf16 → fp32."""
    if orig_input_dtype != weight_dtype and result.dtype != orig_input_dtype:
        if orig_input_dtype == mx.float32:
            return result.astype(mx.float32)
        if (orig_input_dtype, weight_dtype) in {
            (mx.float16, mx.bfloat16), (mx.bfloat16, mx.float16)
        }:
            return result.astype(mx.float32)
        return result.astype(orig_input_dtype)
    return result


def _try_conv3d_spatial_pad_and_slice(input, weight, pad_6tuple):
    """Run the measured SeedVR2 spatial-tail family through the MPP path.

    The MPP kernel requires H/W divisible by 8. Zero-padding only the bottom
    and right edges preserves a pad=1 convolution exactly for the original
    output extent; slicing removes outputs belonging to the padded domain.

    Default-off because the measurements are beta3-indicative. The public gate
    can be revalidated and promoted independently on stable macOS.
    """
    if not get_bool_env("MFA_ENABLE_CONV3D_SPATIAL_PAD_SLICE"):
        return None
    if (not hasattr(input, "shape") or len(input.shape) != 5
            or not hasattr(weight, "shape") or len(weight.shape) != 5):
        return None
    B, T, H, W, C_in = input.shape
    C_out = weight.shape[0]
    if input.dtype != mx.float16 or weight.dtype != mx.float16:
        return None
    family = (int(T), int(H), int(W), int(C_in), int(C_out))
    if family not in _CONV3D_SPATIAL_PAD_FAMILIES:
        return None
    if B != 1 or tuple(weight.shape[1:4]) != (3, 3, 3):
        return None
    if pad_6tuple != (0, 0, 1, 1, 1, 1):
        return None
    H_pad = _roundup(int(H), 8)
    W_pad = _roundup(int(W), 8)
    if H_pad == H and W_pad == W:
        return None
    try:
        from mlx_mfa._ext import conv3d_nax_forward
    except ImportError:
        return None
    try:
        weight_dtype = weight.dtype
        orig_dtype = input.dtype
        x = input.astype(weight_dtype) if orig_dtype != weight_dtype else input
        x = mx.pad(
            x,
            ((0, 0), (0, 0), (0, H_pad - H), (0, W_pad - W), (0, 0)),
        )
        out = conv3d_nax_forward(
            x,
            weight,
            stride=(1, 1, 1),
            padding=pad_6tuple,
            dilation=(1, 1, 1),
            chunk_M=0,
        )
        out = _restore_conv_out_dtype(out[:, :, :H, :W, :], orig_dtype, weight_dtype)
        _record_hook_execution("conv3d_nax_spatial_pad_slice")
        return out
    except Exception as exc:
        _record_hook_fallback(
            "conv3d_nax_spatial_pad_slice",
            f"unexpected runtime error: {type(exc).__name__}: {exc}",
        )
        return None


def _try_conv3d_pad_and_slice(input, weight, pad_6tuple):
    """MLX-style pad-and-slice around the NAX conv datapath
    (csrc/mfa_conv_nax.cpp MPP convolution2d), for Conv3D shapes whose ONLY
    reason for MPP-ineligibility is channel alignment.  Returns the sliced
    output (same shape + baseline dtype as mx.conv_general) or None → caller
    falls back to mx.conv_general.

    The documented NaN trap (conv_nax.py / III-5): the NAX K-loop needs
    K = C_in*27 to be a multiple of the 32-wide K-tile.  We pad C_in UP TO A
    MULTIPLE OF 32 (⇒ K a clean 32-tile multiple, side-stepping the partial-K
    read entirely) and zero-fill BOTH input and weight so the padded channels
    contribute EXACTLY zero (verified cos=1.0, no NaN for C_in ∈ {3,16,31,…}).
    C_out padded to a multiple of 16 and sliced back.

    MIT: pad-and-slice technique adapted from MLX's pad_and_slice_conv_3D_gpu
    (ml-explore/mlx, backend/metal/conv.cpp) — applied here around the NAX
    (matmul2d) datapath, which MLX itself has no equivalent for.

    OPT-IN (default-off): the win is CONFIDENT + correct (cos=1.0, fp16+bf16) but
    β3-indicative (macOS 27 beta).  Per the auto-default principle, a validated-
    but-β3 optimization ships as an env opt-in; flip to default-on after stable-
    macOS re-validation (+ the coordinated dispatch-map/lock update, since it
    reroutes channel-misaligned conv3d off mx.conv_general).
    """
    if not get_bool_env("MFA_ENABLE_CONV3D_PAD_SLICE"):
        return None
    if (not hasattr(input, "shape") or len(input.shape) != 5
            or not hasattr(weight, "shape") or len(weight.shape) != 5):
        return None
    B, _T, H, W, C_in = input.shape
    C_out = weight.shape[0]
    # Every NON-channel MPP constraint must already hold (mirror
    # _conv3d_mpp_eligible); otherwise it is a structural fallback we do not touch.
    if (weight.shape[1], weight.shape[2], weight.shape[3]) != (3, 3, 3):
        return None
    if B != 1 or pad_6tuple not in ((1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 1, 1)):
        return None
    if H % 8 != 0 or W % 8 != 0:
        return None
    # The failing reason must be channel alignment (else the shape was already
    # MPP-eligible and served by the native path upstream — never reaches here).
    if C_in % 16 == 0 and C_in >= 32 and C_out % 16 == 0 and C_out >= 32:
        return None
    C_pad = max(32, _roundup(C_in, 32))   # K = C_pad*27 → clean 32-wide K-tile
    O_pad = max(32, _roundup(C_out, 16))
    pad_ratio = (C_pad / C_in) * (O_pad / C_out)
    try:
        thr = float(os.environ.get("MFA_CONV3D_PAD_RATIO_MAX",
                                   _CONV3D_PAD_RATIO_MAX_DEFAULT))
    except (TypeError, ValueError):
        thr = _CONV3D_PAD_RATIO_MAX_DEFAULT
    if pad_ratio > thr:
        return None  # cost gate: pad overhead not worth it → mx.conv_general
    try:
        from mlx_mfa._ext import conv3d_nax_forward
    except ImportError:
        return None
    try:
        wdt = weight.dtype  # fp16/bf16 (guaranteed by _conv3d_nax_eligible upstream)
        orig_dt = input.dtype
        x = input.astype(wdt) if orig_dt != wdt else input
        w = weight
        if C_pad > C_in:
            x = mx.concatenate(
                [x, mx.zeros((B, _T, H, W, C_pad - C_in), wdt)], axis=-1)
            w = mx.concatenate(
                [w, mx.zeros((C_out, 3, 3, 3, C_pad - C_in), wdt)], axis=-1)
        if O_pad > C_out:
            w = mx.concatenate(
                [w, mx.zeros((O_pad - C_out, 3, 3, 3, C_pad), wdt)], axis=0)
        out = conv3d_nax_forward(x, w, stride=(1, 1, 1), padding=pad_6tuple,
                                 dilation=(1, 1, 1), chunk_M=0)
        out = out[..., :C_out]
        out = _restore_conv_out_dtype(out, orig_dt, wdt)
        _record_hook_execution("conv3d_nax_pad_slice")
        return out
    except Exception:
        return None  # fail-safe: any error → caller falls back to conv_general


def _patched_conv_general(input, weight, stride=1, padding=0,
                          kernel_dilation=1, input_dilation=1,
                          groups=1, flip=False, **kwargs):
    """Auto-route eligible Conv3D shapes to conv3d_nax_forward on M5+.

    Ineligible shapes pass through to original mx.conv_general unchanged.
    """
    # Fast path: original call when not eligible OR not M5+ OR input_dilation != 1
    # Repo review 2026-05: single _is_m5_plus() call captured in a local —
    # the fallback branch previously called it twice per conv invocation.
    _m5 = _is_m5_plus()
    in_dil = _normalize_int_or_tuple(input_dilation, 3) if hasattr(weight, "shape") and len(weight.shape) == 5 else None
    if (not _m5
        or not _conv3d_nax_eligible(weight, stride, padding,
                                     kernel_dilation, groups, flip)
        or (in_dil is not None and in_dil != (1, 1, 1))):
        # Telemetry: record fallback with reason classification.
        if not _m5:
            _record_hook_fallback("conv3d_nax_forward", "not M5+ hardware")
        elif in_dil is not None and in_dil != (1, 1, 1):
            _record_hook_fallback("conv3d_nax_forward",
                                   f"input_dilation {in_dil} != (1,1,1)")
        else:
            # Eligibility failed; surface the specific reason for debugging.
            if not hasattr(weight, "shape") or len(weight.shape) != 5:
                reason = "weight not 5-D (not Conv3D)"
            elif weight.dtype not in (mx.float16, mx.bfloat16):
                reason = f"weight dtype {weight.dtype} not fp16/bf16"
            else:
                reason = "kernel/stride/dilation/groups/flip constraint failed"
            _record_hook_fallback("conv3d_nax_forward", reason)
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
        _record_hook_fallback("conv3d_nax_forward",
                               f"unsupported padding form: {padding}")
        return _ORIGINAL_CONV_GENERAL(
            input, weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # The MPP convolution2d branch is the ONLY numerically-correct NAX
    # conv path (see _conv3d_mpp_eligible).  Both fp16 and bf16 must fall
    # back to the native op outside the MPP envelope — bf16 because the
    # legacy im2col helper fails to compile upstream (KD-7), fp16 because
    # the legacy im2col+matmul2d path silently corrupts (III-5 follow-up:
    # unmasked partial-K-tile reads; C_in=16 -> ~0.11 MAE/RMS, C_in=31 ->
    # NaN).  Previously only bf16 was gated, so small-channel fp16 reached
    # the broken legacy path and returned garbage.
    if not _conv3d_mpp_eligible(input, weight, pad_6tuple):
        # Exact, independently measured SeedVR2 VAE family. This is evaluated
        # before channel pad-and-slice because its only failing axis is H/W.
        _spatial_ps = _try_conv3d_spatial_pad_and_slice(input, weight, pad_6tuple)
        if _spatial_ps is not None:
            return _spatial_ps
        # MLX-style pad-and-slice: if the ONLY reason for MPP-ineligibility is
        # channel alignment, pad C_in/C_out into the MPP envelope (K-tile-clean,
        # zero-filled) and use the NAX datapath instead of forfeiting it.  Keeps
        # mlx-mfa's hardware advantage on shapes MLX would run via STEEL im2col.
        _ps = _try_conv3d_pad_and_slice(input, weight, pad_6tuple)
        if _ps is not None:
            return _ps
        _record_hook_fallback(
            "conv3d_nax_forward",
            "outside MPP gate (need C_in/C_out %16==0 & >=32, H/W %8==0, "
            "B=1, pad=(1,1,1)); legacy NAX conv path is numerically unsafe")
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
        _record_hook_fallback("conv3d_nax_forward",
                               "C++ extension not available")
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
    # III-4 D14 FIX: keep a handle on the UNCAST input — the defensive
    # fallback below must re-run the baseline on the original array, not on
    # a precision-truncated round-trip (fp32 -> fp16 -> fp32 loses bits).
    orig_input = input
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
    except Exception as e:
        # Defensive fallback: if NAX dispatch fails for any reason
        # (unexpected runtime error, hardware feature missing, etc.),
        # revert to MLX baseline rather than propagating.
        # Lot-1 B2: distinguish NotImplementedError (expected unsupported-shape
        # fallback) from an unexpected error; under MFA_HOOK_VERBOSE=1 emit the
        # full traceback for the unexpected kind (the 120-char reason truncates
        # diagnostics). The fallback behaviour itself is unchanged — observability only.
        _kind = "unsupported" if isinstance(e, NotImplementedError) else "unexpected"
        _record_hook_fallback("conv3d_nax_forward",
                               f"NAX dispatch raised [{_kind}]: {type(e).__name__}: {str(e)[:120]}")
        # M4/H-06 FIX (audit, 2026-06-21): warn ONCE on the UNEXPECTED kind in
        # EVERY telemetry mode (was verbose-only → kernel regressions hid under
        # the default `summary`).  Expected unsupported-shape fallbacks stay quiet.
        if _kind == "unexpected" and "conv3d_nax_forward" not in _HOOK_WARNED_UNEXPECTED:
            _HOOK_WARNED_UNEXPECTED.add("conv3d_nax_forward")
            warnings.warn(
                f"[mlx-mfa] conv3d_nax_forward fell back to baseline MLX conv on an "
                f"UNEXPECTED error ({type(e).__name__}: {str(e)[:120]}). The NAX kernel "
                f"may be regressed — set MLX_MFA_HOOK_TELEMETRY=verbose or "
                f"MFA_HOOK_VERBOSE=1 for the full traceback. (warned once)",
                RuntimeWarning, stacklevel=2)
        import os as _os
        if get_bool_env("MFA_HOOK_VERBOSE") and _kind == "unexpected":
            import traceback as _tb, sys as _sys
            print("[mlx-mfa] conv3d_nax_forward unexpected fallback — full traceback:", file=_sys.stderr)
            _tb.print_exc()
        # III-4 D14 FIX: re-run the baseline on the pristine original input
        # (input.astype(orig_input_dtype) was a lossy down-up round trip).
        return _ORIGINAL_CONV_GENERAL(
            orig_input, weight,
            stride=stride, padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups, flip=flip,
            **kwargs,
        )

    # NAX path engaged successfully — record execution.
    _record_hook_execution("conv3d_nax_forward")

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


def _patched_conv3d(input, weight, stride=1, padding=0, dilation=1,
                    groups=1, *, stream=None):
    """Auto-route eligible mx.conv3d calls to conv3d_nax_forward on M5+.

    Phase II-7 (campaign 2026-06): `mlx.nn.Conv3d.__call__` invokes
    `mx.conv3d` DIRECTLY — it never touches `mx.conv_general` — so
    every standard MLX model bypassed the conv_general hook entirely
    (telemetry showed 0 executed / 0 fallback after 20 nn-style conv3d
    calls).  This violated the Auto-default principle (CLAUDE.md): the
    NAX conv path was unreachable for plain `mlx.nn` models.  Delegate
    into `_patched_conv_general`, which owns eligibility, telemetry,
    dtype-contract restoration, and the original-op fallback
    (mx.conv_general with flip=False is mathematically identical to
    mx.conv3d).
    """
    kwargs = {} if stream is None else {"stream": stream}
    return _patched_conv_general(
        input, weight, stride=stride, padding=padding,
        kernel_dilation=dilation, input_dilation=1,
        groups=groups, flip=False, **kwargs,
    )


def install_hooks() -> bool:
    """Install auto-hooks. Idempotent. Returns True if hooks were newly installed."""
    global _HOOKS_INSTALLED, _ORIGINAL_CONV_GENERAL, _ORIGINAL_CONV3D

    if _HOOKS_INSTALLED:
        return False

    if get_bool_env("MFA_DISABLE_AUTO_HOOKS"):
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

    # Phase II-7: also patch mx.conv3d — the surface mlx.nn.Conv3d
    # actually calls (see _patched_conv3d docstring).
    if not hasattr(mx.conv3d, "__mlx_mfa_hook__"):
        _ORIGINAL_CONV3D = mx.conv3d
        _patched_conv3d.__mlx_mfa_hook__ = True
        mx.conv3d = _patched_conv3d

    _HOOKS_INSTALLED = True
    _INSTALL_LOG.append(
        f"mlx_mfa auto-hooks installed: "
        f"mx.conv_general + mx.conv3d -> conv3d_nax_forward for eligible "
        f"Conv3D shapes (M5+={_is_m5_plus()})"
    )
    return True


def uninstall_hooks() -> bool:
    """Restore original behavior. Idempotent."""
    global _HOOKS_INSTALLED, _ORIGINAL_CONV_GENERAL, _ORIGINAL_CONV3D

    if not _HOOKS_INSTALLED:
        return False

    if _ORIGINAL_CONV_GENERAL is not None:
        mx.conv_general = _ORIGINAL_CONV_GENERAL
        _ORIGINAL_CONV_GENERAL = None
    if _ORIGINAL_CONV3D is not None:
        mx.conv3d = _ORIGINAL_CONV3D
        _ORIGINAL_CONV3D = None

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
            get_bool_env("MFA_DISABLE_AUTO_HOOKS")
        ),
    }

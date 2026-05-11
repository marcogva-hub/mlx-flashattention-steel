"""SeedVR2 VAE / generic Conv3D patcher.

Routes eligible `mlx.nn.Conv3d` layers through
`mlx_mfa.conv_nax.conv3d_nax_forward` instead of the default
`mx.conv_general` path. Drop-in: no model code changes required.

Eligibility (matches Sprint C Phase 1.5 ship-default profile):
  - mlx.nn.Conv3d module
  - kernel_size in {(3,3,3), (1,1,1)}
  - stride == (1,1,1), dilation == (1,1,1)
  - weight.dtype in {float16, bfloat16}

Ineligible modules pass through with the original behavior preserved.
Skips are logged with a reason.

Usage::

    from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae
    model = patch_seedvr2_vae(model)
    # ... run inference ...
    model = patch_seedvr2_vae(model, restore=True)  # optional undo

Convention mirrors mlx-mfa's existing patcher precedent
(`mlx_mfa.integrations.mlx_lm.patch_mlx_lm` / `unpatch_mlx_lm`).
The patcher is **in-place modifying** (returns the same `model` object
with `__call__` swapped on eligible Conv3d submodules).
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any
import mlx.core as mx
import mlx.nn as nn

from mlx_mfa.conv_nax import conv3d_nax_forward


_ELIGIBLE_KERNEL_SIZES = {(3, 3, 3), (1, 1, 1)}
_ELIGIBLE_DTYPES = {mx.float16, mx.bfloat16}
# Attribute name on patched modules that points back to the original __call__.
_ORIG_CALL_ATTR = "_conv_nax_orig_call"
_PATCH_MARKER_ATTR = "_conv_nax_patched"


def _module_kernel_size(m: nn.Conv3d) -> Tuple[int, int, int]:
    """Extract (K_T, K_H, K_W) from a Conv3d's weight shape."""
    return (int(m.weight.shape[1]), int(m.weight.shape[2]),
            int(m.weight.shape[3]))


def _module_stride(m: nn.Conv3d) -> Tuple[int, int, int]:
    s = m.stride
    if isinstance(s, int):
        return (s, s, s)
    return tuple(int(v) for v in s)  # type: ignore[return-value]


def _module_dilation(m: nn.Conv3d) -> Tuple[int, int, int]:
    d = getattr(m, "dilation", (1, 1, 1))
    if isinstance(d, int):
        return (d, d, d)
    return tuple(int(v) for v in d)  # type: ignore[return-value]


def _module_padding(m: nn.Conv3d):
    """Extract padding -- conv_nax accepts int, 3-tuple, or per-dim pairs."""
    p = m.padding
    if isinstance(p, int):
        return (p, p, p)
    return tuple(p)


def _eligibility_check(m: Any) -> Tuple[bool, str]:
    """Returns (is_eligible, reason). Reason is human-readable."""
    if not isinstance(m, nn.Conv3d):
        return (False, f"not Conv3d (got {type(m).__name__})")
    if getattr(m, _PATCH_MARKER_ATTR, False):
        return (False, "already patched")
    k = _module_kernel_size(m)
    if k not in _ELIGIBLE_KERNEL_SIZES:
        return (False, f"kernel_size {k} not in {_ELIGIBLE_KERNEL_SIZES}")
    s = _module_stride(m)
    if s != (1, 1, 1):
        return (False, f"stride {s} != (1,1,1)")
    d = _module_dilation(m)
    if d != (1, 1, 1):
        return (False, f"dilation {d} != (1,1,1)")
    dt = m.weight.dtype
    if dt not in _ELIGIBLE_DTYPES:
        return (False, f"weight.dtype {dt} not in {_ELIGIBLE_DTYPES}")
    return (True, "eligible")


def _make_patched_class(orig_class, stride, padding, dilation):
    """Build a dynamic subclass of `orig_class` that overrides __call__.

    Python's __call__ resolution is on the TYPE, not the instance. So
    to intercept `mod(x)`, we swap `mod.__class__` to a subclass whose
    __call__ dispatches via conv3d_nax_forward. The original class is
    saved on the instance for restore.
    """
    def patched_call(self, x):
        # Read weight + bias fresh each call (handles weight swap, quantize).
        y = conv3d_nax_forward(
            x, self.weight,
            stride=stride, padding=padding, dilation=dilation,
        )
        if "bias" in self:
            y = y + self.bias
        return y

    return type(
        f"_NAXPatched_{orig_class.__name__}",
        (orig_class,),
        {"__call__": patched_call},
    )


def patch_seedvr2_vae(
    model: nn.Module,
    *,
    restore: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """Patch (or restore) all eligible `nn.Conv3d` submodules of `model`.

    Args:
        model: a `mlx.nn.Module`. Walked via `named_modules()`.
        restore: if True, restore the original `__call__` on previously
            patched modules. Idempotent — calling on an un-patched model
            is a no-op.
        verbose: if True, print per-module patch/skip decisions.

    Returns:
        The same `model` object (in-place modified).

    Notes:
        Idempotent: calling `patch_seedvr2_vae(model)` twice produces
        the same state as calling it once.

        The patcher reads the module's `weight`, `stride`, `padding`,
        `dilation` attributes at patch time. If the user mutates these
        AFTER patching (e.g. quantizing weights), the patched call sees
        the updated weight but NOT updated stride/padding/dilation. Call
        `patch_seedvr2_vae(model, restore=True)` then re-patch if needed.
    """
    patched: List[str] = []
    skipped: List[Tuple[str, str]] = []
    restored: List[str] = []

    for name, mod in model.named_modules():
        if restore:
            if getattr(mod, _PATCH_MARKER_ATTR, False):
                orig_class = getattr(mod, _ORIG_CALL_ATTR)
                mod.__class__ = orig_class  # type: ignore[assignment]
                setattr(mod, _PATCH_MARKER_ATTR, False)
                if hasattr(mod, _ORIG_CALL_ATTR):
                    delattr(mod, _ORIG_CALL_ATTR)
                restored.append(name)
            continue

        ok, reason = _eligibility_check(mod)
        if ok:
            # Swap __class__ to a dynamic subclass with overridden __call__.
            # Python's __call__ resolution is on TYPE not INSTANCE, so
            # instance-level __call__ override does NOT work; class swap
            # is the canonical Python pattern for per-instance method
            # override.
            stride = _module_stride(mod)
            padding = _module_padding(mod)
            dilation = _module_dilation(mod)
            orig_class = mod.__class__
            patched_class = _make_patched_class(orig_class, stride, padding,
                                                dilation)
            setattr(mod, _ORIG_CALL_ATTR, orig_class)
            mod.__class__ = patched_class  # type: ignore[assignment]
            setattr(mod, _PATCH_MARKER_ATTR, True)
            patched.append(name)
            if verbose:
                print(f"  [patch_seedvr2_vae] PATCHED  {name or '<root>'}: "
                      f"kernel={_module_kernel_size(mod)} "
                      f"dtype={mod.weight.dtype}")
        else:
            # Only log Conv3d-related skips (avoid noise from non-Conv3d modules).
            if isinstance(mod, nn.Conv3d):
                skipped.append((name, reason))
                if verbose:
                    print(f"  [patch_seedvr2_vae] SKIPPED  {name or '<root>'}: "
                          f"{reason}")

    if restore:
        if verbose or restored:
            print(f"[patch_seedvr2_vae] restored {len(restored)} module(s)")
    else:
        print(f"[patch_seedvr2_vae] patched {len(patched)} Conv3d "
              f"module(s), skipped {len(skipped)} ineligible Conv3d module(s)")
        if verbose and skipped:
            for n, r in skipped:
                print(f"    SKIP {n}: {r}")
    return model


def is_patched(model: nn.Module) -> bool:
    """Return True if any submodule was patched by `patch_seedvr2_vae`."""
    for _, mod in model.named_modules():
        if getattr(mod, _PATCH_MARKER_ATTR, False):
            return True
    return False


__all__ = ["patch_seedvr2_vae", "is_patched"]

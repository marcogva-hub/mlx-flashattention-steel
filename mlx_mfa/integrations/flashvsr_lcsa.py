"""LCSA integration patcher - drop-in routing for FlashVSR/VSR-style models.

Sprint B Phase H integration scaffolding. Per docs/lcsa-nax/lcsa-nax-phase1_5-
ship-verdict.md Section H caveat:

    FlashVSR's typical density 0.07-0.24 is OUTSIDE the current narrow niche
    (density < 0.02). Until a future matmul2d cooperative-tensor rewrite
    extends the niche, this patcher routes attention through
    `sparse_attention_dispatch()`, which at moderate density falls THROUGH
    to `mx.fast.scaled_dot_product_attention(mask=bias)` — i.e., the patcher
    is "code-path prep", not an immediate perf win for FlashVSR-typical
    densities.

    At very-sparse calls (density < 0.02), e.g. user-explicit aggressive
    block masks, the patcher delivers the 2.45-4.6x speedup measured in
    Phase 1.4. Sites whose attention masks are constructed with sliding
    window radius <= ~20 tokens fall in this niche.

Usage (recommended):

    from mlx_mfa.integrations.flashvsr_lcsa import patch_flashvsr_lcsa

    # 1. Mark the modules you want to route via Sprint B:
    for blk in model.transformer_blocks:
        blk.attn.lcsa_block_mask = bool_mask_for_this_block  # bool (NQ, NK)
        blk.attn.lcsa_block_tile = 16
        # Optionally cache the float bias to skip dispatcher's internal build:
        blk.attn.lcsa_precomputed_bias = float_bias_for_this_block

    # 2. Patch the model:
    model = patch_flashvsr_lcsa(model)

    # 3. Forward as usual. Patched modules route attention through
    #    `sparse_attention_dispatch` using the bool mask + precomputed bias
    #    set on the module.

    # 4. Optional unpatch:
    model = patch_flashvsr_lcsa(model, restore=True)

The patcher is intentionally OPT-IN per-module: it only patches modules
that have `lcsa_block_mask` set, so a model with mixed dense + sparse
attention blocks gets selective routing.

Pattern mirrors `mlx_mfa.integrations.seedvr2_vae.patch_seedvr2_vae` (Sprint
D D34 `__class__` swap) - see that file for the canonical pattern rationale.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_mfa.lcsa_nax import (
    sparse_attention_dispatch,
    DEFAULT_DENSITY_THRESHOLD,
)


_PATCH_MARKER_ATTR = "_mfa_lcsa_patched"
_ORIG_CLASS_ATTR = "_mfa_lcsa_orig_class"
_ORIG_CALL_ATTR = "_mfa_lcsa_orig_call"

# Attributes the user sets on the module to opt-in + parameterize routing.
LCSA_MASK_ATTR = "lcsa_block_mask"
LCSA_BIAS_ATTR = "lcsa_precomputed_bias"
LCSA_BT_ATTR = "lcsa_block_tile"
LCSA_CAUSAL_ATTR = "lcsa_causal"
LCSA_THRESHOLD_ATTR = "lcsa_density_threshold"
LCSA_DENSITY_ATTR = "lcsa_density"  # caller-cached density


def _eligibility_check(m: Any) -> Tuple[bool, str]:
    """Decide whether `m` is a candidate for LCSA patching.

    Eligibility: module has a callable `__call__` AND user has set
    `lcsa_block_mask` to a bool array. We deliberately DON'T try to infer
    attention modules by structure (q_proj/k_proj/v_proj convention varies
    too much across model families) - the user opts in explicitly per-block.
    """
    if not hasattr(m, "__call__"):
        return False, "not callable"
    mask = getattr(m, LCSA_MASK_ATTR, None)
    if mask is None:
        return False, f"no {LCSA_MASK_ATTR} attribute set"
    if not isinstance(mask, mx.array):
        return False, f"{LCSA_MASK_ATTR} is not an mx.array"
    if mask.dtype != mx.bool_:
        return False, f"{LCSA_MASK_ATTR} dtype must be bool, got {mask.dtype}"
    if mask.ndim < 2 or mask.ndim > 4:
        return False, f"{LCSA_MASK_ATTR} ndim must be 2/3/4, got {mask.ndim}"
    return True, "eligible"


def _make_patched_class(orig_class):
    """Build a dynamic subclass whose __call__ routes through dispatcher.

    The original __call__ is captured on the subclass so the patched method
    can delegate when LCSA routing is disabled or inapplicable. The patched
    call expects the module's __call__ signature to either:
      (a) match a typical attention block: f(self, x, ...) -> y, computing
          Q, K, V internally and calling SDPA. In this case we intercept
          BEFORE the call and just delegate to the original — the user is
          expected to have configured their internal SDPA call to route via
          sparse_attention_dispatch (rare case, kept for forward-compat).
      (b) be a standalone attention primitive accepting Q, K, V positionally.
          In this case the patched call routes via sparse_attention_dispatch.

    Phase 1.5 ship: pattern (b) is the primary expected usage; pattern (a) is
    a forward-compatibility hook (no behavior change in this version).
    """
    orig_call = orig_class.__call__

    def patched_call(self, *args, **kwargs):
        # Pattern (b): explicit Q, K, V positional args.
        # Heuristic: 3+ positional mx.array args of matching trailing shape.
        if (len(args) >= 3 and
                isinstance(args[0], mx.array) and
                isinstance(args[1], mx.array) and
                isinstance(args[2], mx.array) and
                args[0].ndim == 4 and args[1].ndim == 4 and args[2].ndim == 4 and
                args[0].shape[-1] == args[1].shape[-1] == args[2].shape[-1]):
            Q, K, V = args[0], args[1], args[2]
            mask = getattr(self, LCSA_MASK_ATTR)
            BT = getattr(self, LCSA_BT_ATTR, 16)
            causal = getattr(self, LCSA_CAUSAL_ATTR, False)
            threshold = getattr(self, LCSA_THRESHOLD_ATTR,
                                 DEFAULT_DENSITY_THRESHOLD)
            density = getattr(self, LCSA_DENSITY_ATTR, None)
            precomp_bias = getattr(self, LCSA_BIAS_ATTR, None)
            scale = kwargs.get("scale", None)
            return sparse_attention_dispatch(
                Q, K, V, mask,
                block_tile=BT,
                scale=scale,
                causal=causal,
                density_threshold=threshold,
                density=density,
                precomputed_bias=precomp_bias,
            )
        # Pattern (a) or anything else: fall through to original.
        return orig_call(self, *args, **kwargs)

    return type(
        f"_LCSAPatched_{orig_class.__name__}",
        (orig_class,),
        {"__call__": patched_call},
    )


def patch_flashvsr_lcsa(
    model: nn.Module,
    *,
    restore: bool = False,
    verbose: bool = False,
) -> nn.Module:
    """Patch (or restore) eligible attention submodules to route through LCSA dispatcher.

    Eligibility is opt-in per-module: user must set `module.lcsa_block_mask =
    <bool array>` for the patcher to take effect on that module. This lets a
    model with mixed dense + sparse attention blocks get selective routing.

    Args:
        model: an `mlx.nn.Module`. Walked via `named_modules()`.
        restore: if True, restore the original `__class__` on previously
            patched modules. Idempotent on un-patched models.
        verbose: print per-module patch/skip decisions.

    Returns:
        The same `model` object (in-place modified).

    Notes:
        - The patcher reads `lcsa_block_mask` at PATCH TIME for eligibility,
          but the patched `__call__` re-reads it on every forward — so the
          user can update the mask between calls without re-patching.
        - Pattern (b) (explicit Q, K, V positional args) is the primary path.
          Pattern (a) (attention block computing Q/K/V internally) falls
          through to the original call - no behavior change, kept as a
          forward-compatibility hook.
        - At density >= dispatcher threshold (default 0.02), the dispatcher
          internally routes to `mx.fast.scaled_dot_product_attention(mask=bias)`,
          so patching is performance-neutral for FlashVSR-typical density
          0.07-0.24 until a future matmul2d rewrite extends the niche.
    """
    patched: List[str] = []
    skipped: List[Tuple[str, str]] = []
    restored: List[str] = []

    for name, mod in model.named_modules():
        if restore:
            if getattr(mod, _PATCH_MARKER_ATTR, False):
                orig_class = getattr(mod, _ORIG_CLASS_ATTR)
                mod.__class__ = orig_class  # type: ignore[assignment]
                setattr(mod, _PATCH_MARKER_ATTR, False)
                if hasattr(mod, _ORIG_CLASS_ATTR):
                    delattr(mod, _ORIG_CLASS_ATTR)
                restored.append(name)
            continue

        ok, reason = _eligibility_check(mod)
        if ok:
            orig_class = mod.__class__
            patched_class = _make_patched_class(orig_class)
            setattr(mod, _ORIG_CLASS_ATTR, orig_class)
            mod.__class__ = patched_class  # type: ignore[assignment]
            setattr(mod, _PATCH_MARKER_ATTR, True)
            patched.append(name)
            if verbose:
                mask = getattr(mod, LCSA_MASK_ATTR)
                print(f"  [patch_flashvsr_lcsa] PATCHED  {name or '<root>'}: "
                      f"mask.shape={mask.shape}")
        else:
            # Only log if module had a partial LCSA setup (e.g., wrong dtype)
            if hasattr(mod, LCSA_MASK_ATTR):
                skipped.append((name, reason))
                if verbose:
                    print(f"  [patch_flashvsr_lcsa] SKIPPED  "
                          f"{name or '<root>'}: {reason}")

    if restore:
        if verbose or restored:
            print(f"[patch_flashvsr_lcsa] restored {len(restored)} module(s)")
    else:
        print(f"[patch_flashvsr_lcsa] patched {len(patched)} module(s), "
              f"skipped {len(skipped)} ineligible (with partial LCSA setup)")
        if verbose and skipped:
            for n, r in skipped:
                print(f"    SKIP {n}: {r}")
    return model


def is_patched(model: nn.Module) -> bool:
    """True if any submodule was patched by `patch_flashvsr_lcsa`."""
    for _, mod in model.named_modules():
        if getattr(mod, _PATCH_MARKER_ATTR, False):
            return True
    return False


__all__ = [
    "patch_flashvsr_lcsa",
    "is_patched",
    "LCSA_MASK_ATTR",
    "LCSA_BIAS_ATTR",
    "LCSA_BT_ATTR",
    "LCSA_CAUSAL_ATTR",
    "LCSA_THRESHOLD_ATTR",
    "LCSA_DENSITY_ATTR",
]

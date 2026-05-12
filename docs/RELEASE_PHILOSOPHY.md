# Release Philosophy — Auto-Default Principle

**Status**: canonical, in effect 2026-05-12 (Sprint U / v2.36.0).
**Referenced by**: `CLAUDE.md`, `CLAUDE_V6_NAX.md` §6.X pre-tag audit checklist.

## Core principle

Every PyPI release of `mlx-mfa` must be **fully functional transparently**
for users. Validated optimizations activate by default without requiring
user code changes. Opt-in mechanisms (env vars, named patchers) are
transitional (validation pending) or expert-mode (granular control),
**never the primary documented user path**.

## Three usage levels

### Level 1 — Default (auto-on-import)

For 90% of users. Install the package, import it, use MLX normally:

```python
import mlx.core as mx
import mlx_mfa  # auto-installs optimization hooks at import time

# Eligible Conv3D shapes (3x3x3 / 1x1x1, FP16/BF16) on M5+ now route
# through conv3d_nax_forward automatically — 1.02-2.26x speedup vs
# vanilla MLX, zero code change.
y = mx.conv_general(x, weight, padding=(1, 1, 1))
```

This level requires:
- No knowledge of mlx-mfa-specific API
- No patcher calls
- No env var configuration
- No model-specific instrumentation

The user thinks they're using vanilla MLX with a bonus optimization
package installed. The optimization "just happens".

### Level 2 — Explicit API (advanced users)

For users who need direct control over routing or use mlx-mfa-specific
features (varlen, paged, TurboQuant, etc.):

```python
from mlx_mfa import (
    flash_attention,
    flash_attention_sparse,
    sparse_attention_dispatch,
    flash_attention_paged,
    flash_attention_varlen,
    quantize_per_block,
    # ... full public API ...
)

# Call directly when you need:
# - Specific kernel selection
# - mlx-mfa-only features (LSE return, paged KV, etc.)
# - Granular block_mask + bias caching
out = flash_attention_sparse(q, k, v, block_mask, scale=scale, causal=True)
```

This level is for users who know what they're doing and want the explicit
API surface. The auto-hooks remain installed but explicit calls bypass them.

### Level 3 — Expert mode (research/debug)

For research/debug workflows requiring per-module granular control:

```python
from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae
from mlx_mfa.integrations.flashvsr_lcsa import patch_flashvsr_lcsa
from mlx_mfa.integrations.mlx_lm import patch_mlx_lm

# Per-module attribute marking + class swap
for blk in model.transformer_blocks:
    blk.attn.lcsa_block_mask = bool_mask
patch_flashvsr_lcsa(model)

# Granular Conv3D patching with eligibility logging
patch_seedvr2_vae(model, verbose=True)
```

This level is appropriate when:
- The user needs to mark per-module attributes (e.g., LCSA `lcsa_block_mask`)
- The user wants verbose logging of which modules were patched
- The user needs to opt INTO an experimental optimization the auto-hooks
  don't yet enable

## Migration policy

When an optimization graduates from opt-in to default:

1. **The env var control remains** for benchmarking / A/B comparison.
   Example: `MFA_LCSA_KERNEL_VERSION=v1` continues to work post-graduation
   for users who want to compare V1 vs V2 perf.

2. **Existing user code continues to work**. Adding an auto-hook does NOT
   break user code that already calls the explicit API or applies a patcher.
   The auto-hook activates only when neither has been called.

3. **The default activates the optimization without user action**. The
   shift is silent for users who don't read CHANGELOGs. They notice only
   because their code got faster.

4. **CHANGELOG documents the graduation** with explicit "now active by
   default" language. Migration notes provided for users who relied on
   the opt-in name.

## Pre-tag audit checklist

Before tagging any PyPI release, CC verifies (in addition to standard
multi-SoT version audit per the v2.33.x lesson):

- [ ] New optimizations integrated into auto-routing paths (existing
      `flash_attention*`, `sparse_attention*`, `conv3d_nax_forward`, etc.)
- [ ] Auto-on-import hooks register the optimization if it requires
      hooking external `mx.*` surfaces (e.g., `mx.conv_general`)
- [ ] Env-var opt-in present **only** for:
      (a) **Transitional state** — validation pending (e.g., V2 sparse
          SHIP_OPT_IN pre-v2.36.0 while sub-1ms methodology unsolved)
      (b) **Escape hatch** — for benchmarking (e.g., `MFA_DISABLE_AUTO_HOOKS`)
      (c) **A/B comparison knob** — pre-graduation comparison
          (e.g., `MFA_LCSA_KERNEL_VERSION` pre-v2.36 graduation)
- [ ] Named patchers documented as expert-mode, NOT primary path
- [ ] README primary usage path is `import mlx_mfa` + normal MLX usage
- [ ] Migration documented in CHANGELOG if optimization graduates from
      opt-in to default

## When opt-in is appropriate

Opt-in via **env var** is appropriate when:
- **Methodology validation pending** (e.g., V2 sparse SHIP_OPT_IN
  pre-v2.36.0 while sub-1ms protocol unsolved). Use `MFA_<FEATURE>=v2`
  with clear migration timeline.
- **A/B comparison needed** for research / debugging. Use
  `MFA_DISABLE_<FEATURE>` patterns. Both names should describe what's
  being toggled in plain English.
- **Breaking change requires user consent**. Use `MFA_ENABLE_<EXPERIMENTAL_FEATURE>`
  pattern that requires explicit opt-in until graduation.

Opt-in via **patcher** is appropriate when:
- **Granular control needed** (e.g., per-block routing in FlashVSR LCSA
  via `lcsa_block_mask` attribute on specific modules)
- **Hook surface too risky for global auto-application** (e.g., per-module
  attention swap with structural uncertainty about which modules are
  attention modules in a given model)
- **User explicitly opts into experimental behavior** that doesn't have a
  clean auto-detection signal

**All other cases ship as auto-default.** If you find yourself documenting
"call this patcher to get the optimization", reconsider — can the auto-hook
infer the eligibility?

## Anti-patterns

The following are explicit anti-patterns to avoid in future sprints:

1. **Shipping a perf win behind a required env var.** Unless validation is
   pending or the win is unstable, the env var should be the opt-OUT path,
   not the opt-IN.

2. **Requiring `patch_foo(model)` for an optimization that has a clean
   structural eligibility check.** If Conv3D NAX can detect eligible
   shapes via weight.shape, the patcher should not be the primary path.

3. **Documenting `patch_foo()` in README's "Minimal Usage" section.**
   Patchers belong in the Expert mode section. Minimal usage is the
   auto-default path.

4. **Surprising the user via a hook**. If `import mlx_mfa` changes some
   semantic that the user might depend on (output values, numerical
   precision, observable side effects beyond perf), that's a bug, not
   a feature. Auto-hooks must preserve exact MLX semantics for ineligible
   shapes/dtypes.

## Rationale

The auto-default principle codifies a user-experience target: **users
who type `pip install mlx-mfa` get the validated optimizations the
project ships, by default, without homework**. Anything that asks the
user to learn a new API surface or wire something up is friction. The
project absorbs that friction as engineering effort, so the user doesn't
have to.

This principle is informed by:

- **Sprint B v2.34.0 ship** which introduced `sparse_attention_dispatch`
  but required users to call it explicitly. Few users discovered it.
- **Sprint D v2.33.0 Conv3D NAX ship** which required `patch_seedvr2_vae`.
  Discoverable only for users who knew about SeedVR2.
- **Sprint U v2.36.0** auto-on-import design that removes both gates.

The opportunity cost of NOT shipping by default is users not benefiting
from optimizations that ARE validated and SHOULD be active.

## Future-work registry update

When evaluating future-work register items, apply the principle as a
checkpoint:

- An optimization in "tracked future work" should specify whether it
  will ship as default or opt-in. If opt-in, the migration plan to
  default should be sketched.
- Methodology-pending items (sub-1ms protocol) should NOT be merged to
  master's default path until methodology is resolved.
- Architectural items (V34 backward Option β) should ship as default
  via `flash_attention()` VJP auto-routing when validated.

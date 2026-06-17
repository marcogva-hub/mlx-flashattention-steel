# Sprint U — Decisions log

**Date opened**: 2026-05-12
**Branch**: `feat/sprint-u-unification`
**Foundation**: master @ `8ca029f` (v2.35.0)

## DU1 — Codify auto-default principle (Section A)

**Decision**: introduce `docs/RELEASE_PHILOSOPHY.md` as canonical doc;
amend CLAUDE.md and CLAUDE_V6_NAX.md to reference it; add pre-tag
audit checklist.

**Rationale**: Sprint B (v2.34.0) and Sprint D (v2.33.0) both shipped
opt-in surfaces (`sparse_attention_dispatch`, `patch_seedvr2_vae`) that
few users discovered. Auto-default reduces friction.

**Reversibility**: docs-only; trivial.

## DU2 — `flash_attention_sparse` auto-route on M5+ for symmetric-BT masks (Section B)

**Decision**: place auto-route BEFORE STEEL's asymmetric BQ/BK validator
in `flash_attention_sparse`. If mask is symmetric BT ∈ {16, 32, 64},
route through `sparse_attention_dispatch`. Else fall through to STEEL
validator + path (M1-M4) or `_sparse_fallback_sdpa_perhead` (M5+).

**Rationale**: STEEL's asymmetric BQ=32 BK=16 validator would reject
symmetric BT masks pre-Sprint-U, so symmetric masks were never accepted.
Auto-routing them through the dispatcher is purely additive.

**Reversibility**: env var `MFA_DISABLE_AUTO_HOOKS=1` restores pre-Sprint-U
behavior (asymmetric masks accepted, symmetric masks rejected by validator).

## DU3 — Auto-hooks at import time (Section C)

**Decision**: `mlx_mfa/__init__.py` calls `install_hooks()` at import
time unless `MFA_DISABLE_AUTO_HOOKS=1`. Hook patches `mx.conv_general`
to auto-route eligible Conv3D shapes.

**Rationale**: removes `patch_seedvr2_vae(model)` requirement for default
users. Patcher remains available as expert API for verbose logging.

**Reversibility**: `mlx_mfa.disable()` programmatic OR `MFA_DISABLE_AUTO_HOOKS=1`
env. Both idempotent.

## DU4 — Auto-hook eligibility check mirrors patch_seedvr2_vae

**Decision**: `_conv3d_nax_eligible(weight, stride, padding, ...)` checks
the same conditions as `patch_seedvr2_vae`'s `_eligibility_check`:
- 5-D weight (Conv3D)
- kernel_size ∈ {(3,3,3), (1,1,1)}
- dtype ∈ {float16, bfloat16}
- stride == (1,1,1), dilation == (1,1,1), groups == 1, !flip
- M5+ hardware

**Rationale**: same eligibility logic ensures auto-hook routes the same
shapes the explicit patcher would route. Behavioral equivalence.

**Reversibility**: high — single helper function.

## DU5 — `__mlx_mfa_hook__` marker on patched function

**Decision**: tag the patched `mx.conv_general` replacement with
`__mlx_mfa_hook__ = True` attribute. Detection: if `mx.conv_general`
already has this attribute, skip re-installation (idempotent).

**Rationale**: prevent double-wrapping if `mlx_mfa.enable()` is called
twice without an intervening `disable()`. Also lets external libraries
that hook `mx.conv_general` detect our hook and decide what to do.

## DU6 — Three-axis validation applied to Sprint U itself

**Decision**: per the §3.5 rule, Sprint U's 13 new tests cover:
- Axis 1 (output sanity): hooked output correct (Section C test_axis1)
  + auto-routed sparse output correct (Section B test_axis1)
- Axis 2 (path entered): conv3d_nax_forward IS invoked on eligible
  (Section C test_axis2) + sparse_attention_dispatch IS invoked on M5+
  (Section B test_axis2)
- Axis 3 (edges preserved): ineligible shapes pass through to vanilla
  (Conv2D, FP32, 5×5×5, etc.); env disable works; idempotent enable/disable

**Rationale**: the rule we committed in Section A applies retroactively to
Sprint U's own code changes. Catches potential silent failures.

## DU7 — V2 sparse stays opt-in (deferred)

**Decision**: `MFA_LCSA_KERNEL_VERSION=v2` remains opt-in. Sprint U does
NOT flip V2 to default.

**Rationale**: sub-1ms methodology validation pending per
`docs/methodology/sub1ms-protocol-diagnostic.md`. V2 ships in the codebase
but graduation to default awaits methodology resolution. Sprint U
prepares the path: when methodology resolved, the flip is a one-line
change in `csrc/mfa_sparse_attention.cpp:read_kernel_version_env()`.

## DU8 — Patchers preserved as expert API

**Decision**: `patch_seedvr2_vae`, `patch_flashvsr_lcsa`, `patch_mlx_lm`
remain available. README documents them in "Three usage levels" §3
(Expert mode), not in primary usage path.

**Rationale**: patchers provide capabilities auto-hooks can't (verbose
logging, per-module attribute marking for LCSA, granular control).
Their primary use case shifts from "required for the optimization" to
"opt into experimental / verbose mode".

## DU9 — Backward compatibility guaranteed

**Decision**: no public API signature changes. v2.35.0 user code continues
to work without modification on v2.36.0.

Validation: all 52 pre-Sprint-U LCSA + integration tests pass unchanged
(this is the "edges preserved" axis at the project level).

## DU10 — v2.36.0 release as unification milestone

**Decision**: tag v2.36.0 even though no new kernel/perf work. The release
captures the architectural shift (auto-on-import + philosophy doc).

**Rationale**: minor version bump signals new capabilities (auto-hook)
without breaking changes. The shift in posture is significant enough to
deserve its own release rather than being bundled with V6NAX backward
Option β (the next architectural sprint).

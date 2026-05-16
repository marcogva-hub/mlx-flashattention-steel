# Prompt 5h-A — Wheel handoff for SeedVR2 benchmark validation

**Built from**: master tip `b09b9c8` (post-Prompt 5g — KD-6 dtype cast fix + KD-7 bf16 mitigation + hook telemetry + Pattern #8 codification)

**Date**: 2026-05-16

## Wheel artifacts

**Wheel path** (install this):
```
/Users/marcomarcelino/code/mlx-mfa-v2/dist/mlx_mfa-2.50.0-cp311-cp311-macosx_26_0_arm64.whl
```

**Sdist** (source distribution, for reference):
```
/Users/marcomarcelino/code/mlx-mfa-v2/dist/mlx_mfa-2.50.0.tar.gz
```

**Sha256 checksums**:
- wheel: `a66bcad72dc45a925ecb72b194a66ffb1a8ed6f7a8cbde77e51b8ae64aa91b96`
- sdist: `7583ff4a51d91d38f977fd137251b68623587516d31721a2bd3e17ea64362fef`

## Version identifier

The wheel reports `mlx_mfa.__version__ == "2.50.0"` — no version bump was applied
for this validation build.  The **real identifier** is the git commit hash
`b09b9c8`, which contains the post-Prompt-5g code (Phase A KD-6 fix +
Phase C hook telemetry + Phase E Pattern #8 codification).

The actual v2.50.1 release (version bump + tag + PyPI upload + GH release)
happens in **Prompt 5h-C** after Marco provides the benchmark results from
the SeedVR2 session.

## What changed vs PyPI `mlx-mfa==2.50.0`

| Item | PyPI v2.50.0 | This wheel (`b09b9c8`) |
|---|---|---|
| `_auto_hooks.py::_patched_conv_general` | Raises `RuntimeError: conv_nax: x.dtype != w.dtype` on every mismatched call (KD-6) | Dtype cast: input → weight dtype; output cast back to baseline-promoted dtype; defensive try/except (KD-6 RESOLVED) |
| bf16 weight NAX path | Raises Metal compile error at graph eval (KD-7) | Eligibility tightened to fp16-only; bf16 weights fall back to MLX baseline gracefully (KD-7 mitigated) |
| Hook telemetry | Not present | `mlx_mfa.get_hook_stats()` + `mlx_mfa.reset_hook_stats()` public API; `MLX_MFA_HOOK_TELEMETRY` env var with `off`/`summary`/`verbose` modes |
| Pattern #8 codification | Not present | `docs/v50/audit-framing-inversions.md` Pattern #8 + `docs/HOOK_TELEMETRY.md` + skill amendments |

**Net effect for VSR pipelines**: the M5 Neural Engine NAX Conv3D path
(`mx.conv_general` for fp16-weight 3×3×3 or 1×1×1 stride-1 dilation-1
shapes on M5+) now executes natively when input/weight dtype mismatch
occurs.  No caller-side workaround required.

## Verification results

Pre-build (master state verified):
- master tip `b09b9c8`, working tree clean
- 23/23 dtype regression tests pass
  (`tests/test_v50_prompt_5g_conv3d_nax_dtype_compatibility.py`)
- `mlx_mfa.get_hook_stats()` + `mlx_mfa.reset_hook_stats()` API present in dev install

Build:
- `python -m build` → wheel + sdist produced
- `twine check dist/mlx_mfa-2.50.0*` → both PASSED
- Wheel size: 579 KB; sdist size: 1.94 MB

Isolated install smoke test (fresh venv at `/tmp/mlx_mfa_wheel_test`):
- `pip install <wheel>` → no errors
- `import mlx_mfa` → resolves to wheel path (`module file:
  /private/tmp/mlx_mfa_wheel_test/lib/python3.11/site-packages/mlx_mfa/__init__.py`)
- `mlx_mfa.__version__` → `2.50.0`
- `get_hook_stats`, `reset_hook_stats` → both present
- `hooks_status()` → `'installed': True, m5_plus=True`
- `get_device_info()` → Apple M5 Max, gen 17, M5+, `extension_available: True`
- KD-6 functional smoke (fp32 input + fp16 weight Conv3D):
  - Output shape correct, dtype = fp32 (baseline-promoted contract preserved)
  - Output finite
  - `executed.conv3d_nax_forward == 1`, `fallback == 0` (NAX path engaged correctly)

## Install instructions for SeedVR2 session

In the SeedVR2 venv (whichever Python environment SeedVR2's inference runs in):

```bash
# Force reinstall over any existing mlx-mfa, do not touch other packages
<seedvr2-venv>/bin/pip install --force-reinstall --no-deps \
  /Users/marcomarcelino/code/mlx-mfa-v2/dist/mlx_mfa-2.50.0-cp311-cp311-macosx_26_0_arm64.whl

# Verify
<seedvr2-venv>/bin/python -c "
import mlx_mfa
print('version:', mlx_mfa.__version__)
print('telemetry API:', hasattr(mlx_mfa, 'get_hook_stats') and hasattr(mlx_mfa, 'reset_hook_stats'))
print('hooks_status:', mlx_mfa.hooks_status())
"
```

Expected output:
```
version: 2.50.0
telemetry API: True
hooks_status: {'installed': True, 'log': ['mlx_mfa auto-hooks installed: ... M5+=True'], 'm5_plus': True, 'auto_hooks_disabled_env': False}
```

Notes:
- `--no-deps` avoids disturbing other SeedVR2 venv packages (mlx, mlx-metal, etc.)
- `--force-reinstall` overrides any existing `mlx-mfa` (including the PyPI v2.50.0
  install — this is intentional; we want this wheel's post-5g code).
- Python version must be 3.11.x.  The wheel is `cp311` and not compatible with
  Python 3.12 / 3.13 / 3.14.  If SeedVR2 runs on a different Python version,
  build a wheel for that interpreter via:
  `CMAKE_ARGS="-DPython_EXECUTABLE=$(which python3.X)" python -m build`

## Benchmark validation checklist for SeedVR2 session

When the SeedVR2 session runs the Phase 95 equivalent (Phase 96):

1. **Remove/disable the caller-side conv3d workaround** if present.  Inspection
   of `src/mlx_native/mflux/models/seedvr2/model/seedvr2_vae/common/conv3d.py`
   in the SeedVR2 repo shows a Phase 82 dispatch to a custom Metal kernel
   (`causal_conv3d_metal_333`) gated by env var
   `SEEDVR2_MLX_METAL_CONV_ENABLED` (default `0` = disabled, so this path
   is OFF by default).  No action needed unless the env var has been set to
   `1` somewhere — in that case unset/set to `0` so all Conv3D calls go
   through `mx.conv_general` and engage mlx-mfa's NAX hook.

2. **Set telemetry mode (default summary is fine)**:
   ```python
   import os
   # os.environ["MLX_MFA_HOOK_TELEMETRY"] = "summary"  # default
   # or "verbose" for per-fallback warnings during debugging
   ```

3. **Before inference**:
   ```python
   import mlx_mfa
   mlx_mfa.reset_hook_stats()
   ```

4. **Run inference with exact Phase 86 settings**:
   - File: `IMG_0508.mp4` (the original source, 895 frames, 176×144)
   - Target: 528×432 (3× upscale)
   - Config: 3B fp16 · native unified encode+decode · native DiT ·
     batch_size=9 · no quantize · no offload
   - Capture per-phase wall-clock timings: P1 (unified encode), P2 (DiT
     3B fp16), P3 (unified decode), P4 (post-process), Total

5. **After inference**:
   ```python
   stats = mlx_mfa.get_hook_stats()
   print(stats)
   ```
   **GATE checks** (these are the v2.50.1 release gates):
   - `stats["executed"]["conv3d_nax_forward"]` should be **much greater
     than 0** (every VAE encode/decode 3×3×3 fp16 Conv3D layer engaged NAX).
   - `stats["fallback"]["conv3d_nax_forward"]` should be **0** (no silent
     fallback to MLX baseline).
   - `stats["fallback_reasons"]["conv3d_nax_forward"]` should be empty.

   If `fallback > 0`: report the `fallback_reasons` list back to the
   mlx-mfa session — there is a path the dtype cast didn't catch.

6. **Visual quality check**: 5-frame spot-inspection of output vs known-good
   v2.50.0+workaround output.  No new artifacts (banding, flicker, color
   shift, seams).

7. **Report back to Marco** (for relay to mlx-mfa session / Prompt 5h-C):
   ```
   Phase 96 (v2.50.0+post-5g, no workaround):
     P1 = X.Xs   P2 = X.Xs   P3 = X.Xs   P4 = X.Xs   Total = X.Xs
   Hook stats:
     conv3d_nax_forward executed = N, fallback = 0
   Visual quality: PASS / FAIL [+ notes]
   ```

   The Prompt 5h-C release in the mlx-mfa session will:
   - Compare vs Phase 95 baseline (517s, M5 Max, v2.50.0 + workaround).
   - GATE: regression > 5% → halt + escalate; within ±5% or faster →
     proceed to v2.50.1 release.
   - Bump version 2.50.0 → 2.50.1, rebuild wheel with proper version
     string, tag, upload to PyPI, create GH release, push origin.

## What this prompt did NOT do

- No version bump (stays at 2.50.0; real identifier is commit `b09b9c8`).
- No git tag.
- No PyPI upload.
- No GH release.
- No push to origin.
- No additional commits beyond this handoff doc.

The actual v2.50.1 release procedural is **Prompt 5h-C** after the
SeedVR2 session reports benchmark results.

## File list

- `dist/mlx_mfa-2.50.0-cp311-cp311-macosx_26_0_arm64.whl` (the validation wheel)
- `dist/mlx_mfa-2.50.0.tar.gz` (sdist for reference / fallback)
- `docs/v50/prompt-5h-a-wheel-handoff.md` (this document)

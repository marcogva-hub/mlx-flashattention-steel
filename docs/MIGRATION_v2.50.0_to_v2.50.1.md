# Migration guide: v2.50.0 → v2.50.1

**Audience**: existing mlx-mfa v2.50.0 users upgrading to v2.50.1.
**TL;DR**: critical perf-unlock patch.  Zero source changes required.
All M5+ users should upgrade — v2.50.0 silently runs Conv3D on baseline
Metal shaders; v2.50.1 unlocks Neural Engine acceleration.

---

## Breaking changes

**None.**  Bugfix patch.  All public API signatures preserved.

---

## What v2.50.1 delivers

### KD-6 fix — Pattern #8 root-cause closure

The `_auto_hooks.py::conv3d_nax_forward` auto-hook (introduced v2.36.0)
silently blocked the M5 Neural Engine Conv3D acceleration path for
**every VAE encoder call with mismatched input/weight dtypes** (the
canonical fp32 input + fp16 weight pattern in VSR pipelines).

Pre-fix mechanism:
- C++ NAX kernel requires `x.dtype == w.dtype`
- Python eligibility only verified `weight.dtype ∈ {fp16, bf16}`
- Mismatched dtypes raised `RuntimeError: conv_nax: x.dtype != w.dtype`
- User pipelines (SeedVR2, FlashVSR, STCDiT, etc.) silently absorbed
  the exception via downstream `try/except` wrappers, masquerading as
  "everything works at baseline performance"

Post-fix:
- Hook casts input to weight dtype before NAX dispatch
- Restores baseline output dtype after the kernel call (preserves MLX
  promotion contract)
- Defensive `try/except` falls back to MLX baseline on any unexpected
  NAX failure
- Hook telemetry records the engagement vs fallback decision

**Production impact**: M5 Neural Engine fixed-function Conv3D
acceleration now executes natively on every NAX-eligible call.
Flagship benchmark (SeedVR2 3B fp16, 895 frames at 432p) shows **3.10×
speedup over M1 Max optimized baseline**:

| Phase | M1 Max baseline | M5 Max v2.50.1 | Speedup |
|---|---|---|---|
| P1 unified encode (Conv3D-heavy) | 370s | 132s | **2.80×** |
| P2 DiT 3B fp16 | 360s | 72s | **5.00×** |
| P3 unified decode (Conv3D-heavy) | 900s | 322s | **2.80×** |
| P4 post-process | 22s | 7s | **3.14×** |
| Total | 1655s | 533s | **3.10×** |

### KD-7 mitigation — bf16 NAX path fp16-only

While validating KD-6, we discovered that the bf16 NAX Conv3D path has
been broken at the MLX upstream Metal shader level since v2.36.0
(`utils.h:502` im2col helper has a `half` vs `bfloat16_t` type
mismatch).  Zero user reports because no production workload exercised
the bf16 path before this audit.

Mitigation in v2.50.1: NAX eligibility tightened to fp16 weights only.
bf16 weights now fall back to MLX baseline gracefully (which works).
Full fix is a v2.51 task pending upstream MLX coordination OR a
mlx-mfa-specific bf16 NAX kernel path.

### Hook telemetry — Pattern #8 detection infrastructure

New public API:

```python
import mlx_mfa

# Reset counters before measuring a scope
mlx_mfa.reset_hook_stats()

# Run your workload
my_model(my_input)

# Inspect engagement
stats = mlx_mfa.get_hook_stats()
print(stats)
# {
#   'executed': {'conv3d_nax_forward': 12408},
#   'fallback': {'conv3d_nax_forward': 675},
#   'fallback_reasons': {'conv3d_nax_forward': ['kernel/stride/dilation/groups/flip constraint failed']},
#   'mode': 'summary'
# }
```

Three modes via `MLX_MFA_HOOK_TELEMETRY` env var:
- `off`: zero overhead, no counters
- `summary` (default): per-hook counters; ~1% microbench overhead, <0.1% production
- `verbose`: summary + UserWarning per fallback (developer mode)

See `docs/HOOK_TELEMETRY.md` for the full reference.

### Workaround obsolescence note

Pipelines that applied caller-side workarounds for the v2.36.0-v2.50.0
NAX Conv3D break (e.g., SeedVR2's Phase 82 `causal_conv3d_metal_333`
custom Metal kernel dispatch, or any fp16-cast wrapper around
`mx.conv_general`) are now **redundant but harmless** if left in place.

For minimum future maintenance burden, you can remove such workarounds
— mlx-mfa v2.50.1 handles the dtype mismatch natively.  Hook telemetry
confirms NAX engagement on every supported shape.

---

## Upgrade procedure

```bash
pip install --upgrade mlx-mfa==2.50.1
```

Verify:

```bash
python -c "
import mlx_mfa
print('version:', mlx_mfa.__version__)
print('telemetry:', hasattr(mlx_mfa, 'get_hook_stats'))
print('hooks_status:', mlx_mfa.hooks_status())
"
```

Expected on M5+:
```
version: 2.50.1
telemetry: True
hooks_status: {'installed': True, ..., 'm5_plus': True, 'auto_hooks_disabled_env': False}
```

Optional verification with your actual workload:

```python
import mlx_mfa
mlx_mfa.reset_hook_stats()
# run a representative inference call
print(mlx_mfa.get_hook_stats())
# Expect 'executed.conv3d_nax_forward' >> 0
# 'fallback.conv3d_nax_forward' may be > 0 if your model has Conv3D
# layers with non-(3,3,3)/(1,1,1) kernels — that is correct behavior,
# not a bug. Check 'fallback_reasons' to confirm.
```

---

## Yank policy for v2.50.0

v2.50.0 is **NOT yanked** per project policy:
- The KD-6 bug pre-existed since v2.36.0 (not a v2.50.0 regression)
- v2.50.0 functional behavior is preserved (correct output via caller-
  side fallback in user pipelines)
- v2.50.1 release notes prominently flag the perf-unlock to drive
  upgrade
- Yanking would break installs that pin v2.50.0 for reproducibility

v2.50.0 remains installable for users who need it for any reason.

---

## Cross-references

- `CHANGELOG.md [2.50.1]` — full release notes
- `docs/HOOK_TELEMETRY.md` — telemetry API + Pattern #8 detection
- `docs/v50/known-debt-v2.50.md` — KD-6 (resolved) + KD-7 (open)
- `docs/v50/audit-framing-inversions.md` — Pattern #8 codification
- `docs/v50/bench-data/seedvr2-img0508-432p-3bfp16/` — flagship benchmark dataset

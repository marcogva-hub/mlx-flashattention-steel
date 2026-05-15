# Prompt 5g Phase B — Complete hooks inventory + Pattern #8 audit

**Scope**: audit ALL hooks (auto-installed + opt-in) in mlx-mfa for
Pattern #8 vulnerability (tight contract + silent fallback masking
unused optimization path).

**Date**: 2026-05-15

## Inventory

| Hook | Install mechanism | Patched primitive | Contract assertions | Fallback mechanism | Classification (pre-fix) | Classification (post-Phase-A) |
|---|---|---|---|---|---|---|
| `_auto_hooks.py::_patched_conv_general` | **Auto** at import (unless `MFA_DISABLE_AUTO_HOOKS=1`) | `mx.conv_general` | (1) weight 5-D, (2) weight dtype in {fp16, bf16}, (3) kernel ∈ {(3,3,3), (1,1,1)}, (4) stride/dilation/groups/flip constraints, (5) M5+ hardware. **MISSING**: input/weight dtype match check. | None — exceptions from C++ NAX kernel propagate as Python `RuntimeError`. | **(C) Contract tightened + silent fallback** (HIGH Pattern #8 risk; user pipelines absorbed the exception). | **(A) Contract preserved + silent fallback** (Phase A added input cast + try/except; Phase C will add verbose telemetry). |
| `integrations/mlx_lm.py::patch_mlx_lm` | Opt-in (user calls `patch_mlx_lm()`) | `mlx_lm.models.attention_utils.scaled_dot_product_attention` | (1) D ∈ supported head dims, (2) dtype ∈ {fp16, bf16}, (3) mask type checks. | Returns `_original_sdpa(...)` for every unsupported case with verbose dispatch logging (`verbose_dispatch=True`). | **(A) Contract preserved + verbose fallback** — already Pattern #8 safe. | Unchanged. |

## Methodology

1. `grep -rn 'mx\.\(.*\) = ' mlx_mfa/` → only `mx.conv_general` patched
   in `_auto_hooks.py:245` (and restored on uninstall at `:264`).
2. `grep -rn 'def patch_' mlx_mfa/` → `patch_mlx_lm` in
   `integrations/mlx_lm.py`.
3. Each hook inspected for: input contract vs MLX baseline contract;
   fallback path verbose vs silent; performance-critical engagement.

## Phase B verdict

**No additional hooks at Pattern #8 risk.**  The Pattern #8 root cause
was localized to `_patched_conv_general` and is fully addressed by
Phase A:

- Input dtype cast preserves the MLX baseline contract (Phase A)
- Defensive `try/except` falls back to MLX baseline on any unexpected
  NAX failure (Phase A)
- bf16 weight path excluded from eligibility (KD-7 mitigation; Phase A)
- 23 regression tests lock the dtype matrix (Phase A)

**Outstanding work**: add verbose telemetry for the silent fallback
path in `_patched_conv_general` (Phase C scope).  This will mirror the
already-shipped `patch_mlx_lm` telemetry pattern and provide users +
audits a way to detect future hook-engagement failures without needing
a deep code dive.

## Cross-reference

- Pattern #8 codification: `docs/v50/audit-framing-inversions.md` (added Phase E)
- KD-6 (fixed): `docs/v50/known-debt-v2.50.md`
- KD-7 (open): `docs/v50/known-debt-v2.50.md`
- Phase A fix: `mlx_mfa/_auto_hooks.py` + `tests/test_v50_prompt_5g_conv3d_nax_dtype_compatibility.py`
- `mlx_lm` reference pattern: `mlx_mfa/integrations/mlx_lm.py::patch_mlx_lm`

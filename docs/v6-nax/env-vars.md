# V6 NAX environment variables

Reference for all V6 NAX env vars exposed by `csrc/mfa_v6_nax_primitive.cpp`.
Behaviour and defaults reflect v2.32.0 (post-V34 Forward-Max sprint suite).

## Layout

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_BNHD_LEGACY` | unset | If set to anything, forces legacy BNHD layout instead of BHND. The Primitive transposes Q/K/V into Draw Things' [B, N, H, D] kernel layout, dispatches, then transposes O back. Use only for A/B comparison or GQA shapes (Hq != Hk auto-fall back to BNHD anyway). |

## Tile-config overrides

Auto-tuned defaults (Sprint 3.3 + autoresearch on M5 Max):
- `BQ = 16` universally
- `BK = (head_dim == 64) ? 64 : 32`
- `exec_sg = (head_dim == 64) ? 2 : 8`
- `BD = head_dim` (single Otile)

Override via env (used by `bench/v6_*_autoresearch*.py`):

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_BLOCK_R` | auto | BQ — parallelization rows per Q-tile. |
| `MFA_V6_BLOCK_C` | auto | BK — traversal columns per K-tile. |
| `MFA_V6_EXEC_SG` | auto | execution_simdgroups — number of simdgroups per threadgroup. |
| `MFA_V6_BLOCK_D` | head_dim | BD — head-dim block size. Single-Otile assumes BD == head_dim (kBlocks=1). |

## Kernel-variant flags

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_NAX_SINGLE_OTILE` | auto (`Hq == Hk`) | Selects the Apple-style single-Otile kernel (`loopForwardSingleTile()`) over the legacy double-buffered `loopForward()`. Auto-enabled for non-GQA (Hq == Hk); auto-disabled for GQA (the BHND rewriter doesn't yet support per-head K-stride for single-Otile). Explicit `0` or `1` overrides the auto-default. |
| `MFA_V6_BYPASS_TGP` | implied by single-Otile | Forces the always-bypass cP cooperative-tensor PV path (no P_buf threadgroup staging). When `MFA_V6_NAX_SINGLE_OTILE=1`, this is forced on automatically. Sprint 3.2 found bypass alone (without single-Otile) regressed on D=128; with single-Otile + new tiles the combination is the new default. |

## V34 (NAX-direct) flags — added in v2.31.0–v2.32.0

V34 is the Apple `steel_attention_nax.h`-style NAX-direct kernel
(`createV34Source`), bypassing MPP cooperative_tensor entirely. Its
dispatch is shape-aware:

- D=128 (any shape) → V34 default
- D=64 (any N) → V34 default since v2.32.0 (Sprint 4 fix, was D=64 N≥8000 only in v2.31.0)
- D=256+ → not yet ported, falls back to legacy
- Causal forward → V34 supports it since v2.32.0 (Sprint 1)

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_USE_V34` | auto (per dispatch policy) | `1` forces V34 for any V34-eligible shape; `0` forces legacy fallback. Useful for A/B comparison and bench wrappers. |
| `MFA_V6_V34_BQ` | per-D auto (`D=64 → 32`, `D=128 → 64`) | V34 parallelization rows per Q-tile. Constraint: `BQ % (WM × 16) == 0`. |
| `MFA_V6_V34_BK` | per-D auto (`D=64 → 32` since v2.32.0; `D=128 → 32`) | V34 traversal columns per K-tile. v2.32.0 changed D=64 default from 64 → 32 after Sprint 4 sweep showed BK=32 wins +14–20%. |
| `MFA_V6_V34_WM` | per-D auto (`D=64 → 2`, `D=128 → 4`) | V34 simdgroups per threadgroup. |
| `MFA_V6_V34_DISABLE_ALIGN` | unset | If set to `1`, disables Sprint 3's compile-time `align_Q` / `align_K` specialization — V34 always uses the unaligned kernel even on shapes where `qL % BQ == 0` and `kL % BK == 0`. Use for A/B perf comparison; perf-neutral at our pipeline-cache scale (24 entries) per Sprint 3 measurements, kept as escape hatch. |

## Source-generator post-modification flags (from older sprints)

These rewrite the generated MSL source string after Draw Things' generator emits it. Mostly used for parameter-axis paradox testing in earlier sprints; the autoresearch v5 results show they no longer help — kept for backward-compat / diagnostics.

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_FORCE_DYNAMIC_K` | unset | Force `dynamic_length_v<int>` for K dimension in matmul descriptors even when BK%32==0. Hits Apple MPP static_assert if combined with cooperative-left input — keep off unless paradox-testing. |
| `MFA_V6_RELAXED_PRECISION` | `1` (relaxed) | If set to `0`, disables relaxed_precision in matmul2d descriptors. Slightly more accurate, slightly slower. |
| `MFA_V6_UNROLL_MODE` | `full` | Override #pragma clang loop unroll directive. Values: `full`, `none`, `2`, `4`. |

## Diagnostic flags

| Var | Default | Effect |
|---|---|---|
| `MFA_V6_SENTINEL_FILL` | unset | Pre-fills the V6 output array with FP16 sNaN before kernel dispatch, in unified memory. After dispatch, any sNaN cell is one the kernel didn't write — used to verify 100% tile coverage. Use in tandem with the diagnostic scripts in `bench/`. |

## Other (non-V6) env vars relevant to mlx-mfa

For completeness:

| Var | Effect |
|---|---|
| `MFA_DISABLE_GNA_NATIVE` | Forces sparse fallback for GNA backward tests. |
| `MFA_DISABLE_V2`, `MFA_ENABLE_V3`, `MFA_ENABLE_V4`, `MFA_ENABLE_V5` | Per-version kernel toggles for STEEL V2/V3/V4/V5 paths. See CLAUDE.md. |
| `MFA_FORCE_GEN` | Override architecture-gen detection (e.g., `=15` to test M3+ paths on M1). |

## Shipping recommendation

Start with all V6 env vars **unset** to use the auto-tuned defaults. Override only for benchmark sweeps or known-pessimal-default debugging. Future autoresearch sweeps will likely refine the defaults further.

For V34-specific tuning, the per-D defaults committed in v2.32.0 were
validated by Sprint 4 (D=64) and Sprint 5 (D=128) parametric sweeps —
within 1.3% of optimal across the production shape set. Don't override
`MFA_V6_V34_BQ/BK/WM` unless you're specifically reproducing a sweep
result.

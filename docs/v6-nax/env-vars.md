# V6 NAX environment variables

Reference for all V6 NAX env vars exposed by `csrc/mfa_v6_nax_primitive.cpp`.
Behaviour and defaults reflect v2.29.0 (post-Sprint-3.3 + autoresearch).

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

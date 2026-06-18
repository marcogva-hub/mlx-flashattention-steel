# NAMING.md — kernel lineage glossary + V34→V6 rename provenance

This file is the single source of truth for mlx-mfa kernel nomenclature. It is
the **only** living document that intentionally retains the historical `V34`
token (the cartography report below is the second — it is a dated provenance
record). A repo-wide `grep V34` should match only this file and that report.

## Kernel lineage

| Generation | MMA primitive | Notes |
|---|---|---|
| **V1–V5** | `simdgroup_matrix` (STEEL) | `mfa_steel_fwd*.cpp`; threadgroup-loaded MMA. |
| **V6** | `mpp::tensor_ops::matmul2d` cooperative-tensor (**NAX**) | Apple Neural-Accelerator path; `csrc/mfa/v6_nax/`, `csrc/mfa_v6_nax_primitive.cpp`, `csrc/mfa_sparse_attention.cpp`. |

Apple's own SDPA uses a *different* NAX form — raw `metal_simdgroup_matrix`
(`steel_attention_nax.h` / `nax.h`), **not** `matmul2d`. See the cartography
report for why that distinction matters for the dense-vs-sparse perf story.

## The V34→V6 rename (v2.57.0)

`V34` was the internal generator/working name for the V6 NAX kernel during its
development (forward + the 9 backward kernels). It was never a distinct kernel
generation — it is the **same kernel** as V6 NAX. The dual name caused a
documented analysis error (the "port STEEL→NAX" false premise; see the
cartography report). v2.57.0 unifies the nomenclature to **V6**.

**Casing scheme (mechanical, uniform):**

| Old token | New token | Where |
|---|---|---|
| `V34` (any uppercase use: macros, camelCase symbols, prose) | `V6NAX` | MSL `#define`s (`V34_TQ`→`V6NAX_TQ`, `V34BWDF_TK`→`V6NAXBWDF_TK`), C++ symbols (`createV34Source`→`createV6NAXSource`, `MFAV34BwdDV`→`MFAV6NAXBwdDV`, `useV34`→`useV6NAX`) |
| `v34` (lowercase) | `v6nax` | snake_case symbols (`v34_BK`→`v6nax_BK`, `compile_v34_backward_pipeline`→`compile_v6nax_backward_pipeline`, `force_v34`→`force_v6nax`) |

## Environment-variable migration (30 vars)

The **new `MFA_V6*` name is canonical.** The **old `MFA_*V34*` name is a
deprecated alias** — still honored (existing scripts keep working) but it emits
a one-shot `DeprecationWarning` per process. **Aliases are removed in v3.0.0.**
The alias is resolved by `csrc/mfa_env_aliases.hpp` (C++) and
`mlx_mfa/_env_aliases.py` (Python) — those two files deliberately retain the old
names as the deprecation table.

Rename rule: `V34→V6`, EXCEPT where the name already contained `V6` (a
collision), where `V34→NAX` keeps the new name unambiguous.

| Old (deprecated) | New (canonical) |
|---|---|
| `MFA_ENABLE_V34_BACKWARD` | `MFA_ENABLE_V6_BACKWARD` |
| `MFA_DISABLE_V34_BACKWARD` | `MFA_DISABLE_V6_BACKWARD` |
| `MFA_ENABLE_V34_D128` | `MFA_ENABLE_V6_D128` |
| `MFA_V34_BWD_KERNEL` | `MFA_V6_BWD_KERNEL` |
| `MFA_V34_BWD_SPARSE_NATIVE` | `MFA_V6_BWD_SPARSE_NATIVE` |
| `MFA_V34_DUMP_SOURCE` | `MFA_V6_DUMP_SOURCE` |
| `MFA_V34BWD` | `MFA_V6BWD` |
| `MFA_V34BWD_BK` / `_BQ` / `_WM` | `MFA_V6BWD_BK` / `_BQ` / `_WM` |
| `MFA_V34BWD_USE_FUSED` | `MFA_V6BWD_USE_FUSED` |
| `MFA_V34BWD_DUMP_SOURCE` | `MFA_V6BWD_DUMP_SOURCE` |
| `MFA_V34BWDF_BK` / `_BQ` / `_WM` | `MFA_V6BWDF_BK` / `_BQ` / `_WM` |
| `MFA_V34BWDF_DUMP_PATH` / `_DUMP_SOURCE` | `MFA_V6BWDF_DUMP_PATH` / `_DUMP_SOURCE` |
| `MFA_V34BWDK_BK` / `_BQ` / `_WM` | `MFA_V6BWDK_BK` / `_BQ` / `_WM` |
| `MFA_V34BWDV_BK` / `_BQ` / `_WM` | `MFA_V6BWDV_BK` / `_BQ` / `_WM` |
| `MFA_V34BWDKV_BK` / `_BQ` / `_WM` | `MFA_V6BWDKV_BK` / `_BQ` / `_WM` |
| `MFA_V6_USE_V34` *(collision → NAX)* | `MFA_V6_USE_NAX` |
| `MFA_V6_V34_BK` / `_BQ` / `_WM` *(collision → NAX)* | `MFA_V6_NAX_BK` / `_BQ` / `_WM` |

## Provenance

The rename and the analysis that prompted it (the false "port STEEL→NAX"
premise born of the V34/V6 confusion; the sparse forward is already NAX
`matmul2d`) are recorded in
[`.doc-archive/docs/v50/campaign-2026-06/v6-nax/nax-cartography-and-rename-report.md`](.doc-archive/docs/v50/campaign-2026-06/v6-nax/nax-cartography-and-rename-report.md),
which retains the `V34` token to preserve the meaning of that analysis.

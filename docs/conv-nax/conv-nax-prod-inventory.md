# Sprint D — File + Binding Inventory

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-prod-sprint-d` (branched from `feat/conv-nax-prod` ← `experiment/conv-nax-phase1_5` tip 7614ab1)
**Scope:** Sprint C ship-default verdict operationalized — C++ Primitive migration + README/CHANGELOG + `patch_seedvr2_vae` integration wrapper + migration validation.

## Files added

| Path | Lines | Purpose |
|------|------:|---------|
| `csrc/mfa_conv_nax.hpp`                            |  65 | C++ entry-point header (`mlx_mfa::conv3d_nax_forward`) |
| `csrc/mfa_conv_nax.cpp`                            | 446+ | Kernel source builders + chunk layout + dispatch (port of Phase 1.x Python) |
| `mlx_mfa/integrations/seedvr2_vae.py`              | 211 | `patch_seedvr2_vae` + `is_patched` |
| `tests/test_conv_nax_migration.py`                 |  70 | C++ vs Python orchestrator equivalence (6 shapes) |
| `bench/conv_nax_migration_perf.py`                 | 108 | Perf parity sanity bench |
| `bench/conv_nax_patcher_ab.py`                     | 112 | Patcher A/B sanity bench |
| `docs/conv-nax/conv-nax-prod-inventory.md`         | this | This file |
| `docs/conv-nax/conv-nax-prod-decisions.md`         | TBD | D33-D36 |
| `docs/conv-nax/conv-nax-prod-results.md`           | TBD | Per-track validation results |
| `docs/conv-nax/conv-nax-prod-data.json`            | TBD | Aggregate Sprint D data |
| `docs/conv-nax/conv-nax-prod-perf-sanity.json`     | data | Bookend perf parity numbers |
| `docs/conv-nax/conv-nax-prod-patcher-ab.json`      | data | Patcher A/B numbers |

## Files modified

| Path | Δ | Purpose |
|------|--:|---------|
| `mlx_mfa/conv_nax.py`                              | +132/-3 | `conv3d_nax_forward` now C++-routed; legacy Python preserved as `_conv3d_nax_forward_python_legacy` |
| `tests/test_conv_nax.py`                           | +145 | 4 new patcher tests (28 total conv_nax tests) |
| `csrc/bindings.cpp`                                | +35 | `_ext.conv3d_nax_forward` binding |
| `CMakeLists.txt`                                   |  +1 | `csrc/mfa_conv_nax.cpp` added to MFA_SOURCES |
| `README.md`                                        |  +71 | Conv3D NAX support section |
| `CHANGELOG.md`                                     |  +52 | v2.33.0 entry |
| `pyproject.toml`                                   |  ±1 | version 2.32.0 → 2.33.0 |

## Public API additions

```python
# Existing (now C++-routed)
from mlx_mfa.conv_nax import conv3d_nax_forward, get_chunk_plan, estimate_working_set

# NEW Sprint D
from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae, is_patched
```

```cpp
// New C++ entry point (called via nanobind binding _ext.conv3d_nax_forward)
namespace mlx_mfa {
  mlx::core::array conv3d_nax_forward(
      const mlx::core::array& x, const mlx::core::array& w,
      const std::array<int, 3>& stride,
      const ConvPad& padding,           // 6-tuple struct
      const std::array<int, 3>& dilation,
      int chunk_M = 0);
}
```

## Test inventory

**Sprint D adds 4 patcher tests + 6 migration tests = 10 new tests.**

Total `tests/test_conv_nax.py`: 24 tests (was 20 after Phase 1.4).
`tests/test_conv_nax_migration.py`: 6 tests.
**Grand total Conv3D NAX tests: 30.**

Phase 1.1-1.4 (20 unchanged):
- mid_resnet × 4
- up1_resnet × 4
- causal_pad_t × 2
- kt1_routing × 1
- working_set × 3
- multi_chunk_5chunks × 1
- 1×1×1 fast path × 5

Sprint D patcher (4 new):
- test_patcher_correctness
- test_patcher_idempotent
- test_patcher_skips_ineligible
- test_patcher_restore

Sprint D migration (6 new):
- test_cpp_vs_python_equivalence parametrized over 6 production shapes

## Commits on branch (chronological)

1. `8db62ed` — feat(conv-nax): MFAConv3DForward C++ entry point + binding
2. `e8f2755` — refactor(conv-nax): Python orchestrator delegates to C++ binding
3. `c2fc480` — docs(conv-nax): README section + CHANGELOG v2.33.0 + version bump
4. `c282747` — feat+test(conv-nax): patch_seedvr2_vae integration wrapper (4 tests)
5. `33780a0` — test+bench(conv-nax): Sprint D Track D migration validation + Track C patcher fix
6. (next) — docs(conv-nax): Sprint D 5 deliverables + SESSION_LOG

## Validation status

- C++ binding builds + runs: ✓
- 24 conv_nax tests PASS
- 6 migration tests PASS (rel < 1e-5 vs Python orchestrator on all 6 shapes)
- 4 patcher tests PASS (after __class__-swap fix)
- Patcher A/B speedup: 2.29× (mid_resnet-like shape; matches Phase 1.5 2.26×)
- Perf parity: ratio drift -2.04% to +2.61% vs Phase 1.5 (within ±5% bar)
- Full suite: **961 PASS** (was 931 + 30 new), 6 pre-existing failures unchanged, 0 new regressions

# Release-flow validation report — v2.33.0 + v2.33.1

**Date:** 2026-05-12
**Validator:** CC release-prep pass
**Scope:** pre-tag cross-subsystem sanity gate for the v2.33.0 (Sprint D)
and v2.33.1 (sparse fallback) releases.

## Verdict

**Status: BLOCKED on version-string consistency.** Test suite, build,
branch state — all PASS. Three version-string inconsistencies require
Marco's decision before `git tag v2.33.1` can be cut. See §C below.

| Gate | Status | Notes |
|------|:------:|-------|
| A — Full test suite | **PASS** | 964 passed, 6 pre-existing failures unchanged |
| B — Build artifacts | **PASS** | wheel `mlx_mfa-2.33.1-*.whl` + sdist `mlx_mfa-2.33.1.tar.gz` built clean |
| C — Version consistency | **FAIL** | `mlx_mfa/__init__.py:30` still at 2.32.0; README still at 2.32.0; `csrc/bindings.cpp:794` still at 2.22.0 (pre-existing) |
| C — CHANGELOG | **PASS** | v2.33.0 (2026-05-11) + v2.33.1 (2026-05-12) entries present, Keep-a-Changelog format |
| C — README sections | **PASS** | Conv3D NAX section (line 281) + Sparse attention M5+ section (line 358) present |
| D — Branch state | **PASS** | 12 historical branches at expected tips; master unchanged |

## §A — Full test suite

```
.venv/bin/python -m pytest tests/ -q
964 passed, 6 failed, 5 xfailed, 36 xpassed, 2 warnings in 21.41s
```

**964 pass** = 931 pre-existing + 20 conv_nax Phase 1.x + 4 patcher tests
(Sprint D Track C) + 6 migration tests (Sprint D Track D) + 3 sparse-fallback
tests (v2.33.1 TestSparseM5PlusFastFallback) = 964. ✓ Matches expected baseline.

### Six pre-existing failures (unchanged across Sprint C → D → v2.33.1):

| File::Test | Sprint of first appearance |
|------------|----------------------------|
| `test_attention.py::TestTopkAttention::test_topk_ratio_1_matches_dense` | pre-Sprint-C |
| `test_attention.py::TestReturnAttnWeights::test_output_matches_no_return` | pre-Sprint-C |
| `test_attn_bias_native.py::TestBiasMode1::test_d128_causal` | pre-Sprint-C |
| `test_attn_bias_native.py::TestBiasMode2::test_d128_causal` | pre-Sprint-C |
| `test_turboquant.py::TestQRRotation::test_roundtrip` | pre-Sprint-C |
| `test_turboquant.py::TestQRRotation::test_orthogonal` | pre-Sprint-C |

All verified on `master` (pre-Sprint-C) tip; unchanged since.

### Per-subsystem breakdown (new tests added across v2.33.0 + v2.33.1):

| Subsystem | New tests | Source |
|-----------|----------:|--------|
| `tests/test_conv_nax.py` patcher tests | 4 | Sprint D Track C (`patch_seedvr2_vae` correctness/idempotent/skips/restore) |
| `tests/test_conv_nax_migration.py` | 6 | Sprint D Track D (C++ vs Python orchestrator equivalence × 6 production shapes) |
| `tests/test_attention.py::TestSparseM5PlusFastFallback` | 3 | v2.33.1 (correctness equivalence + perf regression guard + M1-M4 unchanged) |
| **Total new** | **13** | |

## §B — Build artifact verification

```bash
rm -rf build/ dist/ *.egg-info/
.venv/bin/python -m build
```

Output:
```
Successfully built mlx_mfa-2.33.1.tar.gz and mlx_mfa-2.33.1-cp311-cp311-macosx_26_0_arm64.whl
```

### Wheel contents inspection

Verified present:
- `mlx_mfa/conv_nax.py` (Sprint D production API surface)
- `mlx_mfa/integrations/seedvr2_vae.py` (Sprint D patcher)
- `mlx_mfa/_ext.cpython-311-darwin.so` (Sprint D C++ binding compiled)
- `mlx_mfa/attention.py` (v2.33.1 patched `_sparse_fallback_sdpa_perhead`)
- `mlx_mfa/precompiled/async_v2.metallib`

Verified absent:
- `.git/`, `build/`, `experiment/*` artifacts
- `dist/`, source-only files outside `mlx_mfa/`

Build artifact: `dist/mlx_mfa-2.33.1-cp311-cp311-macosx_26_0_arm64.whl` (498 KB)
+ `dist/mlx_mfa-2.33.1.tar.gz` (1.4 MB).

### Build version vs runtime version

The wheel filename uses 2.33.1 (from `pyproject.toml`). However, runtime
`mlx_mfa.__version__` returns **`2.32.0`** because `mlx_mfa/__init__.py:30`
was not bumped. **See §C — this is a tag blocker.**

## §C — Version / CHANGELOG / README consistency

### C.1 — pyproject.toml

```
version = "2.33.1"
```

OK ✓.

### C.2 — `mlx_mfa/__init__.py:30` — **STALE 4 versions back**

```python
__version__ = "2.32.0"
```

**Should be `"2.33.1"`.** This is the user-facing version string Python
reports via `import mlx_mfa; mlx_mfa.__version__`. Sprint D bumped
`pyproject.toml` 2.32.0 → 2.33.0 but missed `__init__.py`; v2.33.1 patch
did the same.

**Tag impact:** users who install `mlx_mfa==2.33.1` from PyPI and run
`mlx_mfa.__version__` would see `'2.32.0'`. Confusing; could be reported
as a packaging bug.

**Recommended fix (per Marco's direction):** one-line bump
`__version__ = "2.33.1"`, amend or add a fix-up commit on the v2.33.1
branch BEFORE tagging. Trivial change; minimal risk.

### C.3 — `csrc/bindings.cpp:794` — pre-existing, stale 11 versions

```cpp
m.attr("__version__") = "2.22.0";
```

The `_ext.__version__` attribute (separate from `mlx_mfa.__version__`)
has been at "2.22.0" since v0.2.0-era. **NOT a v2.33.x regression** —
pre-existing inconsistency. Documented here so it doesn't get masked.

Marco's choice: fix in v2.33.1 patch (amends bindings.cpp + rebuilds C++
ext + bumps cache) or defer to a later cleanup pass. Recommendation:
defer — not user-visible enough to block v2.33.1 release.

### C.4 — README header — stale 2 versions

```
Current version: **2.32.0** — SDPA routing for M5+ NAX. …
```

Should be `**2.33.1**` (or `**2.33.0**` if we consider v2.33.1 as a
patch underneath). User-facing but minor; same root cause as C.2 —
release flow incomplete bump.

**Recommended fix:** one-line bump on the v2.33.1 branch BEFORE tagging.

### C.5 — CHANGELOG.md

Top entries (in order):
```
## [Unreleased]
## [2.33.1] — 2026-05-12 — `flash_attention_sparse` M5+ fast-fallback
## [2.33.0] — 2026-05-11 — Conv3D NAX production path
## [2.32.0] — 2026-05-06 — SDPA routing for M5+ NAX
```

- ISO 8601 dates ✓
- Keep-a-Changelog ### sections ✓
- Cross-references to `docs/conv-nax/ship-shelve-decision.md` and
  `docs/lcsa-nax/survey-report.md` ✓

OK ✓.

### C.6 — README sections

| Section | Line | Status |
|---------|-----:|:------:|
| "Conv3D NAX support (M5+ Apple Silicon)" | 281 | ✓ |
| "Sparse attention on M5+ (v2.33.1)" | 358 | ✓ |

Quickstart code examples use public APIs (`mlx_mfa.conv_nax.conv3d_nax_forward`,
`mlx_mfa.integrations.seedvr2_vae.patch_seedvr2_vae`). No internal
`_ext.*` references in user-facing docs.

## §D — Branch state audit

All 12 historical branches at expected tips per sprint summaries:

| Branch | Tip | Description |
|--------|-----|-------------|
| `experiment/v6-nax-backward-phase1_5` | `db5fd8a` | Sprint A close (shelved) |
| `experiment/v6-nax-sprint-a-cleanup` | `91cc26f` | Sprint A cleanup |
| `experiment/conv-nax-phase0-survey` | `f227d2d` | Sprint C Phase 0 |
| `experiment/conv-nax-phase1_0_design` | `401ccd8` | Sprint C Phase 1.0 |
| `experiment/conv-nax-phase1_1` | `ab50e77` | Sprint C Phase 1.1 |
| `experiment/conv-nax-phase1_2` | `0200f59` | Sprint C Phase 1.2 |
| `experiment/conv-nax-phase1_3` | `029510b` | Sprint C Phase 1.3 |
| `experiment/conv-nax-phase1_4` | `8441490` | Sprint C Phase 1.4 |
| `experiment/conv-nax-phase1_5` | `7614ab1` | Sprint C Phase 1.5 |
| `experiment/conv-nax-prod-sprint-d` | `54f22ff` | **Sprint D close (v2.33.0)** |
| `experiment/sparse-attention-m5plus-fallback-fix` | `0b6a3f1` | **v2.33.1 close** |
| `experiment/lcsa-nax-phase0-survey` | `5b328f4` | Sprint B Phase 0 |

Parent branches:

| Branch | Tip | Description |
|--------|-----|-------------|
| `feat/conv-nax` | `479f3a7` | pre-Sprint-C / v2.32.0 release |
| `feat/conv-nax-prod` | `7614ab1` | = Sprint C Phase 1.5 tip; Sprint D base |
| `feat/lcsa-nax` | `39b9ade` | = master tip; Sprint B Phase 0 base |
| `master` | `39b9ade` | v2.28.1 — pre-Sprint-C/D state |

### Lineage verification

`experiment/sparse-attention-m5plus-fallback-fix` merge-base with
`experiment/conv-nax-prod-sprint-d`: `54f22ff` ✓ — v2.33.1 patch
correctly branched from Sprint D close.

### Pending unmerged work

- Sprint D (v2.33.0) → master: **NOT MERGED** (expected — Marco's manual step)
- v2.33.1 patch → master: **NOT MERGED** (expected — Marco's manual step)

## Recommended next steps (for Marco)

1. **Resolve §C blockers**: bump `mlx_mfa/__init__.py:30` to `"2.33.1"`
   and `README.md:7` header version. Optionally bump `csrc/bindings.cpp:794`
   to `"2.33.1"` (would require C++ rebuild). Amend or add fix-up commit
   on `experiment/sparse-attention-m5plus-fallback-fix`.
2. Follow the merge + tag flow in `docs/releases/v2.33.x-release-flow.md`.
3. After tags published, kick off Sprint B Phase 1.0 design prompt
   (takes `docs/lcsa-nax/survey-report.md` as input).

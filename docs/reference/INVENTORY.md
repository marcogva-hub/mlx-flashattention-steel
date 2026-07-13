# Repository inventory

## Published Python surface

`mlx_mfa.__all__` contains 103 names. The executable audit classifies 24 public
computational entry points and verifies all 24. The raw extension exposes 38
computational bindings and the audit inventory accounts for all 38.

The full public grouping is maintained in `API_MANUAL.md`; runtime enumeration
is authoritative when the two differ.

## Python modules

| Area | Main modules |
|---|---|
| public attention and validation | `attention.py`, `dispatch_policy.py`, `lcsa_nax.py` |
| masks and bias construction | `masks.py` |
| transparent hooks | `_auto_hooks.py`, `conv_nax.py` |
| cache/runtime | `inference.py`, `runtime.py`, `kv_cache.py`, `external_cache.py` |
| quantization | `quantize.py`, `turboquant.py`, `svdquant/` |
| configuration | `_knobs.py`, `_env_aliases.py` |

## Native extension

`csrc/bindings.cpp` creates `mlx_mfa._ext`. Dense, sparse, GNA, paged,
quantized, Conv3D and backward primitives are compiled into one arm64 module.
Metal source is mostly generated at runtime and cached by `ShaderCache`.

Build-time probes are excluded unless CMake receives
`-DMFA_BUILD_PROBES=ON`.

## Tests and guards

The suite contains mathematical oracle tests, malformed-input tests,
which-binary locks, route-map locks, cache-coherence tests and publication
guards. The M5/NAX release skip-site count is frozen at 82 so a silently lost
hardware lock fails CI.

The published tree permits root current-state docs and `docs/reference/` only.
Development journals are intentionally excluded from the tracked publication
surface.

## Build and distribution

`pyproject.toml` uses scikit-build-core. CMake builds an arm64 nanobind module
against the installed MLX and links Metal/Foundation. The wheel contains the
Python package; the sdist carries source, tests, scripts, examples, licenses and
the allowlisted current documentation.

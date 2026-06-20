# Contributing to mlx-mfa

## The `_ext` / Python contract (read this before benchmarking)

mlx-mfa's acceleration lives entirely in the compiled C++/Metal extension
`mlx_mfa._ext`. **Every** kernel — STEEL (M1–M4), V6/NAX (M5+), sparse, paged,
conv — is reached through it. If `_ext` does not import, the library is *correct
but unaccelerated*: it silently routes to `mx.fast.scaled_dot_product_attention`
(SDPA). Two hard rules follow.

### 1. `_ext` must be built for the SAME Python as the interpreter you run

`_ext` is an ABI-specific native module: `_ext.cpython-3XY-darwin.so`. A build
for CPython 3.11 **cannot** load under 3.12/3.13/3.14 — `import mlx_mfa._ext`
raises `ModuleNotFoundError`, `has_nax()` returns `False`, and you get SDPA with
no acceleration.

It is also pinned to the **MLX ABI** it was compiled against
(`mlx>=0.31.2` = nanobind 2.12.0 / NB_INTERNALS v19 — see `pyproject.toml` /
`CMakeLists.txt`). Upgrading MLX after building can break the load at runtime;
`mlx_mfa` warns on a major.minor mismatch (`_check_abi`).

### 2. The path-insert-under-mismatched-Python trap (the multi-session footgun)

The exact failure that produced phantom benchmarks across several sessions:

> A bench script ran under a **3.14** venv but did `sys.path.insert(0, repo)`,
> shadowing the 3.14-installed package with the repo checkout whose only built
> `_ext` was for **3.11**. `import mlx_mfa._ext` failed → `has_nax()` False →
> the V6 path never engaged → **both** arms of an A/B ran SDPA → `byteΔ == 0`,
> which read as "V6 looks like SDPA / unverified / parity."

**Symptom to recognize:** an A/B comparison whose two arms are bit-identical
(`byteΔ == 0`), or a "speedup" of exactly 1.0×, or a kernel that "looks like
SDPA." Before debugging the kernel, check whether `_ext` even loaded.

### Always verify NAX is loaded before you bench

```python
import mlx_mfa
assert mlx_mfa.has_nax(), mlx_mfa.has_nax(reason=True)   # raises if the fast path is off
```

`has_nax(reason=True)` returns `(False, code)` with `code` ∈
`{"ext-load-failed", "unsupported-platform", "pre-m5-hardware"}` so you know
*why*. The dev-only bench helper `benchmarks/bench_validity.measured_speedup`
takes `require=lambda: mlx_mfa.has_nax()` and raises `FeatureUnavailable`
(not a misleading "vacuous") when the feature can't engage — use it.

### Building `_ext`

```bash
# canonical dev env — .venv is the single source of truth (CLAUDE.md)
CMAKE_ARGS="-DPython_EXECUTABLE=.venv/bin/python" \
  .venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -c "import mlx_mfa._ext; print('ext OK')"
.venv/bin/python -c "import mlx_mfa; print('has_nax:', mlx_mfa.has_nax(reason=True))"
```

To bench under a *different* Python (e.g. a 3.14 venv), build an `_ext` for
**that** interpreter first — do not point a mismatched Python at a prebuilt
`_ext`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The contract above is locked by `tests/test_nax_availability.py` (warning on
unexpected fallback, silence on expected, strict mode) and
`tests/test_bench_validity_v6_regression.py` (the bench helper's
`FeatureUnavailable` / engagement checks).

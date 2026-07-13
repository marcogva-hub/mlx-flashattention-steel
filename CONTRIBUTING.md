# Contributing

Contributions must preserve correctness, fallback honesty and reproducible
dispatch evidence. A faster result from an unintended binary is not a valid
optimization.

## Local setup

Use an arm64 macOS environment with Python 3.10+, CMake 3.24+ and MLX 0.31.2+.

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e '.[dev]'
CMAKE_ARGS="-DPython_EXECUTABLE=$PWD/.venv/bin/python" \
  .venv/bin/python -m pip install --no-build-isolation -e .
```

The extension links against the installed MLX. Rebuild after changing C++,
Objective-C++ or generated Metal source.

## Validation ladder

Run the narrowest relevant tests while developing, then the complete suite:

```bash
.venv/bin/python -m pytest tests/test_attention.py -q
.venv/bin/python -m pytest tests/ -q
```

Changes to routing or performance documentation also require:

```bash
.venv/bin/python -m pytest \
  tests/test_dispatch_map_lock.py \
  tests/test_doc_accuracy_guards.py \
  tests/test_publish_surface_guard.py \
  tests/test_release_gate_enforcement.py -q
```

## Kernel and dispatch changes

Before editing a kernel:

1. Identify the public caller and every fallback.
2. Establish an independent fp32 oracle.
3. Add a terminal fingerprint that fails when another path runs.
4. Lock malformed inputs and unsupported shapes.
5. Preserve pre-M5 behavior unless the change explicitly targets it.

Routing changes must update `docs/reference/dispatch-map.md` and its executable
lock in the same commit. Expert-only symbols do not prove public engagement.

## Performance evidence

A comparative ratio is accepted only after correctness and engagement. Use
same-dtype arms, absolute milliseconds, five process-isolated sessions in both
orders, and 20 dispatches per sample for sub-millisecond kernels. Record the
runtime versions and hardware. Fine gains require an A-vs-A noise floor.

Do not add a user-facing speed claim unless its public path and expected
terminal have an executable lock.

## Environment variables

Register every new `MFA_*` or `MLX_MFA_*` name in `mlx_mfa/_knobs.py`. Boolean
controls use the shared strict `0`/`1` parser. Document the default, caching
behavior and scope in `ENV_VARS.md`.

## Documentation

Current-state user documentation lives at the repository root and under
`docs/reference/`. Investigation logs and design journals belong in ignored
`devnotes/` or `.doc-archive/`; the publication guard rejects them from the
tracked/published surface.

Claims should cite code or executable tests. Historical changelog entries are
immutable. New release text belongs under `Unreleased` until the maintainer
performs the versioned release.

## Scope and review

Keep commits focused. Do not combine an optimization with unrelated cleanup.
Call out unsupported hardware, beta-OS measurements and any test gap. The
maintainer controls version bumps, tags and publication.

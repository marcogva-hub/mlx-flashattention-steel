# Release Publication Status — v2.9.2 (2026-03-13)

## Release-prep status

- Post-merge verification passed.
- Release artifacts built successfully in `dist/v2.9.2/`.
- `twine check dist/v2.9.2/*` passed.
- Fresh Python 3.11 venv install/import verification passed for:
  - `mlx_mfa.__version__ == "2.9.2"`
  - `flash_attention`
  - `create_inference_context`
  - `create_decode_runtime`
  - `DecodeRuntime`

## Publish-path detection

- Preferred path prepared: Trusted Publishing workflow added at `.github/workflows/publish.yml`.
- Local GitHub CLI auth is unavailable, so the workflow could not be dispatched from this environment.
- Twine fallback is incomplete for the required TestPyPI-first flow:
  - `~/.pypirc` contains a `pypi` section
  - no `testpypi` section was found
  - no publish-related environment credentials were present

## External publication state

- PyPI public state: latest `mlx-mfa` release is `2.6.1`; `2.9.2` is not published.
- TestPyPI public state: `mlx-mfa` project lookup returned `404` (no project/release visible).
- Remote git tag state: `v2.9.2` tag is not present on `origin`.

## Blocking prerequisites

At least one of the following must be completed before a safe publish:

1. Configure Trusted Publishing for `mlx-mfa` on TestPyPI and PyPI to trust this repository/workflow, then trigger `.github/workflows/publish.yml` with `repository=testpypi` first and `repository=pypi` second.
2. Provide secure TestPyPI Twine credentials in addition to the existing PyPI credentials so the required TestPyPI-first fallback flow can be executed safely.

## Decision in this pass

- No publish attempted.
- No `v2.9.2` tag created.
- Reason: avoid creating a release/tag state that implies publication completed when the safe publish prerequisites are still missing.

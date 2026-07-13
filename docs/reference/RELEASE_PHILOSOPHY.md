# Release Philosophy

mlx-mfa publishes useful behavior, not merely compiled kernels.

## Three exposure levels

1. **Default**: the public API selects the path automatically inside a measured and locked envelope.
2. **Opt-in**: a public path exists behind a documented environment variable while platform validation is incomplete.
3. **Expert**: a direct symbol or patcher exposes a correct implementation without promising automatic selection.

Every new optimization enters at the least permissive level justified by correctness, engagement, and measurement.

## Public-path requirement

A release claim must be reproducible through a documented public call. A direct `_ext` benchmark can establish kernel capability but cannot support a public speed claim unless a routing lock proves that users reach the same binary.

## Correctness and fallback

Unsupported inputs either raise with context or use a named fallback. Silent wrong results, silent dtype changes, and tests that claim one terminal while exercising another are release blockers.

Fallbacks remain part of the product: SDPA protects broad correctness, STEEL protects older hardware, and scalar implementations retain coverage for sparse cases outside NAX envelopes.

## Evidence lifetime

Performance is runtime-specific. Every current number includes date, MLX version, hardware, operating system, absolute timings, arm direction, and fingerprints. β3 measurements guide routing conservatively but require stable-OS revalidation.

## Published surface

The source distribution contains current-state documentation only. Campaign journals and development notes are excluded by the publication guard. Historical release records remain in `CHANGELOG.md`; current contracts live under `docs/reference/`.

## Release boundary

Code integration, versioning, tagging, uploading, and announcing are separate actions. A release-prepared commit does not imply publication. The release operator must rerun environment, test, manifest, and artifact checks immediately before tagging.

# Migration: 2.39.1 to 2.50.0

This migration crosses a dispatch-hardening boundary. Applications should retest terminal engagement rather than assuming that a former native route remains selected.

## Required checks

- Run with the canonical `float16` or `bfloat16` dtype expected by native attention.
- Verify GQA divisibility and four-dimensional BHND layout.
- Replace removed environment names with entries listed in [ENV_VARS.md](../../ENV_VARS.md).
- Treat unknown or removed knobs as configuration errors when strict validation is enabled.
- Confirm backward behavior through `mx.grad` or `mx.vjp`; do not call internal primitives as a substitute for a public-path test.

## Behavioral changes

The dispatcher became conservative around unmeasured shapes and began recording terminal paths. Dense M5 routing, backward routing, and sparse routing are separate policies. A fallback is therefore expected on cells outside a measured envelope.

Native GNA is forward-only. Training callers must disable native GNA or use the differentiable sparse fallback.

## Validation recipe

1. capture the pre-upgrade public output;
2. upgrade and rebuild the extension against the installed MLX;
3. enable dispatch tracing;
4. compare outputs to an independent fp32 oracle;
5. compare gradients when the workload trains;
6. record the selected terminal for each production shape.

Historical ratios from the earlier release series are not current performance guarantees.

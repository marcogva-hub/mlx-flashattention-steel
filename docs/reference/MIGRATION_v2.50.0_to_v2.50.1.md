# Migration: 2.50.0 to 2.50.1

This patch transition tightened correctness and dispatch contracts without requiring an API rename.

## Application checklist

- Rebuild the extension; it is compiled against the local MLX installation.
- Run environment validation before import when deploying configurable knobs.
- Verify public terminal engagement for any performance-sensitive path.
- Recheck custom scales, causal asymmetry, and GQA shapes against the application oracle.
- Keep SDPA fallback available for unsupported shapes and dtypes.

## Configuration

Boolean knobs accept only `0` or `1`. An explicitly empty value or any other token is invalid. Removed knobs have no replacement unless [ENV_VARS.md](../../ENV_VARS.md) names one.

## Expected compatibility

Public function signatures remain the compatibility boundary. Direct `_ext` calls are expert interfaces and may reject a shape that the public function serves through delegation.

Use [dispatch-map.md](dispatch-map.md) for the current route table; this migration note does not freeze historical dispatch behavior.

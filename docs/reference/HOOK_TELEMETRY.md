# Hook Telemetry

mlx-mfa can install transparent hooks around selected MLX operations. Telemetry exposes whether those hooks engaged, fell back, or failed.

## Modes

Set `MLX_MFA_HOOK_TELEMETRY` before importing `mlx_mfa`:

| Value | Behavior |
|---|---|
| `off` | no hook summary output |
| `summary` | aggregate counters; default |
| `verbose` | aggregate counters plus detailed diagnostics |

Any other value raises during import. `MFA_HOOK_VERBOSE=1` adds traceback detail for unexpected hook failures.

## Programmatic counters

```python
import mlx_mfa

stats = mlx_mfa.get_hook_stats()
print(stats)
```

Counters are evidence of hook engagement, not a numerical oracle. A correctness test must still compare outputs independently.

## Installation control

Importing the package installs supported hooks unless `MFA_DISABLE_AUTO_HOOKS=1` is set before import. `install_hooks()` is idempotent and can be called explicitly.

## Dispatch traces

Attention routes use a separate dispatch trace. Set `MLX_MFA_VERBOSE_DISPATCH=1` and inspect the emitted terminal name. Examples include `nax_dense`, `v6nax_sparse`, `gna_v6nax`, `varlen_v6nax`, `mfa_primitive`, `v6_split_backward`, `sdpa`, and `varlen_split_concat`.

A benchmark must fingerprint both candidate and baseline. Source inspection alone is insufficient because public gates can replace a lower-level candidate with a fallback.

## Failure policy

Expected ineligibility falls back through the documented route. Unexpected native-hook failures produce a warning once per process and remain visible in counters. Set verbose telemetry when diagnosing a missing route.

# Hook telemetry (Pattern #8 prevention)

mlx-mfa ships with auto-installed hooks that route eligible operations
through accelerated kernels (currently: `mx.conv_general` and
`mx.conv3d` → NAX Conv3D on M5+).  Hook telemetry surfaces whether those optimization paths
actually engage for your workload.

This addresses **Pattern #8** — silent hook fallback masking unused
optimization — documented in
`.doc-archive/docs/v50/audit-framing-inversions.md`.

> **Scope (2.61.0):** `get_hook_stats()` counts **conv hooks only**
> (`conv3d_nax_forward`, via the `mx.conv_general` / `mx.conv3d` patch).
> It does **not** observe `flash_attention*` dispatch. Attention dispatch
> is observed by a separate **test-only** recorder, `mlx_mfa/_dispatch_trace.py`
> (zero-cost no-op unless a `capture()` context is open; not in the public
> `__all__`), used by the routing-equivalence snapshot tests — not by
> `get_hook_stats()`. As of 2.61.0 the conv hook also accepts the causal
> per-axis pad `(0,1,1)` 3×3×3 (not just symmetric `(1,1,1)`).

## When to use

- **Diagnosing perf**: your model runs but seems slower than expected.
  Telemetry confirms whether the NAX path is engaged or your inputs
  are hitting the fallback path.
- **Validating env changes**: after modifying dtypes, shapes, or
  configuration, telemetry confirms the optimization still engages.
- **Pre-release audits**: smoke tests can assert
  `executed[hook_name] > 0` before declaring an optimization
  production-ready.

## Quick start

```python
import mlx_mfa
import mlx.core as mx

# Run your model / inference here
my_model(my_input)
mx.synchronize()

# Inspect what engaged
stats = mlx_mfa.get_hook_stats()
print(stats)
# Example output:
# {
#   "executed": {"conv3d_nax_forward": 128},
#   "fallback": {},
#   "fallback_reasons": {},
#   "mode": "summary"
# }
```

If `fallback[hook_name] > 0`, the NAX path is not being used for
some calls.  `fallback_reasons[hook_name]` lists up to 10 distinct
reasons captured for debugging.

## API

### `mlx_mfa.get_hook_stats() -> dict`

Returns a snapshot dict with keys:
- `executed: dict[str, int]` — calls successfully routed through NAX
- `fallback: dict[str, int]` — calls that fell back to MLX baseline
- `fallback_reasons: dict[str, list[str]]` — up to 10 distinct reason
  strings per hook
- `mode: str` — current telemetry mode ("off" / "summary" / "verbose")

The returned dict is a copy; mutating it does not affect internal
state.

### `mlx_mfa.reset_hook_stats() -> None`

Clears all counters and reasons.  Useful for scoping a measurement to
a specific code block.

## Modes (controlled via `MLX_MFA_HOOK_TELEMETRY` env var)

| Mode | Behavior | Overhead | When to use |
|---|---|---|---|
| `off` | No counters maintained | 0% (early return) | Max-perf production |
| `summary` (default) | Per-hook dict-increment counters | ~1% at microbench, <0.1% at production scale | Default — recommended |
| `verbose` | Summary counters + `UserWarning` on every fallback | Same as summary + warning overhead | Active debugging |

## Example: validating NAX engagement for SeedVR2 VAE

```python
import mlx_mfa, mlx.core as mx
from seedvr2 import load_vae

vae = load_vae(...)
mlx_mfa.reset_hook_stats()

# Run VAE encode on a few frames
frames = mx.random.uniform(-1, 1, (1, 16, 384, 384, 3), dtype=mx.float32)
latent = vae.encode(frames)
mx.eval(latent); mx.synchronize()

stats = mlx_mfa.get_hook_stats()
print(f"NAX Conv3D engaged {stats['executed'].get('conv3d_nax_forward', 0)} times")
print(f"Fallback events: {stats['fallback'].get('conv3d_nax_forward', 0)}")
if stats['fallback']:
    print(f"Fallback reasons: {stats['fallback_reasons']}")
```

Expected output (v2.50.1 onward, current in 2.61.0):
```
NAX Conv3D engaged 32 times
Fallback events: 0
```

If you see `Fallback events > 0`, the eligibility check is rejecting
some calls.  Common reasons:
- `weight dtype {dtype} not fp16/bf16` — unsupported weight dtype
- `bf16 outside MPP gate (KD-7: legacy im2col bf16 broken upstream)` —
  bf16 weight on a shape outside the MPP gate (III-1)
- `not M5+ hardware` — running on M4 or earlier
- `weight not 5-D (not Conv3D)` — input is 2D/4D conv

## See also

- `.doc-archive/docs/v50/audit-framing-inversions.md` — Pattern #8 codification
- `.doc-archive/docs/v50/known-debt-v2.50.md` — KD-6 (resolved) and KD-7 (lifted in v2.51.0 — bf16 routes via MPP)
- `mlx_mfa/_auto_hooks.py` — implementation
- `ENV_VARS.md` — full env var reference

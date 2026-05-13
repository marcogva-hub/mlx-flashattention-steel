# Training with mlx-mfa on M5+ — Quickstart

mlx-mfa v2.37.0 introduces **V34 backward NAX-direct kernels** for
training workloads on M5+ Apple Silicon.  Apple's NAX backward is NYI
in the MLX framework — mlx-mfa V34 backward is the only path for
NAX-accelerated backward attention on M5+.

## Status: SHIP_OPT_IN

V34 backward kernels are correctness-validated (RMSE matches MLX's
autograd of `mx.fast.scaled_dot_product_attention` within FP16/FP32
noise floor) but currently 2.2-2.4× slower than SDPA-vjp on M5 Max at
qL=8192.  Default behavior is unchanged: backward routes through the
existing STEEL backward or SDPA-vjp fallback.

To opt in, set `MFA_ENABLE_V34_BACKWARD=1`.

## Quick example

```python
import os
os.environ["MFA_ENABLE_V34_BACKWARD"] = "1"

import mlx.core as mx
import mlx_mfa

# Eligible shapes: D in {64, 128}, FP16/BF16, no causal/window/softcap.
B, Hq, qL, D = 1, 4, 4096, 128
q = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)
k = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)
v = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)

def loss_fn(q, k, v):
    O = mlx_mfa.flash_attention(q, k, v, backend="mfa")
    return O.sum()

# Backward routes through V34 NAX kernels on M5+ eligible shapes.
dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
# Force materialisation: mx synchronisation primitive
mx.synchronize()
```

## Eligibility (when V34 backward engages)

V34 backward engages when ALL conditions are met:

- `MFA_ENABLE_V34_BACKWARD=1` set
- M5+ hardware (cached check via `_get_has_nax_cached()`)
- `head_dim` in {64, 128}
- dtype is FP16 or BF16
- No causal, window_size, softcap, alibi_slopes
- Forward routes through V34 (D=128 always; D=64 with force_v34=True
  v2.37.0+ post-release patch enables this for D=64 small-Nk too)

Otherwise, backward falls back to STEEL backward / SDPA-vjp
(v2.36.1-exact behavior).

## What V34 backward does (architectural)

Three kernels run during backward:

1. **dQ kernel** (per-Q-tile dispatch, WM=4 Q-row partition):
   recomputes P from forward's lse, computes `dP = dO @ V^T`,
   `dS = P*(dP - D)`, then `dQ += dS @ K`.
2. **dV kernel** (per-K-tile dispatch, WM=4 Q-row partition):
   recomputes P, then `dV += P^T @ dO`.  Writes per-SG partial to
   intermediate buffer.
3. **dK kernel** (per-K-tile dispatch, WM=4 Q-row partition):
   recomputes P + D + dP + dS, then `dK += dS^T @ Q`.  Writes per-SG
   partial.

dV and dK partials are reduced via `mx.sum(axis=2).astype(input_dtype)`
in Python — a 2-3 kernel chain that takes <1ms even at qL=8192.

The V34 forward kernel also writes natural-log lse to device memory
(BLK1 patch), enabling backward to recompute softmax P without
re-running forward.

## Performance characterization (M5 Max FP16 D=128)

| qL | V34 backward | SDPA-vjp | V34 / SDPA |
|---|---:|---:|---:|
| 1024 | 1.07ms | 0.50ms | 2.13× |
| 2048 | 3.22ms | 1.54ms | 2.09× |
| 4096 | 12.77ms | 5.31ms | 2.40× |
| 8192 | 48.93ms | 20.37ms | 2.40× |

V34 backward is currently 2.2-2.4× slower than SDPA-vjp.  This is the
**architectural floor** for this algorithm — dK kernel inherently does
~2× the work of dV (extra `dO @ V^T` matmul required by FA-2 dK
formula).  Apple's SDPA-vjp uses a different algorithm (likely fused
single-kernel dK+dV with TGP cross-SG reduction) that V34 backward
would require major restructure to match.

For now, V34 backward is research infrastructure: useful for studying
NAX backward attention but not a perf win over SDPA-vjp on M5 Max.

## Environment variables (advanced users)

See `ENV_VARS.md` for the full list.  V34 backward-related:

| Variable | Purpose |
|---|---|
| `MFA_ENABLE_V34_BACKWARD=1` | Opt into V34 backward (default off). |
| `MFA_V34BWD_USE_FUSED=1` | Fall back to WM=1 fused dK/dV kernel (vs multi-SG split). |
| `MFA_V34BWD_WM` | WM for multi-SG split (default 4). |
| `MFA_V34BWDV_BQ`, `MFA_V34BWDV_BK`, `MFA_V34BWDV_WM` | dV tile overrides (researchers). |
| `MFA_V34BWDK_BQ`, `MFA_V34BWDK_BK`, `MFA_V34BWDK_WM` | dK tile overrides. |

## What V34 backward does NOT support yet

Deferred to follow-up sprints:

- Block-sparse backward (set `MFA_ENABLE_V34_BACKWARD=1` has no effect
  when `flash_attention_sparse` is used; falls back to STEEL sparse
  backward).
- Causal backward (causal=True → falls back to STEEL causal backward).
- D not in {64, 128} backward (falls back to STEEL).
- Softcap / ALiBi / TurboQuant backward (kept on STEEL).
- Multi-batch GQA where Hq > Hk (dK/dV output is per-Q-head; caller
  must reduce across query-heads sharing each KV-head for proper GQA
  gradient shape — current implementation matches MLX SDPA-vjp layout
  for non-GQA cases).

## References

- `CHANGELOG.md` [2.37.0] entry — full v2.37.0 changes
- `docs/v6-nax/v34-backward-status.md` — full sprint timeline + design
  decisions DC0-DC13
- `docs/v6-nax/v34-backward-option-gamma-design.md` — next-sprint design
  for fused dK+dV (Option γ)
- `docs/v6-nax/v34-backward-decisions.md` — design rationale
- `ENV_VARS.md` — full env var reference
- `docs/API_MANUAL.md` — `flash_attention()` API

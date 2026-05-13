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

## Performance characterization (M5 Max FP16)

> **v2.37.3 audit note (2026-05-13):** the per-shape tables below
> were originally measured via `backend="mfa"` forced path
> (kernel-isolation regime), which is NOT the path users hit by
> default.  Post-v2.37.2 carve-out, the public `backend="auto"` API
> engages V34 backward ONLY for **D=64, qL ≥ 4096, non-causal,
> fp16/bf16, NAX**.  Outside that envelope, the AUTO path falls back
> to SDPA-vjp at parity.  The tables in the D=128 and D=64-small
> subsections below describe **kernel-isolation** behavior accessible
> only via `backend="mfa"`; they are research characterization, not
> user-facing perf.  See `docs/v6-nax/v2.37.x-perf-claim-audit.md`
> for the per-claim reachability audit.

### D=64 — V34 backward wins (user-facing, reachable via public AUTO API)

For D=64, qL ≥ 4096, non-causal training (e.g., FlashVSR class,
LTX2-style cross-attention), V34 backward is genuinely faster than
SDPA-vjp **through the documented public API**.  No `backend` override
needed — just set `MFA_ENABLE_V34_BACKWARD=1` and call
`flash_attention(...)` normally:

| qL=kL | V34 backward (AUTO) | SDPA-vjp | Speedup |
|---|---:|---:|---:|
| **4096** | **2.65 ms** | **4.83 ms** | **1.82× faster** |
| **8192** | **9.78 ms** | **17.67 ms** | **1.81× faster** |

These numbers are reproducible with the snippet in the Usage section
above (M5 Max, B=1, H=4, fp16, canonical methodology §4.2).

### D=64 — small qL (research characterization, NOT user-facing)

For D=64 qL ≤ 2048, V34 backward is at parity or slower than
SDPA-vjp end-to-end.  The v2.37.2 carve-out does NOT engage V34 here,
so the public AUTO API correctly defaults to SDPA-vjp.  Kernel-isolation
numbers (via `backend="mfa"`) for reference:

| qL | V34 (mfa-forced) | SDPA-vjp | Ratio | End-to-end win? |
|---|---:|---:|---:|---:|
| 256 | 0.46 ms | 0.34 ms | 1.35× | No (V34 slower) |
| 512 | 0.51 ms | 0.37 ms | 1.39× | No |
| 1024 | 0.73 ms | 0.64 ms | 1.13× | No |
| 2048 | 1.23 ms | 1.42 ms | 0.87× (1.15× win at kernel level) | ~1.06× — within noise |

The original v2.37.1 release notes claimed "qL=2048: V34 wins 1.44×".
The current canonical-methodology bench shows 1.15× kernel-level win,
≈1.06× end-to-end — within measurement noise.  This row was **retracted
in v2.37.3** (see `docs/v6-nax/v2.37.x-perf-claim-audit.md`).

### D=128 — research-only kernel characterization

V34 backward is 2.0-2.4× SLOWER than SDPA-vjp at D=128 due to an
architectural floor (extra dO@V^T matmul scales with D²; at D=128 the
dK accumulator approaches register-spill threshold on M5 Max).

**The v2.37.2 carve-out does NOT engage V34 for D=128**, so the public
`backend="auto"` API correctly falls back to SDPA-vjp at parity.  These
numbers are reachable only via `backend="mfa"` explicit override and
are kernel-characterization, not user-facing perf:

| qL | V34 (mfa-forced) | SDPA-vjp | Ratio |
|---|---:|---:|---:|
| 1024 | 1.12 ms | 0.50 ms | 2.22× slower |
| 2048 | 3.22 ms | 1.42 ms | 2.27× slower |
| 4096 | 12.33 ms | 5.49 ms | 2.25× slower |
| 8192 | 48.46 ms | 20.18 ms | 2.40× slower |

The previously documented D=128 V34 backward path is **research
infrastructure only**.  Users training at D=128 should not set
`MFA_ENABLE_V34_BACKWARD=1` — the carve-out won't engage and the
default SDPA-vjp path is correct.

### Recommendation

- **D=64 training with qL ≥ 4096**: set `MFA_ENABLE_V34_BACKWARD=1`
  and call `flash_attention(...)` normally — V34 backward engages
  automatically via the v2.37.2 carve-out and delivers **1.81-1.82×
  speedup** over SDPA-vjp.
- **D=64 training with qL < 4096**: don't bother setting the env —
  the carve-out's shape gate keeps you on SDPA-vjp anyway.
- **D=128 training**: leave the env unset; AUTO path defaults to
  SDPA-vjp which is the correct choice.  Setting `MFA_ENABLE_V34_BACKWARD=1`
  has no effect at D=128 (carve-out doesn't engage).

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

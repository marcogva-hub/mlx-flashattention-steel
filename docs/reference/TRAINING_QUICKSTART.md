# Training with mlx-mfa on M5+ — Quickstart

mlx-mfa v2.37.0 introduces **V6NAX backward NAX-direct kernels** for
training workloads on M5+ Apple Silicon.  Apple's NAX backward is NYI
in the MLX framework — mlx-mfa V6NAX backward is the only path for
NAX-accelerated backward attention on M5+.

## Status: DEFAULT-ON for D=64 (since v2.51.0)

V6NAX backward for **D=64 (causal + non-causal)** is **DEFAULT-ON since
v2.51.0**: on M5-class hardware, fp16/bf16, qL ≥ 2048, it engages
automatically — **2.16–3.05× faster than SDPA-vjp at qL≥4096** (~1.5–1.7×
@qL2048; M5 Max / macOS 26.6 / MLX 0.31.2) — no env var needed.  Opt out
with `MFA_DISABLE_V6_BACKWARD=1`.

**D=128 remains opt-in** via `MFA_ENABLE_V6_BACKWARD=1` (parity with
SDPA-vjp, not a speedup).  Kernels are correctness-validated (RMSE
matches MLX's autograd of `mx.fast.scaled_dot_product_attention`
within FP16/FP32 noise floor).

## Quick example

```python
import os
# Only needed for D=128 — D=64 is default-on since v2.51.0.
os.environ["MFA_ENABLE_V6_BACKWARD"] = "1"

import mlx.core as mx
import mlx_mfa

# Eligible shapes: D in {64, 128}, FP16/BF16, no window/softcap.
B, Hq, qL, D = 1, 4, 4096, 128
q = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)
k = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)
v = mx.random.normal((B, Hq, qL, D), dtype=mx.float16)

def loss_fn(q, k, v):
    O = mlx_mfa.flash_attention(q, k, v, backend="mfa")
    return O.sum()

# Backward routes through V6NAX NAX kernels on M5+ eligible shapes.
dq, dk, dv = mx.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
# Force materialisation: mx synchronisation primitive
mx.synchronize()
```

## Eligibility (when V6NAX backward engages)

V6NAX backward engages when ALL conditions are met:

- D=64: on by default since v2.51.0 (opt-out
  `MFA_DISABLE_V6_BACKWARD=1`); D=128: `MFA_ENABLE_V6_BACKWARD=1` set
- M5+ hardware (cached check via `_get_has_nax_cached()`)
- `head_dim` in {64, 128}
- dtype is FP16 or BF16
- qL ≥ 2048
- No window_size, softcap, alibi_slopes (causal IS supported —
  causal + non-causal both default-on at D=64)
- Forward routes through V6NAX (D=128 always; D=64 with force_v6nax=True
  v2.37.0+ post-release patch enables this for D=64 small-Nk too)

Otherwise, backward falls back to STEEL backward / SDPA-vjp
(v2.36.1-exact behavior).

## What V6NAX backward does (architectural)

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

The V6NAX forward kernel also writes natural-log lse to device memory
(BLK1 patch), enabling backward to recompute softmax P without
re-running forward.

## Performance characterization (M5 Max FP16)

> **v2.37.3 audit note (2026-05-13):** the per-shape tables below
> were originally measured via `backend="mfa"` forced path
> (kernel-isolation regime), which is NOT the path users hit by
> default.  Post-v2.37.2 carve-out, the public `backend="auto"` API
> engages V6NAX backward ONLY for **D=64, qL ≥ 4096, non-causal,
> fp16/bf16, NAX**.  Outside that envelope, the AUTO path falls back
> to SDPA-vjp at parity.  The tables in the D=128 and D=64-small
> subsections below describe **kernel-isolation** behavior accessible
> only via `backend="mfa"`; they are research characterization, not
> user-facing perf.  See `.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md`
> for the per-claim reachability audit.  (Superseded by v2.51.0:
> D=64 causal + non-causal is now default-on — see Status above.)

### D=64 — V6NAX backward wins (user-facing, reachable via public AUTO API)

For D=64, qL ≥ 2048, causal or non-causal training (e.g., FlashVSR
class, LTX2-style cross-attention), V6NAX backward is genuinely faster
than SDPA-vjp **through the documented public API** at **1.7-2.7×**.
Since v2.51.0 no env var or `backend` override is needed — just call
`flash_attention(...)` normally.  Representative v2.37.x measurements:

| qL=kL | V6NAX backward (AUTO) | SDPA-vjp | Speedup |
|---|---:|---:|---:|
| **4096** | **2.65 ms** | **4.83 ms** | **1.82× faster** |
| **8192** | **9.78 ms** | **17.67 ms** | **1.81× faster** |

These numbers are reproducible with the snippet in the Usage section
above (M5 Max, B=1, H=4, fp16, canonical methodology §4.2)
within a ~5% measurement-noise band — re-running the same bench
yields values like 2.71 ms / 4.94 ms / 9.91 ms / 18.10 ms (same
1.81-1.82× speedup ratio).  See
`.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md` for the raw audit table.

### D=64 — small qL (research characterization, NOT user-facing)

For D=64 qL ≤ 2048, V6NAX backward is at parity or slower than
SDPA-vjp end-to-end.  The v2.37.2 carve-out does NOT engage V6NAX here,
so the public AUTO API correctly defaults to SDPA-vjp.  Kernel-isolation
numbers (via `backend="mfa"`) for reference:

| qL | V6NAX (mfa-forced) | SDPA-vjp | Ratio | End-to-end win? |
|---|---:|---:|---:|---:|
| 256 | 0.46 ms | 0.34 ms | 1.35× | No (V6NAX slower) |
| 512 | 0.51 ms | 0.37 ms | 1.39× | No |
| 1024 | 0.73 ms | 0.64 ms | 1.13× | No |
| 2048 | 1.23 ms | 1.42 ms | 0.87× (1.15× win at kernel level) | ~1.06× — within noise |

The original v2.37.1 release notes claimed "qL=2048: V6NAX wins 1.44×".
The current canonical-methodology bench shows 1.15× kernel-level win,
≈1.06× end-to-end — within measurement noise.  This row was **retracted
in v2.37.3** (see `.doc-archive/docs/v6-nax/v2.37.x-perf-claim-audit.md`).

### D=128 — research-only kernel characterization

> **v2.50+ update:** the split kernels now engage at **parity** with
> SDPA-vjp for D=128 + qL ≥ 2048 when opted in via
> `MFA_ENABLE_V6_BACKWARD=1`.  The table below is the pre-v2.50
> characterization of the original kernels.

The original V6NAX backward was 2.0-2.4× SLOWER than SDPA-vjp at D=128 due to an
architectural floor (extra dO@V^T matmul scales with D²; at D=128 the
dK accumulator approaches register-spill threshold on M5 Max).

**The v2.37.2 carve-out does NOT engage V6NAX for D=128**, so the public
`backend="auto"` API correctly falls back to SDPA-vjp at parity.  These
numbers are reachable only via `backend="mfa"` explicit override and
are kernel-characterization, not user-facing perf:

| qL | V6NAX (mfa-forced) | SDPA-vjp | Ratio |
|---|---:|---:|---:|
| 1024 | 1.12 ms | 0.50 ms | 2.22× slower |
| 2048 | 3.22 ms | 1.42 ms | 2.27× slower |
| 4096 | 12.33 ms | 5.49 ms | 2.25× slower |
| 8192 | 48.46 ms | 20.18 ms | 2.40× slower |

For D=128, `MFA_ENABLE_V6_BACKWARD=1` engages the split kernels at
parity with SDPA-vjp — correct but not faster.  Leaving the env unset
keeps the default SDPA-vjp path, which is equally correct.

### Recommendation

- **D=64 training (qL ≥ 2048, causal or non-causal)**: no env var
  needed since v2.51.0 — V6NAX backward engages by default and delivers
  **2.16–3.05× speedup over SDPA-vjp at qL≥4096** (~1.5–1.7× @qL2048;
  M5 Max / macOS 26.6 / MLX 0.31.2).  Opt out with
  `MFA_DISABLE_V6_BACKWARD=1` if needed.
- **D=64 training with qL < 2048**: the shape gate keeps you on
  SDPA-vjp; nothing to configure.
- **D=128 training**: optional opt-in via `MFA_ENABLE_V6_BACKWARD=1`
  engages the split kernels at parity (not a speedup); leaving it
  unset keeps SDPA-vjp.  Either way is correct.

## Environment variables (advanced users)

See `ENV_VARS.md` for the full list.  V6NAX backward-related:

| Variable | Purpose |
|---|---|
| `MFA_ENABLE_V6_BACKWARD=1` | Opt-in for D=128 only; D=64 is default-on since v2.51.0. |
| `MFA_DISABLE_V6_BACKWARD=1` | Opt out of the default-on D=64 backward (causal + non-causal). |
| `MFA_V6BWD_USE_FUSED=1` | Fall back to WM=1 fused dK/dV kernel (vs multi-SG split). |
| `MFA_V6BWD_WM` | WM for multi-SG split (default 4). |
| `MFA_V6BWDV_BQ`, `MFA_V6BWDV_BK`, `MFA_V6BWDV_WM` | dV tile overrides (researchers). |
| `MFA_V6BWDK_BQ`, `MFA_V6BWDK_BK`, `MFA_V6BWDK_WM` | dK tile overrides. |

## What V6NAX backward does NOT support yet

Deferred to follow-up sprints:

- Block-sparse backward (set `MFA_ENABLE_V6_BACKWARD=1` has no effect
  when `flash_attention_sparse` is used; falls back to STEEL sparse
  backward).
- ~~Causal backward~~ — no longer deferred: causal backward shipped
  in v2.50 (Sprint 4) and is default-on for D=64 since v2.51.0.
- D not in {64, 128} backward (falls back to STEEL).
- Softcap / ALiBi / TurboQuant backward (kept on STEEL).
- Multi-batch GQA where Hq > Hk (dK/dV output is per-Q-head; caller
  must reduce across query-heads sharing each KV-head for proper GQA
  gradient shape — current implementation matches MLX SDPA-vjp layout
  for non-GQA cases).

## References

- `CHANGELOG.md` [2.37.0] entry — full v2.37.0 changes
- `.doc-archive/docs/v6-nax/v6nax-backward-status.md` — full sprint timeline + design
  decisions DC0-DC13
- `.doc-archive/docs/v6-nax/v6nax-backward-option-gamma-design.md` — next-sprint design
  for fused dK+dV (Option γ)
- `.doc-archive/docs/v6-nax/v6nax-backward-decisions.md` — design rationale
- `ENV_VARS.md` — full env var reference
- `docs/reference/API_MANUAL.md` — `flash_attention()` API

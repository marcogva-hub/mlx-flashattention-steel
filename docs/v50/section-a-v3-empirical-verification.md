# v2.50 Prompt 5d Section A v3 — Empirical verification

**Mandate**: per Marco's Prompt 5d directive, run bench comparing V6NAX
native sparse backward vs Prompt 5c hybrid vs SDPA-vjp dense baseline
at VSR audit shape.

## Bench setup

- Shape: B=1 H=12 qL=4096 D=128 fp16 BT=32 (VSR audit shape)
- Hardware: M5 Max 128GB, macOS 26.4
- Methodology: 3 warmup + 10 measurement iterations per path, median
  reported.  Full materialization (`mx.eval` + `mx.synchronize`)
  between iterations.

## Results (VSR shape)

3 paths via `mx.grad(loss)(q, k, v)`:

| Density | SDPA-vjp | Hybrid (5c) | Full native (5d) | Hybrid/SDPA | Native/SDPA |
|---|---|---|---|---|---|
| 0.1 | 17.41 ms | 34.84 ms | 22.58 ms | 0.50x | 0.77x |
| 0.3 | 17.40 ms | 68.20 ms | 60.67 ms | 0.26x | 0.29x |
| 0.5 | 16.71 ms | 102.01 ms | 98.18 ms | 0.16x | 0.17x |
| 1.0 | 16.93 ms | 175.09 ms | 181.07 ms | 0.10x | 0.09x |

### D=64 small-H shape (B=1 H=4 qL=2048 D=64)

| Density | SDPA-vjp | Full native | Native/SDPA |
|---|---|---|---|
| 0.1 | 1.42 ms | 1.26 ms | **1.13x (faster)** |
| 0.3 | 1.51 ms | 2.33 ms | 0.65x |
| 0.5 | 1.57 ms | 3.28 ms | 0.48x |
| 1.0 | 1.67 ms | 5.98 ms | 0.28x |

## Empirical verdict

**VSR shape (audit target)**: SDPA-vjp dense wins at all densities.
V6NAX native sparse can't outpace Apple SDPA NAX backward on M5+.

**D=64 small-H shape**: V6NAX native sparse marginally wins (1.13x) at
d=0.1 only.  Win envelope too narrow for production AUTO routing.

## Decision tree outcome

Per Marco's directive:
> If confirmed (V6NAX native sparse < SDPA-vjp at all densities tested):
> Prompt 5c hybrid orchestrator is empirically production-optimal.
> Skip Section A v3 entirely.  Document as architectural reality.

**OUTCOME**: confirmed at VSR shape.  Decision: REVERT
flash_attention_sparse routing default from Prompt 5d full native to
Prompt 5c hybrid.

## Implementation change

`mlx_mfa/attention.py` flash_attention_sparse dispatch:
- Default for env=1 eligible: Prompt 5c hybrid (NAX sparse forward +
  native dV + SDPA-vjp dQ/dK)
- Research opt-in `MFA_V6_BWD_SPARSE_NATIVE=1`: full native (Prompt 5d
  4 sparse kernels)

The 3 new sparse kernels (dQ + dK split + fused dKdV) remain SHIPPED
at C++ level — accessible via `_ext.v6_nax_backward_*_sparse_raw`
direct bindings.  NOT the production routing default.

## Pattern #6 verdict

This bench data is the empirical basis for Pattern #6 in
`docs/v50/audit-framing-inversions.md`: Apple SDPA NAX optimization on
M5+ is sufficiently high that custom V6NAX NAX backward kernels (even
with sparse-skip optimization) cannot outpace it at audit-relevant
shapes.

## Scope of native sparse kernel retention

Sparse kernels remain in codebase for:
1. Reference implementation of block-sparse iteration in V6NAX NAX
2. Future hardware (Apple SDPA NAX evolution; M5+ next-gen)
3. Opt-in benchmarking for non-VSR shapes (D=64 small-H low-density)

Production routing = hybrid; opt-in env = `MFA_V6_BWD_SPARSE_NATIVE=1`.

## Skill invocations (§AA.2)

| Skill | Result |
|---|---|
| `/mlx-mfa-bench-methodology` | VSR shape bench documented (single-session per path; breadth scan acceptable per §AA.4) |
| `/mlx-mfa-perf-audit` | No new perf claim for full native; hybrid retains existing Prompt 5c characterization |
| `/mlx-code-review` | Routing revert empirically grounded; preserves Prompt 5c production contract |

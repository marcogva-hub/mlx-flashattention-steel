# Backward Family Specification

## Public behavior

Autograd is attached at Python public surfaces using `mx.custom_function`. This avoids relying on a lower-level primitive to retain an LSE output that the forward caller did not consume.

## Terminals

| Terminal | Work performed |
|---|---|
| `v6_split_backward` | NAX dQ plus split dV and dK, followed by MLX reduction |
| `steel_backward` | native STEEL gradient kernel |
| `sdpa_vjp` | MLX differentiation fallback |

## Default M5 policy

D=64, f16/bf16, qL at least 2048 selects V6 split backward by default unless `MFA_DISABLE_V6_BACKWARD=1` is set. Both causal and non-causal cells are eligible.

D=128 is available only when `MFA_ENABLE_V6_BACKWARD=1` is set. It is not a default performance route. The split dV/dK configuration remains the automatic V6 mode; fused alternatives are expert controls.

## GQA gradients

Native V6 kernels produce per-query-head K/V contributions. The wrapper reshapes these contributions by GQA group and sums them back to `[B, Hkv, S, D]`.

## Sparse training

Sparse backward supports a native STEEL path and an opt-in V6 full-native chain. The V6 chain requires the sparse forward LSE variant and records a distinct terminal from the scalar-LSE implementation.

## Correctness locks

Tests compare dQ, dK, and dV independently against fp32 gradients. Coverage includes causal/non-causal, f16/bf16, GQA, and sequence boundaries. A performance ratio is invalid unless the candidate trace differs from `sdpa_vjp`.

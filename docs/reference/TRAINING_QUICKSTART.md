# Training Quickstart

mlx-mfa exposes differentiable public attention. The forward terminal and backward terminal may differ: a custom VJP can pair a NAX or STEEL forward with an SDPA-based gradient.

## Dense gradients

```python
import mlx.core as mx
from mlx_mfa import flash_attention

q = mx.random.normal((1, 8, 2048, 64)).astype(mx.float16)
k = mx.random.normal((1, 8, 2048, 64)).astype(mx.float16)
v = mx.random.normal((1, 8, 2048, 64)).astype(mx.float16)

def loss(q_, k_, v_):
    return flash_attention(q_, k_, v_, causal=True).square().mean()

value, grads = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)
mx.eval(value, *grads)
```

On M5, D=64 with sequence length at least 2048 uses `v6_split_backward` by default for f16/bf16 unless `MFA_DISABLE_V6_BACKWARD=1` is set. D=128 remains opt-in through `MFA_ENABLE_V6_BACKWARD=1` and is coverage-oriented rather than a default performance route.

The July 2026 engagement harness measured the default D=64 backward at 2.05–2.84x the SDPA-VJP baseline on M5 Max, MLX 0.31.2, macOS 27 beta. This β3-indicative range must be revalidated on stable macOS.

## Sparse gradients

`flash_attention_sparse` offers these backward modes:

| Mode | Behavior |
|---|---|
| `dense` | differentiates a dense SDPA representation of the block mask |
| `steel_sparse` | native sparse backward |
| `sdpa_sparse` | deprecated compatibility mode |

The full-native sparse chain is an opt-in. Enable both the V6 backward capability and the sparse-native selector:

```bash
MFA_ENABLE_V6_BACKWARD=1 MFA_V6_BWD_SPARSE_NATIVE=1 python train.py
```

Inside that envelope, the sparse V6 forward emits the natural-log LSE consumed by the existing dQ/dK/dV kernels. The ordinary routed sparse forward does not request LSE and remains a separate zero-overhead variant.

## GNA gradients

Native GNA kernels are forward-only. For training, set `MFA_DISABLE_GNA_NATIVE=1`; the public API then builds the GNA block mask and uses the differentiable sparse path.

## LSE contract

Public dense `return_lse=True` returns `(output, lse)` and ignores the LSE cotangent in its custom VJP. Sparse full-native backward consumes natural-log LSE. These representations are not interchangeable with historical log2-domain values.

## Verification

Before enabling an opt-in in training:

1. capture the forward and backward terminals;
2. compare `dQ`, `dK`, and `dV` separately to an fp32 oracle;
3. include the real causal, GQA, scale, and sequence shapes;
4. rerun after changing MLX or macOS.

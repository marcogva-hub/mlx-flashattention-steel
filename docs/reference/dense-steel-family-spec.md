# Dense Attention Family Specification

## Public contract

Dense attention accepts BHND tensors `[B, H, N, D]`. Q, K, and V must have matching dtypes; K and V share sequence length and head count; `Hq` must be divisible by `Hkv`.

The public function resolves the default scale to `1/sqrt(D)` and supports causal masking, selected windows, ALiBi, bias modes, optional weights, and optional LSE through separate branches.

## Implementations

| Family | Role |
|---|---|
| MLX SDPA | broad correctness fallback and VJP oracle |
| STEEL V1/V2/V3 | native simdgroup attention for supported shapes and features |
| V6 NAX | cooperative-tensor dense forward on M5 |
| decode primitive | narrow long-context qL=8 or qL=16 carveouts |

V6 is a pure NAX family. The former simdgroup-within-V6 alternative is not a production path.

## M5 auto-route

Plain dense D=128 f16/bf16 self-attention can select terminal `nax_dense` when NAX is available and no incompatible feature is present. D=64 plain forward remains SDPA unless another narrow policy, such as decode, applies.

The NAX forward is wrapped in a custom VJP whose backward uses SDPA-VJP unless the separate backward policy selects `v6_split_backward`.

## Decode carveouts

The `mfa_primitive` terminal is selected only for non-causal D=64 f16/bf16 GQA cells:

- qL=8, GQA=8, 4096 <= kL <= 65536;
- qL=16, GQA in {4, 8, 16}, 16384 <= kL <= 65536.

Adjacent qL, causal calls, D=128, and out-of-range kL use SDPA.

## D=512

The public D=512 path delegates to SDPA. Direct MFA bindings reject D=512 because no corresponding native attention kernel exists. Tests lock both facts.

## LSE

Dense LSE uses natural logarithms in the current public contract. `return_lse=True` rejects combinations whose native shortcut would omit softcap or window semantics.

## Safety properties

Scale values at or below zero are delegated to SDPA because the V6 binding reserves positive values for its scale convention. Unsupported feature combinations do not silently drop the feature.

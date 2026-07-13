# Naming and semantic conventions

This glossary names paths by the datapath that executes them. Runtime trace
terminals are preferred over historical version labels.

## Kernel lineage

| Name | Meaning in current source |
|---|---|
| STEEL V1 | Original simdgroup dense forward family in `mfa_steel_fwd.cpp`. |
| STEEL V2 | Sequential K/V simdgroup family in `mfa_steel_fwd_v2.cpp`. |
| STEEL V3 | Compiled conditional simdgroup family in `mfa_steel_fwd_v3.cpp`. |
| V6 NAX | M5+ Metal-4 path using NAX/matmul2d primitives. |
| V7 | Reserved for a future Metal-4.1 lineage; no production V7 kernel exists. |
| V4 / V5 | Removed forward experiments; their sources and enable controls are not live. |

The old internal label `V34` is not a kernel generation. It was renamed to V6
NAX. Deprecated V34 environment names remain aliases so existing scripts fail
gradually rather than silently changing behavior.

## Runtime terminal names

| Terminal | Executed path |
|---|---|
| `sdpa` | MLX scaled dot-product attention. |
| `nax_dense` | Dense V6 NAX forward. |
| `mfa_primitive` | STEEL-family primitive, including decode/window variants. |
| `v6nax_sparse` | BT32 V6 NAX sparse forward. |
| `scalar_fallback` | Per-query-row scalar sparse coverage kernel. |
| `gna_v6nax` | V6 NAX grouped-neighborhood forward. |
| `gna_native` | STEEL grouped-neighborhood forward. |
| `varlen_v6nax` | Packed-varlen V6 NAX opt-in. |
| `varlen_native` | Packed-varlen STEEL path. |
| `varlen_split_concat` | Per-segment MLX fallback. |
| `v6_split_backward` | V6 NAX split backward family. |

Use these terminal labels in tests and performance evidence. A function name or
requested backend is not proof that its kernel ran.

## Sparse names

`v6nax_sparse` is the canonical cooperative-tensor sparse path. The aliases
`v2`, `v6_nax_sparse`, `v6-nax-sparse` and `v6nax` are accepted by the expert
selector for compatibility.

`scalar_fallback` names the pre-NAX per-thread-row implementation. The aliases
`v1` and `sparse_scalar_fallback` remain accepted. They do not mean that the
kernel belongs to the historical dense V1 lineage.

## Asymmetric causal convention

The canonical name is **bottom-right-aligned, zero-clamped causal masking**.
For a query segment of length `qL` and key segment of length `kL`:

```text
qL_off = max(0, kL - qL)
visible(q_row, k_col) = (k_col <= qL_off + q_row)
```

Indices are segment-local. When `qL <= kL`, this is commonly called
lower-right alignment. When `qL > kL`, the zero clamp makes it top-left
aligned and does not create fully masked leading query rows. Those historical
phrases are aliases for the formula above, not separate conventions.

## Tensor and mask layouts

- Dense attention tensors use BHND: `[batch, heads, sequence, dimension]`.
- Packed-varlen tensors use B=1 BHND storage; cumulative sequence arrays delimit
  independent segments.
- GQA requires `Hq % Hkv == 0` where the selected path supports GQA.
- Sparse block masks may be `[NQ,NK]`, `[H,NQ,NK]` or `[B,H,NQ,NK]`.
- A mask block describes a rectangular query/key tile. The V6 NAX sparse tile
  is BQ32/BK32; BT64 public support expands each source block to a 2x2 BT32
  representation before entering an eligible V6 NAX cell.

## Precision names

Use `fp16`, `bf16` and `fp32` for user-facing dtype names. `f16` and `f32` are
acceptable Metal/source abbreviations. A fallback accepting fp32 does not imply
that the corresponding native kernel supports fp32.

LSE means natural-log log-sum-exp unless a source explicitly states otherwise.
The sparse V6 NAX optional LSE converts the online log2-domain state at store
time; an all-masked row stores negative infinity.

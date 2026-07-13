# Sparse Attention Family Specification

## Mask contract

The public sparse API consumes a block mask whose last two dimensions index query and key blocks. The mask dtype accepted by the public wrapper is path-dependent: boolean is canonical; numeric masks are normalized by paths that explicitly support them. Callers should use boolean masks to avoid route-dependent coercion.

Block tile 32 is native to `v6nax_sparse`. Block tile 64 is expanded exactly to a 2x2 BT32 representation before entering the same kernel when the resulting cell remains eligible. Other block sizes retain their existing fallback.

## Implementations

| Terminal | Meaning |
|---|---|
| `v6nax_sparse` | M5 cooperative-tensor sparse forward |
| `scalar_fallback` | native scalar coverage path |
| `sdpa` | expanded additive-mask MLX fallback |

The scalar path is retained even when NAX is available.

## Hardened M5 route

All NAX cells require self-attention, D in {64,128}, f16/bf16, effective BT32, and density within the stated ceiling.

### Non-causal

| Dtype | N range | B·H | D | Maximum density |
|---|---:|---:|---:|---:|
| fp16 | exactly 8192 | 1, 4, or 12 | 64 or 128 | 0.30 |
| fp16 | 4096 through 8192 | 12 | 128 | 0.30 |
| fp16 | 4096 through 8192 | 12 | 64 | 0.25 |
| fp16 | 4096 through 8192 | 4 | 128 | 0.05 |
| bf16 | 4096 through 8192 | 12 | 128 | 0.30 |

### Causal

| Dtype | N | B·H | D | Maximum density |
|---|---:|---:|---:|---:|
| fp16 | 4096 | 4 | 128 | 0.10 |
| fp16 | 4096 | 12 | 128 | 0.30 |
| fp16 | 8192 | 12 | 64 or 128 | 0.30 |
| bf16 | 4096 | 4 | 128 | 0.10 |

Every unlisted cell routes to SDPA or the scalar coverage path. B·H values are measured values, not an interpolated interval.

## Causal semantics

Causal visibility is bottom-right-aligned and zero-clamped as defined in [NAMING.md](../../NAMING.md). The NAX kernel skips whole out-of-range blocks and applies a triangular mask within the diagonal block.

## Optional LSE

The sparse generator has distinct no-LSE and LSE variants. The LSE variant stores natural-log `m + log(l)` per row and preserves the scalar convention for rows without an active tile. Requesting no LSE leaves the production MSL path unchanged.

## Backward

The public API retains dense-VJP and native sparse choices. The full-native V6 chain is opt-in; it uses `v6nax_sparse_lse` in the forward and existing native dQ/dK/dV kernels in the backward.

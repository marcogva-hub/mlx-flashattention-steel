# Paged Attention — Design Notes

**Status:** Design phase. Not yet implemented. Tracked as v1.0 roadmap item.

---

## Problem

Standard KV cache allocates a contiguous `[B, H, S_max, D]` tensor per layer.
For long-context inference or parallel requests with different sequence lengths,
this wastes significant device memory: every batch item pays the cost of
`S_max` even if its context is much shorter.

Paged Attention (vLLM, 2023) solves this by allocating KV memory in fixed-size
*pages* (blocks), analogous to OS virtual memory paging.

---

## Core Concepts

### Page / Block

A page is a contiguous `[BLOCK_SIZE, D]` tensor for a single head.
Typical `BLOCK_SIZE`: 16, 32, or 64 tokens.

### Block Table

A per-sequence index array `block_table[seq_i]` maps logical block number
`b` → physical block id in a global pool.

```
block_pool: [num_blocks, BLOCK_SIZE, D]   (shared across all sequences)
block_table: [B, max_blocks_per_seq]      (logical → physical mapping)
```

### Variable-length Batching (varlen)

Sequences in a batch have different lengths `[s_0, s_1, ..., s_{B-1}]`.
Instead of padding to `S_max`, store them *packed*:

```
keys_packed:   [total_tokens, H_kv, D]
cu_seqlens_k:  [B+1]   (cumulative sum of kv_lens)
```

---

## Implementation Strategy for mlx-mfa

### Phase 1 — varlen prefill (STEEL cooperative tiling)

**Input layout change:**
- Replace `[B, H, N, D]` with `keys_packed [total, H, D]` + `cu_seqlens`
- Threadgroup grid: `(NQ_per_seq × B_total, H, 1)` where the Q-side is
  similarly packed

**Kernel changes:**
1. Decode the batch item `b` and local sequence position from `tid.x` using
   `cu_seqlens` (binary search or precomputed block offsets).
2. Load Q tile from packed Q at offset `cu_seqlens_q[b] + tile_q * BQ`.
3. Load K/V from packed KV at offset `cu_seqlens_k[b]`.
4. Causal masking must use *local* sequence position, not global tid.

**MLX integration:**
- New free function `mfa_attention_varlen_forward(q_packed, k_packed, v_packed, cu_seqlens_q, cu_seqlens_k, ...)`.
- New `MFAttentionVarlen` primitive or reuse `MFAttention` with a `varlen` Params flag.

### Phase 2 — paged KV decode

**Block table dispatch:**
For each K/V tile in the decode loop, instead of a contiguous pointer, the
kernel must look up the physical block for the current logical K-position:

```metal
// Inside the K-loop:
int logical_block = kb;
int phys_block    = block_table[b * max_blocks + logical_block];
int phys_offset   = phys_block * BLOCK_SIZE * H * D;
// Read K/V page from global pool
```

**Kernel signature (Metal):**
```metal
kernel void mlx_mfa_paged_decode(
    const device T* q              [[buffer(0)]],
    const device T* kv_pool        [[buffer(1)]],  // [num_blocks, BLOCK_SIZE, H, D]
    const device int* block_tables [[buffer(2)]],  // [B, max_blocks]
    const device int* seq_lens     [[buffer(3)]],  // [B] actual KV lengths
    ...
)
```

**Challenges:**
- Block table look-up inside the inner K-loop adds a gather — hardware does
  not coalesce these.  Prefetch or tile multiple pages per iteration.
- `block_table` must be available on-device at dispatch time (not a separate
  Python pass).
- Flash Decoding (Split-KV) integration: splits must align to block boundaries.

---

## Estimated Complexity

| Sub-task | Effort | Dependency |
|----------|--------|------------|
| varlen prefill STEEL kernel | Medium | MFASteelParams varlen extension |
| cu_seqlens dispatch logic | Low | New Params fields |
| Paged KV pool Metal kernel | High | Block table gather primitives |
| MLX array allocator for block pool | Medium | External pool management |
| Python API + bindings | Low | Above C++ work complete |
| Backward pass (varlen) | High | varlen forward working |

---

## Prior Art / References

- **vLLM Paged Attention** — PagedAttention CUDA kernel
  (Kwon et al., SOSP 2023, https://arxiv.org/abs/2309.06180)
- **FlashAttention-3 varlen** — `flash_attn_varlen_func` API (Tri Dao, 2024)
- **MLX issue #1189** — varlen attention request
- **FlashInfer** — block-sparse paged attention kernels for Metal targets

---

## Deferred Work

Paged Attention requires a *block allocator* that manages the physical KV pool
across requests.  For mlx-mfa this would be a Python-level helper (similar to
`mlx-lm`'s `KVCache` class) rather than a kernel concern.

A reference Python allocator sketch:

```python
class PagedKVCache:
    """Minimal page allocator for mlx-mfa paged attention (design sketch)."""

    def __init__(self, num_blocks: int, block_size: int,
                 num_heads: int, head_dim: int, dtype=mx.float16):
        self.block_size = block_size
        # Physical pool: [num_blocks, block_size, num_heads, head_dim]
        self.k_pool = mx.zeros((num_blocks, block_size, num_heads, head_dim),
                               dtype=dtype)
        self.v_pool = mx.zeros_like(self.k_pool)
        self._free = list(range(num_blocks))   # free block ids
        self.block_tables: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """Allocate blocks for a new sequence."""
        import math
        num_blocks = math.ceil(num_tokens / self.block_size)
        self.block_tables[seq_id] = [self._free.pop() for _ in range(num_blocks)]

    def free(self, seq_id: int) -> None:
        """Return blocks of a finished sequence to the free list."""
        self._free.extend(self.block_tables.pop(seq_id, []))
```

This design sketch will be fleshed out when the Metal kernel is implemented.

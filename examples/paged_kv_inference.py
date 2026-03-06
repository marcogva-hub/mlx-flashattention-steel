"""paged_kv_inference.py — Paged KV-cache attention for variable-length batches.

Paged KV caches (inspired by vLLM) store KV blocks in a fixed-size page pool
rather than per-sequence contiguous buffers.  This eliminates fragmentation and
enables continuous batching: multiple sequences of different lengths share the
same page pool.

mlx-mfa exposes paged attention through flash_attention_kvcache():
  - k_cache / v_cache: the shared page pool  [num_pages, block_size, H, D]
  - block_table: integer page assignments     [B, max_pages_per_seq]
  - seq_lens:    actual KV lengths            [B]
  - block_size:  tokens per page (power of 2, typically 16 or 64)

Usage::

    python examples/paged_kv_inference.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention_kvcache

# ── configuration ──────────────────────────────────────────────────────────────

B          = 4       # concurrent sequences (continuous batch)
H          = 8
D          = 128
BLOCK_SIZE = 64      # tokens per page
dtype      = mx.float16
scale      = 1.0 / math.sqrt(D)

# Each sequence has a different KV length (continuous batch scenario).
seq_lengths = [512, 1024, 768, 256]   # tokens already in cache per sequence
max_len     = max(seq_lengths)
N_q         = 1                        # decode step: 1 new query token per seq

mx.random.seed(7)

# ── build paged pool ────────────────────────────────────────────────────────────
#
# For simplicity we allocate a fresh pool with no inter-sequence sharing.
# Production systems reuse freed blocks across sequences.

pages_per_seq = [(l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in seq_lengths]
total_pages   = sum(pages_per_seq)

# Page pool: [total_pages, block_size, H, D]
pool_k = mx.random.normal((total_pages, BLOCK_SIZE, H, D)).astype(dtype)
pool_v = mx.random.normal((total_pages, BLOCK_SIZE, H, D)).astype(dtype)

# Block table: [B, max_pages_per_seq]  (-1 = unused slot)
max_pages = max(pages_per_seq)
block_table_list = []
page_offset = 0
for seq_idx in range(B):
    pages = pages_per_seq[seq_idx]
    row   = list(range(page_offset, page_offset + pages))
    row  += [-1] * (max_pages - pages)   # pad unused slots
    block_table_list.append(row)
    page_offset += pages

block_table = mx.array(block_table_list, dtype=mx.int32)   # [B, max_pages]
seq_lens    = mx.array(seq_lengths,      dtype=mx.int32)   # [B]

print(f"Batch          : B={B} sequences")
print(f"KV seq lengths : {seq_lengths}")
print(f"Page pool size : {total_pages} pages × {BLOCK_SIZE} tokens × H={H} × D={D}")
print(f"block_table    : {block_table.shape}")

# ── single decode step ─────────────────────────────────────────────────────────

# One query token per sequence.
q = mx.random.normal((B, H, N_q, D)).astype(dtype)

out = flash_attention_kvcache(
    q,
    pool_k,        # k_cache = page pool  [total_pages, block_size, H, D]
    pool_v,        # v_cache = page pool  [total_pages, block_size, H, D]
    block_table=block_table,   # page assignments
    seq_lens=seq_lens,         # true KV lengths (avoids attending padding)
    block_size=BLOCK_SIZE,
    scale=scale,
    causal=True,
)
mx.synchronize()

print(f"\nQuery          : {q.shape}")
print(f"Output         : {out.shape}")
assert out.shape == (B, H, N_q, D), f"Unexpected shape {out.shape}"

# ── verify non-NaN output ──────────────────────────────────────────────────────

assert mx.isfinite(out).all().item(), "Output contains NaN/Inf"
print(f"Output finite  : True")

# ── multi-query decode (N_q > 1) ──────────────────────────────────────────────
#
# Speculative decoding or small chunked prefill steps use N_q > 1 per sequence.

N_q4 = 4
q4   = mx.random.normal((B, H, N_q4, D)).astype(dtype)

out4 = flash_attention_kvcache(
    q4, pool_k, pool_v,
    block_table=block_table,
    seq_lens=seq_lens,
    block_size=BLOCK_SIZE,
    scale=scale,
    causal=True,
)
mx.synchronize()
print(f"\nN_q=4 output   : {out4.shape}")
assert out4.shape == (B, H, N_q4, D)

print("\n✓ Paged KV inference example completed")

"""paged_kv_inference.py — Paged decode and scheduler-style remap example.

Demonstrates:
  1) low-level `flash_attention_paged(...)` with paged pools
  2) runtime-level `DecodeRuntime.paged_prefill_batch/paged_step_batch` helpers

Usage::

    python examples/paged_kv_inference.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention_paged, create_decode_runtime

# Configuration
B = 4
H_q = 8
H_kv = 8
D = 128
BLOCK_SIZE = 64
N_q = 1
dtype = mx.float16
scale = 1.0 / math.sqrt(D)

seq_lengths = [512, 1024, 768, 256]
mx.random.seed(7)

# Build paged pool (simple contiguous assignment for demo)
pages_per_seq = [(l + BLOCK_SIZE - 1) // BLOCK_SIZE for l in seq_lengths]
total_pages = sum(pages_per_seq)
max_pages = max(pages_per_seq)

pool_k = mx.random.normal((total_pages, BLOCK_SIZE, H_kv, D)).astype(dtype)
pool_v = mx.random.normal((total_pages, BLOCK_SIZE, H_kv, D)).astype(dtype)

block_table_rows = []
page_offset = 0
for pages in pages_per_seq:
    row = list(range(page_offset, page_offset + pages))
    row += [-1] * (max_pages - pages)
    block_table_rows.append(row)
    page_offset += pages

block_table = mx.array(block_table_rows, dtype=mx.int32)
seq_lens = mx.array(seq_lengths, dtype=mx.int32)

# Low-level paged attention call
q = mx.random.normal((B, H_q, N_q, D)).astype(dtype)
out = flash_attention_paged(
    q,
    pool_k,
    pool_v,
    block_table,
    seq_lens,
    block_size=BLOCK_SIZE,
    scale=scale,
    causal=True,
)
mx.synchronize()
print("Low-level paged output:", out.shape)

# Runtime-level paged decode flow with remap
rt = create_decode_runtime(
    backend="paged",
    paged=True,
    quantized_kv=False,
    B=B,
    H_q=H_q,
    H_kv=H_kv,
    D=D,
    max_seq_len=2048,
    num_blocks=256,
    block_size=BLOCK_SIZE,
)

seq_ids = [101, 102, 103, 104]
q_pre = mx.random.normal((B, H_q, 16, D)).astype(dtype)
k_pre = mx.random.normal((B, H_kv, 16, D)).astype(dtype)
v_pre = mx.random.normal((B, H_kv, 16, D)).astype(dtype)
rt.paged_prefill_batch(q_pre, k_pre, v_pre, seq_ids=seq_ids, causal=True)

q_step = mx.random.normal((2, H_q, 1, D)).astype(dtype)
k_step = mx.random.normal((2, H_kv, 1, D)).astype(dtype)
v_step = mx.random.normal((2, H_kv, 1, D)).astype(dtype)
cache_batch_idx = mx.array([3, 1], dtype=mx.int32)  # active request order
out_step = rt.paged_step_batch(
    q_step,
    k_step,
    v_step,
    seq_ids=seq_ids,
    cache_batch_idx=cache_batch_idx,
    causal=True,
)
mx.synchronize()

print("Runtime paged remap output:", out_step.shape)
print("Runtime metadata:", rt.metadata)
print("\n✓ Paged KV example completed")

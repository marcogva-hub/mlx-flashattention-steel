"""sliding_window.py — Block-sparse sliding-window attention for long sequences.

Long-range attention is O(N²) in compute and memory.  Sliding-window attention
restricts each token to attend only to the W nearest past tokens, reducing
active work to O(N × W / D²).

mlx-mfa exposes this via flash_attention_sparse() + make_sliding_window_mask():
  - make_sliding_window_mask builds a [NQ_tiles, NK_tiles] boolean block mask
  - flash_attention_sparse skips tiles where the mask is False (zero warp divergence)
  - Backward gradients are computed via dense SDPA + block bias (correct, no sparsity)

Usage::

    python examples/sliding_window.py
"""

import math
import time
import mlx.core as mx
from mlx_mfa import flash_attention_sparse, make_sliding_window_mask, flash_attention

# ── configuration ──────────────────────────────────────────────────────────────

B, H, N, D = 1, 8, 8192, 128
WINDOW = 512    # each token attends to ±512 past tokens

dtype  = mx.float16
scale  = 1.0 / math.sqrt(D)

mx.random.seed(1)
q = mx.random.normal((B, H, N, D)).astype(dtype)
k = mx.random.normal((B, H, N, D)).astype(dtype)
v = mx.random.normal((B, H, N, D)).astype(dtype)
mx.synchronize()

# ── build block mask ────────────────────────────────────────────────────────────

mask = make_sliding_window_mask(N, window_size=WINDOW, head_dim=D)
print(f"Block mask shape : {mask.shape}")
active_blocks = mask.sum().item()
total_blocks  = mask.size
sparsity      = 1.0 - active_blocks / total_blocks
print(f"Active blocks    : {int(active_blocks)} / {total_blocks}  "
      f"({sparsity*100:.1f}% sparse)")

# ── sliding-window sparse attention ────────────────────────────────────────────

# Causal = only attend to tokens before or at current position.
out_sparse = flash_attention_sparse(q, k, v, mask, scale=scale, causal=True)
mx.synchronize()
print(f"\nSparse output    : {out_sparse.shape}")

# ── benchmark ──────────────────────────────────────────────────────────────────

def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn(); mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(); mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

ms_sparse = bench(lambda: flash_attention_sparse(q, k, v, mask, scale=scale, causal=True))
ms_dense  = bench(lambda: flash_attention(q, k, v, scale=scale, causal=True))

print(f"\nDense  causal N={N}: {ms_dense:.1f} ms")
print(f"Sparse W={WINDOW}   N={N}: {ms_sparse:.1f} ms  ({ms_dense/ms_sparse:.2f}× speedup)")

# ── gradient check ─────────────────────────────────────────────────────────────

def loss_fn(q_, k_, v_):
    return flash_attention_sparse(q_, k_, v_, mask, scale=scale, causal=True).sum()

_, grads = mx.value_and_grad(loss_fn)(q, k, v)
mx.synchronize()
assert grads.shape == q.shape
print(f"\nGradient dQ      : {grads.shape}  (finite={mx.isfinite(grads).all().item()})")

# ── causal block mask for comparison ──────────────────────────────────────────

from mlx_mfa import make_causal_block_mask
causal_mask   = make_causal_block_mask(N, head_dim=D)
out_causal_bk = flash_attention_sparse(q, k, v, causal_mask, scale=scale, causal=True)
mx.synchronize()
print(f"\nCausal-block output: {out_causal_bk.shape}")

print("\n✓ Sliding-window example completed")

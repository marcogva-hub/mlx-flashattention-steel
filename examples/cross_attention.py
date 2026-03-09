"""cross_attention.py — Encoder-decoder cross-attention with flash_attention_kvcache.

Demonstrates how to use flash_attention_kvcache() for cross-attention:
  - Encoder produces a fixed KV tensor (k_enc, v_enc) once per forward pass.
  - Decoder queries (q_dec) attend to all encoder positions (causal=False).
  - GQA: fewer KV heads than Q heads to reduce encoder memory.
  - Full autograd: verify dQ, dK_enc, dV_enc are finite.

Key insight: passing encoder KV as k_cache/v_cache (without block_table)
routes through the dense attention path, equivalent to flash_attention()
but with the convenient kvcache API.

Usage::

    python examples/cross_attention.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention_kvcache, is_mfa_available

# ── configuration ──────────────────────────────────────────────────────────────

B       = 2      # batch size
H_q     = 8      # decoder query heads
H_kv    = 2      # encoder key/value heads (GQA ratio = H_q / H_kv = 4)
D       = 128    # head dimension
S_enc   = 512    # encoder sequence length (fixed)
S_dec   = 64     # decoder sequence length (target tokens)

dtype = mx.float16
scale = 1.0 / math.sqrt(D)

print(f"mlx-mfa available: {is_mfa_available()}")
print(f"Config: B={B} H_q={H_q} H_kv={H_kv} D={D} S_enc={S_enc} S_dec={S_dec}")

# ── create tensors ──────────────────────────────────────────────────────────────

mx.random.seed(42)

# Encoder output — constant per forward pass
k_enc = mx.random.normal([B, H_kv, S_enc, D]).astype(dtype)
v_enc = mx.random.normal([B, H_kv, S_enc, D]).astype(dtype)

# Decoder queries — one set per decode step (or full S_dec for training)
q_dec = mx.random.normal([B, H_q, S_dec, D]).astype(dtype)

# ── forward pass ───────────────────────────────────────────────────────────────

print("\n--- forward pass ---")
out = flash_attention_kvcache(
    q_dec, k_enc, v_enc,
    scale=scale,
    causal=False,   # decoder attends to ALL encoder positions
)
mx.eval(out)
print(f"output shape: {out.shape}")  # [B, H_q, S_dec, D]
print(f"output dtype: {out.dtype}")

# ── single-token decode step (S_dec = 1) ───────────────────────────────────────

print("\n--- single-token decode ---")
q_single = mx.random.normal([B, H_q, 1, D]).astype(dtype)
out_single = flash_attention_kvcache(
    q_single, k_enc, v_enc,
    scale=scale,
    causal=False,
)
mx.eval(out_single)
print(f"single-token output shape: {out_single.shape}")  # [B, H_q, 1, D]

# ── autograd: verify all gradients are finite ──────────────────────────────────

print("\n--- autograd ---")


def cross_attn_fn(q, k, v):
    return flash_attention_kvcache(q, k, v, scale=scale, causal=False)


cotangent = mx.ones([B, H_q, S_dec, D], dtype=dtype)
out_fwd, grads = mx.vjp(
    cross_attn_fn,
    (q_dec, k_enc, v_enc),
    (cotangent,),
)
dq, dk, dv = grads
mx.eval(dq, dk, dv)

all_fin = {
    "dQ":     bool(mx.all(mx.isfinite(dq)).item()),
    "dK_enc": bool(mx.all(mx.isfinite(dk)).item()),
    "dV_enc": bool(mx.all(mx.isfinite(dv)).item()),
}
print(f"dQ shape:     {dq.shape}  finite={all_fin['dQ']}")
print(f"dK_enc shape: {dk.shape}  finite={all_fin['dK_enc']}")
print(f"dV_enc shape: {dv.shape}  finite={all_fin['dV_enc']}")

# ── GQA note ───────────────────────────────────────────────────────────────────

print(f"\nGQA ratio: {H_q} Q heads / {H_kv} KV heads = {H_q // H_kv}x")
print("dK_enc / dV_enc preserve H_kv shape — no expansion in backward.")

print("\nDone.")

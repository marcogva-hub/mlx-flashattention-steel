"""kvcache_decode.py — Autoregressive decode with a growing KV cache.

Shows the recommended inference pattern:
  - Prefill (full context) → build initial KV cache
  - Decode loop: generate one token at a time, growing the cache via concatenation
  - flash_attention_kvcache() handles both prefill and decode steps

Note: flash_attention_kvcache() takes the *complete* accumulated cache as
k_cache / v_cache and returns only the output tensor.  The caller is
responsible for managing cache growth (concatenation here; paged for
large contexts — see paged_kv_inference.py).

Usage::

    python examples/kvcache_decode.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention_kvcache

# ── configuration ──────────────────────────────────────────────────────────────

B, H, D = 1, 8, 128
PROMPT_LEN   = 512    # prefill tokens
DECODE_STEPS = 8      # autoregressive tokens to generate

dtype = mx.float16
scale = 1.0 / math.sqrt(D)

mx.random.seed(42)

# ── prefill ────────────────────────────────────────────────────────────────────

# Prefill: process the full prompt at once.
# For prefill, N_q == N_k so we use flash_attention_kvcache with causal=True.
q_prefill = mx.random.normal((B, H, PROMPT_LEN, D)).astype(dtype)
k_prefill = mx.random.normal((B, H, PROMPT_LEN, D)).astype(dtype)
v_prefill = mx.random.normal((B, H, PROMPT_LEN, D)).astype(dtype)

out_prefill = flash_attention_kvcache(
    q_prefill, k_prefill, v_prefill,
    scale=scale,
    causal=True,
    # cache_seqlens: how many cached tokens the Q attends *past* its own position.
    # For prefill, no prior cached tokens.
    cache_seqlens=0,
)
mx.synchronize()
print(f"Prefill  N={PROMPT_LEN}  out={out_prefill.shape}")

# Accumulate the KV cache from the prefill step.
k_cache = k_prefill   # [B, H, PROMPT_LEN, D]
v_cache = v_prefill   # [B, H, PROMPT_LEN, D]

# ── decode loop ────────────────────────────────────────────────────────────────

for step in range(DECODE_STEPS):
    # One new query token (the latest generated token).
    q_new = mx.random.normal((B, H, 1, D)).astype(dtype)
    k_new = mx.random.normal((B, H, 1, D)).astype(dtype)
    v_new = mx.random.normal((B, H, 1, D)).astype(dtype)

    # Append new K/V to cache.
    k_cache = mx.concatenate([k_cache, k_new], axis=2)
    v_cache = mx.concatenate([v_cache, v_new], axis=2)
    kv_len  = k_cache.shape[2]   # total cached length including new token

    # Attend: single query over the entire KV cache (causal = attends all).
    out_step = flash_attention_kvcache(
        q_new, k_cache, v_cache,
        scale=scale,
        causal=True,
        cache_seqlens=kv_len - 1,   # Q[0] sits at position kv_len-1
    )
    mx.synchronize()

    print(f"  Step {step+1:2d}  KV_len={kv_len:4d}  out={out_step.shape}")

print("\n✓ Decode loop completed")

"""basic_attention.py — Drop-in replacement for mx.fast.scaled_dot_product_attention.

Shows:
  - flash_attention() for prefill (full sequence, causal or non-causal)
  - GQA (grouped-query attention: H_q != H_kv)
  - Autograd / gradient computation
  - Numerical comparison against MLX SDPA baseline

Usage::

    python examples/basic_attention.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention, is_mfa_available, get_device_info

# ── device info ────────────────────────────────────────────────────────────────

info = get_device_info()
print(f"Device : {info['device_name']}")
print(f"GPU gen: {info['gpu_family_gen']} ({'M3+' if info['is_m3_plus'] else 'M1/M2'})")
print(f"MFA    : {'available' if is_mfa_available() else 'fallback (MLX SDPA)'}")
print()

# ── 1. Basic causal prefill ────────────────────────────────────────────────────

B, H, N, D = 1, 8, 2048, 128
dtype = mx.float16
scale = 1.0 / math.sqrt(D)

mx.random.seed(0)
q = mx.random.normal((B, H, N, D)).astype(dtype)
k = mx.random.normal((B, H, N, D)).astype(dtype)
v = mx.random.normal((B, H, N, D)).astype(dtype)

out_mfa  = flash_attention(q, k, v, scale=scale, causal=True)
out_sdpa = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
mx.synchronize()

err = mx.abs(out_mfa.astype(mx.float32) - out_sdpa.astype(mx.float32)).max().item()
print(f"[1] Causal prefill   B={B} H={H} N={N} D={D}   max_err={err:.2e}")
assert err < 1e-2, f"Numerical error too large: {err}"

# ── 2. Non-causal (cross-attention / encoder) ──────────────────────────────────

S = 1024   # key/value length (encoder sequence)
k2 = mx.random.normal((B, H, S, D)).astype(dtype)
v2 = mx.random.normal((B, H, S, D)).astype(dtype)

out_cross = flash_attention(q[:, :, :256, :], k2, v2, scale=scale, causal=False)
mx.synchronize()
print(f"[2] Cross-attention  Q={256} KV={S} D={D}   shape={out_cross.shape}")

# ── 3. Grouped-query attention (GQA) ──────────────────────────────────────────

H_q, H_kv = 32, 8    # 4× GQA ratio  (matches Llama-3 70B)
q_gqa  = mx.random.normal((B, H_q,  N, D)).astype(dtype)
k_gqa  = mx.random.normal((B, H_kv, N, D)).astype(dtype)
v_gqa  = mx.random.normal((B, H_kv, N, D)).astype(dtype)

out_gqa = flash_attention(q_gqa, k_gqa, v_gqa, scale=scale, causal=True)
mx.synchronize()
print(f"[3] GQA              H_q={H_q} H_kv={H_kv} ratio={H_q//H_kv}×   shape={out_gqa.shape}")

# ── 4. Autograd ────────────────────────────────────────────────────────────────

def attn_fn(q_, k_, v_):
    return flash_attention(q_, k_, v_, scale=scale, causal=True).sum()

loss, grads = mx.value_and_grad(attn_fn, argnums=(0, 1, 2))(q, k, v)
mx.synchronize()
print(f"[4] Gradient         loss={loss.item():.4f}  dQ.shape={grads[0].shape}")
assert grads[0].shape == q.shape
assert grads[1].shape == k.shape

# ── 5. Softcap (Gemma-style) ───────────────────────────────────────────────────

out_cap = flash_attention(q, k, v, scale=scale, causal=True, softcap=30.0)
mx.synchronize()
print(f"[5] Softcap 30.0     shape={out_cap.shape}")

print("\n✓ All basic attention checks passed")

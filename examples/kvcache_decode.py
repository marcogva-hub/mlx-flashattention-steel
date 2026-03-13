"""kvcache_decode.py — Stateful decode using DecodeRuntime (recommended).

Shows current preferred decode usage:
  - prefill via DecodeRuntime
  - step-wise decode with internal cache management
  - optional low-level explicit cache path remains available separately

Usage::

    python examples/kvcache_decode.py
"""

import math
import mlx.core as mx
from mlx_mfa import create_decode_runtime

# Configuration
B, H_q, H_kv, D = 1, 8, 8, 128
PROMPT_LEN = 512
DECODE_STEPS = 8

dtype = mx.float16
scale = 1.0 / math.sqrt(D)

mx.random.seed(42)

rt = create_decode_runtime(
    backend="auto",
    paged=False,
    quantized_kv=False,
    B=B,
    H_q=H_q,
    H_kv=H_kv,
    D=D,
    max_seq_len=4096,
)

# Prefill
q_prefill = mx.random.normal((B, H_q, PROMPT_LEN, D)).astype(dtype)
k_prefill = mx.random.normal((B, H_kv, PROMPT_LEN, D)).astype(dtype)
v_prefill = mx.random.normal((B, H_kv, PROMPT_LEN, D)).astype(dtype)

out_prefill = rt.prefill(q_prefill, k_prefill, v_prefill, scale=scale, causal=True)
mx.synchronize()
print(f"Prefill  N={PROMPT_LEN}  out={out_prefill.shape}  seq_len={rt.seq_length()}")

# Decode loop
for step in range(DECODE_STEPS):
    q_new = mx.random.normal((B, H_q, 1, D)).astype(dtype)
    k_new = mx.random.normal((B, H_kv, 1, D)).astype(dtype)
    v_new = mx.random.normal((B, H_kv, 1, D)).astype(dtype)

    out_step = rt.step(q_new, k_new, v_new, scale=scale)
    mx.synchronize()
    print(
        f"  Step {step+1:2d}  seq_len={rt.seq_length():4d}  out={tuple(out_step.shape)}"
    )

print("\nRuntime metadata snapshot:")
print(rt.metadata)
print("\n✓ DecodeRuntime example completed")

"""varlen_training.py — Variable-length batched attention for training.

In training, sequences in a batch often have different lengths.  The naive
approach pads all sequences to the same length, wasting compute on masked
tokens.  Variable-length (varlen) attention avoids this by:

  1. Packing all sequences end-to-end into a single [1, H, total_tokens, D] tensor.
  2. Passing cumulative-sum arrays (cu_seqlens_q, cu_seqlens_k) that tell the
     kernel where each sequence starts/ends.
  3. Attending independently within each sequence (no cross-sequence leakage).

flash_attention_varlen() is a drop-in for the packed tensor path used in many
training frameworks (Hugging Face Transformers "unpad_input", xFormers, etc.)

Usage::

    python examples/varlen_training.py
"""

import math
import mlx.core as mx
from mlx_mfa import flash_attention_varlen, flash_attention

# ── helpers ────────────────────────────────────────────────────────────────────

def make_cu_seqlens(lengths):
    """Build cumulative seqlen array [0, L0, L0+L1, ...]."""
    cu = [0]
    for l in lengths:
        cu.append(cu[-1] + l)
    return mx.array(cu, dtype=mx.int32)


def _naive_varlen(q_packed, k_packed, v_packed, cu_q, lengths, scale, causal):
    """Reference: iterate over sequences, attend independently."""
    B1, H, _, D = q_packed.shape   # B=1
    outs = []
    for i, L in enumerate(lengths):
        s, e = int(cu_q[i].item()), int(cu_q[i + 1].item())
        qi = q_packed[:, :, s:e, :]
        ki = k_packed[:, :, s:e, :]
        vi = v_packed[:, :, s:e, :]
        oi = flash_attention(qi, ki, vi, scale=scale, causal=causal)
        outs.append(oi)
    return mx.concatenate(outs, axis=2)   # [1, H, total, D]


# ── configuration ──────────────────────────────────────────────────────────────

H, D = 8, 128
scale = 1.0 / math.sqrt(D)
dtype = mx.float16

# Simulate a training mini-batch: 4 sequences of different lengths.
seq_lengths = [256, 512, 128, 384]
total_tokens = sum(seq_lengths)
max_len      = max(seq_lengths)

mx.random.seed(3)

# ── pack sequences ─────────────────────────────────────────────────────────────

# All sequences concatenated along the sequence dimension.
q_pack = mx.random.normal((1, H, total_tokens, D)).astype(dtype)
k_pack = mx.random.normal((1, H, total_tokens, D)).astype(dtype)
v_pack = mx.random.normal((1, H, total_tokens, D)).astype(dtype)

cu_q = make_cu_seqlens(seq_lengths)
cu_k = cu_q   # same packing for self-attention; cross-attention would differ

print(f"Sequences      : {seq_lengths}")
print(f"Total tokens   : {total_tokens}  (vs padded {len(seq_lengths) * max_len})")
print(f"Efficiency     : {total_tokens / (len(seq_lengths) * max_len) * 100:.0f}% utilization")
print(f"cu_seqlens     : {cu_q.tolist()}")

# ── forward pass ───────────────────────────────────────────────────────────────

out = flash_attention_varlen(
    q_pack, k_pack, v_pack,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=max_len,
    max_seqlen_k=max_len,
    scale=scale,
    causal=True,
)
mx.synchronize()

print(f"\nOutput shape   : {out.shape}")    # [1, H, total_tokens, D]
assert out.shape == (1, H, total_tokens, D)

# ── numerical check against naive per-seq reference ──────────────────────────

ref = _naive_varlen(q_pack, k_pack, v_pack, cu_q, seq_lengths, scale, causal=True)
mx.synchronize()
err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
print(f"Max error vs reference: {err:.2e}")
assert err < 5e-2, f"Error too large: {err}"

# ── autograd ───────────────────────────────────────────────────────────────────

def loss_fn(q_, k_, v_):
    return flash_attention_varlen(
        q_, k_, v_, cu_q, cu_k, max_len, max_len, scale=scale, causal=True
    ).sum()

loss, (dq, dk, dv) = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(q_pack, k_pack, v_pack)
mx.synchronize()

print(f"\nLoss           : {loss.item():.4f}")
print(f"dQ shape       : {dq.shape}")
assert mx.isfinite(dq).all().item(), "dQ contains NaN/Inf"
assert mx.isfinite(dk).all().item(), "dK contains NaN/Inf"
print(f"Gradients finite: True")

# ── cross-attention varlen ─────────────────────────────────────────────────────
#
# Q and K/V can have different sequence lengths (e.g., for encoder-decoder).

kv_lengths = [128, 256, 64, 192]   # encoder output lengths (shorter)
cu_q2 = cu_q
cu_k2 = make_cu_seqlens(kv_lengths)
total_kv = sum(kv_lengths)
max_q2   = max(seq_lengths)
max_k2   = max(kv_lengths)

k2_pack = mx.random.normal((1, H, total_kv, D)).astype(dtype)
v2_pack = mx.random.normal((1, H, total_kv, D)).astype(dtype)

out_cross = flash_attention_varlen(
    q_pack, k2_pack, v2_pack,
    cu_seqlens_q=cu_q2,
    cu_seqlens_k=cu_k2,
    max_seqlen_q=max_q2,
    max_seqlen_k=max_k2,
    scale=scale,
    causal=False,   # non-causal for cross-attention
)
mx.synchronize()
print(f"\nCross-attn out : {out_cross.shape}")
assert out_cross.shape == (1, H, total_tokens, D)

print("\n✓ Variable-length training example completed")

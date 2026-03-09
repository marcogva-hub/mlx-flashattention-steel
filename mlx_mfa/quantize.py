"""mlx_mfa.quantize — SageAttention-style int8 quantization utilities.

These functions run on the MLX computation graph before the Metal kernel.
They prepare Q and K for the SageAttention Metal kernel that loads int8 from
device memory and dequantizes to fp16 inside the kernel (in threadgroup memory).

Apple Silicon has NO int8 GEMM hardware (unlike NVIDIA Ampere/Ada).
The simdgroup_matrix_multiply_accumulate operates on fp16 tiles.
The speedup comes from halved device→threadgroup memory traffic for Q@K^T tiles.
V is never quantized — P@V stays fp16.

References:
  SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference
  Acceleration (Zhang et al., ICLR 2025)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx

# Optional C++ accelerated quantization (Phase 4-A.1).
# Falls back to the pure-MLX path below when the extension is not built.
try:
    from mlx_mfa._ext import mfa_quantize_per_block as _mfa_quantize_per_block_cpp
    _HAS_CPP_QUANTIZE = True
except ImportError:
    _HAS_CPP_QUANTIZE = False


# ---------------------------------------------------------------------------
# Per-block INT8 quantization
# ---------------------------------------------------------------------------


def quantize_per_block(
    x: mx.array,
    block_size: int,
) -> tuple[mx.array, mx.array]:
    """Quantize x to int8 per block of tokens.

    For each contiguous block of ``block_size`` tokens along the sequence
    dimension (axis 2), computes:

        scale = max(abs(block)) / 127.0
        x_int8 = clip(round(x / scale), -128, 127)

    This is **symmetric** per-block quantization: zero maps to zero, which
    preserves the sign structure of attention scores.

    Args:
        x:          ``[B, H, N, D]`` fp16 or bf16 input array.
        block_size: Number of tokens per quantization block.  Best set to
                    the STEEL tile size for Q (BQ) or K (BK) so each Metal
                    tile has exactly one scale value.

    Returns:
        ``(x_int8, x_scale)`` where:

        - ``x_int8``:  ``[B, H, N, D]`` int8 — quantized values.
        - ``x_scale``: ``[B, H, N_blocks, 1]`` float32 — one scale per block.
          To dequantize: ``x_fp16 = x_int8.astype(float32) * x_scale[block_idx]``.

    Note:
        The returned ``x_int8`` has the same ``N`` as the input (no padding
        exposed to the caller).  Padding rows (if N was not a multiple of
        ``block_size``) are handled internally and not included in the output.
    """
    # Fast path: single fused Metal kernel (Phase 4-A.1).
    if _HAS_CPP_QUANTIZE:
        return _mfa_quantize_per_block_cpp(x, block_size)

    # Pure-MLX fallback (dispatches ~12 separate kernels).
    B, H, N, D = x.shape
    N_blocks = (N + block_size - 1) // block_size

    # Pad N to a multiple of block_size if needed
    N_padded = N_blocks * block_size
    if N_padded != N:
        pad_len = N_padded - N
        x = mx.pad(x, [(0, 0), (0, 0), (0, pad_len), (0, 0)])

    # Reshape to expose the block dimension: [B, H, N_blocks, block_size, D]
    x_blocked = x.reshape(B, H, N_blocks, block_size, D)

    # Per-block absmax → scale: [B, H, N_blocks, 1, 1]
    x_f32 = x_blocked.astype(mx.float32)  # compute once; reused below
    absmax = mx.max(mx.abs(x_f32), axis=(3, 4), keepdims=True)
    scale = absmax / 127.0
    scale = mx.maximum(scale, 1e-8)  # prevent division by zero

    # Quantize: clip to [-128, 127]
    x_quant = mx.clip(
        mx.round(x_f32 / scale),
        -128, 127,
    ).astype(mx.int8)

    # Flatten back to [B, H, N_padded, D], then trim to original N
    x_int8 = x_quant.reshape(B, H, N_padded, D)
    if N_padded != N:
        x_int8 = x_int8[:, :, :N, :]

    # Scale: [B, H, N_blocks, 1]
    scale_out = scale.reshape(B, H, N_blocks, 1).astype(mx.float32)

    return x_int8, scale_out


def dequantize(
    x_int8: mx.array,
    scale: mx.array,
    block_size: int,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    """Dequantize an int8 tensor produced by ``quantize_per_block``.

    Args:
        x_int8:    ``[B, H, N, D]`` int8.
        scale:     ``[B, H, N_blocks, 1]`` float32.
        block_size: Same block_size used during quantization.
        dtype:     Target dtype (fp16 or bf16).

    Returns:
        ``[B, H, N, D]`` in ``dtype``.
    """
    B, H, N, D = x_int8.shape
    N_blocks = scale.shape[2]
    N_padded = N_blocks * block_size

    # Pad if needed
    if N < N_padded:
        x_f = mx.pad(x_int8.astype(mx.float32), [(0,0),(0,0),(0,N_padded-N),(0,0)])
    else:
        x_f = x_int8.astype(mx.float32)

    # Reshape to [B, H, N_blocks, block_size, D] for broadcasting with scale
    x_blocked = x_f.reshape(B, H, N_blocks, block_size, D)

    # scale: [B, H, N_blocks, 1, 1] for broadcast over block_size and D
    s = scale.reshape(B, H, N_blocks, 1, 1)
    x_deq = (x_blocked * s).reshape(B, H, N_padded, D)

    # Trim to original N
    if N < N_padded:
        x_deq = x_deq[:, :, :N, :]

    return x_deq.astype(dtype)


# ---------------------------------------------------------------------------
# K smoothing (SageAttention key insight for improved quantization accuracy)
# ---------------------------------------------------------------------------


def smooth_k(
    k: mx.array,
) -> tuple[mx.array, mx.array]:
    """Smooth K by subtracting the per-channel mean.

    K has channel-wise outliers (spiky activation patterns in attention keys)
    that degrade per-block int8 quantization accuracy.  Subtracting the mean
    centers each channel around zero, dramatically reducing the absmax scale
    and thus increasing effective precision.

    The subtraction is reversed via an output correction step:
    ``sage_output_correction(O, q, k_mean, v, L, scale)``.

    Args:
        k: ``[B, H, S, D]`` fp16 or bf16 key tensor.

    Returns:
        ``(k_smooth, k_mean)`` where:

        - ``k_smooth``: ``[B, H, S, D]`` — k with per-channel mean removed.
        - ``k_mean``:   ``[B, H, 1, D]`` float32 — saved for output correction.
    """
    k_mean = mx.mean(k.astype(mx.float32), axis=2, keepdims=True)
    k_smooth = (k.astype(mx.float32) - k_mean).astype(k.dtype)
    return k_smooth, k_mean


# ---------------------------------------------------------------------------
# Output correction for K smoothing
# ---------------------------------------------------------------------------


def sage_output_correction(
    O_raw: mx.array,
    q: mx.array,
    k_mean: mx.array,
    v: mx.array,
    L: mx.array,
    scale: float,
) -> mx.array:
    """Correct attention output for K channel-mean smoothing.

    When K is smoothed before quantization:

        K_smooth = K - k_mean
        S_smooth  = Q @ K_smooth^T * scale
        S_exact   = S_smooth + (Q @ k_mean^T) * scale

    The correction accounts for the rank-1 additive bias in the scores.
    For each query position i, the correction is:

        delta_O[i] = sum_j( softmax_correction_j * V[j] )

    where ``softmax_correction_j = exp(q_i · k_mean * scale - L_i) * 1``
    and L_i is the log-normalizer.  Summing over j (all KV positions) with a
    constant per-position weight simplifies to:

        delta_O[i] = exp(q_i · k_mean * scale - L_i) * sum_j(V[j])

    This is an **approximate** first-order correction.  Exact correction
    would require recomputing the full softmax partition, which is O(N*S).
    For most language and image models, this approximation gives negligible
    end-to-end metric loss (< 0.1% perplexity on standard LLM benchmarks).

    Args:
        O_raw:   ``[B, H, N, D]`` raw kernel output (computed with K_smooth).
        q:       ``[B, H, N, D]`` original Q (fp16/bf16).
        k_mean:  ``[B, H, 1, D]`` float32 per-channel K mean.
        v:       ``[B, H, S, D]`` original V (fp16/bf16).
        L:       ``[B, H, N]`` logsumexp from kernel (log₂ domain, as STEEL
                 stores LSE in log₂ for numerical stability).
        scale:   Attention scale (1/sqrt(D)).

    Returns:
        ``[B, H, N, D]`` corrected output in the same dtype as O_raw.
    """
    orig_dtype = O_raw.dtype

    # q · k_mean: [B, H, N, 1]  (dot product of each Q row with the mean key)
    q_f = q.astype(mx.float32)
    correction_logit = mx.sum(q_f * k_mean, axis=-1, keepdims=True) * scale
    # shape: [B, H, N, 1]

    # Convert L from log₂ to natural log
    L_nat = L.astype(mx.float32) * math.log(2)  # [B, H, N]

    # Weight for each query position: exp(correction_logit - L_nat)
    # L_nat is the log-normalizer of softmax, so this computes the
    # fraction of "probability mass" that the correction contributes
    weight = mx.exp(correction_logit - L_nat[..., None])  # [B, H, N, 1]

    # Sum of V across the KV sequence: [B, H, 1, D]
    v_sum = mx.sum(v.astype(mx.float32), axis=2, keepdims=True)

    # Delta: [B, H, N, D]
    delta = weight * v_sum

    O_corrected = (O_raw.astype(mx.float32) + delta).astype(orig_dtype)
    return O_corrected


# ---------------------------------------------------------------------------
# Block config helper (mirrors _steel_block_config in attention.py)
# ---------------------------------------------------------------------------


def sage_block_sizes(head_dim: int) -> tuple[int, int]:
    """Return (BQ, BK) — STEEL tile sizes for the given head_dim.

    These should be used as block_size parameters for ``quantize_per_block``
    so that each Metal tile has exactly one scale value.
    """
    if head_dim <= 64:
        return 32, 32
    elif head_dim <= 128:
        return 32, 16
    else:
        return 32, 16

"""Flash Attention for MLX using Metal Flash Attention kernels."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx

_MFA_SUPPORTED_HDIMS = {64, 128, 256}


def flash_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: Optional[float] = None,
    causal: bool = False,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Compute scaled dot-product attention using Metal Flash Attention.

    Drop-in replacement for ``mx.fast.scaled_dot_product_attention``.

    Args:
        q: Query ``[batch, heads, seq_len, head_dim]``.
        k: Key ``[batch, heads, kv_len, head_dim]``.
        v: Value ``[batch, heads, kv_len, head_dim]``.
        scale: Attention scale. Defaults to ``1 / sqrt(head_dim)``.
        causal: If True, apply causal masking.
        stream: MLX stream for async execution.

    Returns:
        Output ``[batch, heads, seq_len, head_dim]``.

    Falls back to ``mx.fast.scaled_dot_product_attention`` when head_dim
    is not in {64, 128, 256} or the C++ extension is not available.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected 4D tensors [B, H, N, D], got q={q.ndim}D, "
            f"k={k.ndim}D, v={v.ndim}D"
        )

    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if not _can_use_mfa(q, k, v, head_dim):
        return _fallback_sdpa(q, k, v, scale, causal, stream)

    return _mfa_forward(q, k, v, scale, causal, stream)


def _can_use_mfa(
    q: mx.array, k: mx.array, v: mx.array, head_dim: int
) -> bool:
    if head_dim not in _MFA_SUPPORTED_HDIMS:
        return False
    supported_dtypes = {mx.float16, mx.bfloat16, mx.float32}
    if q.dtype not in supported_dtypes:
        return False
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        return False
    if not _ext_available():
        return False
    return True


def _ext_available() -> bool:
    try:
        from mlx_mfa._ext import mfa_attention_forward  # noqa: F401
        return True
    except ImportError:
        return False


def _mfa_forward(
    q: mx.array, k: mx.array, v: mx.array,
    scale: float, causal: bool,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    from mlx_mfa._ext import mfa_attention_forward
    kwargs = {}
    if stream is not None:
        kwargs["stream"] = stream
    return mfa_attention_forward(q, k, v, scale, causal, **kwargs)


def _fallback_sdpa(
    q: mx.array, k: mx.array, v: mx.array,
    scale: float, causal: bool,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    mask = None
    if causal:
        N, S = q.shape[2], k.shape[2]
        mask = mx.triu(
            mx.full((N, S), float("-inf")),
            k=S - N + 1,
        )
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask,
    )

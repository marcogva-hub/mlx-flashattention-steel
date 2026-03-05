"""mlx-mfa: Metal Flash Attention for MLX.

High-performance FlashAttention on Apple Silicon, based on the Metal Flash
Attention kernels from philipturner/metal-flash-attention (ported to C++ by
liuliu/ccv for production use in Draw Things).

Quick start::

    from mlx_mfa import flash_attention

    # Drop-in replacement for mx.fast.scaled_dot_product_attention
    out = flash_attention(q, k, v, scale=None, causal=False)

Supported configurations:
    - head_dim: 64, 128, 256
    - dtype: float16, bfloat16, float32
    - Shapes: [batch, heads, seq_len, head_dim] (BHND)
    - Causal and non-causal attention
    - Full autograd support (dQ, dK, dV)

When the C++ extension is unavailable (e.g., during CI without a Metal GPU),
all functions fall back to ``mx.fast.scaled_dot_product_attention``.
"""

__version__ = "0.6.0"

from mlx_mfa.attention import (
    flash_attention,
    flash_attention_rope,
    flash_attention_sparse,
    flash_attention_varlen,
    make_causal_block_mask,
    make_sliding_window_mask,
    make_rope_3d_tables,
    is_mfa_available,
    get_device_info,
    get_supported_configs,
)

from mlx_mfa.masks import (
    make_spatial_2d_mask,
    make_spatial_3d_mask,
    make_topk_spatial_mask,
    make_segment_mask,
    make_causal_segment_mask,
    make_adaptive_window_mask,
)

__all__ = [
    # Core attention
    "flash_attention",
    "flash_attention_rope",
    "flash_attention_sparse",
    "flash_attention_varlen",
    # Mask construction
    "make_causal_block_mask",
    "make_sliding_window_mask",
    "make_spatial_2d_mask",
    "make_spatial_3d_mask",
    "make_topk_spatial_mask",
    "make_segment_mask",
    "make_causal_segment_mask",
    "make_adaptive_window_mask",
    # RoPE helpers
    "make_rope_3d_tables",
    # Utilities
    "is_mfa_available",
    "get_device_info",
    "get_supported_configs",
    "__version__",
]

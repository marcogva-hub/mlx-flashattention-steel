"""mlx-mfa: Metal Flash Attention for MLX.

High-performance FlashAttention on Apple Silicon, based on
philipturner/metal-flash-attention kernels (via liuliu/ccv C++ port).

Usage:
    from mlx_mfa import flash_attention

    # Drop-in replacement for mx.fast.scaled_dot_product_attention
    out = flash_attention(q, k, v, scale=None, causal=False)
"""

__version__ = "0.1.0"

from mlx_mfa.attention import flash_attention

__all__ = ["flash_attention", "__version__"]

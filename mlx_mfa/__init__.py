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
    - Softcap, ALiBi, dropout, attention weight return (v0.8.0)
    - STEEL varlen kernel, paged KV cache, packed QKV/KV layouts (v0.9.0)
    - STEEL native backward kernels for f16/bf16 (v0.9.0, 2-3× speedup)

When the C++ extension is unavailable (e.g., during CI without a Metal GPU),
all functions fall back to ``mx.fast.scaled_dot_product_attention``.
"""

__version__ = "0.9.3"

from mlx_mfa.attention import (
    flash_attention,
    flash_attention_rope,
    flash_attention_sparse,
    flash_attention_varlen,
    flash_attention_with_kv_cache,
    flash_attention_kvcache,
    flash_attention_kvcache_rope_append,
    flash_attention_paged,
    flash_attention_qkv_packed,
    flash_attention_kv_packed,
    flash_attention_varlen_qkv_packed,
    flash_attention_varlen_kv_packed,
    PagedKVCache,
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
    make_lcsa_mask,
    make_axial_spatial_mask,
    make_axial_temporal_mask,
    make_dilated_temporal_mask,
    make_sink_window_mask,
    make_reference_frame_mask,
    make_cross_stream_mask,
)

__all__ = [
    # Core attention
    "flash_attention",
    "flash_attention_rope",
    "flash_attention_sparse",
    "flash_attention_varlen",
    "flash_attention_with_kv_cache",
    "flash_attention_kvcache",
    "flash_attention_kvcache_rope_append",
    "flash_attention_paged",
    "flash_attention_qkv_packed",
    "flash_attention_kv_packed",
    "flash_attention_varlen_qkv_packed",
    "flash_attention_varlen_kv_packed",
    "PagedKVCache",
    # Mask construction
    "make_causal_block_mask",
    "make_sliding_window_mask",
    "make_spatial_2d_mask",
    "make_spatial_3d_mask",
    "make_topk_spatial_mask",
    "make_segment_mask",
    "make_causal_segment_mask",
    "make_adaptive_window_mask",
    "make_lcsa_mask",
    "make_axial_spatial_mask",
    "make_axial_temporal_mask",
    "make_dilated_temporal_mask",
    "make_sink_window_mask",
    "make_reference_frame_mask",
    "make_cross_stream_mask",
    # RoPE helpers
    "make_rope_3d_tables",
    # Utilities
    "is_mfa_available",
    "get_device_info",
    "get_supported_configs",
    "__version__",
]

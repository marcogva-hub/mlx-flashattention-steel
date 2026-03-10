"""mlx-mfa: Metal Flash Attention for MLX.

High-performance FlashAttention on Apple Silicon, based on the Metal Flash
Attention kernels from philipturner/metal-flash-attention (ported to C++ by
liuliu/ccv for production use in Draw Things).

Quick start::

    from mlx_mfa import flash_attention

    # Drop-in replacement for mx.fast.scaled_dot_product_attention
    out = flash_attention(q, k, v, scale=None, causal=False)

Supported configurations:
    - head_dim: 64, 128, 256, 512
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

__version__ = "2.2.0"


def _check_abi() -> None:
    """Warn if the compiled extension was built against a different MLX major.minor."""
    try:
        import importlib
        _ext = importlib.import_module("mlx_mfa._ext")
        build_ver = _ext._mlx_build_version()
        if build_ver == "unknown":
            return
        import mlx.core
        runtime_ver = mlx.core.__version__
        # Compare major.minor only — patch releases are ABI-compatible.
        bv = tuple(int(x) for x in build_ver.split(".")[:2])
        rv = tuple(int(x) for x in runtime_ver.split(".")[:2])
        if bv != rv:
            import warnings
            warnings.warn(
                f"mlx-mfa was compiled against MLX {build_ver} but the installed "
                f"MLX is {runtime_ver}.  Rebuild the extension to avoid crashes:\n"
                "  pip install --no-build-isolation -e .",
                RuntimeWarning,
                stacklevel=2,
            )
    except Exception:
        pass


_check_abi()

from mlx_mfa.attention import (
    flash_attention,
    flash_attention_rope,
    flash_attention_rope_unified,
    flash_attention_sparse,
    flash_attention_varlen,
    flash_attention_kvcache,
    flash_attention_kvcache_rope_append,
    flash_attention_paged,
    flash_attention_qkv_packed,
    flash_attention_kv_packed,
    flash_attention_varlen_qkv_packed,
    flash_attention_varlen_kv_packed,
    KVCacheProtocol,
    DenseKVCache,
    PagedKVCache,
    make_causal_block_mask,
    make_sliding_window_mask,
    make_rope_3d_tables,
    is_mfa_available,
    get_device_info,
    get_supported_configs,
    warmup_kernels,
    DispatchPolicy,
    # Track JD: LLM inference helpers
    flash_attention_speculative_verify,
    make_shared_prefix_cache,
    flash_attention_splitfuse,
    # Track KC: SageAttention
    sage_attention,
    # Track LA: SageAttention KV-cache decode
    sage_attention_kvcache,
)

from mlx_mfa.quantize import (
    quantize_per_block,
    dequantize,
    smooth_k,
    sage_output_correction,
    sage_block_sizes,
)

# Track LC/Phase 2: InferenceContext + PagedInferenceContext + SageInferenceContext
from mlx_mfa.inference import InferenceContext, PagedInferenceContext, SageInferenceContext

from mlx_mfa.dispatch_policy import calibrate_dispatch

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
    "flash_attention_rope_unified",
    "flash_attention_sparse",
    "flash_attention_varlen",
    "flash_attention_kvcache",
    "flash_attention_kvcache_rope_append",
    "flash_attention_paged",
    "flash_attention_qkv_packed",
    "flash_attention_kv_packed",
    "flash_attention_varlen_qkv_packed",
    "flash_attention_varlen_kv_packed",
    "KVCacheProtocol",
    "DenseKVCache",
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
    "warmup_kernels",
    "DispatchPolicy",
    "calibrate_dispatch",
    # LLM inference helpers (Track JD)
    "flash_attention_speculative_verify",
    "make_shared_prefix_cache",
    "flash_attention_splitfuse",
    # SageAttention (Track KC / LA)
    "sage_attention",
    "sage_attention_kvcache",
    # Quantization utilities (Track KA/KC)
    "quantize_per_block",
    "dequantize",
    "smooth_k",
    "sage_output_correction",
    "sage_block_sizes",
    # InferenceContext lifecycle (Track LC / Phase 2 / LA)
    "InferenceContext",
    "PagedInferenceContext",
    "SageInferenceContext",
    "__version__",
]

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
    - Full autograd support via SDPA VJP fallback (default)
    - STEEL V2 production routing for dense causal D=64/128
    - D=256 narrow benchmark-backed dense causal promotion (M1/M2 f16)
    - D=512 dense remains SDPA-default in auto mode
    - SageAttention as a specialized decode backend
    - Softcap, ALiBi, window, sparse, varlen, and packed QKV/KV layouts

When the C++ extension is unavailable (e.g., during CI without a Metal GPU),
all functions fall back to ``mx.fast.scaled_dot_product_attention``.
"""

__version__ = "2.51.0"


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

# ── Eager imports — core API (always needed) ────────────────────────────────

from mlx_mfa.attention import (
    flash_attention,
    flash_attention_rope,
    flash_attention_rope_unified,
    flash_attention_sparse,
    flash_attention_gna,
    flash_attention_topk,
    flash_attention_varlen,
    flash_attention_kvcache,
    flash_attention_kvcache_rope_append,
    flash_attention_paged,
    flash_attention_paged_varlen,
    flash_attention_paged_varlen_turboquant,
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
    flash_attention_speculative_verify_paged,
    make_shared_prefix_cache,
    flash_attention_splitfuse,
    # Track KC: SageAttention
    sage_attention,
    # Track LA: SageAttention KV-cache decode
    sage_attention_kvcache,
    # CP6: pre-quantized SageAttention + QuantizedKVCache
    sage_attention_prequantized,
    QuantizedKVCache,
)

from mlx_mfa.quantize import (
    quantize_per_block,
    dequantize,
    smooth_k,
    sage_output_correction,
    sage_block_sizes,
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
    make_gna_mask,
    make_diagonal_mask,
    make_strided_mask,
    make_temporal_group_mask,
    make_temporal_distance_bias,
    temporal_distance_bias_to_mask,
)

from mlx_mfa.dispatch_policy import calibrate_dispatch, _load_calibrated_kernel_config, _invalidate_cached_env
from mlx_mfa.compile_metallib import compile_metallib

# Apply any calibrated kernel config (BK selection etc.) before first kernel dispatch.
_load_calibrated_kernel_config()

# Re-export C++ invalidation for benchmarks/advanced users.
def _invalidate_env_config():
    """Re-read all cached MFA_* env vars in the C++ singleton.

    Call after os.environ mutations of cached vars (MFA_V2_FORCE_BK, etc.).
    Dispatch gates (MFA_ENABLE_V3, etc.) are live-read and don't need this.
    """
    try:
        from mlx_mfa._ext import _invalidate_env_config as _inv
        _inv()
    except (ImportError, AttributeError):
        pass

# ── Lazy imports — serving/runtime/compression (loaded on first access) ─────
#
# These submodules pull in heavier dependencies (inference state machines,
# cache abstractions, quantization utilities) that most users don't need
# for basic flash_attention() calls.  Deferring them avoids ~40% of the
# import-time work for the common case.
#
# __all__ is unchanged — all names remain public.  Only the TIMING of
# the import changes, not the API surface.

_LAZY_IMPORTS: dict[str, str] = {
    # mlx_mfa.inference
    "InferenceContext": "mlx_mfa.inference",
    "PagedInferenceContext": "mlx_mfa.inference",
    "SageInferenceContext": "mlx_mfa.inference",
    "TurboQuantPagedInferenceContext": "mlx_mfa.inference",
    "create_inference_context": "mlx_mfa.inference",
    # mlx_mfa.runtime
    "DecodeRuntime": "mlx_mfa.runtime",
    "create_decode_runtime": "mlx_mfa.runtime",
    # mlx_mfa.kv_cache
    "KVCacheCapabilities": "mlx_mfa.kv_cache",
    "KVCacheOperationUnsupported": "mlx_mfa.kv_cache",
    "KVCacheAdapter": "mlx_mfa.kv_cache",
    "DenseKVCacheAdapter": "mlx_mfa.kv_cache",
    "PagedKVCacheAdapter": "mlx_mfa.kv_cache",
    "QuantizedKVCacheAdapter": "mlx_mfa.kv_cache",
    "HybridKVCache": "mlx_mfa.kv_cache",
    "HybridKVCacheAdapter": "mlx_mfa.kv_cache",
    "adapt_kv_cache": "mlx_mfa.kv_cache",
    "resolve_context_cache": "mlx_mfa.kv_cache",
    "resolve_context_cache_adapter": "mlx_mfa.kv_cache",
    # mlx_mfa.external_cache
    "ExternalKVCacheCapabilities": "mlx_mfa.external_cache",
    "ExternalKVCacheAdapter": "mlx_mfa.external_cache",
    "LocalHostKVStoreAdapter": "mlx_mfa.external_cache",
    # mlx_mfa.turboquant
    "turboquant_compress": "mlx_mfa.turboquant",
    "turboquant_decompress": "mlx_mfa.turboquant",
    "TurboQuantKVCache": "mlx_mfa.turboquant",
    "pack_k_for_metal": "mlx_mfa.turboquant",
    "build_tq_paged_k_pool": "mlx_mfa.turboquant",
    "pack_v_for_metal": "mlx_mfa.turboquant",
    "build_tq_paged_v_pool": "mlx_mfa.turboquant",
    "pack_3bit_optimal": "mlx_mfa.turboquant",
    "unpack_3bit_optimal": "mlx_mfa.turboquant",
    # mlx_mfa.svdquant
    "SVDQuantLinear": "mlx_mfa.svdquant",
    "quantize_model": "mlx_mfa.svdquant",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        obj = getattr(module, name)
        globals()[name] = obj  # cache for subsequent accesses
        return obj
    raise AttributeError(f"module 'mlx_mfa' has no attribute {name!r}")


# ── Auto-hook installation (Sprint U / v2.36.0) ─────────────────────────────
#
# Per docs/RELEASE_PHILOSOPHY.md auto-default principle: import-time hook
# installation that auto-routes eligible mx.conv_general calls through
# conv3d_nax_forward on M5+ hardware. Set MFA_DISABLE_AUTO_HOOKS=1 to skip.

from mlx_mfa._auto_hooks import (
    install_hooks as _install_hooks,
    uninstall_hooks as _uninstall_hooks,
    hooks_status as _hooks_status,
    get_hook_stats,
    reset_hook_stats,
)


def enable():
    """Manually (re-)install mlx-mfa auto-hooks.

    Hooks are automatically installed at import time unless
    MFA_DISABLE_AUTO_HOOKS=1 is set in the environment. This function
    is idempotent (multiple calls are no-ops) and is primarily useful
    after a manual disable() or for explicit control flows.

    Returns True if hooks were newly installed, False if already installed
    or disabled via env var.
    """
    return _install_hooks()


def disable():
    """Uninstall mlx-mfa auto-hooks. Restores vanilla MLX behavior.

    Useful for benchmarking (A/B comparison vs mlx-mfa optimizations).
    Idempotent. Existing explicit API calls (flash_attention*, etc.) and
    patchers (patch_seedvr2_vae, etc.) remain available regardless.

    Returns True if hooks were uninstalled, False if not installed.
    """
    return _uninstall_hooks()


def hooks_status():
    """Return a dict describing current auto-hook state.

    Keys: installed, log (install/uninstall events), m5_plus (whether
    auto-routing eligibility check would pass), auto_hooks_disabled_env
    (whether MFA_DISABLE_AUTO_HOOKS=1 is set).
    """
    return _hooks_status()


def diagnostics() -> dict:
    """Return a structured runtime-diagnostics dict for support / debug.

    Aggregates state that otherwise requires querying multiple modules
    (auto-hooks status, NAX detection, env vars, carve-out eligibility,
    version) into a single call.  Useful for:

    - User bug reports: paste output of `mlx_mfa.diagnostics()` so the
      issue can be triaged against the full runtime state
    - CI checks: `assert mlx_mfa.diagnostics()["auto_hooks"]["installed"]`
    - /mlx-mfa-release-audit Check 3 (auto-default principle): the
      runner can call diagnostics() instead of probing internal
      attributes by name

    Output structure (stable contract; safe to log / serialize as JSON):
        {
            "version": "X.Y.Z",
            "mlx_version": "..." or None,
            "platform": {"is_m3_plus": bool, "is_m5_plus": bool,
                         "has_nax": bool},
            "auto_hooks": {"installed": bool, "log": [...]},
            "active_env_vars": {"MFA_*": "value", ...},
            "carveout_eligibility": {
                "<shape_label>": bool, ...
            },
        }
    """
    import os
    from mlx_mfa.attention import _get_is_m3_plus_cached, _get_has_nax_cached
    from mlx_mfa.dispatch_policy import (
        _v34_backward_carveout,
        _dispatch_dtype_key,
    )

    # MLX version (best-effort; older mlx may not have __version__)
    try:
        import mlx.core
        _mlx_ver = getattr(mlx.core, "__version__", None)
    except Exception:
        _mlx_ver = None

    # Platform capability snapshot.  has_nax IS the M5+ indicator
    # (NAX is M5+ only); we don't ship is_m5_plus as a separate field
    # to avoid the tautology (would always equal has_nax).
    _is_m3 = _get_is_m3_plus_cached()
    _has_nax = _get_has_nax_cached()

    # Auto-hooks state
    _hs = _hooks_status()

    # Active MFA_* env vars (snapshot at call time)
    _env = {k: v for k, v in os.environ.items() if k.startswith("MFA_")}

    # Carve-out eligibility for representative shapes.  Calls the
    # public dispatch_policy hook so the answer matches what
    # flash_attention() would actually decide.
    import mlx.core as mx
    fp16 = _dispatch_dtype_key(mx.float16)
    bf16 = _dispatch_dtype_key(mx.bfloat16)
    _carveout = {
        "d64_qL4096_fp16_noncausal": _v34_backward_carveout(
            head_dim=64, seq_len=4096, causal=False, dtype_key=fp16),
        "d64_qL8192_fp16_noncausal": _v34_backward_carveout(
            head_dim=64, seq_len=8192, causal=False, dtype_key=fp16),
        "d64_qL4096_bf16_noncausal": _v34_backward_carveout(
            head_dim=64, seq_len=4096, causal=False, dtype_key=bf16),
        "d64_qL2048_fp16_noncausal": _v34_backward_carveout(
            head_dim=64, seq_len=2048, causal=False, dtype_key=fp16),
        "d64_qL4096_fp16_causal": _v34_backward_carveout(
            head_dim=64, seq_len=4096, causal=True, dtype_key=fp16),
        # v2.50 Prompt 5b Section D: D=128 entries (broadened carve-out)
        "d128_qL2048_fp16_noncausal": _v34_backward_carveout(
            head_dim=128, seq_len=2048, causal=False, dtype_key=fp16),
        "d128_qL4096_fp16_noncausal": _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=False, dtype_key=fp16),
        "d128_qL8192_fp16_noncausal": _v34_backward_carveout(
            head_dim=128, seq_len=8192, causal=False, dtype_key=fp16),
        "d128_qL4096_fp16_causal": _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=True, dtype_key=fp16),
        "d128_qL4096_bf16_noncausal": _v34_backward_carveout(
            head_dim=128, seq_len=4096, causal=False, dtype_key=bf16),
    }

    return {
        "version": __version__,
        "mlx_version": _mlx_ver,
        "platform": {
            "is_m3_plus": _is_m3,
            "has_nax": _has_nax,  # M5+ indicator; NAX is M5+-only
        },
        "auto_hooks": _hs,
        "active_env_vars": _env,
        "carveout_eligibility": _carveout,
    }


# Auto-install at import unless MFA_DISABLE_AUTO_HOOKS=1
_install_hooks()


__all__ = [
    # Sprint U / v2.36.0 auto-hook public API
    "enable",
    "disable",
    "hooks_status",
    # v2.50.1 Prompt 5g Phase C — hook telemetry (Pattern #8 prevention)
    "get_hook_stats",
    "reset_hook_stats",
    "diagnostics",
    # Core attention
    "flash_attention",
    "flash_attention_rope",
    "flash_attention_rope_unified",
    "flash_attention_sparse",
    "flash_attention_gna",
    "flash_attention_topk",
    "flash_attention_varlen",
    "flash_attention_kvcache",
    "flash_attention_kvcache_rope_append",
    "flash_attention_paged",
    "flash_attention_paged_varlen",
    "flash_attention_paged_varlen_turboquant",
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
    "make_gna_mask",
    "make_diagonal_mask",
    "make_strided_mask",
    "make_temporal_group_mask",
    "make_temporal_distance_bias",
    "temporal_distance_bias_to_mask",
    # RoPE helpers
    "make_rope_3d_tables",
    # Utilities
    "is_mfa_available",
    "get_device_info",
    "get_supported_configs",
    "warmup_kernels",
    "compile_metallib",
    "DispatchPolicy",
    "calibrate_dispatch",
    # LLM inference helpers (Track JD)
    "flash_attention_speculative_verify",
    "flash_attention_speculative_verify_paged",
    "make_shared_prefix_cache",
    "flash_attention_splitfuse",
    # SageAttention (Track KC / LA / CP6)
    "sage_attention",
    "sage_attention_kvcache",
    "sage_attention_prequantized",
    "QuantizedKVCache",
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
    "TurboQuantPagedInferenceContext",
    "create_inference_context",
    "DecodeRuntime",
    "create_decode_runtime",
    # Cache abstraction helpers
    "KVCacheCapabilities",
    "KVCacheOperationUnsupported",
    "KVCacheAdapter",
    "DenseKVCacheAdapter",
    "PagedKVCacheAdapter",
    "QuantizedKVCacheAdapter",
    "HybridKVCache",
    "HybridKVCacheAdapter",
    "adapt_kv_cache",
    "resolve_context_cache",
    "resolve_context_cache_adapter",
    "ExternalKVCacheCapabilities",
    "ExternalKVCacheAdapter",
    "LocalHostKVStoreAdapter",
    # TurboQuant KV cache compression
    "turboquant_compress",
    "turboquant_decompress",
    "TurboQuantKVCache",
    "pack_k_for_metal",
    "build_tq_paged_k_pool",
    "pack_v_for_metal",
    "build_tq_paged_v_pool",
    "pack_3bit_optimal",
    "unpack_3bit_optimal",
    # SVDQuant W4A16 quantization
    "SVDQuantLinear",
    "quantize_model",
    "__version__",
]

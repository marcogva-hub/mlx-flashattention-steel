"""mlx-lm integration — drop-in STEEL attention for LLM inference.

Monkey-patches ``mlx_lm.models.base.scaled_dot_product_attention`` so that
every model loaded by mlx-lm automatically uses the STEEL kernel for causal
self-attention when the configuration is supported.

Usage::

    from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
    patch_mlx_lm()

    from mlx_lm import load, generate
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    generate(model, tokenizer, prompt="Hello", verbose=True)

Call ``unpatch_mlx_lm()`` to restore the original implementation.

Supported configs (use STEEL):
    - Extension available (MFA C++ extension compiled)
    - head_dim in {64, 128, 256, 512}
    - dtype float16 or bfloat16
    - mask == "causal" or mask is None with single-token decode
    - Quantized KV cache (cache.bits set): K/V dequantized before STEEL
    - Sliding window from cache.max_kv_window (Track JE)
    - GQA (H_q != H_kv) — STEEL handles natively
    - No attention sinks (sinks=None)

All other cases fall back to the original mlx_lm SDPA transparently.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_mfa import get_supported_configs, is_mfa_available
from mlx_mfa.attention import _mfa_forward as _steel_dispatch

# The original mlx_lm SDPA function, saved at patch time.
_original_sdpa = None
_SUPPORTED_HDIMS: set[int] = set()
_SUPPORTED_DTYPES: frozenset = frozenset()

# Whether to emit per-call dispatch log lines (set by patch_mlx_lm).
_verbose_dispatch: bool = False

# --------------------------------------------------------------------------- #
# Per-session call statistics — plain int counters (faster than dict lookup)   #
# --------------------------------------------------------------------------- #
_stat_forward_calls: int = 0
_stat_steel_calls: int = 0
_stat_fallback_calls: int = 0
_stat_gqa_calls: int = 0
_stat_sliding_window_calls: int = 0

# Public dict alias (lazy-built by get_stats(); not mutated in the hot path)
_stats: dict = {}

# --------------------------------------------------------------------------- #
# Known model configuration hints (Track JE)                                   #
# --------------------------------------------------------------------------- #
#: Hints for popular model families. Keys are lowercase substrings of model
#: names; values are dicts with optional keys ``head_dim`` (int) and
#: ``sliding_window`` (int, -1 = disabled).
KNOWN_MODEL_CONFIGS: dict[str, dict] = {
    "llama":    {"head_dim": 128},
    "mistral":  {"head_dim": 128, "sliding_window": 4096},
    "mixtral":  {"head_dim": 128, "sliding_window": 4096},
    "gemma":    {"head_dim": 256},
    "gemma2":   {"head_dim": 256, "sliding_window": 4096},
    "phi":      {"head_dim": 96},
    "phi3":     {"head_dim": 96},
    "qwen":     {"head_dim": 128},
    "qwen2":    {"head_dim": 128},
    "deepseek": {"head_dim": 128},
    "falcon":   {"head_dim": 64},
    "mpt":      {"head_dim": 64},
    "opt":      {"head_dim": 64},
    "gpt2":     {"head_dim": 64},
    "gpt-neo":  {"head_dim": 64},
    "bloom":    {"head_dim": 64},
    "yi":       {"head_dim": 128},
    "solar":    {"head_dim": 128},
    "internlm": {"head_dim": 128},
    "baichuan": {"head_dim": 128},
    "cohere":   {"head_dim": 128},
    "olmo":     {"head_dim": 64},
    "starcoder":{"head_dim": 64},
}


def _refresh_supported() -> None:
    """Populate cached supported configs from the extension."""
    global _SUPPORTED_HDIMS, _SUPPORTED_DTYPES
    cfg = get_supported_configs()
    _SUPPORTED_HDIMS = set(cfg.get("head_dims", []))
    _SUPPORTED_DTYPES = frozenset({mx.float16, mx.bfloat16})


def _reset_stats() -> None:
    global _stat_forward_calls, _stat_steel_calls, _stat_fallback_calls
    global _stat_gqa_calls, _stat_sliding_window_calls
    _stat_forward_calls = 0
    _stat_steel_calls = 0
    _stat_fallback_calls = 0
    _stat_gqa_calls = 0
    _stat_sliding_window_calls = 0


def _steel_sdpa(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    """Drop-in replacement for mlx_lm.models.base.scaled_dot_product_attention.

    Routes through the STEEL kernel when the configuration is supported.
    Falls back to the original mlx_lm SDPA otherwise.

    The ``cache`` parameter carries mlx_lm's KV cache object. When
    ``cache.bits`` is set (quantized cache), mlx-lm passes ``keys`` and
    ``values`` as ``(quantized_data, scales, biases)`` tuples.  We
    dequantize them to plain float arrays so STEEL can run, preserving the
    causal tile-skip speedup even for 4-bit models.
    """
    global _stat_forward_calls, _stat_steel_calls, _stat_fallback_calls
    global _stat_gqa_calls, _stat_sliding_window_calls
    _stat_forward_calls += 1

    # Attention sinks: always fall back (STEEL doesn't implement them).
    if sinks is not None:
        _stat_fallback_calls += 1
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Quantized KV cache: dequantize K/V then proceed to STEEL.
    # mlx-lm passes keys/values as (quantized_data, scales, biases) tuples
    # when cache.bits is set.  mx.dequantize returns float16/bf16 matching
    # the query dtype.
    # D.9: use getattr to avoid double attribute lookup (hasattr + attr access)
    _cache_bits = getattr(cache, "bits", None) if cache is not None else None
    if _cache_bits is not None:
        try:
            q_k_data, q_k_scales, q_k_biases = keys
            q_v_data, q_v_scales, q_v_biases = values
            keys = mx.dequantize(
                q_k_data, q_k_scales, q_k_biases,
                cache.group_size, _cache_bits,
                dtype=queries.dtype,
            )
            values = mx.dequantize(
                q_v_data, q_v_scales, q_v_biases,
                cache.group_size, _cache_bits,
                dtype=queries.dtype,
            )
        except Exception:
            # Unexpected format (e.g., future quantization modes): fall back.
            _stat_fallback_calls += 1
            return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    D = queries.shape[-1]
    dtype = queries.dtype

    # Only use STEEL for supported head_dims, dtypes, and extension.
    if D not in _SUPPORTED_HDIMS or dtype not in _SUPPORTED_DTYPES:
        if _verbose_dispatch:
            print(f"[mlx-mfa dispatch] fallback: D={D} not in {sorted(_SUPPORTED_HDIMS)}")
        _stat_fallback_calls += 1
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # GQA detection: track when H_q != H_kv.
    _is_gqa = queries.shape[1] != keys.shape[1]

    # Sliding-window from cache: mlx-lm cache objects may expose max_kv_window.
    # D.9: use getattr to avoid hasattr + access double lookup
    _window_left = -1
    _window_right = -1
    _mw = getattr(cache, "max_kv_window", None) if cache is not None else None
    if _mw is not None and _mw > 0:
        _window_left = _mw - 1
        _window_right = 0

    # Causal detection:
    #   - mask == "causal" (string): prefill with standard causal mask
    #   - mask is None: decode step — attend to all cached K/V (causal=False)
    #   - any other mask type: fall back (boolean array, etc.)
    if mask == "causal" or mask is None:
        _is_causal = (mask == "causal")
        _stat_steel_calls += 1
        if _is_gqa:
            _stat_gqa_calls += 1
        if _window_left >= 0:
            _stat_sliding_window_calls += 1
        if _verbose_dispatch:
            print(f"[mlx-mfa dispatch] STEEL {'causal' if _is_causal else 'no-mask'} "
                  f"D={D} {dtype} "
                  f"{'GQA ' if _is_gqa else ''}"
                  f"{'window=' + str(_window_left) if _window_left >= 0 else ''}")
        # D.4: call _mfa_forward directly — skips flash_attention() wrapper overhead
        # (backend validation, GQA expansion check, window tuple unpacking, etc.)
        return _steel_dispatch(
            queries, keys, values, scale, _is_causal,
            window_left=_window_left, window_right=_window_right,
        )

    # mask is an array (boolean, padding, etc.): fall back.
    if _verbose_dispatch:
        print(f"[mlx-mfa dispatch] fallback: mask type={type(mask).__name__}")
    _stat_fallback_calls += 1
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)


def patch_mlx_lm(verbose: bool = True, verbose_dispatch: bool = False) -> bool:
    """Monkey-patch mlx-lm to use STEEL attention.

    Parameters
    ----------
    verbose:
        If ``True`` (default), print a one-line confirmation message after a
        successful patch and a warning when the extension is unavailable.
        Set to ``False`` for silent operation (e.g., inside library code).
    verbose_dispatch:
        If ``True``, print a log line for every SDPA call dispatched through
        the patched function (steel vs fallback, D, dtype, GQA flag, window).
        Useful for debugging routing decisions.  Default ``False``.

    Returns
    -------
    bool
        ``True`` if patching succeeded, ``False`` if the MFA extension is
        unavailable (the original SDPA is left in place).

    Calling this function multiple times is idempotent — the patch is only
    applied once.  Call statistics are reset at each fresh patch.

    Example::

        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
        patch_mlx_lm(verbose_dispatch=True)  # log every dispatch

        from mlx_lm import load, generate
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        generate(model, tokenizer, prompt="Hello world", verbose=True)
    """
    global _original_sdpa, _verbose_dispatch

    if _original_sdpa is not None:
        # Already patched — idempotent.
        return True

    if not is_mfa_available():
        if verbose:
            print("[mlx-mfa] Warning: MFA extension not available, mlx-lm patch skipped")
        return False

    try:
        import mlx_lm.models.base as base_module
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is not installed. Install with: pip install mlx-lm"
        ) from exc

    _refresh_supported()
    _reset_stats()
    _verbose_dispatch = verbose_dispatch
    _original_sdpa = base_module.scaled_dot_product_attention
    base_module.scaled_dot_product_attention = _steel_sdpa
    if verbose:
        print(
            f"[mlx-mfa] Patched mlx-lm — STEEL kernel active for causal f16/bf16 "
            f"attention (head_dims={sorted(_SUPPORTED_HDIMS)})"
            + (" [verbose_dispatch=True]" if verbose_dispatch else "")
        )
    return True


def unpatch_mlx_lm() -> None:
    """Restore the original mlx-lm SDPA function.

    Safe to call even if :func:`patch_mlx_lm` was never called or failed.

    Example::

        from mlx_mfa.integrations.mlx_lm import unpatch_mlx_lm
        unpatch_mlx_lm()
    """
    global _original_sdpa

    if _original_sdpa is None:
        return

    try:
        import mlx_lm.models.base as base_module
        base_module.scaled_dot_product_attention = _original_sdpa
    except ImportError:
        pass

    _original_sdpa = None
    print("[mlx-mfa] Restored original mlx-lm SDPA")


def is_patched() -> bool:
    """Return ``True`` if mlx-lm has been patched."""
    return _original_sdpa is not None


def get_patch_stats() -> dict:
    """Return a snapshot of SDPA call statistics for the current patch session.

    Returns
    -------
    dict with keys:
        - ``forward_calls`` (int): total calls dispatched through the patched SDPA
        - ``steel_calls`` (int): calls routed to the STEEL kernel
        - ``fallback_calls`` (int): calls delegated to the original SDPA
        - ``steel_ratio`` (float): fraction of calls handled by STEEL (0.0–1.0)

    Statistics are reset to zero on each fresh :func:`patch_mlx_lm` call.
    Returns zeros when the patch is not active.

    Example::

        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm, get_patch_stats
        patch_mlx_lm()
        # ... run inference ...
        print(get_patch_stats())
        # {'forward_calls': 128, 'steel_calls': 120, 'fallback_calls': 8, 'steel_ratio': 0.9375}
    """
    total = _stat_forward_calls
    steel = _stat_steel_calls
    return {
        "forward_calls": total,
        "steel_calls": steel,
        "fallback_calls": _stat_fallback_calls,
        "gqa_calls": _stat_gqa_calls,
        "sliding_window_calls": _stat_sliding_window_calls,
        "steel_ratio": steel / total if total > 0 else 0.0,
    }


def check_model_compatibility(model_name: str) -> dict:
    """Heuristically check if a model family is compatible with STEEL attention.

    Does **not** load the model.  Uses the model name string to infer the
    head_dim and checks against the extension's supported configuration.

    Parameters
    ----------
    model_name:
        A Hugging Face repo ID or local path, e.g.
        ``"mlx-community/Llama-3.2-3B-Instruct-4bit"``.

    Returns
    -------
    dict with keys:
        - ``compatible`` (bool): ``True`` when STEEL can likely be used
        - ``reason`` (str): human-readable explanation
        - ``extension_available`` (bool)
        - ``supported_head_dims`` (list[int]): head_dims supported by the build
        - ``supported_dtypes`` (list[str]): ``["float16", "bfloat16"]``
        - ``notes`` (str): additional guidance

    Example::

        from mlx_mfa.integrations.mlx_lm import check_model_compatibility
        info = check_model_compatibility("mlx-community/Llama-3.2-3B-Instruct-4bit")
        print(info["compatible"], info["reason"])
    """
    ext_ok = is_mfa_available()
    if not ext_ok:
        return {
            "compatible": False,
            "reason": "MFA C++ extension not available — run `pip install mlx-mfa` or build from source",
            "extension_available": False,
            "supported_head_dims": [],
            "supported_dtypes": [],
            "notes": "Install the extension to enable STEEL attention.",
        }

    cfg = get_supported_configs()
    supported_dims: list[int] = sorted(cfg.get("head_dims", []))
    supported_dtypes = ["float16", "bfloat16"]

    if not supported_dims:
        return {
            "compatible": False,
            "reason": "No head_dims supported by current build",
            "extension_available": True,
            "supported_head_dims": [],
            "supported_dtypes": supported_dtypes,
            "notes": "Rebuild the extension for your device.",
        }

    # Quantized suffix detection — STEEL supports these via dequantize path
    name_lower = model_name.lower()
    is_quantized = any(
        q in name_lower for q in ("4bit", "8bit", "2bit", "3bit", "6bit", "awq", "gguf")
    )
    quant_note = " (quantized cache: STEEL uses dequantize path)" if is_quantized else ""

    # Check for known-incompatible architectures (non-standard attention)
    known_unsupported = ["mamba", "rwkv", "ssm", "s4", "hyena", "retnet"]
    for arch in known_unsupported:
        if arch in name_lower:
            return {
                "compatible": False,
                "reason": f"Architecture '{arch}' does not use scaled dot-product attention",
                "extension_available": True,
                "supported_head_dims": supported_dims,
                "supported_dtypes": supported_dtypes,
                "notes": "STEEL only applies to Transformer attention layers.",
            }

    return {
        "compatible": True,
        "reason": (
            f"Standard Transformer attention with head_dim in {supported_dims} "
            f"and float16/bfloat16 dtype will use STEEL{quant_note}"
        ),
        "extension_available": True,
        "supported_head_dims": supported_dims,
        "supported_dtypes": supported_dtypes,
        "notes": (
            "STEEL falls back to original SDPA for: array masks, attention sinks, "
            "unsupported head_dim or dtype."
        ),
    }

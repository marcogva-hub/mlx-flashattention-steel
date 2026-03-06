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
    - head_dim in {64, 128, 256}
    - dtype float16 or bfloat16
    - mask == "causal" or mask is None with single-token decode
    - Quantized KV cache (cache.bits set): K/V dequantized before STEEL
    - No attention sinks (sinks=None)

All other cases fall back to the original mlx_lm SDPA transparently.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_mfa import flash_attention, get_supported_configs, is_mfa_available

# The original mlx_lm SDPA function, saved at patch time.
_original_sdpa = None
_SUPPORTED_HDIMS: set[int] = set()
_SUPPORTED_DTYPES: frozenset = frozenset()

# --------------------------------------------------------------------------- #
# Per-session call statistics                                                   #
# --------------------------------------------------------------------------- #
_stats: dict = {
    "forward_calls": 0,   # total calls to _steel_sdpa
    "steel_calls": 0,     # calls routed to the STEEL kernel
    "fallback_calls": 0,  # calls delegated back to the original SDPA
}


def _refresh_supported() -> None:
    """Populate cached supported configs from the extension."""
    global _SUPPORTED_HDIMS, _SUPPORTED_DTYPES
    cfg = get_supported_configs()
    _SUPPORTED_HDIMS = set(cfg.get("head_dims", []))
    _SUPPORTED_DTYPES = frozenset({mx.float16, mx.bfloat16})


def _reset_stats() -> None:
    _stats["forward_calls"] = 0
    _stats["steel_calls"] = 0
    _stats["fallback_calls"] = 0


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
    _stats["forward_calls"] += 1

    # Attention sinks: always fall back (STEEL doesn't implement them).
    if sinks is not None:
        _stats["fallback_calls"] += 1
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Quantized KV cache: dequantize K/V then proceed to STEEL.
    # mlx-lm passes keys/values as (quantized_data, scales, biases) tuples
    # when cache.bits is set.  mx.dequantize returns float16/bf16 matching
    # the query dtype.
    if cache is not None and hasattr(cache, "bits") and cache.bits is not None:
        try:
            q_k_data, q_k_scales, q_k_biases = keys
            q_v_data, q_v_scales, q_v_biases = values
            keys = mx.dequantize(
                q_k_data, q_k_scales, q_k_biases,
                cache.group_size, cache.bits,
                dtype=queries.dtype,
            )
            values = mx.dequantize(
                q_v_data, q_v_scales, q_v_biases,
                cache.group_size, cache.bits,
                dtype=queries.dtype,
            )
        except Exception:
            # Unexpected format (e.g., future quantization modes): fall back.
            _stats["fallback_calls"] += 1
            return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    D = queries.shape[-1]
    dtype = queries.dtype

    # Only use STEEL for supported head_dims, dtypes, and extension.
    if D not in _SUPPORTED_HDIMS or dtype not in _SUPPORTED_DTYPES:
        _stats["fallback_calls"] += 1
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Causal detection:
    #   - mask == "causal" (string): prefill with standard causal mask
    #   - mask is None and queries.shape[-2] == 1: single-token decode
    #     (attend to all cached K/V — no causal mask needed, use causal=False)
    #   - any other mask type: fall back (boolean array, sliding window, etc.)
    if mask == "causal":
        # Prefill: STEEL causal is 1.6–2.1× faster than SDPA
        _stats["steel_calls"] += 1
        return flash_attention(queries, keys, values, scale=scale, causal=True)

    if mask is None:
        # Decode step (N_q=1) or full-attention prefill with no mask
        _stats["steel_calls"] += 1
        return flash_attention(queries, keys, values, scale=scale, causal=False)

    # mask is an array (boolean causal, sliding-window, padding, etc.):
    # fall back to original — STEEL uses its own causal logic, not additive masks.
    _stats["fallback_calls"] += 1
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)


def patch_mlx_lm(verbose: bool = True) -> bool:
    """Monkey-patch mlx-lm to use STEEL attention.

    Parameters
    ----------
    verbose:
        If ``True`` (default), print a one-line confirmation message after a
        successful patch and a warning when the extension is unavailable.
        Set to ``False`` for silent operation (e.g., inside library code).

    Returns
    -------
    bool
        ``True`` if patching succeeded, ``False`` if the MFA extension is
        unavailable (the original SDPA is left in place).

    Calling this function multiple times is idempotent — the patch is only
    applied once.  Call statistics are reset at each fresh patch.

    Example::

        from mlx_mfa.integrations.mlx_lm import patch_mlx_lm
        patch_mlx_lm()

        from mlx_lm import load, generate
        model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        generate(model, tokenizer, prompt="Hello world", verbose=True)
    """
    global _original_sdpa

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
    _original_sdpa = base_module.scaled_dot_product_attention
    base_module.scaled_dot_product_attention = _steel_sdpa
    if verbose:
        print(
            f"[mlx-mfa] Patched mlx-lm — STEEL kernel active for causal f16/bf16 "
            f"attention (head_dims={sorted(_SUPPORTED_HDIMS)})"
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
    total = _stats["forward_calls"]
    steel = _stats["steel_calls"]
    return {
        "forward_calls": total,
        "steel_calls": steel,
        "fallback_calls": _stats["fallback_calls"],
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

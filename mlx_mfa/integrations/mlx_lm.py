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


def _refresh_supported():
    """Populate cached supported configs from the extension."""
    global _SUPPORTED_HDIMS, _SUPPORTED_DTYPES
    cfg = get_supported_configs()
    _SUPPORTED_HDIMS = set(cfg.get("head_dims", []))
    _SUPPORTED_DTYPES = frozenset({mx.float16, mx.bfloat16})


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
    # Attention sinks: always fall back (STEEL doesn't implement them).
    if sinks is not None:
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
            return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    D = queries.shape[-1]
    dtype = queries.dtype

    # Only use STEEL for supported head_dims, dtypes, and extension.
    if D not in _SUPPORTED_HDIMS or dtype not in _SUPPORTED_DTYPES:
        return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Causal detection:
    #   - mask == "causal" (string): prefill with standard causal mask
    #   - mask is None and queries.shape[-2] == 1: single-token decode
    #     (attend to all cached K/V — no causal mask needed, use causal=False)
    #   - any other mask type: fall back (boolean array, sliding window, etc.)
    if mask == "causal":
        # Prefill: STEEL causal is 1.6–2.1× faster than SDPA
        return flash_attention(queries, keys, values, scale=scale, causal=True)

    if mask is None:
        # Decode step (N_q=1) or full-attention prefill with no mask
        return flash_attention(queries, keys, values, scale=scale, causal=False)

    # mask is an array (boolean causal, sliding-window, padding, etc.):
    # fall back to original — STEEL uses its own causal logic, not additive masks.
    return _original_sdpa(queries, keys, values, cache, scale, mask, sinks)


def patch_mlx_lm() -> bool:
    """Monkey-patch mlx-lm to use STEEL attention.

    Returns ``True`` if patching succeeded, ``False`` if the MFA extension
    is unavailable (the original SDPA is left in place).

    Calling this function multiple times is idempotent — the patch is only
    applied once.

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
        print("[mlx-mfa] Warning: MFA extension not available, mlx-lm patch skipped")
        return False

    try:
        import mlx_lm.models.base as base_module
    except ImportError as exc:
        raise ImportError(
            "mlx-lm is not installed. Install with: pip install mlx-lm"
        ) from exc

    _refresh_supported()
    _original_sdpa = base_module.scaled_dot_product_attention
    base_module.scaled_dot_product_attention = _steel_sdpa
    print("[mlx-mfa] Patched mlx-lm — STEEL kernel active for causal f16/bf16 attention")
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

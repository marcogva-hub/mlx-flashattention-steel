"""Lightweight runtime helpers for decode orchestration.

This module provides a small runtime surface over the existing inference
contexts so callers can use a single object for dense/paged/Sage decode
without rewriting application-side selection logic.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_mfa.inference import create_inference_context, _context_backend_name

__all__ = [
    "DecodeRuntime",
    "create_decode_runtime",
]


class DecodeRuntime:
    """Small wrapper over an inference context with stable decode methods."""

    def __init__(
        self,
        *,
        context,
        backend: str,
        requested_backend: str,
        paged: bool,
        quantized_kv: bool,
    ) -> None:
        self.context = context
        self.backend = backend
        self.requested_backend = requested_backend
        self.paged = paged
        self.quantized_kv = quantized_kv

    def prefill(self, q: mx.array, k: mx.array, v: mx.array, **kwargs):
        """Forward to the underlying context prefill call."""
        return self.context.prefill(q, k, v, **kwargs)

    def step(self, q: mx.array, k_new: mx.array, v_new: mx.array, **kwargs):
        """Forward to the underlying context step call."""
        return self.context.step(q, k_new, v_new, **kwargs)

    def reset(self, **kwargs):
        """Forward reset to the underlying context."""
        return self.context.reset(**kwargs)

    def seq_length(self, seq_id: int = 0) -> int:
        """Return sequence length for dense/paged/sage contexts."""
        if hasattr(self.context, "seq_length"):
            return self.context.seq_length(seq_id)
        if seq_id != 0:
            raise ValueError(
                f"seq_id={seq_id} is unsupported for backend={self.backend!r}"
            )
        if hasattr(self.context, "seqlen"):
            return int(self.context.seqlen)
        raise TypeError(
            "Context does not expose seq_length/seqlen: "
            f"{type(self.context).__name__}"
        )

    def __repr__(self) -> str:
        return (
            f"DecodeRuntime(backend={self.backend!r}, "
            f"requested={self.requested_backend!r}, "
            f"paged={self.paged}, quantized_kv={self.quantized_kv}, "
            f"context={type(self.context).__name__})"
        )


def create_decode_runtime(
    *,
    backend: str = "auto",
    paged: bool = False,
    quantized_kv: bool = False,
    B: Optional[int] = None,
    H_q: Optional[int] = None,
    H_kv: int,
    D: int,
    max_seq_len: int = 8192,
    decode_nq: int = 1,
    expected_cache_len: int = 0,
    causal: bool = True,
    window_size: Optional[tuple] = None,
    num_blocks: Optional[int] = None,
    block_size: int = 16,
    dtype: mx.Dtype = mx.float16,
    stream: Optional[mx.Stream] = None,
) -> DecodeRuntime:
    """Create a unified decode runtime over dense/paged/Sage contexts.

    This is a thin wrapper around :func:`create_inference_context` with two
    extra guarantees:
    - Runtime callers can always use the same methods (`prefill`, `step`, `reset`).
    - Explicit `backend="sage"` requires `quantized_kv=True`.
    """
    requested = backend.lower()
    if requested == "sage" and not quantized_kv:
        raise ValueError(
            "create_decode_runtime: backend='sage' requires quantized_kv=True"
        )
    if paged and quantized_kv:
        raise ValueError(
            "create_decode_runtime: paged=True is incompatible with quantized_kv=True"
        )

    context = create_inference_context(
        backend=backend,
        paged=paged,
        quantized_kv=quantized_kv,
        B=B,
        H_q=H_q,
        H_kv=H_kv,
        D=D,
        max_seq_len=max_seq_len,
        decode_nq=decode_nq,
        expected_cache_len=expected_cache_len,
        causal=causal,
        window_size=window_size,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        stream=stream,
    )
    selected = _context_backend_name(context)
    return DecodeRuntime(
        context=context,
        backend=selected,
        requested_backend=requested,
        paged=paged,
        quantized_kv=quantized_kv,
    )

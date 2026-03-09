"""mlx_mfa/inference.py -- InferenceContext lifecycle object (Track LC).

Provides a stateful wrapper around the KV cache for autoregressive generation.
The context owns the growing K/V cache and exposes clean prefill / step / reset
methods so callers do not need to manage concatenation manually.

Typical usage::

    ctx = InferenceContext(B=1, H_kv=8, D=128, max_seq_len=4096)

    # Prefill
    out_prefill = ctx.prefill(q_prefill, k_prefill, v_prefill, scale=scale)
    mx.eval(out_prefill)

    # Autoregressive decode loop
    for _ in range(max_new_tokens):
        out = ctx.step(q_new, k_new, v_new, scale=scale)
        mx.eval(out)

    # Reuse for a new sequence
    ctx.reset()

Context-manager form::

    with InferenceContext(B=1, H_kv=8, D=128) as ctx:
        out = ctx.prefill(q, k, v, scale=scale)
        for _ in range(steps):
            out = ctx.step(q_t, k_t, v_t, scale=scale)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx


__all__ = ["InferenceContext"]


class InferenceContext:
    """Stateful KV-cache manager for autoregressive generation.

    Manages the growing K/V cache across prefill and decode steps so callers
    don't need to track concatenation manually.  The cache is extended lazily
    via :func:`mlx.core.concatenate`; no pre-allocated max-length buffer is
    required.

    Args:
        B:           Batch size.
        H_kv:        Number of KV heads (may differ from Q heads for GQA).
        D:           Head dimension.
        max_seq_len: Soft limit on total sequence length (prefill + generated).
                     Exceeding it raises :class:`ValueError`.
        dtype:       Cache data type (default: ``mx.float16``).
        stream:      Optional MLX stream for all attention calls.

    Attributes:
        seqlen (int): Current cache fill length (0 after reset).
        k_cache:      Accumulated K cache ``[B, H_kv, seqlen, D]`` or ``None``
                      if empty.
        v_cache:      Accumulated V cache ``[B, H_kv, seqlen, D]`` or ``None``
                      if empty.

    Example::

        ctx = InferenceContext(B=1, H_kv=8, D=128, max_seq_len=2048)
        out = ctx.prefill(q_prefill, k_prefill, v_prefill, scale=scale)
        mx.eval(out)
        for _ in range(100):
            out = ctx.step(q_tok, k_tok, v_tok, scale=scale)
            mx.eval(out)
    """

    def __init__(
        self,
        B: int,
        H_kv: int,
        D: int,
        max_seq_len: int = 8192,
        dtype: mx.Dtype = mx.float16,
        stream: Optional[mx.Stream] = None,
    ) -> None:
        self.B = B
        self.H_kv = H_kv
        self.D = D
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.stream = stream

        self._seqlen: int = 0
        self._k_cache: Optional[mx.array] = None
        self._v_cache: Optional[mx.array] = None

    # -- Properties ----------------------------------------------------------

    @property
    def seqlen(self) -> int:
        """Current KV cache fill length."""
        return self._seqlen

    @property
    def k_cache(self) -> Optional[mx.array]:
        """Accumulated K cache ``[B, H_kv, seqlen, D]`` or ``None`` if empty."""
        return self._k_cache

    @property
    def v_cache(self) -> Optional[mx.array]:
        """Accumulated V cache ``[B, H_kv, seqlen, D]`` or ``None`` if empty."""
        return self._v_cache

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> "InferenceContext":
        """Reset cache to empty state (reuse context for a new sequence).

        Clears the accumulated K/V buffers and resets :attr:`seqlen` to 0.
        Does not re-allocate anything; the next :meth:`prefill` or :meth:`step`
        call will start fresh.

        Returns:
            ``self`` -- enables chaining: ``ctx.reset().prefill(...)``
        """
        self._seqlen = 0
        self._k_cache = None
        self._v_cache = None
        return self

    # -- Attention calls -----------------------------------------------------

    def prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
    ) -> mx.array:
        """Process prefill tokens and initialise the KV cache.

        Resets any previous cache state and runs full-sequence attention over
        the provided ``q / k / v``.

        Args:
            q:           Query ``[B, H_q, N, D]``.
            k:           Key   ``[B, H_kv, N, D]``.
            v:           Value ``[B, H_kv, N, D]``.
            scale:       Attention scale (default: ``1/sqrt(D)``).
            causal:      Causal masking (default: ``True``).
            softcap:     Soft-capping logit bound (0 = disabled).
            window_size: ``(left, right)`` sliding-window radii (``None``
                         disables windowing).

        Returns:
            Attention output ``[B, H_q, N, D]``.
        """
        from mlx_mfa.attention import flash_attention

        N = k.shape[2]
        if N > self.max_seq_len:
            raise ValueError(
                f"Prefill length {N} > max_seq_len={self.max_seq_len}"
            )

        # Reset and store the prefill cache.
        self._k_cache = mx.contiguous(k.astype(self.dtype))
        self._v_cache = mx.contiguous(v.astype(self.dtype))
        self._seqlen = N

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        return flash_attention(
            q, k, v,
            scale=scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            stream=self.stream,
        )

    def step(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        *,
        scale: Optional[float] = None,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
    ) -> mx.array:
        """Append new K/V tokens to the cache and run decode attention.

        Concatenates ``k_new / v_new`` onto the accumulated cache, then calls
        :func:`flash_attention_kvcache` so each new query token attends to the
        full KV history (causal by construction).

        Args:
            q:        Query for the new tokens ``[B, H_q, N_new, D]``.
            k_new:    New key tokens   ``[B, H_kv, N_new, D]``.
            v_new:    New value tokens ``[B, H_kv, N_new, D]``.
            scale:    Attention scale (default: ``1/sqrt(D)``).
            softcap:  Soft-capping logit bound (0 = disabled).
            window_size: ``(left, right)`` sliding-window radii.

        Returns:
            Attention output ``[B, H_q, N_new, D]``.

        Raises:
            ValueError: if the new cache length would exceed ``max_seq_len``.
        """
        from mlx_mfa.attention import flash_attention_kvcache

        n_new = k_new.shape[2]
        new_seqlen = self._seqlen + n_new
        if new_seqlen > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: seqlen {self._seqlen} + n_new {n_new} = "
                f"{new_seqlen} > max_seq_len={self.max_seq_len}"
            )

        k_new = k_new.astype(self.dtype)
        v_new = v_new.astype(self.dtype)

        if self._k_cache is None:
            self._k_cache = mx.contiguous(k_new)
            self._v_cache = mx.contiguous(v_new)
        else:
            self._k_cache = mx.concatenate([self._k_cache, k_new], axis=2)
            self._v_cache = mx.concatenate([self._v_cache, v_new], axis=2)
        self._seqlen = new_seqlen

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        return flash_attention_kvcache(
            q, self._k_cache, self._v_cache,
            scale=scale,
            causal=True,
            softcap=softcap,
            window_size=window_size,
            stream=self.stream,
        )

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "InferenceContext":
        return self

    def __exit__(self, *args) -> None:
        self.reset()

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InferenceContext(B={self.B}, H_kv={self.H_kv}, D={self.D}, "
            f"max_seq_len={self.max_seq_len}, seqlen={self._seqlen}, "
            f"dtype={self.dtype})"
        )

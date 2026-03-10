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


__all__ = ["InferenceContext", "PagedInferenceContext"]


class InferenceContext:
    """Stateful KV-cache manager for autoregressive generation.

    Manages the growing K/V cache across prefill and decode steps so callers
    don't need to track concatenation manually.  Internally uses a
    :class:`DenseKVCache` pre-allocated write-pointer buffer; each decode step
    scatter-writes ``N_new`` tokens in O(N_new × H × D) work with constant
    lazy-graph depth (no ``mx.concatenate`` allocations).

    Args:
        B:           Batch size.
        H_kv:        Number of KV heads (may differ from Q heads for GQA).
        D:           Head dimension.
        max_seq_len: Maximum total sequence length (prefill + generated).
                     Pre-allocated at construction; exceeding raises
                     :class:`ValueError`.
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

        # I.1: Single pre-allocated DenseKVCache; eliminates O(seqlen) concatenate
        # per decode step.  mx.eval() is called inside DenseKVCache.append() to
        # keep the lazy graph at constant depth regardless of decode-loop length.
        from mlx_mfa.attention import DenseKVCache
        self._cache = DenseKVCache(B, H_kv, D, max_seq_len=max_seq_len, dtype=dtype)

    # -- Properties ----------------------------------------------------------

    @property
    def seqlen(self) -> int:
        """Current KV cache fill length."""
        return self._cache.seqlen

    @property
    def k_cache(self) -> Optional[mx.array]:
        """Accumulated K cache ``[B, H_kv, seqlen, D]`` or ``None`` if empty."""
        if self._cache.seqlen == 0:
            return None
        return self._cache.k

    @property
    def v_cache(self) -> Optional[mx.array]:
        """Accumulated V cache ``[B, H_kv, seqlen, D]`` or ``None`` if empty."""
        if self._cache.seqlen == 0:
            return None
        return self._cache.v

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> "InferenceContext":
        """Reset cache to empty state (reuse context for a new sequence).

        Resets the write-pointer to 0 without re-allocating the underlying
        buffer.  The next :meth:`prefill` or :meth:`step` call will start fresh.

        Returns:
            ``self`` -- enables chaining: ``ctx.reset().prefill(...)``
        """
        self._cache.reset()
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

        # Reset write-pointer and scatter prefill tokens into the pre-allocated
        # buffer.  DenseKVCache.append() calls mx.eval() internally.
        self._cache.reset()
        self._cache.append(k, v)

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

        Scatter-writes ``k_new / v_new`` into the pre-allocated cache buffer via
        :meth:`DenseKVCache.append`, then calls :func:`flash_attention_kvcache`
        so each new query token attends to the full KV history (causal by
        construction).

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
        new_seqlen = self._cache.seqlen + n_new
        if new_seqlen > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: seqlen {self._cache.seqlen} + n_new {n_new} = "
                f"{new_seqlen} > max_seq_len={self.max_seq_len}"
            )

        # I.1: Scatter-write into pre-allocated buffer — no mx.concatenate,
        # no explicit mx.eval() here (DenseKVCache.append handles it).
        self._cache.append(k_new, v_new)

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        return flash_attention_kvcache(
            q, self._cache.k, self._cache.v,
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
            f"max_seq_len={self.max_seq_len}, seqlen={self._cache.seqlen}, "
            f"dtype={self.dtype})"
        )


class PagedInferenceContext:
    """Stateful KV-cache manager for paged autoregressive generation.

    Wraps :class:`~mlx_mfa.attention.PagedKVCache` and exposes the same
    prefill / step / reset lifecycle as :class:`InferenceContext`, but
    using a block-paged pool that avoids padding waste for variable-length
    sequences.

    Each sequence is identified by an integer ``seq_id`` (default 0).
    Multiple independent sequences can coexist in the same pool.

    Example::

        ctx = PagedInferenceContext(
            num_blocks=128, block_size=16, H_kv=8, D=128
        )

        # Prefill sequence 0
        out = ctx.prefill(q_pre, k_pre, v_pre, scale=scale, seq_id=0)
        mx.eval(out)

        # Decode
        for _ in range(steps):
            out = ctx.step(q_tok, k_tok, v_tok, scale=scale, seq_id=0)
            mx.eval(out)

        ctx.reset(seq_id=0)   # free blocks for seq 0 only

    Args:
        num_blocks:  Total pool blocks (capacity = ``num_blocks * block_size`` tokens).
        block_size:  Tokens per page (16, 32, or 64 recommended).
        H_kv:        Number of KV heads.
        D:           Head dimension.
        dtype:       Cache data type (default: ``mx.float16``).
        stream:      Optional MLX stream for all attention calls.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        H_kv: int,
        D: int,
        dtype: mx.Dtype = mx.float16,
        stream: Optional[mx.Stream] = None,
    ) -> None:
        from mlx_mfa.attention import PagedKVCache
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.H_kv = H_kv
        self.D = D
        self.dtype = dtype
        self.stream = stream
        self._cache = PagedKVCache(num_blocks, block_size, H_kv, D, dtype=dtype)

    # -- Protocol delegation -------------------------------------------------

    @property
    def cache(self):
        """Underlying :class:`~mlx_mfa.attention.PagedKVCache`."""
        return self._cache

    def seq_length(self, seq_id: int = 0) -> int:
        """Current token count for ``seq_id``."""
        return self._cache.seq_length(seq_id)

    # -- Lifecycle -----------------------------------------------------------

    def reset(self, seq_id: Optional[int] = None) -> "PagedInferenceContext":
        """Free blocks for ``seq_id`` (or all sequences if ``None``).

        Returns:
            ``self`` -- enables chaining.
        """
        self._cache.reset(seq_id)
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
        seq_id: int = 0,
    ) -> mx.array:
        """Process prefill tokens and initialise the paged KV cache.

        Resets ``seq_id`` and fills its blocks with ``k / v``.

        Args:
            q:       Query ``[1, H_q, N, D]``.
            k:       Key   ``[1, H_kv, N, D]``.
            v:       Value ``[1, H_kv, N, D]``.
            scale:   Attention scale (default ``1/sqrt(D)``).
            causal:  Causal mask (default ``True``).
            softcap: Logit softcap (0 = disabled).
            window_size: ``(left, right)`` sliding-window radii.
            seq_id:  Sequence identifier (default 0).

        Returns:
            Attention output ``[1, H_q, N, D]``.
        """
        from mlx_mfa.attention import flash_attention

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        self._cache.reset(seq_id)
        self._cache.append(k, v, seq_id=seq_id)

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
        seq_id: int = 0,
    ) -> mx.array:
        """Append new K/V tokens and run decode attention.

        Writes ``k_new / v_new`` into the paged pool for ``seq_id`` then
        gathers the full K/V history and calls flash_attention_kvcache.

        Args:
            q:        Query for the new tokens ``[1, H_q, N_new, D]``.
            k_new:    New key tokens   ``[1, H_kv, N_new, D]``.
            v_new:    New value tokens ``[1, H_kv, N_new, D]``.
            scale:    Attention scale (default ``1/sqrt(D)``).
            softcap:  Logit softcap (0 = disabled).
            window_size: ``(left, right)`` sliding-window radii.
            seq_id:   Sequence identifier (default 0).

        Returns:
            Attention output ``[1, H_q, N_new, D]``.
        """
        from mlx_mfa.attention import flash_attention_kvcache

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        self._cache.append(k_new, v_new, seq_id=seq_id)

        k_hist = self._cache.k_for_attention(seq_id)
        v_hist = self._cache.v_for_attention(seq_id)

        return flash_attention_kvcache(
            q, k_hist, v_hist,
            scale=scale,
            causal=True,
            softcap=softcap,
            window_size=window_size,
            stream=self.stream,
        )

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "PagedInferenceContext":
        return self

    def __exit__(self, *args) -> None:
        self.reset()

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PagedInferenceContext(num_blocks={self.num_blocks}, "
            f"block_size={self.block_size}, H_kv={self.H_kv}, D={self.D}, "
            f"dtype={self.dtype})"
        )

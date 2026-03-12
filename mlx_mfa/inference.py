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


__all__ = [
    "InferenceContext",
    "PagedInferenceContext",
    "SageInferenceContext",
    "create_inference_context",
]


def _context_backend_name(context) -> str:
    """Return canonical backend name for a context instance."""
    if isinstance(context, PagedInferenceContext):
        return "paged"
    if isinstance(context, SageInferenceContext):
        return "sage"
    if isinstance(context, InferenceContext):
        return "dense"
    raise TypeError(
        "Unsupported inference context type: "
        f"{type(context).__name__}"
    )


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

class SageInferenceContext:
    """Stateful KV-cache manager using SageAttention for decode.

    Prefill runs full-precision :func:`flash_attention`; decode uses
    :func:`sage_attention_prequantized` with an incremental int8 K cache
    to reduce memory bandwidth.

    Internally uses a :class:`QuantizedKVCache` so that only the new block
    at the write frontier is (re-)quantized on each decode step — O(block_size×D)
    instead of O(seqlen×D) as the old :func:`sage_attention_kvcache` path did.
    At S=4096 this reduces quantization work by ~4096×.

    Args:
        B:           Batch size.
        H_kv:        Number of KV heads (may differ from Q heads for GQA).
        D:           Head dimension.
        max_seq_len: Pre-allocated buffer length (default: 8192).
        dtype:       Cache data type (default: ``mx.float16``).
        stream:      Optional MLX stream for all attention calls.

    Example::

        ctx = SageInferenceContext(B=1, H_kv=8, D=128, max_seq_len=4096)
        out_prefill = ctx.prefill(q_pre, k_pre, v_pre, scale=scale)
        mx.eval(out_prefill)
        for _ in range(steps):
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
        from mlx_mfa.attention import QuantizedKVCache
        self._cache = QuantizedKVCache(B, H_kv, D, max_seq_len=max_seq_len, dtype=dtype)

    # -- Properties ----------------------------------------------------------

    @property
    def seqlen(self) -> int:
        """Current KV cache fill length."""
        return self._cache.seqlen

    # -- Lifecycle -----------------------------------------------------------

    def reset(self) -> "SageInferenceContext":
        """Reset cache to empty state.  Returns ``self`` for chaining."""
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
        """Process prefill tokens with full-precision flash attention.

        Args:
            q: Query ``[B, H_q, N, D]``.
            k: Key   ``[B, H_kv, N, D]``.
            v: Value ``[B, H_kv, N, D]``.
            scale: Attention scale (default: ``1/sqrt(D)``).
            causal: Causal masking (default: ``True``).
            softcap: Soft-capping logit bound (0 = disabled).
            window_size: ``(left, right)`` sliding-window radii.

        Returns:
            Attention output ``[B, H_q, N, D]``.
        """
        from mlx_mfa.attention import flash_attention

        N = k.shape[2]
        if N > self.max_seq_len:
            raise ValueError(
                f"Prefill length {N} > max_seq_len={self.max_seq_len}"
            )
        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        self._cache.reset()
        self._cache.append(k, v)

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
        window_size: Optional[tuple] = None,
    ) -> mx.array:
        """Append new K/V tokens and run int8 decode attention.

        Uses :func:`sage_attention_prequantized` with the incremental
        :class:`QuantizedKVCache` — only the new block is (re-)quantized per
        step, not the full K cache.

        Args:
            q:       Query for the new tokens ``[B, H_q, N_new, D]``.
            k_new:   New key tokens   ``[B, H_kv, N_new, D]``.
            v_new:   New value tokens ``[B, H_kv, N_new, D]``.
            scale:   Attention scale (default: ``1/sqrt(D)``).
            window_size: Optional decode window ``(left, right)``.

        Returns:
            Attention output ``[B, H_q, N_new, D]``.

        Raises:
            ValueError: if the new cache length would exceed ``max_seq_len``.
        """
        from mlx_mfa.attention import sage_attention_prequantized

        n_new = k_new.shape[2]
        new_seqlen = self._cache.seqlen + n_new
        if new_seqlen > self.max_seq_len:
            raise ValueError(
                f"Cache overflow: seqlen {self._cache.seqlen} + n_new {n_new} = "
                f"{new_seqlen} > max_seq_len={self.max_seq_len}"
            )

        self._cache.append(k_new, v_new)

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        return sage_attention_prequantized(
            q,
            self._cache.k_int8,
            self._cache.k_scale,
            self._cache.v,
            scale=scale,
            causal=True,
            window_size=window_size,
            stream=self.stream,
        )

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> "SageInferenceContext":
        return self

    def __exit__(self, *args) -> None:
        self.reset()

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SageInferenceContext(B={self.B}, H_kv={self.H_kv}, D={self.D}, "
            f"max_seq_len={self.max_seq_len}, seqlen={self._cache.seqlen}, "
            f"dtype={self.dtype})"
        )


def create_inference_context(
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
):
    """Create a decode context for dense, paged, or Sage backends.

    Routing policy:
      - ``backend="auto"``: paged > benchmark-backed Sage decode > dense
      - ``backend="paged"``: :class:`PagedInferenceContext`
      - ``backend="sage"``:  :class:`SageInferenceContext`
      - ``backend="dense"``: :class:`InferenceContext`

    Args:
        backend: ``"auto"``, ``"dense"``, ``"paged"``, or ``"sage"``.
        paged: Hint for auto mode; when True selects paged context.
        quantized_kv: Enables Sage-eligible auto routing (QuantizedKVCache).
        B: Batch size (required for dense/sage; optional for paged helper sizing).
        H_q: Query head count for auto Sage decode policy.  Defaults to ``H_kv``.
        H_kv: KV head count.
        D: Head dimension.
        max_seq_len: Maximum sequence length for dense/sage buffers.
        decode_nq: Expected decode query length (tokens/step). Auto Sage policy
            is decode-only and expects ``decode_nq <= 4``.
        expected_cache_len: Expected KV cache length in decode mode.
        causal: Expected decode masking mode for auto policy.
        window_size: Expected decode window ``(left, right)`` for auto policy.
        num_blocks: Paged pool blocks; if omitted in paged mode, a conservative
            default is derived from ``B`` and ``max_seq_len``.
        block_size: Tokens per paged block.
        dtype: Cache dtype.
        stream: Optional MLX stream.
    """
    mode = backend.lower()
    requested_mode = mode
    if mode not in {"auto", "dense", "paged", "sage"}:
        raise ValueError(f"backend must be one of auto|dense|paged|sage, got {backend!r}")
    if decode_nq <= 0:
        raise ValueError("decode_nq must be >= 1")
    if expected_cache_len < 0:
        raise ValueError("expected_cache_len must be >= 0")
    if H_kv <= 0:
        raise ValueError("H_kv must be >= 1")

    q_heads = H_kv if H_q is None else H_q
    if q_heads <= 0:
        raise ValueError("H_q must be >= 1")
    if q_heads % H_kv != 0:
        raise ValueError("H_q must be divisible by H_kv for GQA routing")
    gqa_factor = q_heads // H_kv

    if mode == "auto":
        from mlx_mfa.dispatch_policy import should_use_sage_decode

        if paged:
            mode = "paged"
        elif should_use_sage_decode(
            D,
            decode_nq,
            expected_cache_len,
            causal,
            has_quantized_kv=quantized_kv,
            window_size=window_size,
            gqa_factor=gqa_factor,
            dtype=dtype,
        ):
            mode = "sage"
        else:
            mode = "dense"

    if mode == "dense":
        if paged:
            raise ValueError("backend='dense' is incompatible with paged=True")
        if quantized_kv and requested_mode == "dense":
            raise ValueError("backend='dense' is incompatible with quantized_kv=True")
        if B is None:
            raise ValueError("B is required for backend='dense'")
        return InferenceContext(
            B=B,
            H_kv=H_kv,
            D=D,
            max_seq_len=max_seq_len,
            dtype=dtype,
            stream=stream,
        )

    if mode == "sage":
        if paged:
            raise ValueError("backend='sage' is incompatible with paged=True")
        if B is None:
            raise ValueError("B is required for backend='sage'")
        return SageInferenceContext(
            B=B,
            H_kv=H_kv,
            D=D,
            max_seq_len=max_seq_len,
            dtype=dtype,
            stream=stream,
        )

    # mode == "paged"
    if quantized_kv:
        raise ValueError("backend='paged' is incompatible with quantized_kv=True")
    if num_blocks is None:
        effective_b = B if B is not None else 1
        # Default assumes one full max_seq_len stream per batch element.
        num_blocks = max(1, (effective_b * max_seq_len + block_size - 1) // block_size)
    return PagedInferenceContext(
        num_blocks=num_blocks,
        block_size=block_size,
        H_kv=H_kv,
        D=D,
        dtype=dtype,
        stream=stream,
    )

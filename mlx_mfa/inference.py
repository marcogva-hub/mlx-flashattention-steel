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
import os
import warnings
from typing import Optional

import mlx.core as mx
from mlx_mfa.kv_cache import adapt_kv_cache


# CC-09 (audit): an inference context constructed with an off-spec head_dim or
# dtype silently runs every attention call via the SDPA fallback (no MFA/NAX
# acceleration) with no signal.  Warn once (or raise under MFA_REQUIRE_NAX=1),
# consistent with the silent-NAX-fallback hardening philosophy (RULE 8).
_MFA_SUPPORTED_DIMS = (64, 128, 256)
_offspec_warned: set = set()


def _warn_offspec(D: int, dtype, where: str) -> None:
    dim_ok = D in _MFA_SUPPORTED_DIMS
    dtype_ok = dtype in (mx.float16, mx.bfloat16)
    if dim_ok and dtype_ok:
        return
    reasons = []
    if not dim_ok:
        reasons.append(f"head_dim D={D} (MFA supports {_MFA_SUPPORTED_DIMS})")
    if not dtype_ok:
        reasons.append(f"dtype={dtype} (MFA supports float16/bfloat16)")
    msg = (
        f"{where}: off-spec config — {'; '.join(reasons)}. Attention will run via the "
        f"SDPA fallback (no MFA/NAX acceleration). Set MFA_REQUIRE_NAX=1 to raise instead."
    )
    if os.environ.get("MFA_REQUIRE_NAX", "").strip().lower() in ("1", "true", "yes"):
        raise ValueError(msg)
    key = (where, int(D), str(dtype))
    if key not in _offspec_warned:
        _offspec_warned.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


# P8: backends whose KV persistence/compute kernels are float16/bfloat16-ONLY —
# the raw `mfa_scatter_kv` (paged) and `mfa_quantize_per_block` (sage) kernels
# cannot carry fp32, so a fp32 cache would CONSTRUCT (off-spec) then fail LATE
# deep in prefill/step.  Gate (backend, dtype) at construction so every combo
# either runs end-to-end or is rejected up-front with a clear capability error —
# no construct-run-then-fail-deep.  dense (SDPA fallback), hybrid (+offload, the
# byte store is dtype-agnostic post-P7) and turboquant (fp32 routes to fallback)
# DO run fp32 end-to-end → they are NOT gated.  This closes the fp32 late-failure
# class (P7 LocalHost + P8 paged were two instances of it).
#
# P10 adds "turboquant": its decode kernels are fp16/bf16 too — the tq_decode
# gather/dequant kernels reject fp32 (`_msl_type`) and the fused fallback
# (MFA_DISABLE_TQ_DECODE_SDPA=1) emits NON-FINITE fp32.  P8 mis-marked this cell
# "OK (fallback)" because pre-P9 tq_decode FORCED fp16 output, so it *ran* while
# silently emitting fp16 — a forced-dtype masking that "does it run?" could not
# see.  Gate it at construction like the other fp16/bf16-only backends.
_FP16_BF16_ONLY_BACKENDS = ("paged", "sage", "turboquant")


def _assert_construct_dtype_supported(backend: str, dtype, where: str) -> None:
    """Reject, at construction, a dtype a fp16/bf16-only backend cannot carry."""
    if dtype in (mx.float16, mx.bfloat16):
        return
    if backend in _FP16_BF16_ONLY_BACKENDS:
        raise ValueError(
            f"{where}: backend={backend!r} does not support dtype={dtype} — its KV "
            f"scatter/quantize kernels are float16/bfloat16-only (a fp32 cache would "
            f"fail deep in prefill/step). Use dtype=float16 or bfloat16; for fp32 use "
            f"backend='dense' (SDPA fallback) or a hybrid cache with offload."
        )


def _validate_qkv_before_mutate(q, k, v, fn, *, pool_heads=None, pool_dim=None):
    """CC class-method batch — Class B (validate-before-mutate) + Class C (TQ
    geometry). Hoist the full Q/K/V mutual-compat check to BEFORE any cache
    mutation so a malformed call raises atomically (cache byteΔ=0), instead of
    appending K/V and then raising deep in the attention call (which left the
    cache mutated on failure). For TQ contexts, also pin q/k to the fixed backing-
    pool geometry (q_heads a multiple of pool kv-heads, head_dim == pool D) —
    `_assert_qkv_mutual_compat` only checks q-vs-k, not q-vs-pool, so a k that
    matches q but not the pool (e.g. D=128 over a D64 pool) would be finite-wrong.
    """
    from mlx_mfa.attention import _assert_qkv_mutual_compat
    _assert_qkv_mutual_compat(q, k, v, fn)
    if pool_heads is not None and k.shape[1] != pool_heads:
        raise ValueError(
            f"{fn}: k/v heads ({k.shape[1]}) must equal the backing-pool "
            f"kv-heads ({pool_heads}).")
    if pool_dim is not None and q.shape[3] != pool_dim:
        raise ValueError(
            f"{fn}: head_dim ({q.shape[3]}) must equal the backing-pool D "
            f"({pool_dim}).")


__all__ = [
    "InferenceContext",
    "PagedInferenceContext",
    "SageInferenceContext",
    "TurboQuantPagedInferenceContext",
    "create_inference_context",
]


def _context_backend_name(context: object) -> str:
    """Return canonical backend name for a context instance."""
    if isinstance(context, TurboQuantPagedInferenceContext):
        return "turboquant"
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
        _warn_offspec(D, dtype, "InferenceContext")  # CC-09: loud off-spec fallback

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

    @property
    def cache_adapter(self):
        """Capability adapter over the underlying cache implementation."""
        return adapt_kv_cache(self._cache)

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k, v, "InferenceContext.prefill")

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k_new, v_new, "InferenceContext.step")

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

    def chunked_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        chunk_size: int,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
        reset: bool = True,
    ) -> mx.array:
        """Chunked prefill for long prompts (causal-only in this pass)."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not causal:
            raise ValueError(
                "chunked_prefill currently requires causal=True in this pass"
            )
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("chunked_prefill expects 4-D q/k/v tensors")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("chunked_prefill requires matching batch sizes")
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError("chunked_prefill requires matching sequence lengths")

        if reset:
            self.reset()

        N = int(q.shape[2])
        out_chunks = []
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            out_chunks.append(
                self.step(
                    q[:, :, s:e, :],
                    k[:, :, s:e, :],
                    v[:, :, s:e, :],
                    scale=scale,
                    softcap=softcap,
                    window_size=window_size,
                )
            )
        if not out_chunks:
            B, H_q, _, D = q.shape
            return mx.zeros((B, H_q, 0, D), dtype=q.dtype)
        return mx.concatenate(out_chunks, axis=2)

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
        # P8: paged scatter (mfa_scatter_kv) is fp16/bf16-only → reject fp32 here,
        # not late in prefill. (Gate before _warn_offspec so the misleading
        # "SDPA fallback" warning never fires for a combo that cannot fall back.)
        _assert_construct_dtype_supported("paged", dtype, "PagedInferenceContext")
        _warn_offspec(D, dtype, "PagedInferenceContext")  # CC-09: loud off-spec fallback
        self._cache = PagedKVCache(num_blocks, block_size, H_kv, D, dtype=dtype)

    # -- Protocol delegation -------------------------------------------------

    @property
    def cache(self):
        """Underlying :class:`~mlx_mfa.attention.PagedKVCache`."""
        return self._cache

    @property
    def cache_adapter(self):
        """Capability adapter over the underlying cache implementation."""
        return adapt_kv_cache(self._cache)

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k, v, "PagedInferenceContext.prefill")

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k_new, v_new, "PagedInferenceContext.step")

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

    def chunked_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        chunk_size: int,
        scale: Optional[float] = None,
        causal: bool = True,
        seq_id: int = 0,
        reset: bool = True,
    ) -> mx.array:
        """Chunked prefill for paged cache lifecycle (causal-only)."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not causal:
            raise ValueError(
                "chunked_prefill currently requires causal=True in this pass"
            )
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("chunked_prefill expects 4-D q/k/v tensors")
        if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
            raise ValueError(
                "PagedInferenceContext.chunked_prefill currently supports B=1; "
                "use DecodeRuntime.chunked_prefill for batched paged flows."
            )
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError("chunked_prefill requires matching sequence lengths")

        if reset:
            self._cache.reset(seq_id=seq_id)

        N = int(q.shape[2])
        out_chunks = []
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            out_chunks.append(
                self.step(
                    q[:, :, s:e, :],
                    k[:, :, s:e, :],
                    v[:, :, s:e, :],
                    scale=scale,
                    seq_id=seq_id,
                )
            )
        if not out_chunks:
            B, H_q, _, D = q.shape
            return mx.zeros((B, H_q, 0, D), dtype=q.dtype)
        return mx.concatenate(out_chunks, axis=2)

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
        # P8: sage quantizer (mfa_quantize_per_block) is fp16/bf16-only → reject
        # fp32 here, not late in prefill/step.
        _assert_construct_dtype_supported("sage", dtype, "SageInferenceContext")
        _warn_offspec(D, dtype, "SageInferenceContext")  # CC-09: loud off-spec fallback
        from mlx_mfa.attention import QuantizedKVCache
        self._cache = QuantizedKVCache(B, H_kv, D, max_seq_len=max_seq_len, dtype=dtype)

    # -- Properties ----------------------------------------------------------

    @property
    def seqlen(self) -> int:
        """Current KV cache fill length."""
        return self._cache.seqlen

    @property
    def cache_adapter(self):
        """Capability adapter over the underlying cache implementation."""
        return adapt_kv_cache(self._cache)

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k, v, "SageInferenceContext.prefill")

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
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k_new, v_new, "SageInferenceContext.step")

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


# =========================================================================
# TurboQuantPagedInferenceContext — Phase 3B
# =========================================================================

class TurboQuantPagedInferenceContext:
    """Stateful paged KV-cache with TurboQuant compression.

    Stores K and V as TQ-packed uint8 pools, compressing on ``append``.
    Attention uses the fused TQ kernel that dequantifies inline.
    Q is automatically pre-rotated with WHT before the fused kernel call.

    ``turboquant=True`` in :func:`create_decode_runtime` creates this context.

    Args:
        num_blocks:  Total pool blocks.
        block_size:  Tokens per page.
        H_kv:        Number of KV heads.
        D:           Head dimension.
        tq_bits:     Quantization bits (2, 3, or 4).
        tq_v:        If True, V is also TQ-packed (Phase 3A). Default True.
        dtype:       Query/output dtype (default ``mx.float16``).
        stream:      Optional MLX stream.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        H_kv: int,
        D: int,
        tq_bits: int = 3,
        tq_v: bool = True,
        wht_in_kernel: bool = False,
        dtype: mx.Dtype = mx.float16,
        stream: Optional[mx.Stream] = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.H_kv = H_kv
        self.D = D
        self.tq_bits = tq_bits
        self.tq_v = tq_v
        self.wht_in_kernel = wht_in_kernel
        self.dtype = dtype
        self.stream = stream
        # P10: TQ decode kernels are fp16/bf16-only (tq_decode rejects fp32; the
        # fused fallback emits non-finite fp32) → reject fp32 at construction.
        _assert_construct_dtype_supported("turboquant", dtype, "TurboQuantPagedInferenceContext")
        _warn_offspec(D, dtype, "TurboQuantPagedInferenceContext")  # CC-09: loud off-spec fallback
        from mlx_mfa.turboquant import _compute_packed_d
        self.packed_D = _compute_packed_d(D, tq_bits)

        # TQ-packed K pool: [num_blocks, block_size, H_kv, packed_D] uint8
        self._k_pool = mx.zeros(
            (num_blocks, block_size, H_kv, self.packed_D), dtype=mx.uint8
        )
        self._k_scales = mx.zeros(
            (num_blocks, block_size, H_kv), dtype=mx.float32
        )

        # V pool — TQ-packed if tq_v else fp16
        if tq_v:
            self._v_pool_tq = mx.zeros(
                (num_blocks, block_size, H_kv, self.packed_D), dtype=mx.uint8
            )
            self._v_scales = mx.zeros(
                (num_blocks, block_size, H_kv), dtype=mx.float32
            )
        else:
            self._v_pool_tq = None
            self._v_scales = None
        # III-4 R9: the fp16 V pool is maintained UNCONDITIONALLY (even at
        # tq_v=True) because the III-2 default decode path reads it
        # directly (it is faster and more accurate than dequantizing
        # packed V).  At tq_v=True both pools therefore exist and append()
        # packs V into _v_pool_tq AND writes _v_pool_fp16 — the packed V
        # only earns its keep for the FUSED multi-token kernel path
        # (MFA_DISABLE_TQ_DECODE_SDPA=1).  The "~8x KV compression" Phase-3
        # figure is the fused-kernel property, not this context's
        # steady-state footprint.  Marco-queue item: make the packed-V
        # pool lazy/optional when only the III-2 decode path is used.
        self._v_pool_fp16 = mx.zeros(
            (num_blocks, block_size, H_kv, D), dtype=dtype
        )

        # Centroids (constant per bits)
        from mlx_mfa.turboquant import _get_centroids
        _, centroids_f32 = _get_centroids(tq_bits)
        self._k_centroids = centroids_f32.astype(mx.float16)
        self._v_centroids = centroids_f32.astype(mx.float16) if tq_v else None

        # Materialise pool allocations: the paged kernels bind these buffers
        # directly, so the zero-fills must be computed up front.
        # III-4 R8 FIX: mx.synchronize() alone never evaluated the lazy
        # zeros — mx.eval(*pools) is the materialization primitive here.
        pools = [self._k_pool, self._k_scales, self._v_pool_fp16]
        if tq_v:
            pools.extend([self._v_pool_tq, self._v_scales])
        mx.eval(*pools)

        # Block management (same as PagedKVCache)
        self._free: list[int] = list(range(num_blocks))
        self._block_table: dict[int, list[int]] = {}
        self._write_ptr: dict[int, int] = {}
        # III-4 R7 FIX: the GPU block-table cache (keyed on seq_ids tuple,
        # filled by get_block_table, invalidated by _allocate_block) must
        # exist from construction — previously created lazily via setattr.
        self._block_table_cache: Optional[tuple] = None

    def _allocate_block(self) -> int:
        # Campaign 2026-06 Sprint C Track 1 (#4): block allocation changes
        # the table layout — drop the cached GPU table.
        self._block_table_cache = None
        if not self._free:
            raise RuntimeError("TurboQuantPagedInferenceContext: out of blocks")
        return self._free.pop()

    def _ensure_seq(self, seq_id: int) -> None:
        if seq_id not in self._block_table:
            blk = self._allocate_block()
            self._block_table[seq_id] = [blk]
            self._write_ptr[seq_id] = 0

    def seq_length(self, seq_id: int = 0) -> int:
        if seq_id not in self._block_table:
            return 0
        n_full = max(0, len(self._block_table[seq_id]) - 1) * self.block_size
        return n_full + self._write_ptr.get(seq_id, 0)

    def reset(self, seq_id: Optional[int] = None) -> "TurboQuantPagedInferenceContext":
        if seq_id is None:
            self._free = list(range(self.num_blocks))
            self._block_table.clear()
            self._write_ptr.clear()
        elif seq_id in self._block_table:
            self._free.extend(self._block_table.pop(seq_id))
            self._write_ptr.pop(seq_id, None)
        # III-4 R7 FIX: reset frees blocks, so the cached GPU table is stale
        # — a survivor here would serve freed/reassigned block ids.
        self._block_table_cache = None
        return self

    def append(self, k: mx.array, v: mx.array, *, seq_id: int = 0,
               defer_pool_materialize: bool = False) -> None:
        """Compress and append K/V tokens to the TQ pool for ``seq_id``.

        Args:
            k: [1, H_kv, N_new, D] fp16 key tokens.
            v: [1, H_kv, N_new, D] fp16 value tokens.
            defer_pool_materialize: when True, SKIP the per-step eager
                ``mx.eval`` of the pools.  Safe ONLY when the immediately-
                following consumer reads EVERY pool this call writes as an
                MLX graph-input (so the consumer's own ``eval(o)`` materializes
                them in dependency order).  ``step()`` sets this True only on
                the gather/dequant decode branch AND only when ``tq_v=False``
                (the decode path reads k_pool / k_scales / v_pool_fp16 as
                graph-inputs; with tq_v=True the packed-V pools are written-
                but-unread, so deferral would leak an unbounded lazy scatter
                chain — see IV-D1).  Default False = eager (the safe public
                contract: standalone ``append`` callers may bind pools raw).
            seq_id: Sequence identifier.
        """
        from mlx_mfa.turboquant import pack_k_for_metal, pack_v_for_metal
        import numpy as np

        self._ensure_seq(seq_id)
        # P5 (HIGH #2): complete K/V persistence contract via the single shared
        # helper — adds the batch axis the per-site P1 check missed (this paged
        # append is single-sequence). expected_batch=1; configured kv-heads + D.
        from mlx_mfa._persist_validate import assert_kv_persist_compat
        assert_kv_persist_compat(
            k, v, "TurboQuantPagedInferenceContext.append",
            expected_batch=1, expected_heads=self.H_kv, expected_dim=self.D,
            accepted_dtypes=(self.dtype,))
        N_new = k.shape[2]

        # Pack K
        k_packed, k_sc, _ = pack_k_for_metal(k, bits=self.tq_bits)
        # k_packed: [1, H_kv, N_new, packed_D], k_sc: [1, H_kv, N_new]

        # Pack V if tq_v
        v_packed = v_sc = None
        if self.tq_v:
            v_packed, v_sc, _ = pack_v_for_metal(v, bits=self.tq_bits)
        # Campaign 2026-06 Sprint C (#12): the pack-side barrier is removed
        # entirely — pack outputs are lazy MLX arrays consumed by the lazy
        # scatter below; no host-side read intervenes, so no barrier is
        # needed before the scatter graph is built.

        # Also keep fp16 V for the fallback buffer binding
        v_fp16 = v.astype(self.dtype)

        # Write tokens into pool blocks.
        # Repo review 2026-05: previously round-tripped every tensor through
        # numpy (np.array(...) = GPU sync + device→host copy, then per-block
        # mx.array(...) host→device uploads) — 3-5 full transfers per decode
        # token.  Now sliced/transposed natively in MLX; data never leaves
        # the GPU.  [1, H_kv, N_new, X] → [N_new, H_kv, X] via swapaxes.
        k_tok = mx.swapaxes(k_packed[0], 0, 1)            # [N_new, H_kv, packed_D]
        ks_tok = mx.swapaxes(k_sc[0].astype(mx.float32), 0, 1)  # [N_new, H_kv]
        v_tok = mx.swapaxes(v_fp16[0], 0, 1)              # [N_new, H_kv, D]
        if self.tq_v:
            vp_tok = mx.swapaxes(v_packed[0], 0, 1)
            vs_tok = mx.swapaxes(v_sc[0].astype(mx.float32), 0, 1)

        blocks = self._block_table[seq_id]
        wp = self._write_ptr[seq_id]
        written = 0

        while written < N_new:
            if wp == self.block_size:
                blk = self._allocate_block()
                blocks.append(blk)
                wp = 0
            space = min(self.block_size - wp, N_new - written)
            blk_id = blocks[-1]
            end = written + space

            # Scatter into pool (GPU-side slice assignment; no host copies)
            self._k_pool[blk_id, wp:wp + space] = k_tok[written:end]
            self._k_scales[blk_id, wp:wp + space] = ks_tok[written:end]
            self._v_pool_fp16[blk_id, wp:wp + space] = v_tok[written:end]
            if self.tq_v:
                self._v_pool_tq[blk_id, wp:wp + space] = vp_tok[written:end]
                self._v_scales[blk_id, wp:wp + space] = vs_tok[written:end]

            wp += space
            written += space

        self._write_ptr[seq_id] = wp
        # Materialize pool updates eagerly: downstream paged kernels bind these
        # buffers directly (raw set_input_array), so the writes must be
        # resolved — not pending lazy graph nodes — before the next dispatch.
        # mx.eval keeps the lazy-graph depth constant across decode steps AND
        # satisfies the materialization contract for the pool buffers (they
        # are Primitive inputs; the scheduler would also eval them, but doing
        # it here bounds graph growth).  Campaign 2026-06 Sprint C (#12): the
        # trailing mx.synchronize() that followed was a full GPU-queue drain
        # per token with no correctness role — mx.eval already blocks until
        # the writes are computed.  Removed; validated by 300-step decode
        # output equivalence.
        # IV-D1: skip this per-step eager eval when the caller guarantees the
        # next consumer materializes these pools via the graph (decode branch,
        # tq_v=False — the gather reads them as mx.fast.metal_kernel graph-
        # inputs so eval(o) resolves them in dependency order).  ~240us/step
        # recovered (the MLX per-eval round-trip floor).  Default path keeps
        # the eager eval (raw-bound fused consumers + the written-but-unread
        # packed-V pools at tq_v=True both require materialization here).
        if not defer_pool_materialize:
            if self.tq_v:
                mx.eval(self._k_pool, self._k_scales, self._v_pool_fp16,
                        self._v_pool_tq, self._v_scales)
            else:
                mx.eval(self._k_pool, self._k_scales, self._v_pool_fp16)

    def get_block_table(self, seq_ids: Optional[list[int]] = None) -> mx.array:
        """Return block_table [B, max_blocks] for given seq_ids."""
        import numpy as np
        if seq_ids is None:
            seq_ids = sorted(self._block_table.keys())
        if not seq_ids:
            return mx.zeros((0, 0), dtype=mx.int32)
        # Campaign 2026-06 Sprint C Track 1 (#4): the table only changes when
        # a block is allocated (every block_size tokens), but this method runs
        # EVERY decode step — previously a numpy alloc + Python double loop +
        # host->GPU upload per token.  Cache the GPU table keyed on the seq_ids
        # tuple; _allocate_block invalidates.
        cache_key = tuple(seq_ids)
        cached = getattr(self, "_block_table_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]
        # III-4 R15: TQ pads the block table with 0 (vs PagedKVCache's -1
        # sentinel).  Safe here because the fused/decode kernels bound
        # their reads by seq_lens_kv and never index past the active
        # block count, so the padding value is never dereferenced.  Kept
        # at 0 (not -1) deliberately: a -1 physical block id would be an
        # out-of-bounds pool index if a future kernel ever read it, while
        # 0 aliases a valid (if wrong-sequence) block — both rely on the
        # seq_lens bound, but 0 fails safe.
        max_blk = max(len(self._block_table.get(s, [])) for s in seq_ids)
        table = np.zeros((len(seq_ids), max_blk), dtype=np.int32)
        for i, s in enumerate(seq_ids):
            blks = self._block_table.get(s, [])
            for j, b in enumerate(blks):
                table[i, j] = b
        result = mx.array(table, dtype=mx.int32)
        self._block_table_cache = (cache_key, result)
        return result

    def get_seq_lens(self, seq_ids: Optional[list[int]] = None) -> mx.array:
        """Return seq_lens_kv [B] for given seq_ids."""
        if seq_ids is None:
            seq_ids = sorted(self._block_table.keys())
        return mx.array([self.seq_length(s) for s in seq_ids], dtype=mx.int32)

    def prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        scale: Optional[float] = None,
        causal: bool = True,
        seq_id: int = 0,
    ) -> mx.array:
        """Compress K/V, store in TQ pool, and attend with fused kernel.

        Q is WHT-rotated either in Python (default) or in the Metal kernel
        (when ``wht_in_kernel=True``).
        """
        from mlx_mfa.attention import flash_attention_paged_varlen_turboquant
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k, v, "TurboQuantPagedInferenceContext.prefill", pool_heads=self.H_kv, pool_dim=self.D)

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        self.reset(seq_id)
        self.append(k, v, seq_id=seq_id)

        if self.wht_in_kernel:
            q_input = q
        else:
            from mlx_mfa.turboquant import apply_rotation
            q_input = apply_rotation(q.astype(mx.float32), "wht").astype(self.dtype)
        # Campaign 2026-06 Sprint C Track 1 (#3): the full-pipeline drain that
        # sat here is removed.  q_input is a graph INPUT of the paged-TQ
        # Primitive — the MLX scheduler materializes Primitive inputs before
        # eval_gpu, so the explicit barrier added only latency (one GPU drain
        # per decode token).  Validated by 300-step decode output equivalence.

        cu_q = mx.array([0, q.shape[2]], dtype=mx.int32)
        block_table = self.get_block_table([seq_id])
        seq_lens = self.get_seq_lens([seq_id])

        return flash_attention_paged_varlen_turboquant(
            q_input, self._k_pool, self._v_pool_fp16,
            block_table, seq_lens, cu_q,
            self._k_centroids, self._k_scales,
            scale=scale, causal=causal,
            block_size=self.block_size, tq_bits=self.tq_bits,
            tq_v_enabled=self.tq_v,
            tq_wht_enabled=self.wht_in_kernel,
            v_pool_tq=self._v_pool_tq,
            v_centroids=self._v_centroids,
            v_scales=self._v_scales,
            stream=self.stream,
        )

    def step(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        *,
        scale: Optional[float] = None,
        seq_id: int = 0,
    ) -> mx.array:
        """Append new K/V tokens (compressed) and decode with fused TQ kernel."""
        from mlx_mfa.attention import flash_attention_paged_varlen_turboquant
        # CC class-method batch: validate Q/K/V (and TQ pool geometry) BEFORE any mutation.
        _validate_qkv_before_mutate(q, k_new, v_new, "TurboQuantPagedInferenceContext.step", pool_heads=self.H_kv, pool_dim=self.D)

        if scale is None:
            scale = 1.0 / math.sqrt(self.D)

        # IV-D1/IV-D2: the gather/dequant decode branch (N_q==1, not opted
        # out) reads k_pool/k_scales/v_pool_fp16 as MLX graph-inputs, so its
        # eval(o) materializes them — append's per-step eager eval is
        # redundant there.  Defer it on the decode branch for BOTH tq_v:
        #   - tq_v=False (IV-D1): the caller's eval(o) materializes every
        #     written pool (no packed-V exists) — append eval fully dropped.
        #   - tq_v=True (IV-D2): the packed-V pools (_v_pool_tq/_v_scales) are
        #     written-but-UNREAD by the gather, so eval(o) won't pull them;
        #     the decode branch instead folds them into ONE combined eval with
        #     o at step end (below) — collapses the two per-step floors into
        #     one, materializes packed-V EVERY step (no unbounded lazy chain),
        #     and leaves them concrete for any later fused read.
        # All non-decode paths (fused N_q>1 / opt-out) keep the eager eval.
        _decode_branch = (q.shape[2] == 1
                          and os.environ.get("MFA_DISABLE_TQ_DECODE_SDPA") != "1")
        _defer_mat = _decode_branch
        self.append(k_new, v_new, seq_id=seq_id, defer_pool_materialize=_defer_mat)

        if self.wht_in_kernel:
            q_input = q
        else:
            from mlx_mfa.turboquant import apply_rotation
            q_input = apply_rotation(q.astype(mx.float32), "wht").astype(self.dtype)
        # III-4 R5 FIX: the bare mx.synchronize() that followed was a full
        # GPU-queue drain per decode token with no correctness role — q_input
        # is a Primitive graph input, so the scheduler materializes it before
        # dispatch (missed Sprint-C site; same class as the append() drain).

        cu_q = mx.array([0, q.shape[2]], dtype=mx.int32)
        block_table = self.get_block_table([seq_id])
        seq_lens = self.get_seq_lens([seq_id])

        # Sprint III-2 (campaign 2026-06): single-token decode routes to
        # gather/dequant kernels + Apple SDPA by default — §AA.5 inverted
        # the fused-dequant premise on M5 (fused TQ attend = 14x dense;
        # this path lands near the dense decode floor, 7.6-8.3x faster
        # than fused at the II-7 ladder cells before the kernels, more
        # after).  Opt-out restores the fused kernel:
        # MFA_DISABLE_TQ_DECODE_SDPA=1.  N_q > 1 keeps the fused kernel
        # (its causal-offset semantics).  V reads the fp16 pool (always
        # maintained) — faster AND more accurate than packed V.
        if (q.shape[2] == 1
                and os.environ.get("MFA_DISABLE_TQ_DECODE_SDPA") != "1"):
            from mlx_mfa.tq_decode import tq_decode_attend
            if self.wht_in_kernel:
                # Fused path would rotate in-kernel; this path needs the
                # rotated q explicitly.
                from mlx_mfa.turboquant import apply_rotation
                q_rot = apply_rotation(
                    q.astype(mx.float32), "wht").astype(self.dtype)
            else:
                q_rot = q_input  # already rotated above
            S = self.seq_length(seq_id)
            n_blocks = (S + self.block_size - 1) // self.block_size
            _bt_row = block_table[0][:n_blocks]
            # CX-TQ-DECODE-01: loud default validation of the page indices on the
            # public TQ decode path. The in-kernel guard (tq_decode.py) keeps OOB
            # memory-safe; this raises EARLY on malformed metadata (loud-failure
            # default). Opt-out MFA_PAGED_TRUST_INDICES=1 skips the value-sync
            # (same contract as the dense paged path); metadata (dtype) stays on.
            if _bt_row.dtype != mx.int32:
                raise ValueError(
                    "TurboQuantPagedInferenceContext.step: block_table must be "
                    f"int32 (the kernel reads it as int32); got {_bt_row.dtype}.")
            if os.environ.get("MFA_PAGED_TRUST_INDICES") != "1" and _bt_row.size:
                _nb_phys = int(self._k_pool.shape[0])
                _probe = mx.stack([mx.min(_bt_row), mx.max(_bt_row)])
                mx.eval(_probe)                       # single sync
                _bmin, _bmax = (int(x) for x in _probe.tolist())
                if _bmin < -1 or _bmax >= _nb_phys:
                    raise ValueError(
                        "TurboQuantPagedInferenceContext.step: block_table entries "
                        f"must be in [-1, {_nb_phys}) (num_blocks = k_pool.shape[0]); "
                        f"got min={_bmin}, max={_bmax}. '-1' marks padding.")
            o = tq_decode_attend(
                q_rot, self._k_pool, self._v_pool_fp16,
                self._k_scales, self._k_centroids,
                _bt_row, S,
                scale=scale, block_size=self.block_size,
                tq_bits=self.tq_bits, stream=self.stream)
            # IV-D2: with tq_v=True, append wrote _v_pool_tq/_v_scales which the
            # gather does NOT read, so eval(o) alone won't materialize them.
            # Fold them into a SINGLE combined eval with o: eval(o) materializes
            # the read pools (k_pool/k_scales/v_pool_fp16) via their graph-input
            # dependency, and the packed-V pools are materialized explicitly in
            # the same round-trip — one per-step floor (not two), packed-V
            # resolved every step (no lazy-chain growth), concrete for any later
            # fused read.  (tq_v=False: no packed-V; the caller's eval(o)
            # materializes the read pools — IV-D1, lazy return preserved.)
            if self.tq_v:
                mx.eval(o, self._v_pool_tq, self._v_scales)
            return o

        return flash_attention_paged_varlen_turboquant(
            q_input, self._k_pool, self._v_pool_fp16,
            block_table, seq_lens, cu_q,
            self._k_centroids, self._k_scales,
            scale=scale, causal=True,
            block_size=self.block_size, tq_bits=self.tq_bits,
            tq_v_enabled=self.tq_v,
            tq_wht_enabled=self.wht_in_kernel,
            v_pool_tq=self._v_pool_tq,
            v_centroids=self._v_centroids,
            v_scales=self._v_scales,
            stream=self.stream,
        )

    def __repr__(self) -> str:
        return (
            f"TurboQuantPagedInferenceContext(num_blocks={self.num_blocks}, "
            f"block_size={self.block_size}, H_kv={self.H_kv}, D={self.D}, "
            f"tq_bits={self.tq_bits}, tq_v={self.tq_v})"
        )


def _resolve_inference_context_mode(
    *,
    backend: str,
    paged: bool,
    quantized_kv: bool,
    H_q: Optional[int],
    H_kv: int,
    D: int,
    decode_nq: int,
    expected_cache_len: int,
    causal: bool,
    window_size: Optional[tuple],
    dtype: mx.Dtype,
    require_quantized_for_sage: bool = False,
) -> tuple[str, str]:
    """Resolve and validate context backend mode.

    Returns ``(resolved_mode, requested_mode)`` where ``resolved_mode`` is one
    of ``dense|paged|sage``.
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

    if requested_mode == "dense":
        if paged:
            raise ValueError("backend='dense' is incompatible with paged=True")
        if quantized_kv:
            raise ValueError("backend='dense' is incompatible with quantized_kv=True")
    if requested_mode == "paged" and quantized_kv:
        raise ValueError("backend='paged' is incompatible with quantized_kv=True")
    if requested_mode == "sage" and paged:
        raise ValueError("backend='sage' is incompatible with paged=True")
    if requested_mode == "sage" and require_quantized_for_sage and not quantized_kv:
        raise ValueError("backend='sage' requires quantized_kv=True")
    if requested_mode == "auto" and paged and quantized_kv:
        raise ValueError("paged=True is incompatible with quantized_kv=True")

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

    return mode, requested_mode


def _build_inference_context(
    *,
    mode: str,
    B: Optional[int],
    H_kv: int,
    D: int,
    max_seq_len: int,
    num_blocks: Optional[int],
    block_size: int,
    dtype: mx.Dtype,
    stream: Optional[mx.Stream],
) -> "InferenceContext | PagedInferenceContext | SageInferenceContext":
    """Instantiate an inference context from a resolved mode."""
    if mode == "dense":
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
) -> "InferenceContext | PagedInferenceContext | SageInferenceContext":
    """Create a decode context for dense, paged, or Sage backends.

    Routing policy:
      - ``backend="auto"``: paged > benchmark-backed Sage decode > dense
      - ``backend="paged"``: :class:`PagedInferenceContext`
      - ``backend="sage"``:  :class:`SageInferenceContext`
      - ``backend="dense"``: :class:`InferenceContext`

    Notes:
      - ``backend="sage"`` is explicit-only here and does not require
        ``quantized_kv=True``; this preserves backward-compatible behavior for
        callers that manage Sage constraints themselves.
      - For stricter runtime validation (including ``backend="sage"``
        requiring quantized KV intent), use
        :func:`mlx_mfa.runtime.create_decode_runtime`.
    """
    mode, _requested_mode = _resolve_inference_context_mode(
        backend=backend,
        paged=paged,
        quantized_kv=quantized_kv,
        H_q=H_q,
        H_kv=H_kv,
        D=D,
        decode_nq=decode_nq,
        expected_cache_len=expected_cache_len,
        causal=causal,
        window_size=window_size,
        dtype=dtype,
        require_quantized_for_sage=False,
    )
    return _build_inference_context(
        mode=mode,
        B=B,
        H_kv=H_kv,
        D=D,
        max_seq_len=max_seq_len,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        stream=stream,
    )

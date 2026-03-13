"""Lightweight runtime helpers for decode orchestration.

This module provides a small runtime surface over the existing inference
contexts so callers can use a single object for dense/paged/Sage decode
without rewriting application-side selection logic.
"""

from __future__ import annotations

from typing import Optional, Literal

import mlx.core as mx

from mlx_mfa.inference import (
    _build_inference_context,
    _context_backend_name,
    _resolve_inference_context_mode,
)
from mlx_mfa.attention import (
    flash_attention,
    flash_attention_paged,
    flash_attention_paged_varlen,
    make_shared_prefix_cache,
    flash_attention_splitfuse,
    flash_attention_speculative_verify,
)

__all__ = [
    "DecodeRuntime",
    "create_decode_runtime",
]

DecodeBackend = Literal["auto", "dense", "paged", "sage"]
QueryLayout = Literal["batched", "packed"]


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
        query_layout: str,
        default_seq_id: int,
    ) -> None:
        self.context = context
        self.backend = backend
        self.requested_backend = requested_backend
        self.paged = paged
        self.quantized_kv = quantized_kv
        self.query_layout = query_layout
        self.default_seq_id = default_seq_id
        self._prepared_prefix = None
        self._splitfuse_used = False
        self._speculative_verify_used = False
        self._active_seq_ids: Optional[tuple[int, ...]] = None
        self._active_cache_batch_idx: Optional[tuple[int, ...]] = None

    def _with_default_seq_id(self, kwargs: dict) -> dict:
        if self.backend == "paged" and "seq_id" not in kwargs:
            kwargs["seq_id"] = self.default_seq_id
        return kwargs

    @staticmethod
    def _normalize_cache_batch_idx(
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]],
    ) -> Optional[tuple[int, ...]]:
        if cache_batch_idx is None:
            return None
        if isinstance(cache_batch_idx, mx.array):
            if cache_batch_idx.ndim != 1:
                raise ValueError(
                    "cache_batch_idx must be 1-D when provided as mx.array"
                )
            return tuple(int(x) for x in cache_batch_idx.tolist())
        return tuple(int(x) for x in cache_batch_idx)

    def _resolve_active_seq_ids(
        self,
        *,
        seq_ids: Optional[tuple[int, ...] | list[int]],
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]],
        expected_batch: Optional[int] = None,
    ) -> tuple[tuple[int, ...], Optional[tuple[int, ...]]]:
        cache = getattr(self.context, "cache", None)
        if cache is None:
            raise TypeError(
                "Paged runtime requires context.cache for sequence-id resolution"
            )
        base_seq_ids = (
            tuple(sorted(cache.seq_lengths.keys()))
            if seq_ids is None
            else tuple(int(s) for s in seq_ids)
        )
        idx_tuple = self._normalize_cache_batch_idx(cache_batch_idx)
        if idx_tuple is not None:
            max_rows = len(base_seq_ids)
            for i in idx_tuple:
                if i < 0 or i >= max_rows:
                    raise ValueError(
                        "cache_batch_idx contains out-of-range slot "
                        f"{i} for {max_rows} available sequence rows"
                    )
            active_seq_ids = tuple(base_seq_ids[i] for i in idx_tuple)
        else:
            active_seq_ids = base_seq_ids

        if expected_batch is not None and len(active_seq_ids) != expected_batch:
            raise ValueError(
                "Active sequence count must match batch/query count "
                f"(got {len(active_seq_ids)} vs expected {expected_batch})"
            )
        return active_seq_ids, idx_tuple

    def prefill(self, q: mx.array, k: mx.array, v: mx.array, **kwargs):
        """Forward to the underlying context prefill call."""
        if self.query_layout != "batched":
            raise ValueError(
                "prefill() requires query_layout='batched'. "
                "Use paged_varlen() for packed-query paged attention."
            )
        return self.context.prefill(q, k, v, **self._with_default_seq_id(dict(kwargs)))

    def step(self, q: mx.array, k_new: mx.array, v_new: mx.array, **kwargs):
        """Forward to the underlying context step call."""
        if self.query_layout != "batched":
            raise ValueError(
                "step() requires query_layout='batched'. "
                "Use paged_varlen() for packed-query paged attention."
            )
        return self.context.step(
            q,
            k_new,
            v_new,
            **self._with_default_seq_id(dict(kwargs)),
        )

    def reset(self, **kwargs):
        """Forward reset to the underlying context."""
        return self.context.reset(**self._with_default_seq_id(dict(kwargs)))

    def paged_varlen(
        self,
        q: mx.array,
        cu_seqlens_q: mx.array,
        *,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]] = None,
        block_table: Optional[mx.array] = None,
        seq_lens_kv: Optional[mx.array] = None,
        max_seqlen_q: Optional[int] = None,
        scale: Optional[float] = None,
        causal: bool = True,
        block_size: Optional[int] = None,
        stream: Optional[mx.StreamOrDevice] = None,
    ):
        """Run paged attention with packed variable-length queries.

        This method is available only when the runtime backend is paged and
        ``query_layout='packed'``. If ``block_table``/``seq_lens_kv`` are not
        provided, they are derived from the runtime paged cache using
        ``seq_ids`` (or all active sequence ids when omitted).
        """
        if self.backend != "paged":
            raise ValueError(
                "paged_varlen() requires backend='paged', got "
                f"backend={self.backend!r}"
            )
        if self.query_layout != "packed":
            raise ValueError(
                "paged_varlen() requires query_layout='packed'. "
                "Create the runtime with query_layout='packed'."
            )

        cache = getattr(self.context, "cache", None)
        if cache is None:
            raise TypeError(
                "paged_varlen(): runtime context does not expose a paged cache"
            )

        if (block_table is None) != (seq_lens_kv is None):
            raise ValueError(
                "paged_varlen(): block_table and seq_lens_kv must be provided together"
            )

        idx_tuple = self._normalize_cache_batch_idx(cache_batch_idx)
        active_seq_ids: Optional[tuple[int, ...]] = None
        if block_table is None:
            expected = int(cu_seqlens_q.shape[0]) - 1
            active_seq_ids, idx_tuple = self._resolve_active_seq_ids(
                seq_ids=seq_ids,
                cache_batch_idx=cache_batch_idx,
                expected_batch=expected,
            )
            block_table = cache.get_block_table(list(active_seq_ids))
            seq_lens_kv = cache.get_seq_lens(list(active_seq_ids))
            # Mapping already applied when deriving rows from cache.
            idx_for_api = None
        else:
            if seq_ids is not None:
                source_seq_ids = tuple(int(s) for s in seq_ids)
                if idx_tuple is None:
                    active_seq_ids = source_seq_ids
                else:
                    max_rows = len(source_seq_ids)
                    for i in idx_tuple:
                        if i < 0 or i >= max_rows:
                            raise ValueError(
                                "cache_batch_idx contains out-of-range slot "
                                f"{i} for seq_ids length {max_rows}"
                            )
                    active_seq_ids = tuple(source_seq_ids[i] for i in idx_tuple)
            idx_for_api = (
                None if idx_tuple is None else mx.array(idx_tuple, dtype=mx.int32)
            )

        eff_block_size = self.context.block_size if block_size is None else block_size
        eff_stream = getattr(self.context, "stream", None) if stream is None else stream
        out = flash_attention_paged_varlen(
            q,
            cache.k_pool,
            cache.v_pool,
            block_table,
            seq_lens_kv,
            cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            scale=scale,
            causal=causal,
            block_size=eff_block_size,
            cache_batch_idx=idx_for_api,
            stream=eff_stream,
        )
        self._active_seq_ids = active_seq_ids
        self._active_cache_batch_idx = idx_tuple
        return out

    def paged_prefill_batch(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]] = None,
        scale: Optional[float] = None,
        causal: bool = True,
        block_size: Optional[int] = None,
        stream: Optional[mx.StreamOrDevice] = None,
    ):
        """Scheduler-friendly paged prefill with active-request remapping."""
        if self.backend != "paged":
            raise ValueError(
                "paged_prefill_batch() requires backend='paged', got "
                f"backend={self.backend!r}"
            )
        if self.query_layout != "batched":
            raise ValueError(
                "paged_prefill_batch() requires query_layout='batched'. "
                "Use paged_varlen() for packed-query paged attention."
            )
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("paged_prefill_batch(): q/k/v batch sizes must match")

        cache = getattr(self.context, "cache", None)
        if cache is None:
            raise TypeError(
                "paged_prefill_batch(): runtime context does not expose a paged cache"
            )

        active_seq_ids, idx_tuple = self._resolve_active_seq_ids(
            seq_ids=seq_ids,
            cache_batch_idx=cache_batch_idx,
            expected_batch=q.shape[0],
        )
        for b, sid in enumerate(active_seq_ids):
            cache.reset(seq_id=sid)
            cache.append(k[b : b + 1], v[b : b + 1], seq_id=sid)

        table = cache.get_block_table(list(active_seq_ids))
        lens = cache.get_seq_lens(list(active_seq_ids))
        eff_block_size = self.context.block_size if block_size is None else block_size
        eff_stream = getattr(self.context, "stream", None) if stream is None else stream
        out = flash_attention_paged(
            q,
            cache.k_pool,
            cache.v_pool,
            table,
            lens,
            scale=scale,
            causal=causal,
            block_size=eff_block_size,
            stream=eff_stream,
        )
        self._active_seq_ids = active_seq_ids
        self._active_cache_batch_idx = idx_tuple
        return out

    def paged_step_batch(
        self,
        q: mx.array,
        k_new: mx.array,
        v_new: mx.array,
        *,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]] = None,
        scale: Optional[float] = None,
        causal: bool = True,
        block_size: Optional[int] = None,
        stream: Optional[mx.StreamOrDevice] = None,
    ):
        """Scheduler-friendly paged decode step with active-request remapping."""
        if self.backend != "paged":
            raise ValueError(
                "paged_step_batch() requires backend='paged', got "
                f"backend={self.backend!r}"
            )
        if self.query_layout != "batched":
            raise ValueError(
                "paged_step_batch() requires query_layout='batched'. "
                "Use paged_varlen() for packed-query paged attention."
            )
        if q.shape[0] != k_new.shape[0] or q.shape[0] != v_new.shape[0]:
            raise ValueError("paged_step_batch(): q/k_new/v_new batch sizes must match")

        cache = getattr(self.context, "cache", None)
        if cache is None:
            raise TypeError(
                "paged_step_batch(): runtime context does not expose a paged cache"
            )

        active_seq_ids, idx_tuple = self._resolve_active_seq_ids(
            seq_ids=seq_ids,
            cache_batch_idx=cache_batch_idx,
            expected_batch=q.shape[0],
        )
        for b, sid in enumerate(active_seq_ids):
            cache.append(k_new[b : b + 1], v_new[b : b + 1], seq_id=sid)

        table = cache.get_block_table(list(active_seq_ids))
        lens = cache.get_seq_lens(list(active_seq_ids))
        eff_block_size = self.context.block_size if block_size is None else block_size
        eff_stream = getattr(self.context, "stream", None) if stream is None else stream
        out = flash_attention_paged(
            q,
            cache.k_pool,
            cache.v_pool,
            table,
            lens,
            scale=scale,
            causal=causal,
            block_size=eff_block_size,
            stream=eff_stream,
        )
        self._active_seq_ids = active_seq_ids
        self._active_cache_batch_idx = idx_tuple
        return out

    def prefill_shared_prefix(
        self,
        prefix_q: mx.array,
        prefix_k: mx.array,
        prefix_v: mx.array,
        *,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
        seed_runtime_cache: bool = True,
        seq_id: Optional[int] = None,
    ):
        """Prepare a shared prefix and optionally seed runtime KV state.

        This helper removes manual orchestration between
        ``make_shared_prefix_cache(...)`` and runtime ``prefill(...)``.
        """
        prefix_out, kp, vp = self.shared_prefix_cache(
            prefix_q,
            prefix_k,
            prefix_v,
            scale=scale,
        )
        self._prepared_prefix = {
            "q": prefix_q,
            "k": kp,
            "v": vp,
            "scale": scale,
            "causal": causal,
            "softcap": softcap,
            "window_size": window_size,
        }
        if seed_runtime_cache:
            prefill_kwargs = {
                "scale": scale,
                "causal": causal,
                "softcap": softcap,
                "window_size": window_size,
            }
            if seq_id is not None:
                prefill_kwargs["seq_id"] = seq_id
            self.prefill(prefix_q, prefix_k, prefix_v, **prefill_kwargs)
        return prefix_out, kp, vp

    def decode_from_shared_prefix(
        self,
        q_suffix: mx.array,
        k_suffix: mx.array,
        v_suffix: mx.array,
        *,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
    ):
        """Run suffix attention using a prepared shared-prefix cache."""
        if self._prepared_prefix is None:
            raise ValueError(
                "decode_from_shared_prefix requires prefill_shared_prefix() first"
            )
        kp = self._prepared_prefix["k"]
        vp = self._prepared_prefix["v"]
        k_full = mx.concatenate([kp, k_suffix], axis=2)
        v_full = mx.concatenate([vp, v_suffix], axis=2)
        return flash_attention(
            q_suffix,
            k_full,
            v_full,
            scale=scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            stream=getattr(self.context, "stream", None),
        )

    def shared_prefix_cache(
        self,
        prefix_q: mx.array,
        prefix_k: mx.array,
        prefix_v: mx.array,
        **kwargs,
    ):
        """Expose make_shared_prefix_cache() through the runtime surface."""
        return make_shared_prefix_cache(prefix_q, prefix_k, prefix_v, **kwargs)

    @property
    def metadata(self) -> dict[str, object]:
        """Lightweight runtime-selection and helper-activation metadata."""
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "context_class": type(self.context).__name__,
            "paged_active": self.backend == "paged",
            "sage_active": self.backend == "sage",
            "query_layout": self.query_layout,
            "shared_prefix_active": self._prepared_prefix is not None,
            "splitfuse_active": self._splitfuse_used,
            "speculative_verify_active": self._speculative_verify_used,
            "default_seq_id": self.default_seq_id,
            "active_seq_ids": self._active_seq_ids,
            "active_cache_batch_idx": self._active_cache_batch_idx,
        }

    @property
    def context_class(self) -> str:
        """Return the concrete wrapped context class name."""
        return type(self.context).__name__

    def splitfuse(
        self,
        q_prefill: Optional[mx.array],
        k_prefill: Optional[mx.array],
        v_prefill: Optional[mx.array],
        q_decode: Optional[mx.array],
        k_cache_decode: Optional[mx.array],
        v_cache_decode: Optional[mx.array],
        *,
        use_prepared_prefix: bool = False,
        **kwargs,
    ):
        """Expose flash_attention_splitfuse() through the runtime surface."""
        if use_prepared_prefix:
            if self._prepared_prefix is None:
                raise ValueError(
                    "splitfuse(use_prepared_prefix=True) requires "
                    "prefill_shared_prefix() first"
                )
            q_prefill = self._prepared_prefix["q"] if q_prefill is None else q_prefill
            k_prefill = self._prepared_prefix["k"] if k_prefill is None else k_prefill
            v_prefill = self._prepared_prefix["v"] if v_prefill is None else v_prefill

        prefill_present = [q_prefill is not None, k_prefill is not None, v_prefill is not None]
        if any(prefill_present) and not all(prefill_present):
            raise ValueError(
                "splitfuse prefill inputs must be all provided or all None"
            )
        decode_present = [q_decode is not None, k_cache_decode is not None, v_cache_decode is not None]
        if any(decode_present) and not all(decode_present):
            raise ValueError(
                "splitfuse decode inputs must be all provided or all None"
            )
        out = flash_attention_splitfuse(
            q_prefill,
            k_prefill,
            v_prefill,
            q_decode,
            k_cache_decode,
            v_cache_decode,
            **kwargs,
        )
        self._splitfuse_used = True
        return out

    def speculative_verify(
        self,
        q_target: mx.array,
        draft_ids: mx.array,
        *,
        k_cache: Optional[mx.array] = None,
        v_cache: Optional[mx.array] = None,
        **kwargs,
    ):
        """Expose flash_attention_speculative_verify() through runtime.

        If ``k_cache``/``v_cache`` are omitted, dense runtime uses its own
        internal cache. Other backends must pass explicit dense caches.
        """
        if (k_cache is None) != (v_cache is None):
            raise ValueError(
                "speculative_verify: k_cache and v_cache must be provided together"
            )

        if k_cache is None:
            if self.backend != "dense":
                raise ValueError(
                    "speculative_verify without explicit k_cache/v_cache requires "
                    f"dense backend runtime, got backend={self.backend!r}"
                )
            k_cache = self.context.k_cache
            v_cache = self.context.v_cache
            if k_cache is None or v_cache is None:
                raise ValueError(
                    "speculative_verify: dense runtime cache is empty; run prefill/step "
                    "first or pass explicit k_cache/v_cache"
                )

        out = flash_attention_speculative_verify(
            q_target,
            k_cache,
            v_cache,
            draft_ids,
            **kwargs,
        )
        self._speculative_verify_used = True
        return out

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
            f"query_layout={self.query_layout!r}, "
            f"default_seq_id={self.default_seq_id}, "
            f"active_seq_ids={self.metadata['active_seq_ids']}, "
            f"active_cache_batch_idx={self.metadata['active_cache_batch_idx']}, "
            f"shared_prefix_active={self.metadata['shared_prefix_active']}, "
            f"splitfuse_active={self.metadata['splitfuse_active']}, "
            f"speculative_verify_active={self.metadata['speculative_verify_active']}, "
            f"context={type(self.context).__name__})"
        )


def create_decode_runtime(
    *,
    backend: DecodeBackend = "auto",
    paged: bool = False,
    quantized_kv: bool = False,
    query_layout: QueryLayout = "batched",
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
    default_seq_id: int = 0,
) -> DecodeRuntime:
    """Create a unified decode runtime over dense/paged/Sage contexts.

    This is a thin wrapper around :func:`create_inference_context` with two
    extra guarantees:
    - Runtime callers can always use the same methods (`prefill`, `step`, `reset`).
    - Explicit `backend="sage"` requires `quantized_kv=True`.
    """
    if default_seq_id < 0:
        raise ValueError("default_seq_id must be >= 0")
    if query_layout not in ("batched", "packed"):
        raise ValueError("query_layout must be one of 'batched' or 'packed'")

    mode, requested = _resolve_inference_context_mode(
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
        require_quantized_for_sage=True,
    )

    context = _build_inference_context(
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
    selected = _context_backend_name(context)
    if query_layout == "packed" and selected != "paged":
        raise ValueError(
            "query_layout='packed' is currently supported only with paged runtime "
            f"(resolved backend={selected!r})"
        )
    return DecodeRuntime(
        context=context,
        backend=selected,
        requested_backend=requested,
        paged=paged,
        quantized_kv=quantized_kv,
        query_layout=query_layout,
        default_seq_id=default_seq_id,
    )

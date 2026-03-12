"""Lightweight runtime helpers for decode orchestration.

This module provides a small runtime surface over the existing inference
contexts so callers can use a single object for dense/paged/Sage decode
without rewriting application-side selection logic.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from mlx_mfa.inference import (
    _build_inference_context,
    _context_backend_name,
    _resolve_inference_context_mode,
)
from mlx_mfa.attention import (
    flash_attention,
    make_shared_prefix_cache,
    flash_attention_splitfuse,
    flash_attention_speculative_verify,
)

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
        default_seq_id: int,
    ) -> None:
        self.context = context
        self.backend = backend
        self.requested_backend = requested_backend
        self.paged = paged
        self.quantized_kv = quantized_kv
        self.default_seq_id = default_seq_id
        self._prepared_prefix = None
        self._splitfuse_used = False
        self._speculative_verify_used = False

    def _with_default_seq_id(self, kwargs: dict) -> dict:
        if self.backend == "paged" and "seq_id" not in kwargs:
            kwargs["seq_id"] = self.default_seq_id
        return kwargs

    def prefill(self, q: mx.array, k: mx.array, v: mx.array, **kwargs):
        """Forward to the underlying context prefill call."""
        return self.context.prefill(q, k, v, **self._with_default_seq_id(dict(kwargs)))

    def step(self, q: mx.array, k_new: mx.array, v_new: mx.array, **kwargs):
        """Forward to the underlying context step call."""
        return self.context.step(
            q,
            k_new,
            v_new,
            **self._with_default_seq_id(dict(kwargs)),
        )

    def reset(self, **kwargs):
        """Forward reset to the underlying context."""
        return self.context.reset(**self._with_default_seq_id(dict(kwargs)))

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
    def metadata(self) -> dict:
        """Lightweight runtime-selection and helper-activation metadata."""
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "context_class": type(self.context).__name__,
            "paged_active": self.backend == "paged",
            "sage_active": self.backend == "sage",
            "shared_prefix_active": self._prepared_prefix is not None,
            "splitfuse_active": self._splitfuse_used,
            "speculative_verify_active": self._speculative_verify_used,
            "default_seq_id": self.default_seq_id,
        }

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
            f"default_seq_id={self.default_seq_id}, "
            f"shared_prefix_active={self.metadata['shared_prefix_active']}, "
            f"splitfuse_active={self.metadata['splitfuse_active']}, "
            f"speculative_verify_active={self.metadata['speculative_verify_active']}, "
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
    return DecodeRuntime(
        context=context,
        backend=selected,
        requested_backend=requested,
        paged=paged,
        quantized_kv=quantized_kv,
        default_seq_id=default_seq_id,
    )

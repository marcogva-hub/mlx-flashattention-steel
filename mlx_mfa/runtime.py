"""Lightweight runtime helpers for decode orchestration.

This module provides a small runtime surface over the existing inference
contexts so callers can use a single object for dense/paged/Sage decode
without rewriting application-side selection logic.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

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
        self._speculative_step_used = False
        self._active_seq_ids: Optional[tuple[int, ...]] = None
        self._active_cache_batch_idx: Optional[tuple[int, ...]] = None
        self._prefix_cache: dict[str, dict[str, Any]] = {}
        self._active_prefix_id: Optional[str] = None
        self._last_prefix_reuse: Optional[dict[str, Any]] = None
        self._last_speculative_step: Optional[dict[str, Any]] = None

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

    @staticmethod
    def _normalize_prefix_id(prefix_id: str) -> str:
        pid = str(prefix_id).strip()
        if not pid:
            raise ValueError("prefix_id must be a non-empty string")
        return pid

    def _resolve_prefix_id_or_active(self, prefix_id: Optional[str]) -> str:
        pid = self._active_prefix_id if prefix_id is None else prefix_id
        if pid is None:
            raise ValueError(
                "No active prefix is set; call register_prefix(...) first "
                "or pass prefix_id explicitly."
            )
        pid = self._normalize_prefix_id(pid)
        if pid not in self._prefix_cache:
            raise ValueError(
                f"Unknown prefix_id={pid!r}. Known ids: "
                f"{tuple(sorted(self._prefix_cache.keys()))}"
            )
        return pid

    @staticmethod
    def _accepted_prefix_lens_from_mask(mask: mx.array) -> mx.array:
        """Compute contiguous accepted prefix lengths from per-token bool mask."""
        if mask.ndim != 2:
            raise ValueError(
                f"accept mask must be 2-D [B, N], got ndim={mask.ndim}"
            )
        lens: list[int] = []
        for row in mask.tolist():
            accepted = 0
            for flag in row:
                if bool(flag):
                    accepted += 1
                else:
                    break
            lens.append(accepted)
        return mx.array(lens, dtype=mx.int32)

    def _seed_dense_prefix(self, entry: dict[str, Any], *, reset: bool) -> None:
        cache = getattr(self.context, "_cache", None)
        if cache is None or not hasattr(cache, "append"):
            raise TypeError(
                "Dense/Sage prefix seeding requires an append-capable runtime cache"
            )
        if reset:
            self.reset()
        cache.append(entry["k"], entry["v"])

    def _seed_paged_prefixes(
        self,
        entries: tuple[dict[str, Any], ...],
        seq_ids: tuple[int, ...],
        *,
        reset: bool,
    ) -> None:
        cache = getattr(self.context, "cache", None)
        if cache is None:
            raise TypeError(
                "Paged prefix seeding requires runtime context.cache"
            )
        if len(entries) != len(seq_ids):
            raise ValueError(
                "entries/seq_ids length mismatch in paged prefix seeding "
                f"({len(entries)} vs {len(seq_ids)})"
            )
        if reset:
            for sid in seq_ids:
                cache.reset(seq_id=sid)
        for sid, entry in zip(seq_ids, entries):
            k = entry["k"]
            v = entry["v"]
            if int(k.shape[0]) != 1 or int(v.shape[0]) != 1:
                raise ValueError(
                    "Paged prefix seeding currently requires prefix tensors "
                    "with batch=1"
                )
            cache.append(k, v, seq_id=sid)

    def register_prefix(
        self,
        prefix_id: str,
        prefix_q: mx.array,
        prefix_k: mx.array,
        prefix_v: mx.array,
        *,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
        overwrite: bool = False,
    ):
        """Register reusable shared-prefix state under ``prefix_id``."""
        pid = self._normalize_prefix_id(prefix_id)
        if (not overwrite) and pid in self._prefix_cache:
            raise ValueError(
                f"prefix_id={pid!r} already exists. Pass overwrite=True to replace it."
            )
        prefix_out, kp, vp = self.shared_prefix_cache(
            prefix_q,
            prefix_k,
            prefix_v,
            scale=scale,
        )
        entry = {
            "id": pid,
            "q": prefix_q,
            "k": kp,
            "v": vp,
            "scale": scale,
            "causal": causal,
            "softcap": softcap,
            "window_size": window_size,
        }
        self._prefix_cache[pid] = entry
        self._active_prefix_id = pid
        return prefix_out, kp, vp

    def list_registered_prefix_ids(self) -> tuple[str, ...]:
        """Return registered prefix ids sorted lexicographically."""
        return tuple(sorted(self._prefix_cache.keys()))

    def drop_prefix(self, prefix_id: str) -> None:
        """Remove one registered prefix by id."""
        pid = self._normalize_prefix_id(prefix_id)
        if pid in self._prefix_cache:
            self._prefix_cache.pop(pid, None)
            if self._active_prefix_id == pid:
                self._active_prefix_id = None
            if self._prepared_prefix is not None and self._prepared_prefix.get("id") == pid:
                self._prepared_prefix = None

    def clear_registered_prefixes(self) -> None:
        """Remove all registered prefixes and clear active prepared state."""
        self._prefix_cache.clear()
        self._active_prefix_id = None
        self._prepared_prefix = None

    def seed_prefix(
        self,
        *,
        prefix_id: Optional[str] = None,
        prefix_ids: Optional[tuple[str, ...] | list[str]] = None,
        seq_id: Optional[int] = None,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        reset: bool = True,
    ) -> Optional[tuple[int, ...]]:
        """Seed runtime cache from registered prefix entries.

        Returns paged ``seq_ids`` when seeding paged runtime, else ``None``.
        """
        if prefix_id is not None and prefix_ids is not None:
            raise ValueError("seed_prefix: pass either prefix_id or prefix_ids, not both")

        if prefix_ids is not None:
            if self.backend != "paged":
                raise ValueError(
                    "seed_prefix(prefix_ids=...) is supported only on paged runtime"
                )
            ids = tuple(self._normalize_prefix_id(pid) for pid in prefix_ids)
            if seq_ids is None:
                raise ValueError(
                    "seed_prefix with prefix_ids requires explicit seq_ids"
                )
            sids = tuple(int(s) for s in seq_ids)
            if len(ids) != len(sids):
                raise ValueError(
                    "seed_prefix prefix_ids/seq_ids length mismatch "
                    f"({len(ids)} vs {len(sids)})"
                )
            entries = tuple(self._prefix_cache[pid] for pid in ids)
            self._seed_paged_prefixes(entries, sids, reset=reset)
            self._active_prefix_id = ids[-1] if ids else self._active_prefix_id
            self._active_seq_ids = sids
            self._last_prefix_reuse = {
                "prefix_ids": ids,
                "seq_ids": sids,
                "reset": bool(reset),
            }
            return sids

        pid = self._resolve_prefix_id_or_active(prefix_id)
        entry = self._prefix_cache[pid]

        if self.backend == "paged":
            if seq_ids is not None:
                sids = tuple(int(s) for s in seq_ids)
                entries = tuple(entry for _ in sids)
                self._seed_paged_prefixes(entries, sids, reset=reset)
            else:
                sid = self.default_seq_id if seq_id is None else int(seq_id)
                sids = (sid,)
                self._seed_paged_prefixes((entry,), sids, reset=reset)
            self._active_prefix_id = pid
            self._active_seq_ids = sids
            self._last_prefix_reuse = {
                "prefix_ids": (pid,),
                "seq_ids": sids,
                "reset": bool(reset),
            }
            return sids

        if seq_ids is not None:
            raise ValueError(
                "seed_prefix(seq_ids=...) is unsupported for non-paged runtime"
            )
        if seq_id not in (None, 0):
            raise ValueError(
                "seed_prefix(seq_id=...) is unsupported for non-paged runtime"
            )
        self._seed_dense_prefix(entry, reset=reset)
        self._active_prefix_id = pid
        self._last_prefix_reuse = {
            "prefix_ids": (pid,),
            "seq_ids": None,
            "reset": bool(reset),
        }
        return None

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

    def prefill_with_prefix(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        prefix_id: Optional[str] = None,
        prefix_ids: Optional[tuple[str, ...] | list[str]] = None,
        seq_id: Optional[int] = None,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        chunk_size: Optional[int] = None,
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]] = None,
        cu_seqlens_q: Optional[mx.array] = None,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
        block_size: Optional[int] = None,
        stream: Optional[mx.StreamOrDevice] = None,
        reset: bool = True,
    ):
        """Seed registered prefix state then run suffix prefill.

        This is an explicit serving helper:
        1) seed runtime cache from registered prefix entry/entries,
        2) route suffix through ``chunked_prefill(..., reset=False)``.
        """
        seeded_seq_ids = self.seed_prefix(
            prefix_id=prefix_id,
            prefix_ids=prefix_ids,
            seq_id=seq_id,
            seq_ids=seq_ids,
            reset=reset,
        )

        if q.ndim != 4:
            raise ValueError("prefill_with_prefix expects 4-D q/k/v tensors")
        total_q = int(q.shape[2])
        eff_chunk_size = max(1, total_q) if chunk_size is None else int(chunk_size)

        eff_seq_ids = seq_ids
        if eff_seq_ids is None and seeded_seq_ids is not None:
            eff_seq_ids = list(seeded_seq_ids)

        out = self.chunked_prefill(
            q,
            k,
            v,
            chunk_size=eff_chunk_size,
            seq_ids=eff_seq_ids,
            cache_batch_idx=cache_batch_idx,
            cu_seqlens_q=cu_seqlens_q,
            scale=scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            block_size=block_size,
            stream=stream,
            reset=False,
        )

        prev = self._last_prefix_reuse or {}
        self._last_prefix_reuse = {
            **prev,
            "op": "prefill_with_prefix",
            "chunk_size": eff_chunk_size,
            "query_layout": self.query_layout,
            "backend": self.backend,
        }
        return out

    def chunked_prefill(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        *,
        chunk_size: int,
        seq_ids: Optional[tuple[int, ...] | list[int]] = None,
        cache_batch_idx: Optional[mx.array | tuple[int, ...] | list[int]] = None,
        cu_seqlens_q: Optional[mx.array] = None,
        scale: Optional[float] = None,
        causal: bool = True,
        softcap: float = 0.0,
        window_size: Optional[tuple] = None,
        block_size: Optional[int] = None,
        stream: Optional[mx.StreamOrDevice] = None,
        reset: bool = True,
    ):
        """Chunk a long prefill into multiple cache-updating prefill steps.

        This is a serving-oriented helper for prefill scheduling. It preserves
        existing prefill/step paths and adds explicit chunk semantics.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not causal:
            raise ValueError(
                "chunked_prefill currently requires causal=True; "
                "non-causal chunked prefill is not supported."
            )

        if self.query_layout == "packed":
            if self.backend != "paged":
                raise ValueError(
                    "chunked_prefill with query_layout='packed' requires "
                    "backend='paged'"
                )
            if cache_batch_idx is not None:
                raise ValueError(
                    "chunked_prefill packed path does not yet support "
                    "cache_batch_idx; pass seq_ids in packed order"
                )
            if cu_seqlens_q is None:
                raise ValueError(
                    "chunked_prefill packed path requires cu_seqlens_q"
                )
            if seq_ids is None:
                raise ValueError(
                    "chunked_prefill packed path requires explicit seq_ids"
                )
            if q.ndim != 4 or q.shape[0] != 1:
                raise ValueError(
                    "chunked_prefill packed path expects q shape [1,H,total_q,D]"
                )
            if k.ndim != 4 or v.ndim != 4 or k.shape[0] != 1 or v.shape[0] != 1:
                raise ValueError(
                    "chunked_prefill packed path expects k/v shape [1,H_kv,total_q,D]"
                )
            if k.shape[2] != q.shape[2] or v.shape[2] != q.shape[2]:
                raise ValueError(
                    "chunked_prefill packed path requires matching total_q across q/k/v"
                )
            if k.shape[3] != q.shape[3] or v.shape[3] != q.shape[3]:
                raise ValueError(
                    "chunked_prefill packed path requires matching head dim D across q/k/v"
                )
            if k.shape[1] != v.shape[1]:
                raise ValueError(
                    "chunked_prefill packed path requires matching H_kv between k and v"
                )
            if cu_seqlens_q.ndim != 1:
                raise ValueError("cu_seqlens_q must be 1-D [B+1]")

            seq_ids_t = tuple(int(s) for s in seq_ids)
            B = len(seq_ids_t)
            if cu_seqlens_q.shape[0] != B + 1:
                raise ValueError(
                    "cu_seqlens_q must have shape [B+1] matching seq_ids "
                    f"(got {cu_seqlens_q.shape[0]} vs expected {B + 1})"
                )
            cu = [int(x) for x in cu_seqlens_q.tolist()]
            if cu[0] != 0:
                raise ValueError("cu_seqlens_q[0] must be 0")
            if cu[-1] != q.shape[2]:
                raise ValueError(
                    "cu_seqlens_q[-1] must equal total_q in packed input"
                )

            cache = getattr(self.context, "cache", None)
            if cache is None:
                raise TypeError(
                    "chunked_prefill packed path requires runtime context.cache"
                )

            if reset:
                for sid in seq_ids_t:
                    cache.reset(seq_id=sid)

            lengths = [cu[i + 1] - cu[i] for i in range(B)]
            consumed = [0] * B
            out_parts: list[list[mx.array]] = [[] for _ in range(B)]

            while any(consumed[i] < lengths[i] for i in range(B)):
                active_rows = [i for i in range(B) if consumed[i] < lengths[i]]
                if not active_rows:
                    break
                active_seq_ids = [seq_ids_t[i] for i in active_rows]

                q_parts = []
                k_parts = []
                v_parts = []
                chunk_offsets = [0]
                for i in active_rows:
                    base = cu[i]
                    remain = lengths[i] - consumed[i]
                    clen = min(chunk_size, remain)
                    s = base + consumed[i]
                    e = s + clen
                    q_i = q[:, :, s:e, :]
                    k_i = k[:, :, s:e, :]
                    v_i = v[:, :, s:e, :]
                    q_parts.append(q_i)
                    k_parts.append(k_i)
                    v_parts.append(v_i)
                    chunk_offsets.append(chunk_offsets[-1] + clen)

                q_chunk = mx.concatenate(q_parts, axis=2)
                cu_chunk = mx.array(chunk_offsets, dtype=mx.int32)

                for sid, k_i, v_i in zip(active_seq_ids, k_parts, v_parts):
                    cache.append(k_i, v_i, seq_id=sid)

                out_chunk = self.paged_varlen(
                    q_chunk,
                    cu_chunk,
                    seq_ids=active_seq_ids,
                    scale=scale,
                    causal=causal,
                    block_size=block_size,
                    stream=stream,
                )

                for local_idx, i in enumerate(active_rows):
                    s = chunk_offsets[local_idx]
                    e = chunk_offsets[local_idx + 1]
                    out_parts[i].append(out_chunk[:, :, s:e, :])
                    consumed[i] += e - s

            flat_parts = []
            for i in range(B):
                if out_parts[i]:
                    flat_parts.append(mx.concatenate(out_parts[i], axis=2))
                else:
                    H_q, D = q.shape[1], q.shape[3]
                    flat_parts.append(mx.zeros((1, H_q, 0, D), dtype=q.dtype))
            return mx.concatenate(flat_parts, axis=2)

        # query_layout == "batched"
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("chunked_prefill batched path requires 4-D q/k/v")
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("chunked_prefill batched path requires matching batch sizes")
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise ValueError(
                "chunked_prefill batched path requires matching sequence lengths"
            )

        if self.backend == "sage":
            raise ValueError(
                "chunked_prefill is not implemented for backend='sage' in this pass"
            )

        N = int(q.shape[2])

        if (
            self.backend == "dense"
            and seq_ids is None
            and cache_batch_idx is None
            and hasattr(self.context, "chunked_prefill")
        ):
            return self.context.chunked_prefill(
                q,
                k,
                v,
                chunk_size=chunk_size,
                scale=scale,
                causal=causal,
                softcap=softcap,
                window_size=window_size,
                reset=reset,
            )

        if (
            self.backend == "paged"
            and int(q.shape[0]) == 1
            and cache_batch_idx is None
            and hasattr(self.context, "chunked_prefill")
            and (seq_ids is None or len(seq_ids) == 1)
        ):
            sid = self.default_seq_id if seq_ids is None else int(seq_ids[0])
            return self.context.chunked_prefill(
                q,
                k,
                v,
                chunk_size=chunk_size,
                scale=scale,
                causal=causal,
                seq_id=sid,
                reset=reset,
            )

        if reset:
            if self.backend == "paged":
                cache = getattr(self.context, "cache", None)
                if cache is None:
                    raise TypeError(
                        "chunked_prefill paged path requires runtime context.cache"
                    )
                active_seq_ids, _ = self._resolve_active_seq_ids(
                    seq_ids=seq_ids,
                    cache_batch_idx=cache_batch_idx,
                    expected_batch=int(q.shape[0]),
                )
                for sid in active_seq_ids:
                    cache.reset(seq_id=sid)
            else:
                self.reset()

        out_chunks = []
        for s in range(0, N, chunk_size):
            e = min(N, s + chunk_size)
            q_c = q[:, :, s:e, :]
            k_c = k[:, :, s:e, :]
            v_c = v[:, :, s:e, :]
            if self.backend == "paged":
                out_c = self.paged_step_batch(
                    q_c,
                    k_c,
                    v_c,
                    seq_ids=seq_ids,
                    cache_batch_idx=cache_batch_idx,
                    scale=scale,
                    causal=causal,
                    block_size=block_size,
                    stream=stream,
                )
            else:
                out_c = self.step(
                    q_c,
                    k_c,
                    v_c,
                    scale=scale,
                    softcap=softcap,
                    window_size=window_size,
                )
            out_chunks.append(out_c)

        if not out_chunks:
            B, H_q, _, D = q.shape
            return mx.zeros((B, H_q, 0, D), dtype=q.dtype)
        return mx.concatenate(out_chunks, axis=2)

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
        prefix_id: Optional[str] = None,
    ):
        """Prepare a shared prefix and optionally seed runtime KV state.

        This helper removes manual orchestration between
        ``make_shared_prefix_cache(...)`` and runtime ``prefill(...)``.
        """
        pid = "__prepared__" if prefix_id is None else self._normalize_prefix_id(prefix_id)
        prefix_out, kp, vp = self.register_prefix(
            pid,
            prefix_q,
            prefix_k,
            prefix_v,
            scale=scale,
            causal=causal,
            softcap=softcap,
            window_size=window_size,
            overwrite=True,
        )
        self._prepared_prefix = self._prefix_cache[pid]
        self._active_prefix_id = pid
        if seed_runtime_cache:
            seed_seq = self.default_seq_id if seq_id is None else int(seq_id)
            self.seed_prefix(prefix_id=pid, seq_id=seed_seq, reset=True)
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
            "speculative_step_active": self._speculative_step_used,
            "default_seq_id": self.default_seq_id,
            "active_seq_ids": self._active_seq_ids,
            "active_cache_batch_idx": self._active_cache_batch_idx,
            "prefix_cache_size": len(self._prefix_cache),
            "registered_prefix_ids": tuple(sorted(self._prefix_cache.keys())),
            "active_prefix_id": self._active_prefix_id,
            "last_prefix_reuse": self._last_prefix_reuse,
            "last_speculative_step": self._last_speculative_step,
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
        internal cache. Paged runtime can also source cache by ``seq_id`` for
        batched-layout single-sequence verify. Other combinations must pass
        explicit dense cache tensors.
        """
        if (k_cache is None) != (v_cache is None):
            raise ValueError(
                "speculative_verify: k_cache and v_cache must be provided together"
            )

        if k_cache is None:
            if self.backend == "dense":
                k_cache = self.context.k_cache
                v_cache = self.context.v_cache
                if k_cache is None or v_cache is None:
                    raise ValueError(
                        "speculative_verify: dense runtime cache is empty; run prefill/step "
                        "first or pass explicit k_cache/v_cache"
                    )
            elif self.backend == "paged":
                if self.query_layout != "batched":
                    raise ValueError(
                        "speculative_verify paged runtime fallback requires "
                        "query_layout='batched'"
                    )
                if int(q_target.shape[0]) != 1:
                    raise ValueError(
                        "speculative_verify paged runtime fallback currently requires "
                        "batch size 1"
                    )
                seq_id = int(kwargs.pop("seq_id", self.default_seq_id))
                cache = getattr(self.context, "cache", None)
                if cache is None:
                    raise TypeError(
                        "speculative_verify paged runtime fallback requires context.cache"
                    )
                if int(cache.seq_length(seq_id)) <= 0:
                    raise ValueError(
                        "speculative_verify: paged runtime cache is empty for "
                        f"seq_id={seq_id}; run prefill/step first or pass explicit k_cache/v_cache"
                    )
                k_cache = cache.k_for_attention(seq_id)
                v_cache = cache.v_for_attention(seq_id)
            else:
                raise ValueError(
                    "speculative_verify without explicit k_cache/v_cache requires "
                    f"dense or paged runtime, got backend={self.backend!r}"
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

    def speculative_step(
        self,
        q_target: mx.array,
        draft_ids: mx.array,
        *,
        draft_logprobs: Optional[mx.array] = None,
        accept_logprob_delta: float = 0.0,
        k_cache: Optional[mx.array] = None,
        v_cache: Optional[mx.array] = None,
        **kwargs,
    ) -> dict[str, mx.array]:
        """Run verify + contiguous-prefix accept/reject bookkeeping.

        This is a lightweight runtime integration wrapper around
        ``flash_attention_speculative_verify``. It does not implement a full
        scheduler; it computes inspectable acceptance outputs for serving code.

        Acceptance rule:
        - if ``draft_logprobs`` is provided: accept token i when
          ``target_logprobs[i] - draft_logprobs[i] >= accept_logprob_delta``
        - else: accept token i when
          ``target_logprobs[i] >= accept_logprob_delta``

        The returned accepted prefix length is contiguous from token 0.
        """
        if q_target.ndim != 4:
            raise ValueError("speculative_step expects q_target shape [B,H,N,D]")
        if draft_ids.ndim != 2:
            raise ValueError("speculative_step expects draft_ids shape [B,N]")
        if int(q_target.shape[0]) != int(draft_ids.shape[0]):
            raise ValueError(
                "speculative_step batch mismatch between q_target and draft_ids "
                f"({q_target.shape[0]} vs {draft_ids.shape[0]})"
            )
        if int(q_target.shape[2]) != int(draft_ids.shape[1]):
            raise ValueError(
                "speculative_step token-count mismatch between q_target and draft_ids "
                f"({q_target.shape[2]} vs {draft_ids.shape[1]})"
            )
        if draft_logprobs is not None:
            if draft_logprobs.ndim != 2:
                raise ValueError(
                    "speculative_step expects draft_logprobs shape [B,N]"
                )
            if draft_logprobs.shape != draft_ids.shape:
                raise ValueError(
                    "speculative_step requires draft_logprobs shape to match draft_ids "
                    f"({tuple(draft_logprobs.shape)} vs {tuple(draft_ids.shape)})"
                )

        out, lse, target_logprobs = self.speculative_verify(
            q_target,
            draft_ids,
            k_cache=k_cache,
            v_cache=v_cache,
            **kwargs,
        )

        if draft_logprobs is None:
            accept_mask = target_logprobs >= float(accept_logprob_delta)
        else:
            accept_mask = (
                target_logprobs.astype(mx.float32)
                - draft_logprobs.astype(mx.float32)
            ) >= float(accept_logprob_delta)

        accepted_prefix_lens = self._accepted_prefix_lens_from_mask(accept_mask)
        token_idx = mx.arange(int(draft_ids.shape[1]), dtype=mx.int32)[None, :]
        prefix_mask = token_idx < accepted_prefix_lens[:, None]
        minus_one = mx.full(draft_ids.shape, -1, dtype=draft_ids.dtype)

        accepted_ids = mx.where(prefix_mask, draft_ids, minus_one)
        rejected_ids = mx.where(
            prefix_mask,
            minus_one,
            draft_ids,
        )

        self._speculative_step_used = True
        self._last_speculative_step = {
            "batch": int(q_target.shape[0]),
            "tokens": int(q_target.shape[2]),
            "accept_logprob_delta": float(accept_logprob_delta),
            "used_explicit_cache": bool(k_cache is not None),
            "query_layout": self.query_layout,
            "backend": self.backend,
        }
        return {
            "out": out,
            "lse": lse,
            "target_logprobs": target_logprobs,
            "accept_mask": accept_mask,
            "accepted_prefix_lens": accepted_prefix_lens,
            "accepted_ids": accepted_ids,
            "rejected_ids": rejected_ids,
        }

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
            f"prefix_cache_size={self.metadata['prefix_cache_size']}, "
            f"active_prefix_id={self.metadata['active_prefix_id']!r}, "
            f"splitfuse_active={self.metadata['splitfuse_active']}, "
            f"speculative_verify_active={self.metadata['speculative_verify_active']}, "
            f"speculative_step_active={self.metadata['speculative_step_active']}, "
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

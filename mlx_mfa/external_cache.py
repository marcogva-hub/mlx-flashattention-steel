"""External/offload-oriented KV cache adapter interfaces.

This module provides a small extension-point for LMCache-like future backends.
The first concrete backend in this repo is local host-memory storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class ExternalKVCacheCapabilities:
    put: bool = True
    fetch: bool = True
    prefetch: bool = True
    evict: bool = True
    multi_seq: bool = True


class ExternalKVCacheAdapter:
    """Abstract external-cache adapter surface for hybrid/offload flows."""

    kind: str = "external"

    @property
    def capabilities(self) -> ExternalKVCacheCapabilities:
        return ExternalKVCacheCapabilities()

    def put(self, seq_id: int, k, v, *, meta: Optional[dict[str, Any]] = None) -> None:
        raise NotImplementedError

    def fetch(self, seq_id: int):
        raise NotImplementedError

    def prefetch(self, seq_id: int) -> None:
        raise NotImplementedError

    def evict(self, seq_id: int) -> None:
        raise NotImplementedError

    def has(self, seq_id: int) -> bool:
        raise NotImplementedError

    def seq_length(self, seq_id: int) -> int:
        raise NotImplementedError

    @property
    def offloaded_seq_ids(self) -> tuple[int, ...]:
        return ()

    @property
    def state(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "offloaded_seq_ids": self.offloaded_seq_ids,
            "capabilities": self.capabilities.__dict__,
        }


class LocalHostKVStoreAdapter(ExternalKVCacheAdapter):
    """Local host-memory adapter used as the first real offload backend.

    Stores per-sequence K/V payloads as numpy arrays in process memory.
    This is intentionally local-only, but exercises real offload/reload state
    transitions for HybridKVCache.
    """

    kind = "local_host"

    def __init__(self) -> None:
        self._records: dict[int, dict[str, Any]] = {}
        self._last_prefetch: Optional[dict[str, Any]] = None

    def put(self, seq_id: int, k, v, *, meta: Optional[dict[str, Any]] = None) -> None:
        sid = int(seq_id)
        # Store mx.array directly — zero-copy on Apple Silicon unified memory.
        # The numpy roundtrip was only needed for a future remote/network backend;
        # LocalHostKVStoreAdapter is explicitly local so we skip the copy.
        # P5: complete K/V persistence contract via the single shared helper —
        # this persists K/V offload state for reload→attention; an inconsistent
        # K/V pair is rejected at the put surface. No fixed geometry → K↔V mutual.
        from mlx_mfa._persist_validate import assert_kv_persist_compat
        # host KV-store, dtype-agnostic storage → accepts the supported input
        # precisions; K↔V dtype consistency still enforced (a seq's K/V share dtype).
        assert_kv_persist_compat(k, v, "LocalHostKVStoreAdapter.put",
                                 accepted_dtypes=(mx.float16, mx.bfloat16))
        self._records[sid] = {
            "k": k,
            "v": v,
            "shape": tuple(k.shape),
            "dtype": str(k.dtype),
            "stored_at": time.time(),
            "meta": dict(meta or {}),
        }

    def fetch(self, seq_id: int):
        sid = int(seq_id)
        rec = self._records.get(sid)
        if rec is None:
            raise KeyError(f"No offloaded KV payload for seq_id={sid}")
        return rec["k"], rec["v"]  # zero-copy return

    def prefetch(self, seq_id: int) -> None:
        sid = int(seq_id)
        if sid not in self._records:
            raise KeyError(f"No offloaded KV payload for seq_id={sid}")
        self._last_prefetch = {
            "seq_id": sid,
            "at": time.time(),
        }

    def evict(self, seq_id: int) -> None:
        self._records.pop(int(seq_id), None)

    def has(self, seq_id: int) -> bool:
        return int(seq_id) in self._records

    def seq_length(self, seq_id: int) -> int:
        rec = self._records.get(int(seq_id))
        if rec is None:
            return 0
        return int(rec["shape"][2])

    @property
    def offloaded_seq_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._records.keys()))

    @property
    def state(self) -> dict[str, Any]:
        entries = {
            sid: {
                "shape": tuple(rec["shape"]),
                "dtype": rec["dtype"],
                "stored_at": rec["stored_at"],
                "meta": dict(rec.get("meta", {})),
            }
            for sid, rec in sorted(self._records.items())
        }
        return {
            "kind": self.kind,
            "offloaded_seq_ids": self.offloaded_seq_ids,
            "num_records": len(self._records),
            "entries": entries,
            "last_prefetch": self._last_prefetch,
            "capabilities": self.capabilities.__dict__,
        }


__all__ = [
    "ExternalKVCacheAdapter",
    "ExternalKVCacheCapabilities",
    "LocalHostKVStoreAdapter",
]

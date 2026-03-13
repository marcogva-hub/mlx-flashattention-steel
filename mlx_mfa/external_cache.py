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

    @staticmethod
    def _to_numpy_preserve(arr) -> tuple[np.ndarray, str]:
        if arr.dtype == mx.bfloat16:
            # NumPy lacks native bfloat16 support; store as float32 and restore.
            return np.array(arr.astype(mx.float32)), "bfloat16"
        if arr.dtype == mx.float16:
            return np.array(arr), "float16"
        if arr.dtype == mx.float32:
            return np.array(arr), "float32"
        # Fallback for uncommon dtypes.
        return np.array(arr.astype(mx.float32)), str(arr.dtype)

    @staticmethod
    def _restore_mx(arr_np: np.ndarray, original_dtype: str):
        out = mx.array(arr_np)
        if original_dtype == "bfloat16":
            return out.astype(mx.bfloat16)
        if original_dtype == "float16":
            return out.astype(mx.float16)
        if original_dtype == "float32":
            return out.astype(mx.float32)
        return out

    def put(self, seq_id: int, k, v, *, meta: Optional[dict[str, Any]] = None) -> None:
        sid = int(seq_id)
        k_np, k_dtype = self._to_numpy_preserve(k)
        v_np, v_dtype = self._to_numpy_preserve(v)
        if k_np.ndim != 4 or v_np.ndim != 4:
            raise ValueError("LocalHostKVStoreAdapter.put expects 4-D K/V arrays")
        if k_np.shape[:3] != v_np.shape[:3] or k_np.shape[-1] != v_np.shape[-1]:
            raise ValueError("LocalHostKVStoreAdapter.put requires matching K/V shapes")
        self._records[sid] = {
            "k": k_np,
            "v": v_np,
            "k_dtype": k_dtype,
            "v_dtype": v_dtype,
            "shape": tuple(k_np.shape),
            "stored_at": time.time(),
            "meta": dict(meta or {}),
        }

    def fetch(self, seq_id: int):
        sid = int(seq_id)
        rec = self._records.get(sid)
        if rec is None:
            raise KeyError(f"No offloaded KV payload for seq_id={sid}")
        k = self._restore_mx(rec["k"], rec["k_dtype"])
        v = self._restore_mx(rec["v"], rec["v_dtype"])
        return k, v

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
                "k_dtype": rec["k_dtype"],
                "v_dtype": rec["v_dtype"],
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

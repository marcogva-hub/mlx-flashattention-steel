"""Cache abstraction helpers for serving-oriented runtime flows.

This module provides a small capability-driven adapter surface over concrete
cache implementations (`DenseKVCache`, `PagedKVCache`, `QuantizedKVCache`).

The goal is structural clarity and future extension points (hybrid/offload
cache work), without forcing all cache types to expose identical internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


class KVCacheOperationUnsupported(RuntimeError):
    """Raised when a cache adapter operation is not supported."""


@dataclass(frozen=True)
class KVCacheCapabilities:
    """Capability flags for a cache adapter."""

    append: bool = True
    reset: bool = True
    seq_length: bool = True
    attention_view: bool = False
    paged_pool: bool = False
    quantized_view: bool = False
    multi_seq: bool = False


class KVCacheAdapter:
    """Capability-oriented wrapper around a concrete cache object."""

    kind: str = "unknown"

    def __init__(self, cache: Any):
        self.cache = cache

    @property
    def capabilities(self) -> KVCacheCapabilities:
        return KVCacheCapabilities()

    def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
        self.cache.append(k_new, v_new, seq_id=seq_id)

    def reset(self, *, seq_id: Optional[int] = None) -> None:
        if seq_id is None:
            self.cache.reset()
        else:
            self.cache.reset(seq_id=seq_id)

    def seq_length(self, seq_id: int = 0) -> int:
        if hasattr(self.cache, "seq_length"):
            return int(self.cache.seq_length(seq_id))
        if hasattr(self.cache, "seqlen"):
            if seq_id != 0:
                raise KVCacheOperationUnsupported(
                    "This cache tracks a single sequence (seq_id must be 0)"
                )
            return int(self.cache.seqlen)
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose seq length"
        )

    def attention_k(self, seq_id: int = 0):
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose attention K view"
        )

    def attention_v(self, seq_id: int = 0):
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose attention V view"
        )

    def paged_pool(self):
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose paged pools"
        )

    def paged_tables(self, seq_ids: list[int]):
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose paged tables"
        )

    def active_seq_ids(self) -> tuple[int, ...]:
        return ()

    def quantized_view(self):
        raise KVCacheOperationUnsupported(
            f"Cache type {type(self.cache).__name__} does not expose quantized view"
        )


class DenseKVCacheAdapter(KVCacheAdapter):
    kind = "dense"

    @property
    def capabilities(self) -> KVCacheCapabilities:
        return KVCacheCapabilities(attention_view=True, multi_seq=False)

    def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
        # Dense cache accepts seq_id for compatibility, but tracks one sequence.
        self.cache.append(k_new, v_new, seq_id=seq_id)

    def reset(self, *, seq_id: Optional[int] = None) -> None:
        # Dense cache ignores seq_id in current implementation.
        self.cache.reset(seq_id=seq_id)

    def attention_k(self, seq_id: int = 0):
        if hasattr(self.cache, "k_for_attention"):
            return self.cache.k_for_attention(seq_id)
        if seq_id != 0:
            raise KVCacheOperationUnsupported(
                "Dense cache adapter supports only seq_id=0 attention views"
            )
        return self.cache.k

    def attention_v(self, seq_id: int = 0):
        if hasattr(self.cache, "v_for_attention"):
            return self.cache.v_for_attention(seq_id)
        if seq_id != 0:
            raise KVCacheOperationUnsupported(
                "Dense cache adapter supports only seq_id=0 attention views"
            )
        return self.cache.v


class PagedKVCacheAdapter(KVCacheAdapter):
    kind = "paged"

    @property
    def capabilities(self) -> KVCacheCapabilities:
        return KVCacheCapabilities(
            attention_view=True,
            paged_pool=True,
            multi_seq=True,
        )

    def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
        self.cache.append(k_new, v_new, seq_id=seq_id)

    def reset(self, *, seq_id: Optional[int] = None) -> None:
        self.cache.reset(seq_id=seq_id)

    def attention_k(self, seq_id: int = 0):
        return self.cache.k_for_attention(seq_id)

    def attention_v(self, seq_id: int = 0):
        return self.cache.v_for_attention(seq_id)

    def paged_pool(self):
        return self.cache.k_pool, self.cache.v_pool, int(self.cache.block_size)

    def paged_tables(self, seq_ids: list[int]):
        return self.cache.get_block_table(seq_ids), self.cache.get_seq_lens(seq_ids)

    def active_seq_ids(self) -> tuple[int, ...]:
        seq_lengths = getattr(self.cache, "seq_lengths", {})
        return tuple(sorted(int(sid) for sid in seq_lengths.keys()))


class QuantizedKVCacheAdapter(KVCacheAdapter):
    kind = "quantized"

    @property
    def capabilities(self) -> KVCacheCapabilities:
        return KVCacheCapabilities(quantized_view=True, multi_seq=False)

    def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
        if seq_id != 0:
            raise KVCacheOperationUnsupported(
                "QuantizedKVCache currently supports only seq_id=0"
            )
        self.cache.append(k_new, v_new)

    def reset(self, *, seq_id: Optional[int] = None) -> None:
        if seq_id not in (None, 0):
            raise KVCacheOperationUnsupported(
                "QuantizedKVCache currently supports only seq_id=0"
            )
        self.cache.reset()

    def quantized_view(self):
        return self.cache.k_int8, self.cache.k_scale, self.cache.v


class HybridKVCache:
    """Tiered cache wrapper with explicit residency metadata.

    Tier semantics in this pass:
    - primary cache = hot tier
    - secondary cache = cold tier (optional)

    This class now tracks real residency/recency state and exposes inspectable
    cache metadata, but initial behavior remains conservative until promotion /
    demotion policy is enabled in follow-up steps.
    """

    def __init__(
        self,
        primary_cache: Any,
        secondary_cache: Optional[Any] = None,
        *,
        policy: str = "lru",
        hot_seq_capacity: int = 1,
    ) -> None:
        if hot_seq_capacity <= 0:
            raise ValueError("hot_seq_capacity must be > 0")
        self.primary_cache = primary_cache
        self.secondary_cache = secondary_cache
        self.policy = str(policy)
        self.hot_seq_capacity = int(hot_seq_capacity)
        self._primary_adapter = adapt_kv_cache(primary_cache)
        self._secondary_adapter = (
            None if secondary_cache is None else adapt_kv_cache(secondary_cache)
        )

        # Tier/residency state.
        self._residency: dict[int, str] = {}
        self._hot_seq_ids: set[int] = set()
        self._cold_seq_ids: set[int] = set()
        self._pinned_seq_ids: set[int] = set()
        self._prefetch_intent: set[int] = set()

        # Recency / event metadata.
        self._tick = 0
        self._last_access_tick: dict[int, int] = {}
        self._promotion_count = 0
        self._demotion_count = 0
        self._eviction_count = 0
        self._last_promotion: Optional[dict[str, Any]] = None
        self._last_demotion: Optional[dict[str, Any]] = None
        self._last_eviction: Optional[dict[str, Any]] = None
        self._last_prefetch_intent: Optional[dict[str, Any]] = None

    @property
    def ready_for_production(self) -> bool:
        """Hybrid behavior is available in this pass (local tiering only)."""
        return True

    def _touch(self, seq_id: int, *, reason: str) -> None:
        self._tick += 1
        self._last_access_tick[int(seq_id)] = int(self._tick)
        self._last_prefetch_intent = {
            "seq_id": int(seq_id),
            "reason": str(reason),
            "tick": int(self._tick),
        }

    def _set_residency(self, seq_id: int, tier: str, *, reason: str) -> None:
        sid = int(seq_id)
        if tier not in ("hot", "cold"):
            raise ValueError(f"Unknown residency tier: {tier!r}")
        self._residency[sid] = tier
        if tier == "hot":
            self._hot_seq_ids.add(sid)
            self._cold_seq_ids.discard(sid)
        else:
            self._cold_seq_ids.add(sid)
            self._hot_seq_ids.discard(sid)
        self._touch(sid, reason=reason)

    @property
    def residency_map(self) -> dict[int, str]:
        return dict(sorted(self._residency.items()))

    @property
    def hot_occupancy(self) -> int:
        return len(self._hot_seq_ids)

    @property
    def cold_occupancy(self) -> int:
        return len(self._cold_seq_ids)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "hot_seq_capacity": self.hot_seq_capacity,
            "hot_seq_ids": tuple(sorted(self._hot_seq_ids)),
            "cold_seq_ids": tuple(sorted(self._cold_seq_ids)),
            "residency_map": self.residency_map,
            "pinned_seq_ids": tuple(sorted(self._pinned_seq_ids)),
            "prefetch_intent_seq_ids": tuple(sorted(self._prefetch_intent)),
            "last_access_tick": dict(sorted(self._last_access_tick.items())),
            "promotion_count": self._promotion_count,
            "demotion_count": self._demotion_count,
            "eviction_count": self._eviction_count,
            "last_promotion": self._last_promotion,
            "last_demotion": self._last_demotion,
            "last_eviction": self._last_eviction,
            "last_prefetch_intent": self._last_prefetch_intent,
            "has_secondary_tier": self._secondary_adapter is not None,
            "ready_for_production": self.ready_for_production,
        }

    def append(self, k_new, v_new, seq_id: int = 0) -> None:
        sid = int(seq_id)
        self._primary_adapter.append(k_new, v_new, seq_id=sid)
        self._set_residency(sid, "hot", reason="append")

    def reset(self, seq_id: Optional[int] = None):
        self._primary_adapter.reset(seq_id=seq_id)
        if self._secondary_adapter is not None:
            try:
                self._secondary_adapter.reset(seq_id=seq_id)
            except KVCacheOperationUnsupported:
                # Secondary tier may not support seq-id scoped reset yet.
                self._secondary_adapter.reset(seq_id=None)

        if seq_id is None:
            self._residency.clear()
            self._hot_seq_ids.clear()
            self._cold_seq_ids.clear()
            self._pinned_seq_ids.clear()
            self._prefetch_intent.clear()
            self._last_access_tick.clear()
        else:
            sid = int(seq_id)
            self._residency.pop(sid, None)
            self._hot_seq_ids.discard(sid)
            self._cold_seq_ids.discard(sid)
            self._pinned_seq_ids.discard(sid)
            self._prefetch_intent.discard(sid)
            self._last_access_tick.pop(sid, None)
        return self

    def seq_length(self, seq_id: int = 0) -> int:
        sid = int(seq_id)
        tier = self._residency.get(sid)
        if tier == "hot":
            return self._primary_adapter.seq_length(sid)
        if tier == "cold" and self._secondary_adapter is not None:
            return self._secondary_adapter.seq_length(sid)
        # Conservative fallback for legacy state.
        length = self._primary_adapter.seq_length(sid)
        if length > 0:
            return length
        if self._secondary_adapter is not None:
            try:
                return self._secondary_adapter.seq_length(sid)
            except KVCacheOperationUnsupported:
                return 0
        return 0

    def k_for_attention(self, seq_id: int = 0):
        sid = int(seq_id)
        self._touch(sid, reason="k_for_attention")
        return self._primary_adapter.attention_k(sid)

    def v_for_attention(self, seq_id: int = 0):
        sid = int(seq_id)
        self._touch(sid, reason="v_for_attention")
        return self._primary_adapter.attention_v(sid)

    def offload_seq(self, seq_id: int) -> None:
        raise NotImplementedError(
            "HybridKVCache.offload_seq will be backed by real tier transitions "
            "in this branch; remote/offload backends remain future work."
        )

    def prefetch_seq(self, seq_id: int) -> None:
        raise NotImplementedError(
            "HybridKVCache.prefetch_seq will be wired to local prefetch intent "
            "and hot-tier warmup in this branch."
        )

    def promote_seq(self, seq_id: int) -> None:
        raise NotImplementedError(
            "HybridKVCache.promote_seq will be backed by explicit residency "
            "transitions in this branch."
        )

    def __repr__(self) -> str:
        sec = "none" if self.secondary_cache is None else type(self.secondary_cache).__name__
        return (
            f"HybridKVCache(primary={type(self.primary_cache).__name__}, "
            f"secondary={sec}, policy={self.policy!r}, "
            f"hot={sorted(self._hot_seq_ids)}, cold={sorted(self._cold_seq_ids)}, "
            f"ready_for_production={self.ready_for_production})"
        )


class HybridKVCacheAdapter(KVCacheAdapter):
    kind = "hybrid"

    @property
    def capabilities(self) -> KVCacheCapabilities:
        inner = self.cache._primary_adapter.capabilities
        return KVCacheCapabilities(
            append=inner.append,
            reset=inner.reset,
            seq_length=inner.seq_length,
            attention_view=inner.attention_view,
            paged_pool=inner.paged_pool,
            quantized_view=inner.quantized_view,
            multi_seq=inner.multi_seq,
        )

    def append(self, k_new, v_new, *, seq_id: int = 0) -> None:
        self.cache.append(k_new, v_new, seq_id=seq_id)

    def reset(self, *, seq_id: Optional[int] = None) -> None:
        self.cache.reset(seq_id=seq_id)

    def seq_length(self, seq_id: int = 0) -> int:
        return self.cache.seq_length(seq_id)

    def attention_k(self, seq_id: int = 0):
        return self.cache.k_for_attention(seq_id)

    def attention_v(self, seq_id: int = 0):
        return self.cache.v_for_attention(seq_id)

    def paged_pool(self):
        return self.cache._primary_adapter.paged_pool()

    def paged_tables(self, seq_ids: list[int]):
        return self.cache._primary_adapter.paged_tables(seq_ids)

    def active_seq_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(s) for s in self.cache.residency_map.keys()))

    def quantized_view(self):
        return self.cache._primary_adapter.quantized_view()


def adapt_kv_cache(cache: Any) -> KVCacheAdapter:
    """Build a capability adapter for a concrete cache object."""
    if cache is None:
        raise ValueError("adapt_kv_cache requires a non-None cache")

    cls_name = type(cache).__name__
    if cls_name == "PagedKVCache":
        return PagedKVCacheAdapter(cache)
    if cls_name == "DenseKVCache":
        return DenseKVCacheAdapter(cache)
    if cls_name == "QuantizedKVCache":
        return QuantizedKVCacheAdapter(cache)
    if cls_name == "HybridKVCache":
        return HybridKVCacheAdapter(cache)

    # Fallback duck-typed adaptation for cache-like custom implementations.
    if hasattr(cache, "get_block_table") and hasattr(cache, "k_pool"):
        return PagedKVCacheAdapter(cache)
    if hasattr(cache, "k_int8") and hasattr(cache, "k_scale"):
        return QuantizedKVCacheAdapter(cache)
    if hasattr(cache, "k") and hasattr(cache, "v"):
        return DenseKVCacheAdapter(cache)

    return KVCacheAdapter(cache)


def resolve_context_cache(context: Any) -> Any:
    """Resolve the primary cache object from an inference/runtime context."""
    if hasattr(context, "cache"):
        return context.cache
    if hasattr(context, "_cache"):
        return context._cache
    return None


def resolve_context_cache_adapter(context: Any) -> KVCacheAdapter:
    """Resolve and adapt the primary cache object from context."""
    cache = resolve_context_cache(context)
    if cache is None:
        raise ValueError(
            f"Context {type(context).__name__} does not expose cache or _cache"
        )
    return adapt_kv_cache(cache)

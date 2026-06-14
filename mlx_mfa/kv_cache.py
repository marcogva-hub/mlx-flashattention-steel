"""Cache abstraction helpers for serving-oriented runtime flows.

This module provides a small capability-driven adapter surface over concrete
cache implementations (`DenseKVCache`, `PagedKVCache`, `QuantizedKVCache`).

The goal is structural clarity and future extension points (hybrid/offload
cache work), without forcing all cache types to expose identical internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from mlx_mfa.external_cache import ExternalKVCacheAdapter


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
    external_offload: bool = False


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
        external_adapter: Optional[ExternalKVCacheAdapter] = None,
        *,
        policy: str = "lru",
        hot_seq_capacity: int = 1,
    ) -> None:
        if hot_seq_capacity <= 0:
            raise ValueError("hot_seq_capacity must be > 0")
        self.primary_cache = primary_cache
        self.secondary_cache = secondary_cache
        self.external_adapter = external_adapter
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
        self._offloaded_seq_ids: set[int] = set()
        self._pinned_seq_ids: set[int] = set()
        self._prefetch_intent: set[int] = set()
        # III-4 R4 FIX: tombstones for sequences evicted without a secondary/
        # external tier (drop_no_secondary) — their history is gone, so any
        # later touch must fail loudly instead of silently restarting the seq.
        self._evicted_seq_ids: set[int] = set()

        # Recency / event metadata.
        self._tick = 0
        self._last_access_tick: dict[int, int] = {}
        self._promotion_count = 0
        self._demotion_count = 0
        self._eviction_count = 0
        self._reload_count = 0
        self._last_promotion: Optional[dict[str, Any]] = None
        self._last_demotion: Optional[dict[str, Any]] = None
        self._last_eviction: Optional[dict[str, Any]] = None
        self._last_reload: Optional[dict[str, Any]] = None
        self._last_access_event: Optional[dict[str, Any]] = None
        self._last_prefetch_intent: Optional[dict[str, Any]] = None
        self._last_prefetch_action: Optional[dict[str, Any]] = None

    @property
    def ready_for_production(self) -> bool:
        """Hybrid behavior is available in this pass (local tiering only)."""
        return True

    def _touch(self, seq_id: int, *, reason: str) -> None:
        self._tick += 1
        sid = int(seq_id)
        self._last_access_tick[sid] = int(self._tick)
        self._last_access_event = {
            "seq_id": sid,
            "reason": str(reason),
            "tick": int(self._tick),
        }

    def _copy_seq(self, src: KVCacheAdapter, dst: KVCacheAdapter, seq_id: int) -> None:
        sid = int(seq_id)
        # Skip copy when src and dst are the same adapter (promote/demote no-op)
        if src is dst:
            return
        s = int(src.seq_length(sid))
        if s <= 0:
            dst.reset(seq_id=sid)
            return
        k = src.attention_k(sid)
        v = src.attention_v(sid)
        dst.reset(seq_id=sid)
        dst.append(k, v, seq_id=sid)

    def _choose_demotion_victim(self, *, exclude_seq: int) -> Optional[int]:
        candidates = [
            sid for sid in self._hot_seq_ids
            if sid != exclude_seq and sid not in self._pinned_seq_ids
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda sid: self._last_access_tick.get(sid, -1))

    def _demote_seq(self, seq_id: int, *, reason: str) -> None:
        sid = int(seq_id)
        if sid not in self._hot_seq_ids:
            return
        if self.external_adapter is not None:
            k = self._primary_adapter.attention_k(sid)
            v = self._primary_adapter.attention_v(sid)
            self.external_adapter.put(
                sid,
                k,
                v,
                meta={
                    "reason": str(reason),
                    "tick": int(self._tick),
                },
            )
            self._primary_adapter.reset(seq_id=sid)
            self._set_residency(sid, "offloaded", reason=reason)
            self._demotion_count += 1
            self._last_demotion = {
                "seq_id": sid,
                "reason": str(reason),
                "tick": int(self._tick),
                "target": "external",
            }
            return
        if self._secondary_adapter is not None:
            self._copy_seq(self._primary_adapter, self._secondary_adapter, sid)
            # III-4 R2 FIX: the demoted sequence must vacate the primary (hot)
            # tier after the copy — without this reset its blocks stay
            # allocated and the hot tier exhausts (mirrors the
            # external-offload branch above).
            self._primary_adapter.reset(seq_id=sid)
            self._set_residency(sid, "cold", reason=reason)
            self._demotion_count += 1
            self._last_demotion = {
                "seq_id": sid,
                "reason": str(reason),
                "tick": int(self._tick),
            }
        else:
            self._primary_adapter.reset(seq_id=sid)
            self._residency.pop(sid, None)
            self._hot_seq_ids.discard(sid)
            self._cold_seq_ids.discard(sid)
            # III-4 R4 FIX: record the dropped sequence as a tombstone — its
            # KV history is irrecoverable (no secondary/external tier), so a
            # later append/access must raise instead of silently restarting.
            self._evicted_seq_ids.add(sid)
            self._eviction_count += 1
            self._last_eviction = {
                "seq_id": sid,
                "reason": str(reason),
                "tick": int(self._tick),
                "mode": "drop_no_secondary",
            }

    def _reload_offloaded_seq(self, seq_id: int, *, reason: str) -> None:
        sid = int(seq_id)
        if self.external_adapter is None:
            raise KVCacheOperationUnsupported(
                "Cannot reload offloaded sequence without external adapter"
            )
        if not self.external_adapter.has(sid):
            raise KVCacheOperationUnsupported(
                f"Offloaded sequence {sid} is missing from external adapter"
            )
        self._ensure_hot_capacity(incoming_seq=sid)
        k, v = self.external_adapter.fetch(sid)
        self._primary_adapter.reset(seq_id=sid)
        self._primary_adapter.append(k, v, seq_id=sid)
        self._set_residency(sid, "hot", reason=reason)
        self._promotion_count += 1
        self._reload_count += 1
        self._last_promotion = {
            "seq_id": sid,
            "reason": str(reason),
            "tick": int(self._tick),
            "source": "external",
        }
        self._last_reload = {
            "seq_id": sid,
            "reason": str(reason),
            "tick": int(self._tick),
        }

    def _ensure_hot_capacity(self, *, incoming_seq: int) -> None:
        if len(self._hot_seq_ids) < self.hot_seq_capacity:
            return
        if incoming_seq in self._hot_seq_ids:
            return
        victim = self._choose_demotion_victim(exclude_seq=incoming_seq)
        if victim is None:
            raise KVCacheOperationUnsupported(
                "HybridKVCache hot tier is full and no demotion victim is "
                "available (all candidates pinned or excluded)."
            )
        self._demote_seq(victim, reason="capacity_pressure")

    def _promote_seq(self, seq_id: int, *, reason: str) -> None:
        sid = int(seq_id)
        self._ensure_hot_capacity(incoming_seq=sid)
        if self._secondary_adapter is not None:
            self._copy_seq(self._secondary_adapter, self._primary_adapter, sid)
        self._set_residency(sid, "hot", reason=reason)
        self._promotion_count += 1
        self._last_promotion = {
            "seq_id": sid,
            "reason": str(reason),
            "tick": int(self._tick),
        }

    def _ensure_hot(self, seq_id: int, *, reason: str) -> None:
        sid = int(seq_id)
        # III-4 R4 FIX: sequences evicted via drop_no_secondary lost their KV
        # history permanently — touching them again must fail loudly (the
        # tombstone is cleared only by an explicit reset of that seq_id).
        if sid in self._evicted_seq_ids:
            raise KVCacheOperationUnsupported(
                f"Sequence {sid} history was evicted (no secondary/external "
                f"tier to demote to); its KV state is irrecoverable. Call "
                f"reset(seq_id={sid}) to explicitly restart the sequence."
            )
        # III-4 R3 FIX: a single-seq primary (e.g. DenseKVCache ignores
        # seq_id) silently interleaves distinct sequences into one buffer —
        # the unknown-residency path below would infer a second sid as "hot"
        # from the first sequence's length. Reject a second distinct seq_id
        # before any state is touched (a single non-zero sid is fine: the
        # primary ignores the id, so one tracked sequence stays coherent).
        if not self._primary_adapter.capabilities.multi_seq:
            other_sids = [s for s in self._residency if s != sid]
            if other_sids:
                raise KVCacheOperationUnsupported(
                    f"HybridKVCache primary tier "
                    f"({self._primary_adapter.kind}) tracks a single "
                    f"sequence; seq_id={sid} would silently interleave with "
                    f"already-tracked seq_id(s) {sorted(other_sids)}. Use a "
                    f"multi-seq primary (e.g. PagedKVCache) for "
                    f"multi-sequence workloads."
                )
        tier = self._residency.get(sid)
        if tier == "hot":
            self._touch(sid, reason=reason)
            return
        if tier == "offloaded":
            self._reload_offloaded_seq(sid, reason=reason)
            return
        if tier == "cold":
            self._promote_seq(sid, reason=reason)
            return
        # Unknown residency: infer from adapters.
        if int(self._primary_adapter.seq_length(sid)) > 0:
            self._set_residency(sid, "hot", reason=f"{reason}:infer_hot")
            return
        if self._secondary_adapter is not None and int(self._secondary_adapter.seq_length(sid)) > 0:
            self._set_residency(sid, "cold", reason=f"{reason}:infer_cold")
            self._promote_seq(sid, reason=reason)
            return
        if self.external_adapter is not None and self.external_adapter.has(sid):
            self._set_residency(sid, "offloaded", reason=f"{reason}:infer_offloaded")
            self._reload_offloaded_seq(sid, reason=reason)
            return
        # Fresh sequence path.
        self._ensure_hot_capacity(incoming_seq=sid)
        self._set_residency(sid, "hot", reason=f"{reason}:new")

    def mark_pinned(self, seq_id: int, *, pinned: bool = True) -> None:
        sid = int(seq_id)
        if pinned:
            self._pinned_seq_ids.add(sid)
        else:
            self._pinned_seq_ids.discard(sid)

    def mark_for_prefetch(self, seq_id: int, *, reason: str = "manual") -> None:
        sid = int(seq_id)
        self._prefetch_intent.add(sid)
        self._last_prefetch_intent = {
            "seq_id": int(seq_id),
            "reason": str(reason),
            "tick": int(self._tick),
        }

    def clear_prefetch_intent(self, seq_id: Optional[int] = None) -> None:
        if seq_id is None:
            self._prefetch_intent.clear()
            return
        self._prefetch_intent.discard(int(seq_id))

    @property
    def pending_prefetch_seq_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(sid) for sid in self._prefetch_intent))

    def _set_residency(self, seq_id: int, tier: str, *, reason: str) -> None:
        sid = int(seq_id)
        if tier not in ("hot", "cold", "offloaded"):
            raise ValueError(f"Unknown residency tier: {tier!r}")
        self._residency[sid] = tier
        if tier == "hot":
            self._hot_seq_ids.add(sid)
            self._cold_seq_ids.discard(sid)
            self._offloaded_seq_ids.discard(sid)
        elif tier == "cold":
            self._cold_seq_ids.add(sid)
            self._hot_seq_ids.discard(sid)
            self._offloaded_seq_ids.discard(sid)
        else:
            self._offloaded_seq_ids.add(sid)
            self._hot_seq_ids.discard(sid)
            self._cold_seq_ids.discard(sid)
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
    def offloaded_occupancy(self) -> int:
        return len(self._offloaded_seq_ids)

    @property
    def state(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "hot_seq_capacity": self.hot_seq_capacity,
            "hot_seq_ids": tuple(sorted(self._hot_seq_ids)),
            "cold_seq_ids": tuple(sorted(self._cold_seq_ids)),
            "offloaded_seq_ids": tuple(sorted(self._offloaded_seq_ids)),
            "residency_map": self.residency_map,
            "pinned_seq_ids": tuple(sorted(self._pinned_seq_ids)),
            "prefetch_intent_seq_ids": tuple(sorted(self._prefetch_intent)),
            "last_access_tick": dict(sorted(self._last_access_tick.items())),
            "promotion_count": self._promotion_count,
            "demotion_count": self._demotion_count,
            "eviction_count": self._eviction_count,
            "reload_count": self._reload_count,
            "last_promotion": self._last_promotion,
            "last_demotion": self._last_demotion,
            "last_eviction": self._last_eviction,
            "last_reload": self._last_reload,
            "last_access_event": self._last_access_event,
            "last_prefetch_intent": self._last_prefetch_intent,
            "last_prefetch_action": self._last_prefetch_action,
            "has_secondary_tier": self._secondary_adapter is not None,
            "has_external_offload": self.external_adapter is not None,
            "external_offload_state": (
                None if self.external_adapter is None else self.external_adapter.state
            ),
            "ready_for_production": self.ready_for_production,
        }

    @property
    def debug_state(self) -> dict[str, Any]:
        """Alias used by runtime/debug tooling."""
        return self.state

    def append(self, k_new, v_new, seq_id: int = 0) -> None:
        sid = int(seq_id)
        self._ensure_hot(sid, reason="append")
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
        if self.external_adapter is not None:
            if seq_id is None:
                for sid in list(self.external_adapter.offloaded_seq_ids):
                    self.external_adapter.evict(int(sid))
            else:
                self.external_adapter.evict(int(seq_id))

        if seq_id is None:
            self._residency.clear()
            self._hot_seq_ids.clear()
            self._cold_seq_ids.clear()
            self._offloaded_seq_ids.clear()
            self._pinned_seq_ids.clear()
            self._prefetch_intent.clear()
            # III-4 R4 FIX: explicit full reset clears eviction tombstones.
            self._evicted_seq_ids.clear()
            self._last_access_tick.clear()
            self._last_access_event = None
            self._last_prefetch_action = None
            # Repo review 2026-05: a full reset previously left these event
            # records (and the tick counter) from the prior session — `state`
            # reported an empty residency map alongside stale last_promotion/
            # last_eviction entries.  Clear everything on full reset.
            self._last_promotion = None
            self._last_demotion = None
            self._last_eviction = None
            self._last_reload = None
            self._last_prefetch_intent = None
            self._tick = 0
        else:
            sid = int(seq_id)
            self._residency.pop(sid, None)
            self._hot_seq_ids.discard(sid)
            self._cold_seq_ids.discard(sid)
            self._offloaded_seq_ids.discard(sid)
            self._pinned_seq_ids.discard(sid)
            self._prefetch_intent.discard(sid)
            # III-4 R4 FIX: explicit per-seq reset clears the tombstone.
            self._evicted_seq_ids.discard(sid)
            self._last_access_tick.pop(sid, None)
        return self

    def seq_length(self, seq_id: int = 0) -> int:
        sid = int(seq_id)
        tier = self._residency.get(sid)
        if tier == "hot":
            return self._primary_adapter.seq_length(sid)
        if tier == "cold" and self._secondary_adapter is not None:
            return self._secondary_adapter.seq_length(sid)
        if tier == "offloaded" and self.external_adapter is not None:
            return int(self.external_adapter.seq_length(sid))
        # Conservative fallback for legacy state.
        length = self._primary_adapter.seq_length(sid)
        if length > 0:
            return length
        if self._secondary_adapter is not None:
            try:
                return self._secondary_adapter.seq_length(sid)
            except KVCacheOperationUnsupported:
                pass
        if self.external_adapter is not None:
            return int(self.external_adapter.seq_length(sid))
        return 0

    def k_for_attention(self, seq_id: int = 0):
        sid = int(seq_id)
        self._ensure_hot(sid, reason="k_for_attention")
        return self._primary_adapter.attention_k(sid)

    def v_for_attention(self, seq_id: int = 0):
        sid = int(seq_id)
        self._ensure_hot(sid, reason="v_for_attention")
        return self._primary_adapter.attention_v(sid)

    def paged_pool(self):
        if not self._primary_adapter.capabilities.paged_pool:
            raise KVCacheOperationUnsupported(
                "HybridKVCache primary tier does not expose paged_pool capability"
            )
        return self._primary_adapter.paged_pool()

    def paged_tables(self, seq_ids: list[int]):
        if not self._primary_adapter.capabilities.paged_pool:
            raise KVCacheOperationUnsupported(
                "HybridKVCache primary tier does not expose paged table capability"
            )
        norm_ids = [int(sid) for sid in seq_ids]
        for sid in norm_ids:
            self._ensure_hot(sid, reason="paged_tables")
        return self._primary_adapter.paged_tables(norm_ids)

    def active_seq_ids(self) -> tuple[int, ...]:
        ids = set(int(sid) for sid in self._residency.keys())
        if self.external_adapter is not None:
            ids.update(int(sid) for sid in self.external_adapter.offloaded_seq_ids)
        return tuple(sorted(ids))

    def quantized_view(self):
        if not self._primary_adapter.capabilities.quantized_view:
            raise KVCacheOperationUnsupported(
                "HybridKVCache primary tier does not expose quantized_view capability"
            )
        self._ensure_hot(0, reason="quantized_view")
        return self._primary_adapter.quantized_view()

    # Compatibility properties used by existing runtime/context codepaths.
    @property
    def k(self):
        return self.k_for_attention(0)

    @property
    def v(self):
        return self.v_for_attention(0)

    @property
    def seqlen(self) -> int:
        return self.seq_length(0)

    @property
    def k_pool(self):
        k_pool, _, _ = self.paged_pool()
        return k_pool

    @property
    def v_pool(self):
        _, v_pool, _ = self.paged_pool()
        return v_pool

    @property
    def block_size(self) -> int:
        _, _, block_size = self.paged_pool()
        return int(block_size)

    def get_block_table(self, seq_ids: Optional[list[int]] = None):
        ids = list(self.active_seq_ids()) if seq_ids is None else [int(s) for s in seq_ids]
        table, _ = self.paged_tables(ids)
        return table

    def get_seq_lens(self, seq_ids: Optional[list[int]] = None):
        ids = list(self.active_seq_ids()) if seq_ids is None else [int(s) for s in seq_ids]
        _, lens = self.paged_tables(ids)
        return lens

    @property
    def seq_lengths(self) -> dict[int, int]:
        return {sid: self.seq_length(sid) for sid in self.active_seq_ids()}

    @property
    def k_int8(self):
        k_int8, _, _ = self.quantized_view()
        return k_int8

    @property
    def k_scale(self):
        _, k_scale, _ = self.quantized_view()
        return k_scale

    def offload_seq(self, seq_id: int) -> None:
        sid = int(seq_id)
        self._demote_seq(sid, reason="offload_seq")

    def reload_seq(self, seq_id: int, *, reason: str = "manual_reload") -> None:
        sid = int(seq_id)
        tier = self._residency.get(sid)
        if tier == "offloaded":
            self._reload_offloaded_seq(sid, reason=reason)
            return
        self._ensure_hot(sid, reason=reason)

    def prefetch_seq(self, seq_id: int, *, reason: str = "manual") -> None:
        sid = int(seq_id)
        self.mark_for_prefetch(sid, reason=reason)
        if self.external_adapter is not None and self.external_adapter.has(sid):
            self.external_adapter.prefetch(sid)
        self._ensure_hot(sid, reason=f"prefetch:{reason}")
        self._prefetch_intent.discard(sid)
        self._last_prefetch_action = {
            "seq_id": sid,
            "reason": str(reason),
            "tick": int(self._tick),
            "result_tier": self._residency.get(sid, "unknown"),
        }

    def prefetch(
        self,
        seq_ids: list[int] | tuple[int, ...],
        *,
        reason: str = "batch",
    ) -> tuple[int, ...]:
        warmed: list[int] = []
        for sid in seq_ids:
            self.prefetch_seq(int(sid), reason=reason)
            warmed.append(int(sid))
        return tuple(warmed)

    def prepare_hot_window(
        self,
        seq_ids: list[int] | tuple[int, ...],
        *,
        pin: bool = False,
        reason: str = "window",
    ) -> tuple[int, ...]:
        warmed = self.prefetch(seq_ids, reason=reason)
        if pin:
            for sid in warmed:
                self.mark_pinned(sid, pinned=True)
        return warmed

    def promote_seq(self, seq_id: int) -> None:
        sid = int(seq_id)
        self._ensure_hot(sid, reason="promote_seq")

    def __repr__(self) -> str:
        sec = "none" if self.secondary_cache is None else type(self.secondary_cache).__name__
        ext = "none" if self.external_adapter is None else type(self.external_adapter).__name__
        return (
            f"HybridKVCache(primary={type(self.primary_cache).__name__}, "
            f"secondary={sec}, external={ext}, policy={self.policy!r}, "
            f"hot={sorted(self._hot_seq_ids)}, cold={sorted(self._cold_seq_ids)}, "
            f"offloaded={sorted(self._offloaded_seq_ids)}, "
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
            external_offload=self.cache.external_adapter is not None,
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
        return self.cache.paged_pool()

    def paged_tables(self, seq_ids: list[int]):
        return self.cache.paged_tables(seq_ids)

    def active_seq_ids(self) -> tuple[int, ...]:
        return self.cache.active_seq_ids()

    def quantized_view(self):
        return self.cache.quantized_view()


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
    if cls_name == "TurboQuantKVCache":
        from mlx_mfa.turboquant import _make_adapter
        return _make_adapter(cache)

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

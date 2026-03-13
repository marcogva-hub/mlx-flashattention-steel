# Minimal Real KV Offloading Design (Final Serving Completion Pass)

## Scope

This pass implements a **local-only real offload tier** for `HybridKVCache`.
It does not add distributed cache infrastructure or LMCache remote backends.

## Tier Model

- **Hot tier (primary)**:
  - Existing runtime-attention-ready cache (`DenseKVCache` or `PagedKVCache`)
  - This is the tier directly consumed by runtime attention paths.
- **Offloaded tier (secondary external)**:
  - Host-memory materialized K/V payloads stored by a local external adapter.
  - Sequence payloads are serialized as contiguous host arrays.

## Offload Representation

First implementation uses a local host adapter:
- Stores per-sequence K/V payloads in host memory (numpy-backed records).
- Stores metadata (`dtype`, `shape`, timestamps, last prefetch hint).
- Supports `put`, `fetch`, `prefetch`, `evict`, and capability introspection.

## Residency States

`HybridKVCache` residency now distinguishes:
- `hot` — sequence present in primary runtime cache
- `offloaded` — sequence materialized in external offload adapter

(`cold` remains available for legacy local-secondary behavior where used.)

## State Transitions

- **Demotion (capacity pressure / offload request):**
  1. Read attention-ready K/V from hot tier.
  2. Persist to external adapter (`put`).
  3. Reset that sequence in hot tier.
  4. Mark residency as `offloaded`.
- **Promotion / reload (attention access / explicit prefetch):**
  1. Fetch K/V from external adapter.
  2. Ensure hot-tier capacity (demote victim if needed).
  3. Append fetched K/V to primary tier.
  4. Mark residency as `hot`.

## Prefetch Semantics

- `mark_for_prefetch(seq_id)` records intent.
- `prefetch_seq(seq_id)` can trigger proactive reload into hot tier.
- `prepare_hot_window(seq_ids, pin=...)` warms a set of sequences and optionally pins them.

## Runtime Integration

Runtime can opt into hybrid behavior and offload-aware control surfaces.
Metadata exposes residency map and last offload/reload/prefetch events.

## Explicit Non-Goals in This Pass

- No remote/distributed storage backend.
- No network transport.
- No full LMCache protocol implementation.
- No automatic global policy tuning from workload traces.

## Future Work

- Remote adapters (LMCache-like) behind the same external adapter interface.
- Page-level offload granularity for paged pools.
- Asynchronous prefetch pipelines.
- Hardware-aware offload heuristics.

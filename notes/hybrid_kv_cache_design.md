# Hybrid KV Cache Abstraction Design (Serving-Oriented)

Date: 2026-03-13
Branch: `codex/hybrid-kv-cache-abstraction`

## 1) Current cache landscape

Current concrete cache types:

- `DenseKVCache`
  - single-sequence dense history
  - attention-ready contiguous views via `k` / `v`
  - append/reset/seq_length lifecycle
- `PagedKVCache`
  - multi-sequence paged pool
  - block-table + seq-lens metadata
  - contiguous gather path (`k_for_attention` / `v_for_attention`)
- `QuantizedKVCache`
  - dense history + pre-quantized K (`k_int8`, `k_scale`)
  - append/reset/seqlen lifecycle
  - currently not exposed through `KVCacheProtocol`

Current runtime assumptions are mixed:

- dense/sage paths frequently touch `context._cache` directly
- paged paths frequently touch `context.cache` and paged-only members
  (`k_pool`, `v_pool`, `get_block_table`, `get_seq_lens`, `seq_lengths`)
- speculative verify fallback reads either dense fields (`k_cache` / `v_cache`) or
  paged gather methods (`k_for_attention` / `v_for_attention`)

## 2) Problem statement

The runtime is serving-oriented but cache interactions are still partially tied to
concrete class internals. This makes future hybrid/offload work harder because
new cache types would need to mimic many implicit assumptions.

## 3) Minimal abstraction for this pass

Introduce a small capability-driven adapter layer over concrete caches.

Core capability buckets:

- lifecycle:
  - `append(k, v, seq_id=...)`
  - `reset(seq_id=None)`
  - `seq_length(seq_id=...)`
- attention view:
  - `attention_k(seq_id=...)`
  - `attention_v(seq_id=...)`
- paged pool view (optional):
  - `k_pool`, `v_pool`, `block_size`
  - `block_table(seq_ids)`
  - `seq_lens(seq_ids)`
  - `active_seq_ids()`
- quantized K view (optional):
  - `k_int8`, `k_scale`, `v`, `block_size`

The adapter must make unsupported operations explicit (raise clear errors),
not fake-generic.

## 4) Required now vs future-facing

Required now:

- cover existing dense/paged/quantized caches without behavior changes
- let runtime logic depend on adapter capabilities rather than concrete fields
- keep current public APIs intact

Future-facing only (not fully implemented in this pass):

- hybrid tiering (dense+paged, device+host)
- offload adapters (LMCache-like or remote)
- eviction/promotion policies
- async prefetch/commit hooks

## 5) Runtime integration scope in this pass

Use adapters where runtime currently hardcodes concrete cache internals:

- prefix seeding
- paged varlen and paged batch helpers
- chunked prefill packed path cache updates
- speculative verify fallback cache lookup

Keep explicit path constraints where needed (e.g. paged-only operations).

## 6) Hybrid skeleton scope

Add a future-facing skeleton (`HybridKVCache`) that conforms structurally to the
new abstraction surface but is intentionally non-production:

- explicit `NotImplementedError` for unsupported policy operations
- no auto-routing integration
- docs clearly label as groundwork only

## 7) Expected outcome

This pass should improve structural clarity and extension points while keeping
runtime behavior and benchmarks stable. Primary value is maintainability and
future integration readiness, not immediate speedup.

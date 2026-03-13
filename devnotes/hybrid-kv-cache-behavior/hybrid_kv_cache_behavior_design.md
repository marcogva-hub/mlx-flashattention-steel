# Hybrid KV Cache Behavior Model (Real Tiered Semantics)

Date: 2026-03-13
Branch: `codex/hybrid-kv-cache-behavior`

## 1) Tier model

`HybridKVCache` will manage two explicit tiers:

- **hot tier (primary)**
  - actively used sequences
  - always used for attention-ready reads
- **cold tier (secondary)**
  - demoted sequences
  - promotion source when a cold sequence is needed

Both tiers are local cache implementations (dense/paged/quantized compatible
where feasible in this pass).

## 2) Data placement

- `append(seq_id)` writes to hot tier.
- New sequences start in hot tier.
- If `seq_id` is currently cold, append triggers promotion first.

## 3) Promotion and demotion policy

Policy in this pass:

- configurable `hot_seq_capacity` (default 1)
- LRU-style recency tracking over hot-resident sequences
- when hot tier is full and a new/promoted sequence must enter hot:
  1. choose least-recently-used unpinned hot sequence
  2. demote it to cold tier (`copy hot -> cold`, then remove from hot)
  3. promote requested sequence to hot (`copy cold -> hot` when needed)

If no demotion target exists (all hot sequences pinned), raise explicit error.

## 4) Residency tracking

Maintain explicit state:

- `residency_map: seq_id -> {"hot"|"cold"}`
- recency counters / last-access timestamp
- pinned sequence set (optional for active request protection)
- counters and last-event metadata:
  - promotion_count, demotion_count, eviction_count
  - last_promotion, last_demotion, last_eviction
  - last_prefetch_intent

Expose this state through a debug metadata API for runtime/tests.

## 5) Attention-ready views

Attention reads (`k_for_attention`, `v_for_attention`, paged table views)
operate on hot tier.

When requested sequence is cold:
- trigger promotion (cold -> hot)
- then serve from hot tier.

## 6) Prefetch intent semantics (this pass)

`prefetch` is local-tier residency warmup, not remote I/O:

- `mark_for_prefetch(seq_id)` records intent
- `prefetch(seq_id)` ensures seq is hot now
- `prepare_hot_window(seq_ids, pin=False)` prefetches a set and optionally pins

These methods update inspectable prefetch metadata for future offload control
surface compatibility.

## 7) Runtime integration scope

Supported in this pass:

- dense runtime flows (prefill/step/chunked/prefix/speculative)
- paged runtime flows where adapter-capability path already exists

Potentially limited in this pass:

- some quantized multi-sequence semantics (quantized cache is single-sequence)
- hybrid+packed+paged corner combinations under heavy remap pressure

Unsupported combinations must raise explicit, clear errors.

## 8) Future work left intentionally out of this pass

- remote/offloaded tier implementation
- asynchronous transfer orchestration
- background eviction workers
- distributed cache coordination
- LMCache protocol integration

This pass aims for a real, deterministic local tiered behavior milestone.

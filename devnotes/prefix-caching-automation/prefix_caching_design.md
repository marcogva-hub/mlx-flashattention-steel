# Runtime-Integrated Prefix Caching Design

## Current state (before this pass)

Prefix-related capabilities already exist, but are mostly helper-driven:

- `make_shared_prefix_cache(...)`
- `DecodeRuntime.shared_prefix_cache(...)`
- `DecodeRuntime.prefill_shared_prefix(...)`
- `DecodeRuntime.decode_from_shared_prefix(...)`
- `flash_attention_splitfuse(...)` and `DecodeRuntime.splitfuse(...)`

What is still manual:

- Prefix objects are effectively single-use/runtime-local (`_prepared_prefix` only).
- No runtime-managed registry for multiple reusable prefixes.
- Reusing prefix state across multiple later prefill/decode calls requires caller orchestration.
- Prefix reuse is not unified with chunked prefill, paged scheduler flows, or packed-query flows.

## Target semantics for this branch

Make prefix reuse a first-class runtime capability with explicit and inspectable behavior:

1. Runtime-managed prefix registry
   - Register prefix state by `prefix_id`.
   - Keep precomputed `(k_prefix, v_prefix)` and related metadata.
   - Allow explicit activation/selection and removal.

2. Runtime-managed seeding
   - Seed runtime cache from a registered prefix without re-running full prefix prefill orchestration.
   - Dense runtime: seed dense cache directly.
   - Paged runtime: seed one or more `seq_id` rows explicitly.

3. Prefix-aware serving entry point
   - Add a wrapper that seeds a registered prefix then runs suffix prefill via existing runtime methods.
   - Reuse existing `chunked_prefill(...)` implementation for scheduler-friendly chunking.

4. Metadata and observability
   - Runtime metadata should expose:
     - number of registered prefixes,
     - active prefix id,
     - last prefix-reuse operation summary.

## Flow-level integration targets

- Dense runtime
  - Supported: register -> seed -> suffix prefill/chunked prefill.

- Paged runtime
  - Supported: register -> seed by `seq_id` or `seq_ids` -> suffix prefill/chunked prefill.

- Packed query layout (`query_layout="packed"`)
  - Supported where practical through the same prefix-seeding wrapper + existing packed chunked path.
  - Requires explicit `seq_ids` and `cu_seqlens_q` for packed suffix calls.

- Chunked prefill
  - Prefix-aware wrapper should route through `chunked_prefill(..., reset=False)` after prefix seeding.

- Continuous batching
  - Runtime remains scheduler-friendly by keeping explicit `seq_ids` / mapping arguments.
  - No hidden automatic scheduler is introduced in this pass.

## Explicit non-goals in this pass

- No full scheduler framework.
- No speculative broad auto prefix matching/hashing of request payloads.
- No kernel changes.
- No forced support for brittle combinations; unsupported combinations should fail clearly.

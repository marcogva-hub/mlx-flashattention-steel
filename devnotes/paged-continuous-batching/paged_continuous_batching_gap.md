# Paged Continuous Batching Gap Audit

Date: 2026-03-13  
Branch: `codex/paged-continuous-batching`

## Current limitation summary

Continuous-batching-style request-slot remapping is first-class in dense mode
(`cache_batch_idx` in `flash_attention_kvcache`), but still incomplete in paged
mode.

### 1) API layer

- `flash_attention_kvcache(..., cache_batch_idx=...)`
  - dense mode: supported (`k_cache = k_cache[idx]`, `v_cache = v_cache[idx]`)
  - paged mode (with `block_table`):
    - non-append path: no remap support exposed
    - append path (`k_new/v_new`): explicitly raises `NotImplementedError`
- `flash_attention_paged(...)` currently has no remap/index argument.
- `flash_attention_paged_varlen(...)` currently has no remap/index argument.

### 2) Runtime layer

- `DecodeRuntime.prefill/step` in paged mode forwards to
  `PagedInferenceContext.prefill/step`, which are effectively single-sequence
  (`seq_id`) lifecycle methods.
- `DecodeRuntime.paged_varlen(...)` accepts `seq_ids`, which helps packed
  queries, but there is no explicit `cache_batch_idx`-style remap for
  scheduler-maintained slot arrays.
- Runtime metadata does not expose active remap/slot state.

### 3) Cache layer

- `PagedKVCache` stores state by `seq_id -> block list` and can build
  `block_table/seq_lens` for requested `seq_ids`.
- There is no explicit helper to apply a logical-batch remap over an existing
  table/lens matrix (slot-order -> active-order).

### 4) Kernel/dispatch layer

- Native paged kernels already consume `block_table` + `seq_lens` rows.
- No kernel change is required for request-slot remap; remap can be applied in
  Python by row-gathering metadata before dispatch.

## Minimum clean implementation strategy for this branch

1. Add explicit optional remap argument in paged APIs:
   - `flash_attention_paged(..., cache_batch_idx=None)`
   - `flash_attention_paged_varlen(..., cache_batch_idx=None)`
   where `cache_batch_idx` indexes rows of `block_table` and `seq_lens(_kv)`
   (same concept as dense continuous batching).

2. Keep behavior backward-compatible:
   - If no remap is provided, existing behavior is unchanged.
   - Validate remap shape/dtype/range and fail clearly on invalid input.

3. Runtime integration:
   - Extend `DecodeRuntime.paged_varlen(...)` with remap passthrough.
   - Add batched paged decode helper(s) that accept explicit active-request
     order (`seq_ids`) and/or remap (`cache_batch_idx`) for changing active
     sets.

4. Scope guard:
   - Do not attempt fused scheduler kernel rewrite in this pass.
   - Keep append+remap in `flash_attention_kvcache` paged append explicitly
     unsupported unless safely implementable in this pass.

This closes the scheduler-facing gap at API/runtime/caching integration level
while preserving existing paged functionality and kernel stability.

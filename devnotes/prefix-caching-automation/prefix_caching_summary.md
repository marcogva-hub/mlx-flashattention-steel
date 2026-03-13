# Prefix Caching Automation Summary

## What was added

This branch adds runtime-integrated prefix reuse as a first-class capability in
`DecodeRuntime`.

New runtime APIs:
- `register_prefix(...)`
- `list_registered_prefix_ids()`
- `seed_prefix(...)`
- `prefill_with_prefix(...)`
- `drop_prefix(...)`
- `clear_registered_prefixes()`

Runtime metadata now reports:
- `prefix_cache_size`
- `registered_prefix_ids`
- `active_prefix_id`
- `last_prefix_reuse`

## Supported flows

- Dense runtime: prefix register/seed/reuse via `prefill_with_prefix(...)`.
- Paged runtime (batched): per-sequence prefix seeding (`seq_id`/`seq_ids`) and
  suffix execution through chunked prefill routing.
- Paged runtime (packed): prefix-aware suffix path through
  `prefill_with_prefix(..., query_layout="packed")` where `seq_ids` and
  `cu_seqlens_q` are provided.
- Chunked prefill integration: prefix-aware wrapper routes suffix work through
  `chunked_prefill(..., reset=False)`.

## Correctness status

- Added coverage for:
  - prefix registration + metadata,
  - dense prefix reuse parity vs manual seed + chunked suffix path,
  - paged batched prefix reuse parity,
  - paged packed (single-seq) prefix reuse parity,
  - invalid combinations and clear failures.
- Full suite status after this pass: `718 passed`.

## Benchmark outcome

Matrix:
- script: `benchmarks/bench_prefix_caching_runtime.py`
- artifact: `devnotes/prefix_caching_runtime_matrix_latest.json`

Summary:
- Runtime-managed path matches explicit helper correctness (`max_err=0.0` in
  measured rows).
- Paged scenarios show clear wins vs no-reuse baseline.
- Dense scenarios in this setup do not show speedups; main value is runtime/API
  integration and lower orchestration complexity.

## Remaining serving-native gaps

- No fully automatic scheduler layer is introduced in this pass.
- Prefix-id assignment/matching remains explicit (no implicit request hashing).
- Packed multi-sequence prefix automation remains conservative and explicit to
  avoid brittle implicit behavior.

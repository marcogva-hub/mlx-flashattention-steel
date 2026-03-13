# Speculative Decode Runtime Branch Summary

Date: 2026-03-13
Branch: `codex/speculative-decode-runtime`

## What landed

- Added runtime-level speculative API:
  - `DecodeRuntime.speculative_step(...)`
- Preserved low-level primitive:
  - `flash_attention_speculative_verify(...)`
- Added explicit accept/reject bookkeeping outputs from runtime flow:
  - `accept_mask`
  - `accepted_prefix_lens`
  - `accepted_ids`
  - `rejected_ids`
- Extended runtime metadata:
  - `speculative_step_active`
  - `last_speculative_step`

## Supported flows

- Dense runtime cache fallback (default): supported.
- Paged runtime cache fallback: supported for batched query layout with `seq_id`.
- Explicit `k_cache/v_cache`: supported for all backends.

## Explicit limitations (by design in this pass)

- No full speculative scheduler engine.
- No automatic paged packed-query speculative fallback without explicit caches.
- No broad throughput auto-promotion policy from this pass.

## Correctness status

Added tests cover:
- full accept / partial accept / reject-all behavior
- accepted-prefix bookkeeping and token partition outputs
- dense and paged runtime-cache integration
- invalid combinations and shape validation
- metadata visibility for speculative flow activation

## Benchmark status

Matrix:
- script: `benchmarks/bench_speculative_decode_runtime.py`
- artifact: `notes/speculative_decode_runtime_matrix_latest.json`

Observed outcome:
- runtime and manual helper flow match acceptance outputs in all measured rows
- performance deltas are mixed (scenario and accept-rate dependent)
- best interpretation is capability/runtime integration milestone rather than
  broad speedup promotion

## Remaining future work

- Optional: add a higher-level scheduler-oriented speculative loop manager
  (request queue + cache mutation policy) on top of `speculative_step`.
- Optional: evaluate packed-query speculative path integration if a clean
  cache/state contract is defined for paged packed flows.
- Optional: revisit throughput claims only with larger acceptance-rate sweeps
  and end-to-end serving traces.

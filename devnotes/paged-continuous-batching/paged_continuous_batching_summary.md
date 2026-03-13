# Paged Continuous Batching — Branch Summary

Date: 2026-03-13  
Branch: `codex/paged-continuous-batching`

## What was added

1. API-level remap support for paged attention:
   - `flash_attention_paged(..., cache_batch_idx=...)`
   - `flash_attention_paged_varlen(..., cache_batch_idx=...)`
   - paged non-append path in `flash_attention_kvcache(...)` now forwards remap.

2. Runtime-level scheduler-friendly helpers:
   - `DecodeRuntime.paged_prefill_batch(...)`
   - `DecodeRuntime.paged_step_batch(...)`
   - remap-aware `DecodeRuntime.paged_varlen(...)`

3. Runtime metadata additions:
   - `active_seq_ids`
   - `active_cache_batch_idx`

## Remapping model

- `seq_ids` defines stable slot order.
- `cache_batch_idx` projects current active-request order over those slots.
- In paged APIs this is implemented as row-gather over
  `block_table` / `seq_lens(_kv)` before dispatch.
- No kernel-family rewrite is required for this capability.

## vLLM-like usage now supported

- Dynamic active-request reorder for batched paged decode via
  `paged_step_batch(..., seq_ids=..., cache_batch_idx=...)`.
- Packed varlen query dispatch with paged KV + active-order remap via
  `paged_varlen(..., seq_ids=..., cache_batch_idx=...)`.

## Correctness and benchmark outcome

- New tests cover:
  - paged remap parity vs explicit row-gather reference
  - changing active-request order across steps
  - packed-query paged remap parity
  - invalid remap inputs

- Benchmark matrix:
  - script: `benchmarks/bench_paged_continuous_batching.py`
  - artifact: `devnotes/paged_continuous_batching_latest.json`

- Result summary (M1 Max, f16):
  - `paged_step_batch` remap: ~1.02–1.05x vs manual baseline
  - `paged_varlen` remap: ~0.90–0.98x vs manual baseline
  - correctness parity (`max_err = 0.0`) on reported rows

Interpretation: this branch closes a scheduler/runtime capability gap first.
Performance impact is mixed and should not be treated as broad speedup evidence.

## Remaining gaps toward serving-native fusion

- Paged append + remap in `flash_attention_kvcache` remains unsupported.
- Heterogeneous packed-query path is still a runtime bridge, not a fully fused
  single native paged-varlen scheduler kernel.
- Future work can target lower per-step Python/runtime overhead in remap-heavy
  packed varlen workloads.

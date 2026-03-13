# Splitfuse Runtime Integration Design (Deeper Pass)

## Current State

- Low-level helper exists: `flash_attention_splitfuse(...)`.
- `DecodeRuntime.splitfuse(...)` exposes it, but callers still need to assemble
  many inputs manually (especially decode cache tensors).
- Metadata currently tracks only a boolean `splitfuse_active`.

## Gap

Splitfuse is helper-accessible but not yet a first-class runtime route with:
- cache-aware defaults
- prefix-aware convenience wiring
- clear route metadata for serving traces

## Target in This Pass

Add a deeper runtime surface that remains explicit and low-risk:
- `DecodeRuntime.splitfuse_step(...)`:
  - resolves decode cache tensors from runtime cache when feasible
  - supports dense runtime directly
  - supports paged runtime in a narrow path by deriving dense attention views
    for a provided single `seq_id`
- prefix-aware convenience option:
  - `use_registered_prefix=True` + `prefix_id` for prefill side
- richer metadata:
  - route payload (`backend`, `used_registered_prefix`, `used_runtime_cache`,
    `seq_id`, `query_layout`)

## Supported Flows (This Pass)

- Dense runtime: supported for runtime-cache-derived decode side.
- Prefix-related dense flow: supported.
- Paged runtime: supported in narrow single-sequence path by materializing
  attention-ready K/V from paged cache adapter.

## Explicit Non-Goals

- No full scheduler-level splitfuse orchestration.
- No new fused kernel path.
- No broad packed-query splitfuse integration in this pass.

## Validation

- tests: runtime splitfuse reachability + metadata + parity vs helper call.
- benchmark: helper-only vs runtime-integrated splitfuse path (focused matrix).

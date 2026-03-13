# Chunked Prefill Design (Serving-Oriented)

Date: 2026-03-13  
Branch: `codex/chunked-prefill`

## Goal

Add explicit chunked-prefill support so long prefills can be split into smaller
units and scheduled without monopolizing decode-oriented serving loops.

## Chunk semantics

- A **chunk** is a contiguous token span along the sequence axis.
- For batched layout (`query_layout="batched"`):
  - input shape: `q/k/v = [B, H, N, D]`
  - chunk boundaries are implicit: `[0:chunk_size)`, `[chunk_size:2*chunk_size)`, ...
- For packed layout (`query_layout="packed"`):
  - input shape: `q/k/v = [1, H, total_q, D]`
  - sequence boundaries come from `cu_seqlens_q`
  - each sequence is chunked independently by `chunk_size`

## Causal constraint

This pass defines chunked prefill as **causal-only** (`causal=True`).
Non-causal chunked prefill is left unsupported for now because simple chunked
execution cannot reproduce monolithic non-causal prefill semantics.

## Runtime/cache interactions

### Dense runtime

- Chunked prefill routes through repeated cache-updating decode-style steps.
- Cache grows chunk-by-chunk.
- Concatenated chunk outputs represent full prefill output.

### Paged runtime

- Chunked prefill routes through paged batch append+attend loops.
- Supports scheduler-style active-order remap via `seq_ids` and
  `cache_batch_idx` (batched layout).
- Uses paged cache growth per chunk.

### Packed query layout

- Supported for paged runtime with explicit `seq_ids` and `cu_seqlens_q`.
- Current packed chunked path intentionally requires packed-order `seq_ids`
  (no `cache_batch_idx` in this first pass) to keep behavior explicit.

### Shared-prefix

- No special shared-prefix chunk API in this pass.
- Shared-prefix remains available through existing runtime helpers and can be
  combined externally with chunked prefill scheduling.

## Minimal clean implementation for this branch

1. Add explicit runtime API: `DecodeRuntime.chunked_prefill(...)`.
2. Keep existing prefill/step APIs unchanged.
3. Validate chunk params and fail clearly on unsupported combinations.
4. Provide dense + paged batched support first.
5. Provide paged packed support where straightforward; document any limits.

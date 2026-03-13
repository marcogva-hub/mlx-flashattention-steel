# Chunked Prefill Summary (Serving-Oriented)

## What was added

- New explicit runtime API: `DecodeRuntime.chunked_prefill(...)`.
- Supported in this branch for:
  - dense batched runtime (`query_layout="batched"`),
  - paged batched runtime (`query_layout="batched"`),
  - paged packed-varlen runtime (`query_layout="packed"`, requires `seq_ids` + `cu_seqlens_q`).
- Validation is explicit and currently causal-only (`causal=True` required).

## How chunked prefill works

- Long prefill is split into fixed-size chunks (`chunk_size`).
- Each chunk appends K/V to cache and computes attention for that chunk only.
- Output chunks are concatenated in original query order.
- This provides scheduler-friendly prefill units that can be interleaved with decode.

## Correctness status

- Added test coverage for:
  - dense chunked vs monolithic parity,
  - paged batched chunked vs incremental manual reference,
  - paged packed-varlen chunked multi-chunk behavior,
  - invalid parameter handling,
  - cache growth with `reset=False`.
- Full suite status after this pass: `713 passed`.

## Benchmark status

- Matrix script: `benchmarks/bench_chunked_prefill.py`
- Latest artifact: `notes/chunked_prefill_matrix_latest.json`

Observed on M1 Max:
- Monolithic prefill remains faster across dense/paged scenarios.
- Best chunk-size rows in this run were generally at `chunk_size=512`.
- Chunked mode is a serving/runtime capability milestone (interleavable chunks),
  not a raw throughput win in current measurements.

## vLLM-style serving implications

- This closes a runtime scheduling gap by making prefill chunking explicit and
  available through the unified runtime surface.
- Combined with paged varlen + remap support, the runtime is now better aligned
  with mixed request-state serving loops.

## Remaining gaps / future work

- Non-causal chunked prefill is intentionally unsupported in this pass.
- Chunked packed-varlen and paged batched remain bridge-level serving paths;
  a more fused serving-native path can be evaluated later if benchmark evidence
  justifies it.

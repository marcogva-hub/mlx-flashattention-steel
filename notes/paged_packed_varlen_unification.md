# Paged + Packed Varlen Query Unification (Branch Summary)

Date: 2026-03-13  
Branch: `codex/paged-packed-varlen-unification`

## What was added

- Public API: `flash_attention_paged_varlen(...)`
  - packed query layout: `q=[1,H_q,total_q,D]`
  - query boundaries: `cu_seqlens_q`
  - paged KV inputs: `k_pages`, `v_pages`, `block_table`, `seq_lens_kv`
- Runtime integration:
  - `DecodeRuntime(..., query_layout="packed")`
  - `DecodeRuntime.paged_varlen(...)`
- Runtime metadata and validation now include query layout semantics.

## Implementation strategy

This pass implemented a correctness-first bridge that preserves existing APIs:

- uniform `q_len` batches: single batched paged dispatch
- heterogeneous `q_len` batches: per-sequence paged dispatch + packed concat

This closes the paged-KV + packed-query capability gap for vLLM-like scheduling
without forcing a risky kernel-family rewrite in the same pass.

## Correctness status

Added tests cover:

- basic heterogeneous correctness
- mixed `q_len` / `kv_len` handling
- packed output shape (`[1,H,total_q,D]`)
- zero-length query segments
- invalid `cu_seqlens_q` validation
- runtime integration and query-layout guards

Latest full suite run on this branch: `703 passed`.

## Benchmark status

Matrix script: `benchmarks/bench_paged_varlen.py`  
Artifact: `notes/paged_varlen_matrix_latest.json`

Observed on M1 Max (f16):

- GQA hetero rows: near parity/slight win vs padded paged (`1.01x` to `1.07x`)
- MQA hetero rows: below padded paged (`0.37x` to `0.81x`)

Interpretation:

- capability and runtime-unification milestone achieved
- not yet a broad performance win
- no broad auto-routing promotion from this matrix

## Future work (if revisited)

- evaluate a fused heterogeneous-query paged kernel path
- reduce per-sequence bridge overhead in MQA-heavy shapes
- keep compatibility with current batched paged API/runtime behavior

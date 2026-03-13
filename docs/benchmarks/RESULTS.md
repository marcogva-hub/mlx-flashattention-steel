# Benchmark Interpretation (Freeze Prep)

Version target: **2.10.0**  
Primary hardware in these summaries: **Apple M1 Max**

This document is a concise interpretation layer over benchmark artifacts stored
under `devnotes/`.

## 1) Core Dense Conclusions

- V2 dense remains the production default for causal D=64/128 regimes.
- D=256 is narrow benchmark-backed only.
- D=512 remains SDPA-default.
- Native dense backward was evaluated and not promoted.

Representative rows:

| Area | Representative row | Outcome |
|---|---|---|
| Dense V2 | D=64, N=8192 causal | ~1.82x vs SDPA |
| Dense V2 | D=128, N=16384 causal | ~1.75x vs SDPA |
| D=256 | causal long-N only | narrow win regime |
| D=512 | decision pass | no broad wins, SDPA-default |

Supporting artifacts:
- `devnotes/d256-design-track/`
- `devnotes/d512-decision-pass/`
- `devnotes/native-backward-pass/`

## 2) Paged + Packed + Continuous Batching

### Paged + packed varlen (`flash_attention_paged_varlen`)

Artifact: `devnotes/paged-packed-varlen-unification/paged_varlen_matrix_latest.json`

| Scenario | D | varlen/padded paged |
|---|---:|---:|
| GQA hetero | 64 | 1.75x |
| GQA hetero | 128 | 2.26x |
| MQA hetero | 64 | 0.65x |
| MQA hetero | 128 | 0.97x |

Interpretation: capability gap closed; performance remains workload-sensitive.

### Continuous batching remap

Artifact: `devnotes/paged-continuous-batching/paged_continuous_batching_latest.json`

| Scenario | D | runtime-remap/manual |
|---|---:|---:|
| paged_step_batch remap | 64 | 1.24x |
| paged_step_batch remap | 128 | 1.04x |
| paged_varlen remap | 64 | 1.05x |
| paged_varlen remap | 128 | 1.01x |

Interpretation: explicit remap semantics are operational and near-parity-to-win
in the sampled scheduler-style rows.

## 3) Chunked Prefill + Prefix + Speculative

### Chunked prefill

Artifact: `devnotes/chunked-prefill/chunked_prefill_matrix_latest.json`

| Group | D | chunked/monolithic |
|---|---:|---:|
| dense | 64 | 2.64x |
| dense | 128 | 2.03x |
| paged batched | 64 | 1.31x |
| paged batched | 128 | 1.28x |

Interpretation: scheduling/interleaving capability milestone; not a throughput
win in this hardware profile.

### Runtime-managed prefix reuse

Artifact: `devnotes/prefix-caching-automation/prefix_caching_runtime_matrix_latest.json`

| Scenario | D | runtime/explicit-helper | no-reuse/runtime |
|---|---:|---:|---:|
| dense prefix reuse | 64 | 0.95x | 0.43x |
| dense prefix reuse | 128 | 0.97x | 0.65x |
| paged prefix reuse | 64 | 1.00x | 2.12x |
| paged prefix reuse | 128 | 1.02x | 1.70x |

Interpretation: runtime-managed path tracks helper path closely; strongest
benefit appears in paged reuse scenarios.

### Runtime speculative decode

Artifact: `devnotes/speculative-decode-runtime/speculative_decode_runtime_matrix_latest.json`

Summary:
- dense mean `manual/runtime` ~1.015x
- paged mean `manual/runtime` ~1.002x

Interpretation: integration milestone with parity-level behavior, not a broad
throughput promotion claim.

## 4) Hybrid Cache / Offload / Splitfuse Runtime

### Hybrid cache behavior smoke

Artifact: `devnotes/hybrid-kv-cache-behavior/hybrid_kv_cache_bench_latest.json`

Representative interpretation:
- mixed overhead profile; several rows near parity or better, one row slightly
  slower;
- primary value is cache-control capability and extension readiness.

### Final serving completion focused rows

Artifacts:
- `devnotes/final-serving-completion/splitfuse_runtime_matrix_latest.json`
- `devnotes/final-serving-completion/paged_page_native_runtime_latest.json`
- `devnotes/final-serving-completion/final_serving_capabilities_summary.md`

Key rows:

| Scenario | D | runtime/baseline |
|---|---:|---:|
| splitfuse decode-step (dense) | 64 | 1.11x |
| splitfuse decode-step (dense) | 128 | 1.04x |
| paged decode-only splitfuse | 64 | 0.84x |
| paged decode-only splitfuse | 128 | 1.02x |

Interpretation: deeper runtime integration is real and test-covered, with
shape-sensitive performance.

## 5) Experimental Path Triage

Artifacts:
- `devnotes/experimental-triage/experimental_path_triage_latest.json`
- `devnotes/experimental-triage/experimental_path_status_matrix.md`

Outcome:
- V2 remains production path.
- V3/V4/V5 remain experimental/hardware-dependent.
- selective advanced-kernel AOT expansion was evaluated and deferred.

## 6) Historical JSON Archive

Legacy JSON snapshots previously mixed in `docs/benchmarks/` were moved to:
- `docs/benchmarks/archive/benchmarks_v2.0.0/`

These files are historical-only and not part of the active benchmark status
surface.

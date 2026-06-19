# mlx-mfa Results Summary

> **⚠ VERIFIED M5/26.6 PERF (audit Phase E, 2026-06-18) — read this first.** Measured on the
> now-known real dispatch (each number annotated with its fingerprinted binary; lesson #15 + Pattern #6).
> (Note: the M5 NA fp16/bf16 matmul peak was recalibrated to ~62 TFLOPS in 2026-06-19; the older
> "effective-FLOP ≤51.8 TFLOPS" plausibility gate these numbers were checked against is a lower estimate
> and is superseded — it does not invalidate the dated absolute-ms numbers below.)
> Perf is **Verified-at-date, NOT executable-locked** (timing is
> CI-flaky — re-measure is the anti-drift). Full table + methodology:
> `.doc-archive/docs/v50/campaign-2026-06/audit/phase-E-rebench-report.md`. Headlines (B2 H8 N4096, M5/26.6):
> - **sparse V2 (matmul2d) is the right sparse kernel** — 19–59× faster than the V1 scalar; the V1/V2
>   `2^31` work-threshold mis-routes D=64 (always) + D=128 N<4096 to the slow V1 (Phase-F target).
> - **symmetric NAX-sparse beats SDPA at D=128**: 4.16×@d=0.06 → 1.61×@d=0.5 (crossover ~d=0.78) — the
>   win is reachable only via a symmetric mask today (Phase-F D=128-sparse-routing fix justified).
> - **Dense D=128 `backend="auto"` now routes the NAX matmul2d forward** (`v6_nax_forward`, F-2 Change 3):
>   **parity-to-modest-win vs SDPA at D=128** (0.89–1.03× across N, never loses), all scales (scale plumbed),
>   backward=SDPA-vjp. **D=64 stays SDPA** (NAX loses 1.17–1.22×). `backend="mfa"` simdgroup-STEEL remains
>   legacy on M5 (SDPA 2–4× faster — a different kernel family). The ~5–7pp ALU gap to a larger D=128 win is
>   a future single-O-accumulator source-gen rewrite.
> - **sage int8 is 4.7× slower than SDPA on M5** (cos ~0.997) — not worth auto-routing here.
> - The numbers BELOW are historical (M1 Max + pre-26.6 M5); treat as indicative, not current.

Current library version: **2.61.0**
Benchmark hardware for the tables in §2–§7 below: **Apple M1 Max** · **Apple M4 Max**
(historical, pre-M5; the verified M5/26.6 numbers are in the boxed note above and in
`docs/reference/PERF_CLAIMS.md`).

> The "Version: 2.26.0" stamp this doc previously carried referred to the benchmark
> vintage of the §2–§7 M1/M4 tables, not the current release. The §2–§7 tables are
> **dated historical** M1-Max/M4-Max data; treat them as indicative, not current-M5.

For complete benchmark tables and architectural notes, see
`docs/reference/BENCHMARKS.md`.

## 1) Production Interpretation

- **V2 dense** remains the production default for causal D=64/128 on M1/M2.
- **V1 double-buffer** is the production default for causal D≤128 on M3+.
- **D=256** promoted for f16 causal (both chips) and bf16 causal (M3+ only).
- **D=512** remains SDPA-default.
- **Non-causal D=64/128** enabled on M1/M2 only (1.06-1.56×); M3+ stays SDPA.
- **Native dense backward** was benchmarked and not promoted.
- **Sage** remains a specialized decode backend (narrow policy).
- **V3** is **conditionally auto-routed** (not opt-in) for causal, N≥4096 (D=64) /
  N≥2048 (D=128), B·H≥4, f16/bf16 — re-validated on **M5 Max / macOS 26.6** (Queue
  Closure Sprint, 3-session §4-strict, V3 vs V2 the fallback): V3 is faster or at
  parity at every measured cell — windowed D=64 N=4096 `~3.4 ms vs ~4.9 ms` (0.68×,
  V3 ~32% faster), D=64 N=8192 0.92×, D=128 N=4096 0.97×, D=128 N=8192 ~parity
  (0.998×); `backend="mfa"` dense D=64 N=4096 0.86×, D=128 N=4096 ~parity (1.02×).
  Numbers are compute-bound / OS-sensitive (one cell, D=128 N=2048, was HIGH_VARIANCE
  r=0.43 but V3-faster-or-parity in all 3 sessions). The M1-2026-03 "1.015× vs V2"
  verdict holds on M5, stronger at D=64.
- **V4/V5** STEEL forward prototypes were **removed from the build (Lot-2, archived at
  tag `archive/v4-v5-prototypes`)**; the routed STEEL forwards are V1/V2/V3/V6_NAX only.
  (Historically these were experimental opt-in via `MFA_ENABLE_V4`/`MFA_ENABLE_V5`, never
  auto-routed; those env vars are no longer recognized.)
- **TurboQuant** KV cache compression (Phase 1–4) production-ready.
- **SVDQuantLinear** W4A16 + optional SVD low-rank correction for DiT quantization.
- **GNA native kernel** inline 3D window attention (D=128, f16/bf16, forward-only).

## 2) Representative Results

| Scenario | M1 Max | M4 Max |
|---|---:|---:|
| D=64 N=8192 causal | 1.69× | **2.07×** |
| D=128 N=8192 causal | 1.58× | **1.62×** |
| D=256 N=8192 causal f16 | 1.01× | **1.81×** |
| D=256 N=8192 causal bf16 | SDPA default | **1.68×** |
| D=64/128 non-causal | up to 1.51× | SDPA default |
| Sliding-window D=128 N=8192 win=256 | **18.4×** | **20.8×** |
| D=64 backward | 0.60-0.72× | **1.29-1.45×** |
| Softcap D=128 | **1.37×** | **1.34×** |

## 3) Serving/Runtime Capability Outcomes

### Paged + packed varlen queries

**PagedVarlenForward fused kernel** (v2.14.1): single dispatch for heterogeneous
query lengths + paged KV. Benchmark results (M1 Max, H_q=32 H_kv=8 D=128 f16):

| Config | Fused | Bridge | Speedup |
|--------|------:|-------:|--------:|
| B=4 decode kv hetero | 0.09ms | 0.43ms | 4.7× |
| B=8 decode kv hetero | 0.16ms | 0.77ms | 4.8× |
| B=16 decode kv hetero | 0.27ms | 1.48ms | 5.6× |
| B=4 prefill q+kv hetero | 0.03ms | 0.37ms | 10.9× |
| B=8 mixed q hetero | 0.04ms | 0.90ms | 25.6× |

Full data: `.doc-archive/devnotes/paged_varlen_fused_bench.json`

Previous artifact: `.doc-archive/devnotes/paged-packed-varlen-unification/paged_varlen_matrix_latest.json`

- GQA hetero rows show strong wins (up to ~2.26x vs padded paged baseline).
- MQA hetero rows are mixed/near parity.
- Treated as capability closure with workload-sensitive performance.

### Paged continuous batching remap

Artifact: `.doc-archive/devnotes/paged-continuous-batching/paged_continuous_batching_latest.json`

- Runtime remap path is operational and parity-to-win in sampled rows
  (`~1.01x` to `~1.24x` vs manual baseline).

### Chunked prefill

Artifact: `.doc-archive/devnotes/chunked-prefill/chunked_prefill_matrix_latest.json`

- Operational scheduling capability.
- Monolithic prefill remains faster in current M1 Max rows.

### Prefix caching runtime integration

Artifact: `.doc-archive/devnotes/prefix-caching-automation/prefix_caching_runtime_matrix_latest.json`

- Runtime-managed path tracks explicit helper orchestration closely.
- Strongest practical benefit appears in paged shared-prefix scenarios.

### Speculative runtime integration

Artifact: `.doc-archive/devnotes/speculative-decode-runtime/speculative_decode_runtime_matrix_latest.json`

- Runtime `speculative_step` path is parity-level with manual helper wiring in
  aggregate.
- Interpreted as capability/integration milestone.

### Hybrid/offload + splitfuse/page-native final pass

Artifacts:
- `.doc-archive/devnotes/hybrid-kv-cache-behavior/hybrid_kv_cache_bench_latest.json`
- `.doc-archive/devnotes/final-serving-completion/splitfuse_runtime_matrix_latest.json`
- `.doc-archive/devnotes/final-serving-completion/paged_page_native_runtime_latest.json`
- `.doc-archive/devnotes/final-serving-completion/final_serving_capabilities_summary.md`

Interpretation:
- `HybridKVCache` now has real local offload-capable behavior
  (hot/cold/offloaded residency + reload/promotion).
- External adapter extension point exists with concrete local backend
  (`LocalHostKVStoreAdapter`).
- Splitfuse/runtime deepening and page-native reductions are real but
  shape-sensitive.

## 4) TurboQuant KV Cache Compression (v2.21.0–v2.24.0)

Training-free, data-oblivious KV compression based on Google's TurboQuant (ICLR 2026).
Four phases: Phase 1 (non-fused decompress), Phase 2 (K fused in kernel),
Phase 3 (K+V fused in kernel), Phase 4 (optimal 3-bit packing + WHT fusion).

### Benchmark matrix (M1 Max, f16)

| Config | fp16 (ms) | P2/K-only (ms) | P3/K+V (ms) | P3/fp16 | cos(P2) | cos(P3) | fp16 MB | KV-TQ MB |
|--------|----------:|----------------:|-------------:|--------:|--------:|--------:|--------:|---------:|
| Llama-8B 1seq 2K | 0.59 | 4.04 | 4.36 | 7.4× | 0.9825 | 0.9654 | 8.0 | 2.1 |
| Llama-8B 1seq 8K | 1.40 | 11.29 | 12.23 | 8.7× | 0.9811 | 0.9647 | 32.0 | 8.5 |
| Llama-8B 4seq 2K | 1.24 | 3.33 | 3.90 | 3.1× | 0.9811 | 0.9652 | 32.0 | 8.5 |
| Llama-8B 8seq 4K | 3.86 | 13.53 | 14.94 | 3.9× | 0.9824 | 0.9656 | 128.0 | 34.0 |
| Llama-8B prefill 512 | 3.68 | 3.06 | 5.00 | 1.4× | 0.9875 | 0.9707 | 2.0 | 0.5 |
| Llama-8B prefill 2K | 40.33 | 25.40 | 37.06 | 0.9× | 0.9863 | 0.9694 | 8.0 | 2.1 |
| Qwen-7B 1seq 8K | 0.90 | 11.68 | 12.92 | 14.4× | 0.9826 | 0.9647 | 16.0 | 4.3 |
| Qwen-7B 4seq 4K | 1.51 | 6.30 | 7.55 | 5.0× | 0.9822 | 0.9650 | 32.0 | 8.5 |
| Mixed 8seq hetero | 4.05 | 12.49 | 14.52 | 3.6× | 0.9818 | 0.9649 | 60.0 | 15.9 |

**Interpretation:**
- Memory savings: ~3.8× with both K+V compressed (Phase 3).
- Quality: cosine similarity 0.9647–0.9707 vs fp16 (Phase 3), 0.9811–0.9875 (Phase 2 K-only).
- Latency: Phase 2/3 are currently slower than fp16 due to Python-side pack/unpack overhead.
  The fused Metal kernel avoids decompression but packing still happens in Python.
- Primary value is memory savings for long-context serving, not latency reduction.

Full data: `.doc-archive/devnotes/turboquant_full_bench.json`

## 5) Practical Ceiling Statement (Current Hardware)

For this architecture on M1 Max and M4 Max:
- dense kernel family appears close to practical ceiling in the main production
  design space;
- M3/M4 optimizations (V1 routing, direct reads) recover hardware-specific wins;
- remaining improvements are mostly serving/runtime integration and future
  hardware-dependent opportunities (M5+ tensor API).

## 6) Artifact Map

- Dense/backward/large-D decisions: `.doc-archive/devnotes/native-backward-pass/`,
  `.doc-archive/devnotes/d256-design-track/`, `.doc-archive/devnotes/d512-decision-pass/`
- Sage/runtime/paged passes: `.doc-archive/devnotes/sage-decode-productionization/`,
  `.doc-archive/devnotes/runtime-unification/`,
  `.doc-archive/devnotes/paged-shared-prefix-productionization/`
- Serving completion tracks:
  - `.doc-archive/devnotes/paged-packed-varlen-unification/`
  - `.doc-archive/devnotes/paged-continuous-batching/`
  - `.doc-archive/devnotes/chunked-prefill/`
  - `.doc-archive/devnotes/prefix-caching-automation/`
  - `.doc-archive/devnotes/speculative-decode-runtime/`
  - `.doc-archive/devnotes/hybrid-kv-cache-abstraction/`
  - `.doc-archive/devnotes/hybrid-kv-cache-behavior/`
  - `.doc-archive/devnotes/final-serving-completion/`
- Experimental triage: `.doc-archive/devnotes/experimental-triage/`
- TurboQuant: `.doc-archive/devnotes/turboquant_full_bench.json`

## 7) Related Documentation

- Benchmark details: `docs/reference/BENCHMARKS.md`
- API reference: `docs/reference/API_MANUAL.md`
- Architecture: `docs/reference/ARCHITECTURE.md`
- Historical archive index: `.doc-archive/devnotes/README.md`

# V34 causal forward — Sprint 1 results

**Date:** 2026-05-06
**Sprint:** V34-FORWARD-MAX Sprint 1 (causal port)
**Branch:** `experiment/v34-forward-max`

## Summary

V34 NAX-direct path now supports causal forward attention. The Apple
`steel_attention_nax.h:175-303` causal pattern is ported to our V34
generator with all three optimizations: skip-end (`kb_lim`), skip-mask-clean
(`kb_min_causal`), and per-element fragment masking on the NAX frags
directly.

**Correctness validated** across 6 LLM-style and VSR-style causal shapes:
RMSE FP32 vs SDPA reference 5.99e-06 to 5.64e-05 (well within 1e-3 target).

**Performance** (warm single-process timing — proper cross-session A/B/A
deferred to merge phase):

| Shape | D | N | V34 ms | Legacy ms | V34/SDPA |
|---|---|---|---:|---:|---:|
| Llama-prefill-2k | 128 | 2048 | 1.42 | 1.21 | 1.34× |
| Llama-prefill-4k | 128 | 4096 | **3.35** | 3.43 | **1.04×** ⭐ |

V34 gains over legacy at N≥4k. Below that, V34's per-kernel overhead is
unfavorable (same pattern as non-causal FlashVSR-dense at D=64 small-N).
The existing shape-aware dispatch policy (`mfa_v6_nax_primitive.cpp`)
correctly routes:
- D=128 causal → V34 (parity at N=4k+)
- D=64 N_kv > 8000 causal → V34
- D=64 small-N causal → legacy

## Apple pattern citations

| Component | Apple file:line |
|---|---|
| `kb_lim` / `kb_min_causal` setup | `steel_attention_nax.h:175-187` |
| Per-element fragment causal mask | `steel_attention_nax.h:278-303` |
| `qL_off` semantics (decode offset) | `steel_attention_nax.h:180,184` |

## Implementation diff

### Generator (`createV34Source`)

1. New `#define V34_DO_CAUSAL` macro (compile-time per kernel — separate
   pipeline for causal vs non-causal so the per-element mask is
   completely dead-code on non-causal).
2. `V34Params` struct gains `int qL_off` field (between `kL_rem` and
   strides).
3. After `lim_rows_q` / `lim_rows_k` / `is_last_q` setup, before the K-loop:
   ```c
   int kb_lim = params.NK;
   int kb_min_causal = params.NK;
   #if V34_DO_CAUSAL
   {
     int q_max = (int(tid.x) + 1) * V34_BQ + params.qL_off;
     kb_lim = min(params.NK, (q_max + V34_BK - 1) / V34_BK);
     int q_min = max(0, int(tid.x) * V34_BQ + params.qL_off);
     kb_min_causal = q_min / V34_BK;
   }
   #endif
   ```
4. K-loop bound becomes `for (int kb = 0; kb < kb_lim; kb++)`.
5. After scale + kL_rem mask, before softmax:
   ```c
   #if V34_DO_CAUSAL
   if (kb >= kb_min_causal) {
     // Per-element mask: fg[loc] = (r < c) ? -inf : fg[loc]
     // r = base_row + iq*16 + ii*kFragRowsJump + sm
     // c = base_col + ik*16 + jj + sn
     ...
   }
   #endif
   ```

### Host dispatch (`v6_nax_compile.mm`)

`V34ParamsHost` adds `int qL_off` matching the kernel struct.
`v34_dispatch` accepts `bool causal` parameter; sets
`params.qL_off = causal ? max(0, kL - qL) : 0`.

For prefill (qL == kL), qL_off = 0 (standard lower-triangular mask).
For decode-style (qL < kL, e.g. partial KV cache + small Q batch),
qL_off = kL - qL so query position 0 maps to key position (kL - qL).

### Primitive dispatch (`mfa_v6_nax_primitive.cpp`)

Removed the `params_.causal` exclusion from V34 dispatch gates.
Both `generate_v6_source` and `eval_gpu` now allow V34 for causal
forward when single-Otile-eligible. Dispatch passes `params_.causal`
to `v34_dispatch`.

## Correctness data

| Shape | D | N_q | causal | RMSE FP32 |
|---|---|---|---|---:|
| FlashVSR-dense | 64 | 4096 | False | 3.60e-06 (regression-check) |
| LTX2-cross | 64 | 2048 | False | 1.76e-06 (regression-check) |
| SeedVR2-small | 128 | 26730 | False | 1.75e-06 (regression-check) |
| Causal-D64-prefill | 64 | 1024 | True | 1.74e-05 |
| Causal-D128-prefill | 128 | 1024 | True | 1.27e-05 |
| Llama-prefill-2k | 128 | 2048 | True | 9.82e-06 |
| Llama-prefill-4k | 128 | 4096 | True | 7.69e-06 |
| Llama-prefill-8k | 128 | 8192 | True | 5.99e-06 |
| Mistral-prefill-4k | 128 | 4096 | True | 7.69e-06 |
| Default-causal-D128 | 128 | 1024 | True | 1.27e-05 (V34 dispatched) |
| Default-causal-D64-small | 64 | 1024 | True | 5.64e-05 (legacy dispatched) |
| Default-causal-D64-long | 64 | 16384 | True | 5.43e-06 (V34 dispatched) |

Default dispatch (no env var) correctly routes by shape via the
existing per-D + Nk policy: causal V34 for D=128 and D=64 N_kv>8000,
legacy for D=64 small-N (where V34 perf regresses).

## Open follow-ups

- Cross-session A/B/A with thermal validation (Marco's preferred
  protocol) — deferred to release phase. Current bench is single-process
  warm timings; sufficient for correctness + perf sanity but not for
  shipping decisions on borderline cases.
- LSE writeback for causal V34 — same as non-causal V34 (Sprint 2).
- Sprint 4 may close the FlashVSR-dense + small-causal-D=64 gap if a
  smaller V34 tile config (BQ=16, WM=1) beats legacy.

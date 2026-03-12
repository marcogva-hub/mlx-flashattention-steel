# Native Backward Targeted Viability (2026-03-12)

## Scope
- Device: Apple M1 Max (gen 13)
- Shapes: `B=2`, `H=8`, `causal=True`
- D: `{64, 128}`
- N: `{2048, 4096, 8192, 16384}`
- dtype: `f16`, `bf16`
- Benchmark script: `benchmarks/bench_backward_targeted.py`
- Raw data: `notes/native_backward_targeted_latest.json`

Measured variants:
1. **current** = `flash_attention(..., backend="mfa")` VJP path
2. **native** = `mfa_forward_with_lse + mfa_steel_backward`
3. **sdpa** = `mx.vjp(_fallback_sdpa, ...)`

## Classification Summary
- Promising: **0**
- Neutral: **0**
- Losing: **16**

No benchmark-backed winning regime was found for native STEEL backward in this target set.

## Full Results

| dtype | D | N | current total ms | native total ms | sdpa total ms | native/sdpa | max abs err | class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| f16 | 64 | 2048 | 29.31 | 21.68 | 12.32 | 0.57x | 1.953e-03 | losing |
| f16 | 64 | 4096 | 66.97 | 71.91 | 44.75 | 0.62x | 3.906e-03 | losing |
| f16 | 64 | 8192 | 274.58 | 287.42 | 180.73 | 0.63x | 3.906e-03 | losing |
| f16 | 64 | 16384 | 1335.53 | 1100.40 | 909.02 | 0.83x | 2.930e-03 | losing |
| f16 | 128 | 2048 | 22.41 | 65.97 | 17.48 | 0.26x | 3.418e-03 | losing |
| f16 | 128 | 4096 | 74.88 | 253.75 | 67.80 | 0.27x | 3.906e-03 | losing |
| f16 | 128 | 8192 | 271.27 | 995.13 | 280.98 | 0.28x | 3.906e-03 | losing |
| f16 | 128 | 16384 | 1230.04 | 3884.41 | 1235.97 | 0.32x | 2.686e-03 | losing |
| bf16 | 64 | 2048 | 18.76 | 34.73 | 16.21 | 0.47x | 1.562e-02 | losing |
| bf16 | 64 | 4096 | 77.27 | 106.47 | 55.46 | 0.52x | 3.125e-02 | losing |
| bf16 | 64 | 8192 | 503.27 | 385.63 | 200.48 | 0.52x | 2.344e-02 | losing |
| bf16 | 64 | 16384 | 1042.17 | 1398.88 | 1002.86 | 0.72x | 3.125e-02 | losing |
| bf16 | 128 | 2048 | 113.38 | 106.62 | 23.93 | 0.22x | 2.734e-02 | losing |
| bf16 | 128 | 4096 | 84.60 | 375.19 | 74.77 | 0.20x | 2.344e-02 | losing |
| bf16 | 128 | 8192 | 315.42 | 1448.63 | 311.83 | 0.22x | 3.125e-02 | losing |
| bf16 | 128 | 16384 | 1345.56 | 5747.27 | 1379.49 | 0.24x | 3.125e-02 | losing |

## Decision
- Keep dispatch unchanged for now (no native backward auto-enable yet).
- Proceed with a **narrow policy gate** in code that defaults to SDPA VJP and
  supports explicit override for targeted re-evaluation.

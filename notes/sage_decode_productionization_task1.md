# Sage Decode Regime Map (Task 1)

Date: 2026-03-12
Benchmark script: `benchmarks/bench_sage_decode_matrix.py`
Output: `notes/sage_decode_matrix_post_bwd_latest.json`

## Matrix

- Decode-only: `N_q in {1,2,4}`
- Cache length: `N_cache in {512,1024,2048,4096,8192}`
- Head dim: `D in {64,128}`
- Dtype: `f16` and `bf16` (if supported)
- Window: `None` and `(256, 0)`
- Profiles:
  - `prod_gqa_b2_hq8_hkv4` (production-like GQA)
  - `under_b1_hq1_hkv1` (under-occupied)

## Summary

- Total rows: 240
- `sage_win`: 13
- `maybe`: 4
- `losing`: 223

## Clear no-win regions

- `window=None` is overwhelmingly losing for Sage vs dense STEEL in this matrix.
- `prod_gqa_b2_hq8_hkv4` is mostly losing even when windowed, with only sparse wins.
- `D=64` production-like GQA decode is generally not competitive for Sage auto defaulting.

## Maybe-win regions

- Near parity appears on a few windowed rows:
  - `prod_gqa_b2_hq8_hkv4`, `D=128`, `N_q=4`, `N_cache=1024`, `f16`.
  - `prod_gqa_b2_hq8_hkv4`, `D=128`, `N_q=1`, `N_cache=8192`, `bf16`.
  - `under_b1_hq1_hkv1`, `D=64`, `N_q=1`, `N_cache=512`, `bf16`.

## Clear Sage-winning regions (narrow)

- Wins cluster in **windowed decode** (`window=(256,0)`), mostly under-occupied profiles.
- Production-like wins exist but are narrow, mainly around `D=128` windowed cases:
  - `prod_gqa_b2_hq8_hkv4`, `D=128`, `N_q=4`, `N_cache=4096`, `f16`.
  - `prod_gqa_b2_hq8_hkv4`, `D=128`, `N_q in {1,2}`, selected `bf16` rows.

## Quantized cache reuse impact

- Reusing quantized KV cache is frequently material.
- `sage_requant_each_call / sage_cache` is often >1x and can be much higher on long-cache windowed rows.
- Conclusion: any Sage promotion should require quantized KV reuse and should stay decode-narrow.

## Task 1 decision

- Proceed with a **decode-only**, **narrow**, benchmark-backed Sage auto route candidate.
- Keep STEEL V2 as default path globally.

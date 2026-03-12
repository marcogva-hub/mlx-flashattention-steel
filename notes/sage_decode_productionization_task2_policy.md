# Sage Decode Auto-Routing Policy Note (Task 2/3)

Date: 2026-03-12
Input matrix: `notes/sage_decode_matrix_post_bwd_latest.json`

## Policy shape

Auto routing to Sage decode is intentionally strict and decode-only:

- requires `QuantizedKVCache` (`quantized_kv=True`)
- requires causal decode with `N_q <= 4`
- requires a decode window (`window_size` enabled)
- requires `D=128`
- requires `H_q / H_kv = 2`
- requires `N_cache = 4096`
- dtype-specific:
  - `float16`: route only when `N_q = 4`
  - `bfloat16`: route only when `N_q = 1`

`MFA_FORCE_SAGE_DECODE=0|1` overrides the heuristic policy, but still
respects decode-safety constraints.

## Matrix hit quality

Applying the policy across the full 240-row matrix selects 2 rows:

- selected: 2
- `sage_win`: 2
- `maybe`: 0
- `losing`: 0

This keeps auto mode conservative while still enabling benchmark-backed,
narrow Sage decode wins.

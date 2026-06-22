# Raw-Backward Binding Validation Enumeration (Volet H2 — CX-04)

Branch `fix/audit-remediation`, base HEAD `8e61a97` (after volet H). M5 Max /
macOS 26.6 / MLX 0.31.2. Verify-first (RULE 16): each binding's actual current
validation was read at source before editing.

CX-04 (HIGH): exported raw backward `_ext` bindings accepted malformed
auxiliaries / mismatched K-V and returned finite, materially-wrong gradients
(dQ delta 174). Volet C added the common backward validator to some bindings but
missed others and the **K↔V mutual-shape** dimension on the V6-NAX family.

## Validators
- `mfa_check_backward_inputs(name,q,k,v,O,L,dO)` (bindings.cpp) — full: ranks,
  batch, k_seq==v_seq, k_heads==v_heads, head_dim, GQA, dtype, O/dO=[B,Hq,Nq,D],
  L=[B,Hq,Nq]. `mfa_check_bnq` — [B,Hq,Nq] aux.
- `v6_check_bwd_gqa(name,q,k,v)` — **EXTENDED in H2**: was GQA-only (q↔k); now also
  ranks + K↔V mutual (k_seq==v_seq, heads, head_dim) + batch. `v6_aux_bnq`/
  `v6_aux_bnqd` — [B,Hq,Nq] / [B,Hq,Nq,D] aux vs Q.

## Per-binding enumeration (had → now)

| binding | had (pre-H2) | now (post-H2) |
|---|---|---|
| `mfa_backward_query_debug` | full `mfa_check_backward_inputs` | unchanged ✓ |
| `mfa_backward_kv_debug` | full + `mfa_check_bnq(D)` | unchanged ✓ |
| `mfa_steel_backward` | full `mfa_check_backward_inputs` | unchanged ✓ |
| **`mfa_steel_backward_sparse`** | **NONE** (CX-04 lead #1) | **`mfa_check_backward_inputs`** ✓ |
| **`v6_nax_backward_query`** | Q-rank + GQA only (CX-04 lead #2) | **GQA+K↔V + aux o/lse/d_o/d_vec** ✓ |
| `v6_nax_backward_kv` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| `v6_nax_backward_dk_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| `v6_nax_backward_dv_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| **`v6_nax_backward_fused_dkdv_raw`** | GQA(q↔k) + aux, **no K↔V** (CX-04 lead #3) | **+ K↔V** ✓ |
| `v6_nax_backward_query_sparse_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| `v6_nax_backward_dk_sparse_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| `v6_nax_backward_dv_sparse_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |
| `v6_nax_backward_fused_dkdv_sparse_raw` | GQA(q↔k) + aux | **+ K↔V** ✓ |

All 13 raw backward bindings now validate GQA + K↔V mutual shape + aux shapes
(every binding either via `mfa_check_backward_inputs` or `v6_check_bwd_gqa(...,v)`
+ the `v6_aux_*` checks). No fourth under-validated binding remained after the
K↔V extension covered the whole V6-NAX family in one helper change.

## Validation
- CX-04 repros now raise (was finite-wrong): `mfa_steel_backward_sparse` undersized
  L → `ValueError: L (logsumexp) must be [B,Hq,Nq]`; `v6_nax_backward_query`
  undersized lse → `RuntimeError: lse must be [B,Hq,Nq]`;
  `v6_nax_backward_fused_dkdv_raw` V=[1,2,1,64] vs K=[1,2,64,64] → `RuntimeError:
  k and v must share the kv sequence length`.
- byteΔ-identity on valid backward: dense + GQA, D∈{64,128}, causal — grads
  oracle-correct (relerr ≤2.8e-3), unchanged (validation adds raises only).
- Lock: `tests/test_raw_backward_validation_h2.py` (11 malformed → raise; 7 valid →
  run + finite). Bite-proven: neutralize the K↔V seq check in `v6_check_bwd_gqa`
  + rebuild → the 5 v6 KV-mismatch cells FAIL (steel_sparse, which uses
  `mfa_check_backward_inputs`, correctly still raises — bite is guard-specific);
  restore + rebuild → all pass.
- Full suite: 2596 passed / 91 skipped / 0 failed / 0 XPASS.

Validation-only host gate; no valid backward gradient changed. Commit on
`fix/audit-remediation` only.

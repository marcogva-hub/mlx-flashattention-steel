# Complete API-Surface Inventory (Volet I)

Branch `fix/audit-remediation`, base HEAD `6208dc6`. Host M5 Max / macOS 26.6 /
MLX 0.31.2. **Every cell below was exercised first-hand THIS pass** (no delegation
to existing tests; determinism = 8 identical-input runs; validation probed per
buffer/dtype the entry actually reads). Built to break the round-by-round sibling
cycle — the durable matrix round-7 verifies cell-by-cell.

Axes: **C**=correctness (relerr vs independent fp32/fp64 oracle), **AV**=accept-valid
(legit input runs), **RM**=reject-malformed (raises, per buffer/dtype/shape),
**D**=determinism (max pairwise byteΔ over 8 runs).

## Entry list (derived from code, not reports)
Public (`mlx_mfa/attention.py`): `flash_attention`, `flash_attention_sparse`,
`flash_attention_gna`, `sage_attention`, `flash_attention_varlen`,
`flash_attention_paged`, `flash_attention_paged_varlen`,
`flash_attention_paged_varlen_turboquant`.
Raw (`csrc/bindings.cpp`): `mfa_paged_steel_forward`, `mfa_paged_varlen_forward`,
`mfa_paged_varlen_tq_forward`, `mfa_paged_kv_gather`, backward family
(`mfa_steel_backward[_sparse]`, `mfa_backward_{query,kv}_debug`,
`v6_nax_backward_{query,kv,dk,dv,fused_dkdv}[_sparse]_raw`).

## Dense / sparse / sage / GNA

| entry | features probed | C (relerr) | AV | RM | D |
|---|---|---|---|---|---|
| `flash_attention` | D{64,128,256}, f16/bf16, causal±, GQA, softcap, window, asym-`D_v` | GQA-causal 3.4e-4; softcap=30 7.3e-4; window(256,0) 2.3e-4 | GQA q8/k2, q6/k2, asym-`D_v`, D256-bf16 all run ✓ | batch/k_seq/k_heads mismatch, Hk=0, GQA-indivisible, empty-KV, dropout=1.0 all raise ✓ | 0.00 |
| `flash_attention_sparse` | D{64,128}, causal + **non-causal**, GQA | causal 2.5e-4; non-causal (vs expanded-block oracle) 3.8e-4 | GQA q8/k2 runs ✓ | k_seq≠v_seq raises ✓ | 0.00 |
| `sage_attention` | D{64,128}, GQA, int8 | within int8 floor (≤1.5e-1, lossy) | GQA q8/k2 runs ✓ | asym-`D_v` raises ✓ | 0.00 (in-proc; CX-R6-02 cross-proc → volet S) |
| `flash_attention_gna` | D=128 3-D window | vs exact 3D-window oracle 1.8e-4 | GQA runs ✓ | batch/k_seq/head mismatch raise ✓ | 0.00 |

## Backward (dense + raw bindings)

| entry | C | AV | RM | D |
|---|---|---|---|---|
| `flash_attention` vjp | dense GQA D128 ≤2.2e-3 (caus±) vs fp32 vjp oracle | GQA runs ✓ | — | 0.00 |
| raw `v6_nax_backward_{query,kv,dk,dv,fused}[_sparse]_raw`, `mfa_steel_backward[_sparse]` | valid grads finite | GQA valid runs ✓ | undersized `lse`/`L`, K↔V mismatch, invalid GQA all raise ✓ (volet H2) | n/a |

## Paged (Reinforcement A — full cross-product, 72 correctness cells)

`flash_attention_paged` × D∈{64,128,256} × {f16,bf16} × {MHA 4/4, GQA 8/2} ×
{homo Nq1, homo Nq17, hetero Nq<Nk, hetero Nq>Nk, hetero nc, hetero3}: **all 72
oracle-correct** (per-sequence fp64 oracle; f16 ≤4.0e-4, bf16 ≤3.1e-3). Per-sequence
causal offset verified (matches per-row oracle, diverges from batch-global).

| entry | C | AV | RM (per buffer/dtype) | D |
|---|---|---|---|---|
| `flash_attention_paged` | 72/72 (above) + sliding-window 3.0e-4 | B>1 matched pools run ✓ | seq_lens-card, block_table int64, seq_lens float, OOB page, V fewer heads/blocks all raise ✓ | 0.00 |
| `flash_attention_paged_varlen` | valid hetero correct | valid runs ✓ | cu int64, seq_lens short, V fewer heads raise ✓ | 0.00 |
| `flash_attention_paged_varlen_turboquant` | valid (lossy, ground-truth-locked) | valid runs ✓ | **v_pages fewer blocks/heads/head_dim, k_scales short, incompatible packed_D, cu int64 all raise ✓ (CX-R6-01 FIXED this volet)** | 0.00 |
| raw `mfa_paged_steel_forward` | valid B>1 correct | valid runs ✓ | seq-card, V mismatch, **block_table/seq_lens int64/float raise ✓ (CX-R6-03 FIXED; float was a HANG)** | n/a |
| raw `mfa_paged_varlen_forward` | valid | valid runs ✓ | card, V mismatch, cu/**block_table/seq_lens_kv int64/float raise ✓ (CX-R6-03 FIXED; float seq_lens was a HANG)** | n/a |
| raw `mfa_paged_varlen_tq_forward` | valid | valid runs ✓ | **TQ buffer shapes + metadata int32 raise ✓ (CX-R6-01/03 FIXED)** | n/a |
| raw `mfa_paged_kv_gather` | — | — | batch-card + int32 (volet C2) ✓ | n/a |

## Findings fixed this volet

| id | sev | repro (before) | after |
|---|---|---|---|
| **CX-R6-01** | CRITICAL | TQ public+raw: undersized `v_pages` → OOB finite-wrong; smaller head_dim → **NaN**; undersized `k_scales` → OOB; incompatible `packed_D` → garbage | shape-lock v_pages/k_scales/packed_D (+v_pool_tq/v_scales) vs k_pool_tq → all **raise** |
| **CX-R6-03** | HIGH | raw steel/varlen: int64 block_table → silent int32 cast; **float seq_lens → HANG** | int32 enforced on block_table/seq_lens(_kv)/cu → all **raise**, no hang |

## Validation
- Bite-proven: neutralize the TQ v_pages-dims guard (Python + C++) → `v_pages fewer
  blocks` NO-RAISE; restore → raises. CX-R6-03 before/after (all 5 raw int64/float
  cases NO-RAISE/HANG → all RAISE) demonstrates the dtype guards load-bearing.
- byteΔ-identity: validation-only — valid TQ/raw output unchanged (the TQ
  ground-truth lock `tests/test_phase3_iii2_tq_decode.py` still passes).
- Determinism: every deterministic kernel byteΔ=0 over 8 identical-input runs.
- Lock: `tests/test_surface_inventory_i.py` (8 cells).

## Determinism axis — RE-SPECIFIED at N≥512 (volet I2)

Volet I sampled determinism at **N=256 (single K-tile)** — below the shared-buffer
reuse threshold — so it certified "byteΔ=0" without ever exercising multi-tile
`KV_smem` reuse (the sage race needed N≥512). Re-specified here: every forward
kernel that aliases `Ks==Vs==KV_smem` audited (a) statically for the
inter-iteration barrier and (b) at runtime over **20 fresh-but-identical-input runs
at N∈{512,1024}** × D{64,128} × {f16,bf16} × {MHA,GQA}.

| kernel (generator) | KV_smem alias | inter-iteration barrier (static) | runtime byteΔ N≥512 |
|---|---|---|---|
| dense STEEL V1 (`mfa_steel_fwd`) | yes | present (double-buf end-of-iter barrier) | 0 |
| dense STEEL V2 (`mfa_steel_fwd_v2`) | yes | present ("barrier X: P@V reads done → overwrite K") | 0 |
| varlen STEEL (`mfa_steel_fwd`) | yes | present (start-of-iter, explicit) | 0 |
| paged STEEL (`mfa_steel_fwd`) | yes | present | 0 (incl. S=1024 = 64 tiles) |
| paged-varlen STEEL (`mfa_steel_paged_varlen_fwd`) | yes | present (explicit) | 0 |
| paged TQ (`mfa_steel_paged_varlen_tq_fwd`) | yes | present (start-of-loop) | 0 |
| GNA (`mfa_gna_fwd`) | yes | present (pre-load-next-K, barriered both sides) | 0 |
| **sage (`mfa_sage_fwd`)** | yes | **ADDED in volet S** (was the sole missing one) | 0 (post-S) |
| v3 (`mfa_steel_fwd_v3`), v6-NAX (`mfa_steel_fwd_v6_nax`) | **no** (separate K/V smem) | n/a | 0 |

**Verdict: no new sibling race. Sage was the only kernel that had dropped the
inter-iteration barrier; every other shared-buffer kernel already had it** (confirmed
static + runtime). Multi-tile outputs are deterministic AND oracle-correct (dense-mfa
& sparse GQA D128 N1024 relerr ~3.5e-4). Lock: `tests/test_multitile_determinism_i2.py`
(56 cells) + `tests/test_sage_determinism_s.py` (51 cells).

## Notes (RULE 16)
- CX-R6-02 (sage nondeterminism): RESOLVED in volet S. My volet-I "byteΔ=0 over 8
  runs" used a config (N=256) below the multi-tile threshold — the race only fires at
  **N≥512** (KV_smem reuse across K-tiles). Root cause: missing start-of-loop barrier
  in `mfa_sage_forward`; NOT GQA-specific (MHA Hq2Hk2 N≥512 also raced). Fixed +
  locked (`tests/test_sage_determinism_s.py`).
- All findings reproduced first-hand before fixing; the Codex report was not taken
  as gospel — both CX-R6-01 and CX-R6-03 reproduced exactly as described.

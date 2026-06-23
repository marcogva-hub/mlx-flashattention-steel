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
| `flash_attention_paged` | **108/108** (D∈{64,128,256} since volet J — was claimed, now executable) | B>1 matched pools run ✓ | seq_lens-card, block_table int64, seq_lens float, OOB page, V fewer heads/blocks all raise ✓ | **0 post-volet-J** (CX-J-02 fixed a real race; see axis below) |
| `flash_attention_paged_varlen` | valid hetero correct | valid runs ✓ | cu int64, seq_lens short, V fewer heads raise ✓ | 0.00 |
| `flash_attention_paged_varlen_turboquant` | valid (lossy, ground-truth-locked) | valid runs ✓ | **v_pages fewer blocks/heads/head_dim, k_scales short, incompatible packed_D, cu int64 all raise ✓ (CX-R6-01 FIXED this volet)** | 0.00 |
| raw `mfa_paged_steel_forward` | valid B>1 correct; **sliding-window 3.0e-4 (window_left/right ARE on THIS raw entry, NOT public paged)** | valid runs ✓ | seq-card, V mismatch, **block_table/seq_lens int64/float raise ✓ (CX-R6-03 FIXED; float was a HANG)** | n/a |
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
| paged STEEL (`mfa_steel_fwd`) | yes | **MISSING → ADDED volet J (CX-J-02)** | **0 post-fix** (pre-fix: intermittent race, I2 runtime got LUCKY) |
| paged-varlen STEEL (`mfa_steel_paged_varlen_fwd`) | yes | present (explicit) | 0 |
| paged TQ (`mfa_steel_paged_varlen_tq_fwd`) | yes | present (start-of-loop) | 0 |
| GNA (`mfa_gna_fwd`) | yes | present (pre-load-next-K, barriered both sides) | 0 |
| **sage (`mfa_sage_fwd`)** | yes | **ADDED in volet S** (was the sole missing one) | 0 (post-S) |
| v3 (`mfa_steel_fwd_v3`), v6-NAX (`mfa_steel_fwd_v6_nax`) | **no** (separate K/V smem) | n/a | 0 |

**Verdict (CORRECTED in volet J): the paged STEEL forward was ALSO a sibling race.**
I2's original verdict ("no new sibling; paged STEEL barrier present") was WRONG — it
trusted the I2 *runtime* probe (byteΔ=0) over the *source*. The paged K-gather reuses
KV_smem with NO inter-iteration barrier; the race is **intermittent** (pytest-context-
triggered: 1–6 of 8 cells flake across 5 runs) so a single 20-run probe got lucky.
RULE 16 #2 lesson: source-verify the barrier, never infer "present" from a green
runtime. **Fixed in volet J (CX-J-02)** — start-of-loop barrier added; paged now
byteΔ=0 across D{64,128,256}×S{256,512,1024}, 8/8 + 12/12 stable ×5. Structural reason
the dense forwards were genuinely fine: on M5 they use `MFA_DIRECT_READS` (K via device
pointer, no Ks reuse) or emit barrier X in the gather path; paged/sage MUST gather
scattered K into Ks every iteration, so they always reuse and always need the barrier.
The remaining shared-buffer kernels (dense V1/V2, varlen, flash-decode, paged-varlen,
paged-TQ, GNA) were RE-VERIFIED AT SOURCE this volet — each emits the P@V→next-K
barrier (start-of-loop or end-of-body preload). Multi-tile outputs deterministic AND
oracle-correct. Locks: `tests/test_multitile_determinism_i2.py` (now incl. paged D256 +
paged-varlen + paged-TQ) + `tests/test_sage_determinism_s.py` + the paged_envelope
D=256 cells.

## Volet J — mechanical enumeration + CX-R7-01 (2026-06-23)
- **This inventory's row-set was the FAMILIAR SUBSET labeled complete** (round-7
  CX-R7-02): 8 of ~22 computational public entries, 16 of 34 computational raw. The
  authoritative row-set now comes from `scripts/enumerate_api_surface.py` (AST of
  `__all__` + regex of `m.def`) → `api_surface_enumeration.md`. **True scope: 103
  public exports (22 computational), 51 raw (34 computational); OMITTED computational
  = 13 public + 18 raw = 31** entries still needing the 4-axis treatment (input to the
  scope decision; this is a SEPARATE phase).
- **CX-R7-01 (CRITICAL) FIXED:** `sage_attention_prequantized` + raw `mfa_sage_forward`
  accepted malformed buffers (half-length V → OOB, batch mismatch → NaN, wrong
  k_int8/k_scale dtype → garbage, short k_scale → OOB). Now validated both surfaces
  (`_assert_sage_prequant_buffers` Python + C++ guards); all raise, valid byteΔ-identical,
  bite-proven. Lock: `tests/test_sage_prequant_validation_j.py` (12 cells).
- **CX-J-02 (CRITICAL, found via the CX-R7-03 D=256 matrix-honesty work):** paged STEEL
  nondeterminism (see corrected determinism axis above) — fixed.
- **CX-R7-03 matrix honesty:** D=256 paged cells added & executable; paged-varlen +
  paged-TQ determinism cells added; window attribution corrected (raw not public).

## Volet K1 — priority groups 1–6 hardened (2026-06-23)

4-axis sweep + fix of the 6 highest-priority K0 groups (10 raw entries incl. R1
retrofit). **Every entry that read multiple buffers was missing mutual checks**
(the systemic class round-7 predicted) — confirmed first-hand, all fixed via a
shared `validate_dense_qkv` C++ helper + per-entry residual.

| entry (K0) | correctness | accept-valid | reject-malformed (was→now) | determinism |
|---|---|---|---|---|
| `v6_nax_forward` (R15) | fp64 oracle relerr <3e-3 ✓ | GQA runs ✓ | batch→NaN, k_seq/k_heads/q_D/dtype **all no-raise → all RAISE** | N-A (cooperative tensor, only P staged) |
| `mfa_attention_varlen_forward` (R7) | — | valid runs ✓ | **NO validation + silent int32 cast → full Q/K/V + int32-reject (no cast)** | byteΔ=0 N≥512 (packed staging barrier, src-verified) |
| `mfa_attention_rope_forward` (R5) | — | valid runs ✓ | all dense no-raise + no cos/sin checks **→ all RAISE + cos/sin f32/shape/width** | byteΔ=0 N≥512 (barrier X, src-verified) |
| `mfa_attention_alibi_forward` (R3) | — | valid runs ✓ | all no-raise incl. invalid-GQA **→ all RAISE + slopes len Hq** | N-A (M5 direct reads) |
| `mfa_attention_bias_forward` (R4) | — | valid runs ✓ | Q/K/V mutual unchecked **→ RAISE** (bias mode/shape kept) | N-A |
| `mfa_attention_sparse_forward[_with_lse]` (R6) | — | valid runs ✓ | all no-raise **→ all RAISE** (mask checks kept) | N-A (contiguous tile jumps) |
| `sparse_attention_forward_with_lse` (R10) | — | valid runs (mask ≥4096B) ✓ | batch/k_seq/q_D no-raise **→ RAISE** (matched R9) | N-A (device row reads) |
| `mfa_scatter_kv` (R13) | — | valid runs ✓ | tokens-dtype≠pool unchecked **→ RAISE** | N-A (copy/scatter) |
| `mfa_attention_forward` (R1) | — | asym-D_v runs ✓ | was missing GQA/q↔k-D/dtype **→ added (shared validator)** | N-A |

Validation: bite-proven (neutralize shared `q.D==k.D` → alibi q_D≠k_D no-raise;
restore → raises). Validation-only → valid output byteΔ-identical by construction
(no math touched; v6 oracle relerr unchanged). Lock:
`tests/test_hardening_k1.py` (59 cells). RULE 16 catch: the plan called R9
"comprehensive" and R10 a subset — verified at source (R9 HAS the batch/K↔V/D
checks, R10 lacked them; plan correct) and the R6 `with_lse` variant was unbraced
so the first shared-helper edit missed it (caught by the lock failing).

**K1 closes 10 of 31 omitted entries → 21 remain (13 public + 8 raw)** for K2+:
public P1–P7 residuals (packed/speculative/splitfuse/rope-family/kvcache/topk),
raw R2 (`mfa_forward_with_lse`), R8 (`mfa_gna_forward`), R9 (`sparse_attention_forward`,
already comprehensive — verify-only), R11/R12 (quantizers), R14 (`conv3d_nax_forward`),
R16 (`v6_nax_backward_{query,kv}`).

## Volet K2 — remaining raw entries hardened (2026-06-23)

4-axis sweep of the last 8 raw entries. **2 had defects (R8, R16 — fixed); 5
verify-only** (R2/R9/R11/R12/R14 already comprehensive). HARD GUARD honored — and
it BIT: my first pass added "R2 f16/bf16-only", but R2's dense primitive SUPPORTS
float32 (the return_lse path upcasts to f32) — 7 suite tests failed, the check was
reverted. The lesson recurred exactly as warned; the suite caught it, then the
valid-space enumeration confirmed f32 is valid for R2 (verify-only).

| entry (K0) | correctness | accept-valid | reject-malformed (was→now) | determinism |
|---|---|---|---|---|
| `mfa_forward_with_lse` (R2) | fp64 oracle <3e-3 ✓ | f16/bf16/**f32** GQA ✓ | batch/k_seq/q_D/dtype/GQA **already RAISE** (verify-only) | N-A (dense direct/barriered) |
| `mfa_gna_forward` (R8) | (GNA, covered by public) | valid runs ✓ | **dtype-mismatch + window/stride≤0 no-raise (win-neg=NaN) → RAISE** | N-A (M5 direct reads; barriered legacy) |
| `sparse_attention_forward` (R9) | — | valid runs ✓ | batch/k_seq/q_D **already RAISE** (verify-only, comprehensive) | N-A (device row reads) |
| `mfa_quantize_per_block` (R11) | — | valid runs ✓ | block=0/non-pow2/f32 **already RAISE** | N-A (reduction) |
| `mfa_smooth_quantize_k` (R12) | — | valid runs ✓ | block=0/f32 **already RAISE** | N-A (reduction) |
| `conv3d_nax_forward` (R14) | — | valid runs ✓ | Cin/dtype/stride0/neg-pad **already RAISE** | N-A (conv/im2col) |
| `v6_nax_backward_query` (R16) | fp32 vjp oracle dQ <5e-2 ✓ | D=64 GQA ✓ | **dtype-mismatch + f16 lse/d_vec no-raise → RAISE** (shapes already raised) | N-A (coop tile) |
| `v6_nax_backward_kv` (R16) | fp32 vjp oracle dK/dV <5e-2 ✓ | D=64 GQA ✓ | same as query → RAISE | N-A |

Fixes: R2 +f16/bf16-supported; R8 +mutual dtype +positive lattice/window/stride;
R16 +`v6_check_bwd_dtypes` (q/k/v mutual f16/bf16 + lse/d_vec float32, called from
both query+kv). Bite-proven (3 checks neutralized → repros stop raising; restore →
raise). Validation-only → valid output/grads byteΔ-identical by construction (R2
oracle <3e-3, R16 vjp-oracle <5e-2 unchanged). Lock: `tests/test_hardening_k2.py`
(40 cells). Accept-valid proof: R16 lse/d_vec float32 confirmed at the production
call site (`_v6nax_backward`: `D = sum(dO*O).astype(float32)`, lse = forward
float32 output) BEFORE adding the check (HARD GUARD).

**K2 closes the raw surface → only the 13 PUBLIC entries (P1–P7 residuals) remain
for K3** (packed/speculative/splitfuse/rope-family/kvcache/topk).

## Volet K3 — final 13 public adapters hardened — INVENTORY COMPLETE (2026-06-23)

4-axis sweep of the last 13 public entries (adapters over the now-hardened cores).
Defect density continued to fall (K1 10/10 → K2 2/8 → K3 4/13). Dispatch
inheritance traced per entry: packed/speculative/splitfuse/sage_kvcache/rope all
route through hardened cores (Q/K/V/dtype inherited), so only **adapter residuals**
needed fixing.

| entry | inherits core | correctness | accept-valid | reject-malformed (was→now) | determ. |
|---|---|---|---|---|---|
| `flash_attention_qkv_packed` (P1) | flash_attention | ✓ | 3D/5D, kv≤Hq ✓ | bad-fused/no-heads raise; **5D num_kv_heads>buf silent-truncate → RAISE** | N-A |
| `flash_attention_kv_packed` (P1) | flash_attention | ✓ | ✓ | flat-reshape/5D-shape-derived already raise (verify) | N-A |
| `flash_attention_speculative_verify` (P1) | flash_attention | ✓ | int32/int64 ids, temp>0 ✓ | **float ids + temp≤0/inf no-raise → RAISE**; +ids shape | N-A |
| `flash_attention_splitfuse` (P1) | flash_attention | ✓ | prefill/decode/both ✓ | **partial-triple AttributeError → clean ValueError** | N-A |
| `flash_attention_rope_unified` (P2) | MFA rope (R5) | ✓ | **f16 cos/sin** ✓ | k_seq/etc inherited RAISE | gather (R5 barrier) |
| `flash_attention_rope` (P2) | rope_unified | ✓ | ✓ | inherited RAISE | gather |
| `flash_attention_kvcache_rope_append` (P2) | rope/append | ✓ | append@valid ✓ | OOB-append RAISES (memory-safe) | gather |
| `flash_attention_kvcache` (P3) | flash_attention/paged | ✓ | dense+paged ✓ | dense-append **concatenates (no OOB)**; paged-append OOB RAISES | gather (paged) |
| `flash_attention_speculative_verify_paged` (P4) | flash_attention_paged | ✓ | ✓ | **float ids + temp≤0 → RAISE**; pool inherited | gather (paged) |
| `flash_attention_varlen_qkv_packed` (P5) | flash_attention_varlen | ✓ | ✓ | **5D capacity → RAISE**; layout already raised | gather |
| `flash_attention_varlen_kv_packed` (P5) | flash_attention_varlen | ✓ | ✓ | flat/5D already raise (verify) | gather |
| `sage_attention_kvcache` (P6) | sage_attention | ✓ | ✓ | k_seq/batch inherited RAISE | gather (sage barrier) |
| `flash_attention_topk` (P7) | SDPA/MLX ref | ✓ | MHA ratio∈(0,1] ✓ | ratio bounds RAISE; GQA loud-raises | N-A |

**Defects fixed (4):** packed-5D capacity (qkv + varlen_qkv); speculative draft_ids
dtype + temperature (dense + paged); splitfuse partial-triple. **Cache-append family
probed for OOB first-hand and is MEMORY-SAFE** (dense concatenates; paged + rope-append
raise on out-of-range slots) — the highest-risk family is clean. Bite-proven
(temperature check neutralized → temp=0 no-raise; restore → raise). Validation-only →
valid output byteΔ-identical. HARD GUARD applied to EVERY new check with an accept-valid
cell (int64 ids, kv=Hq boundary, f16 cos, MHA-topk) — no over-strictness. Lock:
`tests/test_hardening_k3.py` (24 cells).

## ✅ INVENTORY COMPLETE — OMITTED computational = 0
`scripts/enumerate_api_surface.py` reports **0 omitted** computational entries (22
public + 34 raw, all AUDITED). The full computational attention surface — public +
raw — now has a first-hand 4-axis matrix row. The round-by-round sibling cycle is
closed: round-8 (Codex) is the convergence check.

## Notes (RULE 16)
- CX-R6-02 (sage nondeterminism): RESOLVED in volet S. My volet-I "byteΔ=0 over 8
  runs" used a config (N=256) below the multi-tile threshold — the race only fires at
  **N≥512** (KV_smem reuse across K-tiles). Root cause: missing start-of-loop barrier
  in `mfa_sage_forward`; NOT GQA-specific (MHA Hq2Hk2 N≥512 also raced). Fixed +
  locked (`tests/test_sage_determinism_s.py`).
- All findings reproduced first-hand before fixing; the Codex report was not taken
  as gospel — both CX-R6-01 and CX-R6-03 reproduced exactly as described.

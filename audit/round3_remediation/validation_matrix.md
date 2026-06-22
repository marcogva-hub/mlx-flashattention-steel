# Host-Side Validation Matrix — Volet C

> Branch `fix/audit-remediation`, host **M5 Max / macOS 26.6 / MLX 0.31.2**,
> base HEAD `7477672` (after volet A). One uniform host-side input-validation
> layer over **every public entry point AND every raw `_ext` binding × every
> edge-input class**, each raising a loud, specific error (Rule 8).  The valid
> envelope is **byteΔ-identical** before/after (validation adds raises only —
> proven over a 48-cell forward+grad sweep across D∈{64,128,256} × {f16,bf16} ×
> causal × N, plus the rebuild).  Line numbers verified at source (RULE 16).

## Helpers (factored, not copy-pasted)
- Python: `_validate_cu_seqlens(cu_q, cu_k, total_q, total_k, name)` (varlen
  family); inline gates in `flash_attention` (dropout×feature, zero-query arity,
  window).
- C++ `bindings.cpp`: `mfa_check_backward_inputs(name,q,k,v,O,L,dO)` +
  `mfa_check_bnq(name,what,q,a)` (raw/debug backward family).
- C++ `mfa_v6_nax_primitive.cpp`: `v6_check_bwd_gqa`, `v6_aux_bnq`,
  `v6_aux_bnqd` (all 8 raw V6-NAX backward wrappers, dense + sparse).
- C++ `mfa_paged_gather.cpp`: inline shape+dtype gate.

## Matrix — `RC` = raises-cleanly (post-fix) · `SW` = was silent-wrong/NaN/OOB · `N/A` = not applicable · `OK` = correct before

Edge classes: **eKV**=empty-KV(S=0) · **eQ**=empty-query(Nq=0) · **nm**=non-monotone cu · **cb**=cu[0]≠0/[-1]≠total · **sc**=q/k seg-count · **dt**=dtype mismatch · **nd**=wrong ndim · **D**=D∉supported · **gq**=GQA Hq%Hk≠0 · **h0**=Hk=0 · **win**=window<-1 · **drf**=dropout×feature · **psB**=paged seq_lens.shape≠B · **pdt**=paged meta dtype≠int32 · **aux**=raw-bwd aux shape≠q

| entry point | eKV | eQ | nm | cb | sc | dt | nd | D | gq | h0 | win | drf | psB | pdt | aux |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flash_attention` | RC† | **RC**‡ | N/A | N/A | N/A | OK | OK | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A |
| `flash_attention_varlen` | **RC** | **RC** | **RC** | **RC** | **RC** | OK | OK | OK | OK | OK | N/A | N/A | N/A | N/A | N/A |
| `flash_attention_paged_varlen` | OK | OK | OK | OK | OK | OK | OK | OK | OK | OK | N/A | N/A | OK | OK | N/A |
| `_ext.mfa_paged_kv_gather` | N/A | N/A | N/A | N/A | N/A | — | OK | N/A | N/A | N/A | N/A | N/A | **RC** | **RC** | N/A |
| `_ext.v6_nax_backward_kv` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_dv_raw` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_dk_raw` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_fused_dkdv_raw` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_{dv,dk,fused}_sparse_raw` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_query_sparse_raw` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.v6_nax_backward_query` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | OK* | OK* | N/A | N/A | N/A | N/A | **RC** |
| `_ext.mfa_backward_query_debug` | N/A | N/A | N/A | N/A | N/A | **RC** | **RC** | **RC** | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.mfa_backward_kv_debug` | N/A | N/A | N/A | N/A | N/A | **RC** | **RC** | **RC** | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.mfa_steel_backward` | N/A | N/A | N/A | N/A | N/A | **RC** | **RC** | **RC** | **RC** | **RC** | N/A | N/A | N/A | N/A | **RC** |
| `_ext.mfa_forward_with_lse` | N/A | N/A | N/A | N/A | N/A | OK | OK | OK | OK | OK | N/A | N/A | N/A | N/A | N/A |

† `flash_attention` empty-KV already raised (audit CC-10, pre-volet-C). ‡ empty-query previously returned a **bare array** breaking (O,L)/(O,weights) arity → now arity-correct (CX-04). `wm≤0` (CC-17): RC on the 6 wm-taking V6 wrappers. *`v6_nax_backward_query` GQA/Hk0 already guarded pre-volet-C; aux added now.

`—` (`mfa_paged_kv_gather` `dt`): the data **pool** dtype is f16/bf16/f32 (unconstrained, correct); only the **metadata** (block_table/seq_lens) is dtype-gated → see `pdt`.

## Findings closed (silent-wrong/NaN/OOB → raises-cleanly)

| finding | sev | entry × class | was | now |
|---|---|---|---|---|
| **CC-01** | CRIT | varlen × {eKV,nm,cb,sc} | empty KV segment → all-`-inf` row → silent **NaN** | `_validate_cu_seqlens` raises ValueError |
| **CX-01** | CRIT | flash_attention × drf | dropout + bias/softcap/window/alibi → feature **silently dropped** | raises (safe default; FLAG below) |
| **CX-02** | CRIT(host) | mfa_paged_kv_gather × psB | `seq_lens` shorter than B → **OOB read** of seq_lens | host raises if `seq_lens.shape[0]≠B` |
| **CX-03/CC-05** | HIGH/MED | debug bindings × {D,dt,nd,aux} | zero validation → kernel on malformed buffers | `mfa_check_backward_inputs` raises |
| **CX-04** | HIGH | flash_attention × eQ | bare array → broke (O,L)/(O,weights) **arity** | returns correctly-shaped empty tuple |
| **CX-05** | HIGH | mfa_paged_kv_gather × pdt | float/int64 metadata read as int32 → silent-wrong indices | host raises if not int32 |
| **CC-03** | HIGH(host) | 8 V6-bwd wrappers × {gq,h0,aux} | invalid GQA → **OOB KV-head read**; undersized aux over-read | `v6_check_bwd_gqa` + aux guards raise |
| **CC-04** | HIGH | _fallback_sdpa_with_lse | LSE convention flipped by head_dim (`log2 Σ2ˢ` vs MFA `log2 Σeˢ`) | unified on MFA convention + docstring (FLAG below) |
| **CC-17** | LOW | flash_attention × win; V6 `wm` | window<-1 silently disabled; `wm` unchecked | both raise |
| **CX-06** | MED | raw bindings × stream | real `mx.Stream`/`Device` → confusing TypeError | documented limitation in binding docstring (only None; runs on default GPU stream) |

**Total: ~50 cells across 15 entry points × 15 edge classes moved silent-wrong/NaN/OOB → raises-cleanly** (CC-04 is a value-unification behavior change, not a raise).

## Validation (bite-proven)
1. **Full suite:** `2333 passed, 91 skipped, 0 failed, 0 XPASS` (54s); collection ≥1800. Plus `tests/test_volet_c_input_validation.py` (33 cells) all green.
2. **byteΔ-identity:** 48-cell forward+grad hash sweep over the supported (D,dtype,causal,N) envelope — **0 diffs** before vs after (incl. the C++ rebuild).
3. **CC-01 exact repro** (`q=k=v=(1,4,32,128)`, `cu_seqlens_k=[0,16,16]`): now **raises** `ValueError`, no NaN (was 8192/16384 nonfinite).
4. **3 CRITICAL bite proofs** (non-destructive — monkeypatch / direct-call, never mutate-then-checkout):
   - **CC-01** (`test_bite_cc01_*`): monkeypatch `_validate_cu_seqlens`→no-op → the empty-KV varlen reproduces the silent NaN (n_nan>0). The guard is load-bearing.
   - **CX-01** (`test_bite_cx01_*`): `_dropout_sdpa(...)` output is byte-identical with/without an intended softcap (it has no softcap parameter) → combining them WOULD silently drop softcap; the public API now raises.
   - **CX-02** (C++): post-commit rebuild bite — `seq_lens.shape[0]≠B` test FAILS (no raise → OOB) when the guard is removed, passes when restored. Result recorded below.

   CX-02 rebuild-bite result: **PASS** (post-commit, RULE-2b-safe). Neutralizing
   `if (seq_lens.shape(0) != B)` → `if (false)` in `mfa_paged_gather.cpp` +
   rebuild → `test_seq_lens_shape_mismatch_raises` **FAILED** (no raise);
   `git checkout` + rebuild restored it → passes, byteΔ-identical. The guard is
   load-bearing.

## FLAG-FOR-SIGNOFF (2 — safe default implemented, capability/breaking decision deferred to Marco)

**[FLAG-1 — CX-01 dropout × feature composition].** Safe default shipped = **raise loud** when `dropout_p>0` is combined with `attn_bias`/`softcap`/`window_size`/`alibi_slopes` (the plain dropout path drops them silently). The alternative is to **implement full dropout∘feature composition** (apply dropout on top of the bias/softcap/windowed/alibi softmax). That is real work + a capability decision — NOT implemented. If you want the capability, it should compose dropout inside `_dropout_sdpa` (or route to `_sdpa_with_weights`, which already threads bias/alibi/window) and add per-combo oracle tests.

**[FLAG-2 — CC-04 return_lse convention].** Safe default shipped = **unify on the MFA-path convention** `L = log2(Σ_j exp(score_j))` (zero change for f16/bf16 D∈{64,128,256}, who already take the MFA path; **behaviour change for fp32 / unsupported-D return_lse users**, who used the fallback's prior `log2(Σ_j 2^{score_j})`). The alternative is to **standardize on natural-log LSE** `L = ln(Σ_j exp(score_j))` (the more interop-standard choice, e.g. for external online-softmax recombination) — but that **also changes the MFA-path value** for *all* return_lse users (a wider breaking change). Implemented the minimal unify-on-MFA + docstring fix; the natural-log option is yours to call.

---
*Telemetry/validation-only host gate; no kernel math, dispatch, or valid-path
output changed (byteΔ-identity #2).  Commit on `fix/audit-remediation` only.*

---

## Volet C2 — widening (round-4 CX-02/CC-01 · CX-03 · CX-04 · CX-05 · CC)

> Branch `fix/audit-remediation`, host **M5 Max / macOS 26.6 / MLX 0.31.2**,
> base HEAD `<after volet G>`. The round-4 re-audit showed the volet-C
> enumeration above had **completeness gaps**: it locked `mfa_paged_kv_gather`
> but **missed the shared `_validate_paged_block_table` validator** (used by
> `flash_attention_paged` + paged-varlen + TQ) and **lacked a Q/K/V
> mutual-shape-compat column** entirely. Closed here. Line numbers verified at
> source (RULE 16).

### The round-3 gaps, explicitly

1. **Missing rows (psB/pdt columns existed, but these entries were never rowed):**
   `flash_attention_paged` (+ the shared `_validate_paged_block_table`),
   `_ext.mfa_paged_steel_forward`, `_ext.mfa_paged_varlen_forward`,
   `_ext.mfa_paged_varlen_tq_forward`, `flash_attention_gna` (public + native).
   The matrix had `mfa_paged_kv_gather` only — so the *public* paged path and the
   *raw kernel-level* paged paths were unguarded for `psB`/`pdt`.
2. **Missing column `qkv`** = Q/K/V mutual-shape-compat (batch equal; `k_seq==v_seq`
   [`==N` for GNA]; head counts consistent; `head_dim` equal). The dense
   `flash_attention` got this in volet C (C-01) but it was never a *column*, so no
   other entry was swept for it — GNA (CX-03) fell through.

### New column + rows (`qkv` = Q/K/V mutual-shape-compat)

| entry point | qkv | psB | pdt | round-3 status |
|---|---|---|---|---|
| `flash_attention` | OK (C-01) | N/A | N/A | had qkv-check, not columned |
| `_validate_paged_block_table` (shared) | N/A | **RC** | **RC** | **row missing** |
| `flash_attention_paged` (public) | N/A | **RC** | **RC** | **row missing** (psB→NaN, pdt→trunc) |
| `_ext.mfa_paged_steel_forward` | N/A | **RC** | OK* | **row missing** (psB→NaN/OOB) |
| `_ext.mfa_paged_varlen_forward` | N/A | **RC** | N/A‡ | **row missing** (psB→NaN) |
| `_ext.mfa_paged_varlen_tq_forward` | N/A | **RC** | N/A‡ | **row missing** (psB→NaN) |
| `flash_attention_paged_varlen_turboquant` | N/A | **RC** | **RC** | psB via shared validator |
| `flash_attention_gna` (public + native) | **RC** | N/A | N/A | **column missing** (batch/seq/head/D → finite-wrong) |

`*` raw STEEL auto-casts metadata to int32 (pre-existing convenience) → `pdt` not
gated there; batch-cardinality (`psB`) is the OOB fix. `‡` raw varlen `pdt` left
to the kernel's existing handling; `psB` (the OOB) is gated via num_seqs =
`cu_seqlens_q.shape[0]-1`.

### Findings closed (round-4)

| finding | sev | entry × class | was | now |
|---|---|---|---|---|
| **CX-02 / CC-01** | CRIT | `_validate_paged_block_table` + raw `mfa_paged_steel_forward` × psB | rows < batch → **OOB read** → silent NaN/finite-wrong | shared validator: `block_table.shape[0]==expected_batch` (non-remap) + `seq_lens.shape[0]==block_table.shape[0]`; raw STEEL host guard mirrors it |
| **CX-05** | HIGH | `_validate_paged_block_table` × pdt | float/int64 metadata read as int32 → silent-wrong indices | validator rejects non-int32 metadata |
| **CX-04** | HIGH | raw `mfa_paged_varlen_forward` + TQ × psB | only Q rank checked → short `seq_lens_kv` → **OOB** → NaN | host guards: metadata rank + `block_table.shape[0]==seq_lens_kv.shape[0]==num_seqs` |
| **CX-03** | CRIT | `flash_attention_gna` (native) × qkv | no Q/K/V batch/seq/head/D check → **OOB** → finite-wrong | rejects batch/seq(`==N`)/head/`head_dim` mismatch pre-dispatch |
| **CC** | LOW | `flash_attention` × h0 | `Hk=0` → raw `ZeroDivisionError` | clean `ValueError` (q/k must have ≥1 head) |

### Validation (bite-proven)
1. **Full suite:** `2513 passed, 91 skipped, 0 failed, 0 XPASS` (~69s); collection
   ≥1800. `tests/test_validation_matrix_c2.py` (14 cells) green. Oracle envelope
   (`test_oracle_envelope.py`, 61 cells) unchanged → **byteΔ-identity** on the
   valid envelope (the diff is `throw`/`raise` *before* any compute).
2. **Each round-4 repro now raises** (was silent): `flash_attention_paged` B=2 /
   `seq_lens=[48]` (NaN); `block_table` B=1 / q B=2 (wrong-finite); float metadata
   (int32 trunc); raw `mfa_paged_steel_forward` short seq/bt (NaN/wrong); raw
   `mfa_paged_varlen_forward` short `seq_lens_kv` (NaN); GNA `q=[2,2,64,128]`/
   `kv=[1,2,64,128]` and `k_seq!=v_seq` (finite-wrong); `Hk=0` (ZeroDivisionError).
3. **Bite proofs** (non-destructive Edit-restore; sole-guard cells):
   - **CX-03 (Python):** neutralize GNA batch check → `gna_batch_mismatch` **FAILS**;
     restore → passes.
   - **CX-02 (C++, rebuild):** neutralize `if (seq_lens.shape(0) != B)` in
     `mfa_paged_steel_forward` + rebuild → `paged_raw_steel_seq_short` **FAILS**;
     restore + rebuild → passes.
   - The *public* paged cells are **doubly-guarded** (shared validator + C++ gather
     guard) — neutralizing one layer is masked by the other (defense-in-depth), so
     bite proofs target sole-guard cells.

*Volet C2 is a validation-only host-gate widening; no kernel math, dispatch, or
valid-path output changed.  Commit on `fix/audit-remediation` only.*

---

## Volet C2b — closing the siblings C2 exposed (base HEAD `1056f0b`)

Three completeness items a round-5 audit would re-find, **verified-first** (RULE 16).

### `qkv` column — completed for EVERY public q/k/v entry

| entry point | qkv | how |
|---|---|---|
| `flash_attention` | **RC** | inline (C-01) |
| `flash_attention_gna` (public) | **RC** | inline; **GQA-aware** (C2b fix — was wrongly `q==k==v`) |
| `_ext.mfa_gna_forward` (raw) | **RC** | C++ host guard (C2b item 2 — was unguarded) |
| `flash_attention_sparse` | **RC** | `_assert_qkv_mutual_compat` (C2b item 3 — was unguarded for `k_seq!=v_seq`) |
| `sage_attention` | **RC** | `_assert_qkv_mutual_compat` (C2b item 3 — was fully unguarded) |
| `flash_attention_varlen` | **RC** | `_assert_qkv_mutual_compat` in `_varlen_setup` — K/V packed-total/heads/head_dim (per-segment `q!=k` len stays the cu_seqlens contract) |
| `flash_attention_paged` | N/A | K/V are **pools** indexed by `block_table`, not q-shaped — invariant is the paged validator (C2) + `cache_batch_idx` bounds (below) |
| `flash_attention_paged_varlen` | N/A | pools — as above |
| `flash_attention_paged_varlen_turboquant` | N/A | pools — as above |

Shared helper `_assert_qkv_mutual_compat(q,k,v,fn)`: batch equal · `k_seq==v_seq` ·
`k_heads==v_heads` · `q_heads % kv_heads == 0` (GQA) · `q_dim==k_dim`. `v` head_dim
intentionally unconstrained (`D_v != D_qk` → SDPA fallback on some paths).

### `cache_batch_idx` edge-input row (Item 1 — verified already guarded → pinned)

| entry | cbi-OOB (`≥ rows`) | cbi-neg | ground truth |
|---|---|---|---|
| `flash_attention_paged` | **RC** | **RC** | already validated `[0, block_table.shape[0])` (pre-C2b); now lock-pinned |
| `flash_attention_paged_varlen` | **RC** | **RC** | already validated; now lock-pinned |
| `flash_attention_paged_varlen_turboquant` | N/A | N/A | does not take `cache_batch_idx` |

**No new raise added** for Item 1 (would be redundant); the existing guards are
pinned by `cbi_oob_paged` / `cbi_neg_paged` and the valid remap by `cbi_remap`.
The C2 `expected_batch` carve-out (suppressed under remap) is unchanged — the
*index value* is bounded, not the row count.

### Findings closed (C2b)

| item | sev | entry × class | was | now |
|---|---|---|---|---|
| Item 2 | HIGH | `_ext.mfa_gna_forward` × qkv | no Q/K/V check → OOB → finite-wrong | C++ host guard raises |
| Item 3 | HIGH | `flash_attention_sparse` / `sage_attention` / `flash_attention_varlen` × qkv | unguarded (k_seq!=v_seq / batch / heads) → OOB/finite-wrong | `_assert_qkv_mutual_compat` raises |
| Item 1 | — | paged × cbi | (already guarded) | pinned by lock (no redundant raise) |
| GNA-GQA | regression | `flash_attention_gna` (public) × valid GQA | C2 check wrongly rejected GQA `q!=k` heads (undetected — GQA test uses raw `_ext`) | GQA-aware check; valid-GQA pinned |

### Validation (bite-proven)
1. Full suite: `2526 passed, 91 skipped, 0 failed, 0 XPASS`; `test_validation_matrix_c2.py`
   27 malformed + 5 valid cells green. Oracle envelope (61 cells) unchanged →
   **byteΔ-identity** on the valid envelope.
2. Bite proofs (sole-guard, non-destructive Edit-restore):
   - **Item 3 (Python):** neutralize `_assert_qkv_mutual_compat` in `sage_attention`
     → `sage_batch` + `sage_kseq_ne_vseq` **FAIL**; restore → pass.
   - **Item 2 (C++, rebuild):** neutralize the raw-GNA batch guard + rebuild →
     `raw_gna_batch` **FAILS**; restore + rebuild → passes.
3. GNA-GQA regression: `_ok_gna_gqa` (H_q=8/H_kv=2) must NOT raise — pins the fix.

*Validation-only host-gate completeness; no kernel math, dispatch, or valid-path
output changed.  Commit on `fix/audit-remediation` only.*

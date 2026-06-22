# Kernel Buffer-Read Bounds Audit — Volet B

> Branch `fix/audit-remediation`, host **M5 Max / macOS 26.6 / MLX 0.31.2**,
> base HEAD `5222ff1` (after volets A, C). Every device-buffer read in every
> Metal kernel under `csrc/`, classified by guard ordering: **guarded-correctly**
> (check precedes read) / **mis-ordered** (read precedes check) / **unguarded**
> (no in-kernel guard; relies on a host invariant). Line numbers verified at
> source (RULE 16). This is the defense-in-depth layer behind volet C's host
> validation: even a raw `_ext` call bypassing host checks must not OOB-read.

## Summary
- **~22 JIT kernel emitters** (13 generator files) + 3 static `.metal` (1 placeholder, 2 dead-on-platform) scanned; **~70 distinct device-buffer read sites** classified.
- **mis-ordered: 2** (both fixed) · **OOB-on-partial-tile: 2** (both fixed) · **unguarded-on-host-invariant: ~14 classes** (none a live bug; standard FA grid/shape/divisibility contracts, several mechanically enforced by volet C) · **guarded-correctly: the large remainder**.
- **4 reads fixed this volet** (all byteΔ-identical on valid input — proven over a 60-cell partial-tile + 48-cell aligned sweep):

| # | file:line | buffer | defect | fix |
|---|---|---|---|---|
| 1 | `mfa_steel_paged_varlen_tq_fwd.cpp:370` (K-gather) | `block_table` | **mis-ordered**: read precedes `blk_idx<max_blocks` | nested-if, check-before-read (CC-02) |
| 2 | `mfa_steel_paged_varlen_tq_fwd.cpp:501` (V-gather) | `block_table` | **mis-ordered** (same) | nested-if (CC-02) |
| 3 | `mfa_steel_fwd_v2.cpp:541` (K direct-read) | `K_cur` | **OOB**: key-row `sn` unclamped on partial final tile (V was clamped, K wasn't — III-9 sibling) | clamp `k_row=min(sn,kL_rem-1)` |
| 4 | `mfa_gna_fwd.cpp:363` (K direct-read) | `K_cur` | **OOB** (same III-9 K sibling) | clamp `k_row` |

Findings #3/#4 are **live on M5** (`use_direct_reads = is_m3_plus[ && !has_rope]` → the default forward path), masked-to-`-INF` by the K-boundary mask (so result-neutral) but a genuine UB OOB read — fixed per "do not rely on Apple-GPU-returns-0" (it is UB).

## Full table

### Group 1 — Paged-varlen forward

**`mfa_steel_paged_varlen_tq_fwd.cpp` (TurboQuant — the core finding, FIXED)**

| buffer | index | bound | enforced where | class |
|---|---|---|---|---|
| `tile_offsets`/`cu_seqlens_q`/`seq_lens_kv` | `[seq_id(+1)]` | seq_id<num_seqs | host: tables sized num_seqs(+1) | unguarded (host) |
| `centroids`/`v_centroids` (smem load) | `[thread_idx]` | `if(thread_idx<n_centroids)` precedes | guarded | guarded-correctly |
| **`block_table` K** | `[seq_id*max_blocks+blk_idx]` | blk_idx<max_blocks | **was read@370 BEFORE guard@372 → now `if(blk_idx<max_blocks){phys=block_table[..];if(phys..)..}`** | **mis-ordered → FIXED** |
| **`block_table` V** | same | blk_idx<max_blocks | **was@501 → FIXED** | **mis-ordered → FIXED** |
| `k_pool_tq`/`v_pool_tq`/`k_scales`/`v_scales`/`v_pool` | `[phys*stride+..]` | phys∈[0,num_blocks) | inside the (now correctly-nested) phys guard | guarded-correctly |
| `k_centroids_smem`/`v_centroids_smem` | `[idx]` | idx from `&3`/`&15`/3 bits ≤15<16 | construction | guarded-correctly |
| Q | loader `load_safe(qL_rem)` | partial-tile clamp | loader | guarded-correctly |

**`mfa_steel_paged_varlen_fwd.cpp` (non-TQ — the REFERENCE pattern)** — `block_table` K@287/V@391 read AFTER `blk_idx<max_blocks`; pool reads after phys guard. **All guarded-correctly.**

### Group 2 — Paged gather / scatter

**`mfa_paged_gather.cpp`**

| buffer | index | bound | enforced where | class |
|---|---|---|---|---|
| `seq_lens` | `[b]`, b∈[0,B) | b<B | **volet-C host invariant `seq_lens.shape(0)==B`** (`mfa_paged_gather.cpp:247-261`, added volet C / CX-02 kernel half) | unguarded (host — covered) |
| `block_table` | `[b*max_blocks+log_blk]` | log_blk<max_blocks | guard@70 precedes read@75 | guarded-correctly |
| `k_pool`/`v_pool` | `[phys_blk*stride+..]` | phys_blk∈[0,num_blocks) | guard@78 precedes read | guarded-correctly |
| (CPU fallback) | `table_ptr[b*max_blocks+log_blk]` | log_blk<max_blocks | guard@121 precedes@123 | guarded-correctly |

**`mfa_scatter.cpp`** — `pool_in/out[elem]` (elem<total guard@60), `blk_ids/blk_offs/tokens[n..]` (loop bound n<N_write). No block_table→pool gather. **All guarded-correctly.**

### Group 3 — STEEL forward v1/v2/v3

- Q/K/V/O stream via `BlockLoaderT::load_safe/load_unsafe/store_safe` (partial-tile clamped). **guarded-correctly.**
- **PagedSteelForward gather** (`mfa_steel_fwd.cpp:~3119-3289`): `global_tok<kL` → `blk_idx<max_blocks` → phys range, guards precede reads. **guarded-correctly.**
- **`mfa_steel_fwd_v2.cpp:541` K direct-read (`MFA_DIRECT_READS`, M5 default)** — key-row `sn` **was unclamped** on the partial final K-tile (V sibling@790 clamps). **OOB → FIXED** (`k_row=min(sn,kL_rem-1)`).
- Unguarded-on-host (not bugs): RoPE `rotary_cos/sin` (rely on host BQ/BK-padded tables), `attn_bias`, `block_mask`, `alibi_slopes[tid.y]`, `seq_lens[b_idx]` (v1 paged) — host shape/grid contracts.

### Group 4 — GNA / Sage / Sparse / Conv

- **`mfa_gna_fwd.cpp:363` K direct-read** — same III-9 K sibling. **OOB → FIXED.** (V sibling@499 clamps.)
- Sage `K_scale_bh[kb]` (kb<kb_lim≤NK, host scale shape); Q/Kb/V partial-tile guarded. **guarded-correctly / unguarded-on-host.**
- Sparse (V6NAX coop + scalar ref): loop-bound / early-return / coop-extent guarded, or unguarded-on host divisibility `qL%BT==0 ∧ kL%BT==0` (host-enforced). No partial tiles by construction. **guarded-correctly / unguarded-on-host.**
- Conv NAX (matmul2d / im2col3d / conv3d_mpp): im2col `X` gather 3D-bounds-guarded (guard@206-208 precedes@211); time axis `if(tf<0||tf>=T)continue`; matmul coop-extent-bounded + JIT-throw on unaligned K. **guarded-correctly.**

### Group 5 — V6 NAX / ccv dense / STEEL backward

- **`NAAttentionKernel.cpp` `kv_head_idx = tid.y/gqa_factor; K/V += kv_head_idx*strides[1]`** (~15 sites incl. :2816-2818, :3958, :4440) — **unclamped**, relies on host `Hq%Hk==0 ∧ grid.y==Hq` ⇒ `kv_head_idx∈[0,Hk)`. **This is the CC-03 kernel half.** The host invariant is **mechanically enforced by volet C**: `v6_check_bwd_gqa` (`mfa_v6_nax_primitive.cpp:39-43`, called at every raw backward wrapper) raises on `Hq%Hk≠0 ∨ Hk==0`; the forward + `v6_nax_backward_query` guard it too. **unguarded (host — now covered by volet C).** (RULE-16 correction: the prompt's `v6_nax_compile.mm:464` is host param-packing, NOT a kernel read — the real reads are in `NAAttentionKernel.cpp`.)
- `mfa/AttentionKernel.cpp`: Q/O/L by gid.y/z, K/V by gid.y/(Hq/Hk) — unguarded-on-host-grid (standard FA contract); traversal tiles `min(BK,C-off)`-clamped. **guarded-correctly / unguarded-on-host.**
- `mfa_steel_bwd.cpp` (dQ + dKV): base advances by dispatched grid id (host grid==shape); L/delta predicated `(q_row<qL)?:0`; tiles via load_safe/store_safe. **guarded-correctly / unguarded-on-host.**
- `mfa_steel_fwd_v6_nax.cpp` (standalone `_ext`-only): coop slices `get_mask` + `if(k_row_lo>=kL)break`. **guarded-correctly.**

### Group 6 — Static `.metal`
- `csrc/kernels/attention_forward.metal` — no-op placeholder, zero device reads.
- `csrc/async_v2_{noasm,kernel}.metal` — **NOT dispatched on macOS≥26** (`shader_cache.mm:97-104` returns nullptr → JIT path). Edge reads use load_safe/store_safe; `L[..]` is `q_idx<qL`-guarded. Dead-on-platform; excluded from the live tally.

## Reads whose bound depends on a volet-C host invariant (with exact host check)
- **CX-02 kernel half** — `mfa_paged_gather.cpp:62` `seq_lens[b]` (b∈[0,B)): bound = `seq_lens.shape(0)==B`, enforced host-side at `mfa_paged_gather.cpp:247-261` (volet C). Confirmed the kernel performs no read with `b≥B` beyond this.
- **CC-03 kernel half** — `NAAttentionKernel.cpp` `kv_head_idx=tid.y/gqa_factor`: bound = `Hq%Hk==0 ∧ Hk>0`, enforced host-side at `mfa_v6_nax_primitive.cpp:39-43` (`v6_check_bwd_gqa`, volet C) + the forward GQA guard. No additional in-kernel unguarded read remains (aux arrays lse/o/d_o/d_vec are q-shape-checked by volet C's `v6_aux_bnq`/`v6_aux_bnqd`).
- **CC-02 raw path** — `mfa_paged_varlen_tq_forward` (raw `_ext`): the **kernel reorder (#1/#2)** is the defense (no OOB read even if `blk_idx≥max_blocks`). The **public** path additionally raises via `_validate_paged_block_table` capacity branch (`attention.py:7212-7217`, reached at `flash_attention_paged_varlen_turboquant:8235`) — **RULE-16 correction:** this host capacity invariant ALREADY EXISTED for the public path (not missing as the prompt's premise stated); only the raw C++ free function lacks it, and the kernel reorder makes the raw path safe. A host value-read in the C++ binding was deliberately NOT added (it would force a per-call GPU sync and duplicate the public guard; the kernel guard is the correct defense-in-depth layer).

## Re-confirmed CLEAN (not taken on faith — source-verified)
`mfa_forward_with_lse` (full validation, volet C); `mfa_paged_gather`/`mfa_scatter` block_table→phys→pool ordering; the non-TQ paged-varlen kernel; PagedSteelForward gather; the BK=16/TK=1 paired-MMA guard (`mfa_v6_nax_primitive.cpp`, raises on BK=16 except the II-8-tail dense-fused site); int64 offset casts (`(long)phys * stride`) throughout the paged/TQ pools.

## Validation (bite-proven)
1. **Full suite:** `2366 passed, 91 skipped, 0 failed, 0 XPASS` (57s); collection ≥1800. `tests/test_volet_b_buffer_bounds.py` 9 cells green.
2. **Source-order lock BITES (CC-02):** `_tq_read_is_guarded(real file)`=True; a scratch reconstruction of the read-before-check order (never mutating the tracked file — RULE 2b) → False. `tests/test_volet_b_buffer_bounds.py::test_tq_ordering_lock_bites`.
3. **K-clamp lock BITES:** real v2/gna use `k_row`; a scratch `k_row→sn` revert → lock fails. `::test_k_clamp_lock_bites[*]`.
4. **Host capacity invariant test + bite:** over-capacity `seq_lens (100 > 4*16)` raises; monkeypatching `_validate_paged_block_table`→no-op removes the raise (load-bearing). `::TestHostCapacityInvariant`.
5. **byteΔ-identity on valid envelope:** K-clamp partial-tile sweep (D∈{64,128,256} × N∈{100,250,513,1000,4095} (non-BK-aligned) × causal × f16/bf16, forced `backend="mfa"`) = **60 cells, 0 diffs** before/after (the pre-fix path was deterministic ⇒ OOB was masked, not leaking); dense aligned sweep = 48 cells, 0 diffs. TQ reorder result-preservation additionally covered by `test_phase3_iii2_tq_decode` + `test_turboquant` (unchanged).

---
*Bounds-ordering only; no kernel math or in-bounds result changed (#5). Commit on `fix/audit-remediation` only.*

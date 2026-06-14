# III-4 fresh-eyes audit — findings ledger (pass 1, 2026-06-12)

Working queue. Status: OPEN / FIXED / DECLINED / VERIFIED-FALSE.
Agents: C++ (1 finding), docs (15), tests (17), dispatch-core (18); runtime agent pending.

## C++ agent
- [FIXED] CXX-1 MEDIUM: TQ packed-pool offsets int32-truncated (K row_off/base_off, V vrow_off) → long end-to-end. Rebuilt; 96 TQ tests green.

## Docs agent (15)
- [FIXED] DOC-1 HIGH: `from mlx_mfa import sparse_attention_dispatch` ImportError → added to _LAZY_IMPORTS + __all__.
- [DELEGATED→agent aff9c7b] DOC-2..10, 12..15: README v2.39.1/v2.50/sparse/conv staleness, ENV_VARS V34 rows + missing opt-outs, HOOK_TELEMETRY, TRAINING_QUICKSTART, INVENTORY/INDEX headers, PERF_CLAIMS header, CLAUDE.md status, known-debt KD-7.
- [FIXED] DOC-11 MEDIUM: perf-claims `documented_in` fields cite README content not present (ii12 optout, v2.38.1 figures) → repointed to CHANGELOG.md (+ v38-1 audit doc, figures verified present); 3rd v2.38.1 row (qL16384) never cited README.

## Tests agent (17)
- [FIXED] F1 HIGH: fused V-TQ GT locks added — test_phase3_iii2_tq_decode.py::TestDecodePathGroundTruth::test_fused_v_tq_matches_ground_truth, bits∈{2,3,4}, tq_v_enabled=True, GT = apply_rotation(sdpa(q_rot, K_deq, V_rot_deq), "wht") (wrapper de-rotates; WHT self-inverse). Measured ~2e-4, bar 5e-3. All 3 pass.
- [FIXED] F2: test_wht_fused_matches_python_wht parametrized over bits∈{2,3,4} (×D∈{64,128} = 6 cells); all pass.
- [FIXED] F3: test_fused_other_bitwidths upgraded to decompress-path GT (turboquant_compress/decompress -> SDPA, same quantization); max-abs < 0.1; passes for bits 2 and 4.
- [FIXED] F4: all 15 tests in TestPagedSteelForward + TestPagedFlashDecode now reference mx.fast.scaled_dot_product_attention (mask="causal" for causal; native GQA — k_exp/v_exp expansion removed). Original tolerances kept; 15/15 pass.
- [FIXED] F5 LOW: renamed test_fp16_path_unchanged_bitwise → test_fp16_path_deterministic; docstring states it is a determinism lock (correctness lives in test_conv_nax.py torch GT).
- [FIXED] F6 HIGH: ii6 TestUnitScaleCorrectness::test_default_on_matches_sdpa_vjp_elementwise + TestNonCausalPromotionII12::test_unit_scale_elementwise parametrized over {fp16,bf16}. Measured floors: fp16 0.004-0.008, bf16 0.016-0.0625 → bounds fp16 0.1, bf16 0.25 (commented in file). All pass.
- [FIXED] F7 HIGH: unit-scale retrofit (normal std 1.0) in all 6 LCSA files + one std-8 adversarial-finiteness test per file. NO kernel corruption discovered — all GT comparisons pass at unit scale at the original bounds; one tolerance-only recalibration: phase1_4 dispatch_routes_moderate_density (1e-5 → 1e-3; cross-path NAX-vs-SDPA at Sprint-1 threshold 1.01, measured floor max-abs 2.44e-4). Measured floors commented per assert (rmse 4e-6..8e-5).
- [DEFERRED→Marco queue] F8 MEDIUM: not implemented per III-4 scope — M1-M4-only flash-decode kernel is unreachable on M5 dev hardware; adding a force-knob is Marco's call (hardware-coverage class).
- [FIXED] F9 MEDIUM: unit-scale retrofit in all 6 V34 files + std-8 adversarial-finiteness test each. NO corruption discovered. Recalibrations (commented): kv dk_bound 1e-4→5e-4 (floor 5.7e-5, old margin 1.8x); dq bf16_bound default 1e-4→1e-3. v50_v34_causal: 0.1-defending comment removed — N(0,1) does NOT overflow fp16 softmax on M5 (measured floors dQ/dK 2.9e-3, dV 2.0e-3 vs bound 1e-2).
- [FIXED] F10 HIGH: fixture moved inside TestGNABackward (class-scoped method); whole module now exercises the production GNA-native path. No newly-exposed failures: test_attention.py 696 passed ×3 consecutive runs, no tolerance changes needed.
- [FIXED] F11 MEDIUM: both fixtures restore pre-test hook state in teardown (uninstall only if hooks weren't installed before — preserves the import-time auto-install baseline; test_conv_nax uses torch/conv_general so uninstall is safe). iii1 engagement test now monkeypatches ah._HOOK_TELEMETRY_MODE to "summary" (valid non-off mode) instead of the vacuous if-guard.
- [FIXED] F12 MEDIUM: mx.clear_cache() added after test_all_false_mask_row_gives_nan_or_zero (matches the :1495 steel_sparse pattern). Zeros-or-NaN assert left as-is (it documents the kernel's empty-softmax contract).
- [FIXED] F13 LOW: comments added — sage 0.30/0.50 (lossy int8 + documented ±0.2 Metal nondeterminism on causal), tq 1.0 (lossy 3-bit quality test backed by Pearson-correlation assert; kernel correctness locked by decompress-GT tests).
- [FIXED] F14 LOW: module docstring rewritten — causal eligibility gate was lifted in Phase 4b-complete; no longer claims SDPA-vjp fallback.
- [FIXED] F15 MEDIUM: ghost-env arm deleted with comment (grep-verified read nowhere in mlx_mfa/csrc); MFA_ENABLE_V34_BACKWARD (already set) is the real knob; opt_in kept as report annotation only.
- [FIXED] F16 MEDIUM: mfa_alibi row now uses flash_attention(alibi_slopes=...); silent ImportError guard removed. Smoke-verified: row measures again (0.239ms at B1 H4 N256 D64).
- [FIXED] F17 LOW: dead env write deleted with comment (backend="mfa" kwarg is the real mechanism; env grep-verified read nowhere).

## Dispatch-core agent (18)
- [FIXED] D1 CRITICAL: backend="sdpa" early return (attention.py:318-336) silently drops softcap/alibi/window/dropout/return_lse/return_attn_weights; later sdpa branch (503) is dead code → delete early return, let use_mfa=False path handle.
- [FIXED] D2 HIGH: softcap+window backward (_windowed_sdpa, attention.py:4794) never applies softcap → wrong grads for the combo.
- [FIXED+D6-partial] D3 HIGH: return_lse=True MFA path (attention.py:599) drops softcap/window → gate shortcut on softcap==0 and no window.
- [FIXED fwd+bwdDQ+bwdDKV, M3-sim tests pass] D4 HIGH: D=128 sparse block-mask geometry: Python builds BK=16, C++ M3/M4 runs BK=32 → silently wrong sparse on M3/M4 D=128. Fix: C++ force non-m3plus block config when has_block_mask (mirrors V2/V5 exclusions); audit steel_sparse backward same class (§AA.5.x).
- [FIXED] D5 HIGH: _paged_batched_bwd causal mask aligned to max_kv_len not per-seq kv_len → wrong grads for heterogeneous seq_lens N_q>1.
- [FIXED — attn_bias×softcap/alibi + alibi×softcap raise; return_lse combos raise (D3)] D6 MEDIUM: no mutual-exclusion raises for feature combos (attn_bias×softcap×alibi×window×return_lse, sage drops) → validation block.
- [FIXED+locked — CONFIRMED real (fwd 0.67/grads 1.1 at N=100); _expansion_tile + 6 sites + head_dim threading; tests/test_phase3_iii4_d7_mask_tiling.py] D7 MEDIUM: block-mask→bias expansion helpers re-derive BQ_actual=ceil(N/NQ) ≠ kernel BQ for non-divisible N (4 sites: attention.py:3708,3750,3832,3900) → pass true kernel BQ/BK.
- [FIXED+locked — force_kernel threaded; backend='mfa' runs MFA kernel (fp16 floor 0.001, not bit-SDPA); tests/test_phase3_iii4_dispatch_guards.py] D8 MEDIUM: backend="mfa" silently routed to SDPA forward on V34-eligible cells (II-8 fusion branch) → forced-backend measurement corruption class; thread backend/force flag.
- [FIXED — v.shape[3]==D and kv_len==qL gates added to carve-out] D9 MEDIUM: V34 carve-out misses v_dim_mismatch guard + ignores kv_len (cross-attention can enter V34 backward) → add guards.
- [FIXED — both RoPE offset sites raise on heterogeneous offsets] D10 MEDIUM: paged RoPE offsets use min/first seq_len for heterogeneous batches (attention.py:1999, 2180) → raise or per-row.
- [FIXED] D11 MEDIUM: flash_attention_kvcache paged drops softcap silently; paged-append drops softcap/alibi/window → add raises.
- [FIXED] D12 MEDIUM: flash_attention_sparse(backward=...) ignored on M5+, invalid values silently accepted → validate at entry.
- [FIXED+locked — _rope_tables_match_base10000 probe-verify; custom-base routes to STEEL bit-identical to opt-out; D8/D13 lock file] D13 MEDIUM: NAX RoPE fast path ignores user cos/sin tables (non-10000 theta wrong) + drops explicit k_offset (attention.py:991, 2195) → table check + use _k_off.
- [FIXED] D14 MEDIUM: auto-hook fallback re-runs baseline on precision-truncated input (_auto_hooks.py:345) → keep orig_input.
- [FIXED] D15 MEDIUM: bare except around native attn_bias kernel (attention.py:394) → narrow + telemetry.
- [FIXED+locked — CONFIRMED real (dV RMSE 0.506 at bt=32); bt>=64 eligibility gate; tests/test_phase3_iii4_d16_sparse_bwd.py] D16 MEDIUM(conf HIGH): KD-1 OR-downsample may contaminate V34 sparse backward grads (env-gated path) → probe numerically vs SDPA-vjp; restrict or document.
- [FIXED] D17 LOW: stale docs (attn_bias "always falls back", make_causal_block_mask summary, _M5_NAX_THRESHOLDS comment, _v34_eligible qL=4096 docstring).
- [FIXED] D18 LOW: silent catches — _load_calibrated_kernel_config except-pass; GNA native except-pass retires kernel silently (§Z class).

## Runtime agent (15)
- [FIXED, repro now bit-exact] R1 CRITICAL (VERIFIED w/ repro): sliding-window decode via patch_mlx_lm attends ONLY to key 0 — decode branch passes causal=False + window, but kernel qL_off only applies when causal (csrc/mfa_attention.cpp:1203; integrations/mlx_lm.py:189-215). Latent with stock mlx-lm 0.30.7 (no max_kv_window attr). Fix: causal=True when window active in decode branch (+§AA.5.x sweep of V1/V2/V5/splitK window sites). Test gap: test_mlx_lm_integration.py:643 asserts counter only.
- [FIXED] R2 HIGH: HybridKVCache._demote_seq to secondary never resets primary tier (kv_cache.py:312-320) → pool exhaustion; with R3 corruption. Fix: primary reset after copy + occupancy regression test.
- [FIXED] R3 HIGH: HybridKVCache multi-seq over single-seq primary silently interleaves (kv_cache.py:514; DenseKVCache ignores seq_id) → raise when sid!=0 and not capabilities.multi_seq.
- [FIXED] R4 MEDIUM: eviction without secondary/external silently truncates history (drop_no_secondary, kv_cache.py:321) → raise/tombstone.
- [FIXED] R5 MEDIUM: residual mx.synchronize() per decode token in TQ step (inference.py:1085) — missed Sprint-C site. Delete + 300-step equivalence.
- [FIXED] R6 MEDIUM (VERIFIED w/ repro): make_sink_window_mask under-approximates window left edge (masks.py:949: q_end-ws instead of q_start-ws) → drops active tiles (6/42 at S=256 ws=64).
- [FIXED] R7 MEDIUM: TQ _block_table_cache survives reset() + uninitialized in __init__ → clear in reset, init in __init__.
- [FIXED] R8 LOW: pools "materialise" comment wrong — mx.eval(*pools) missing (inference.py:853).
- [FIXED — doc note: fp16 V pool unconditional for III-2 decode; lazy packed-V → Marco queue] R9 LOW: tq_v=True stores V both packed AND fp16; per-token pack_v cost with III-2 decode never reading packed V → document/conditional.
- [FIXED — lcsa density_threshold (1.01) + block_tile (16 vs 32) docstrings corrected] R10 LOW: lcsa doc drift (0.02 vs 1.01 threshold; block_tile 16 vs 32).
- [FIXED] R11 LOW: make_strided_mask boundary excludes position 0 from global-stride set (masks.py:1412).
- [FIXED] R12 LOW: gqa_decode_cider no D%32 validation (D=80 silently wrong) → loud check.
- [DEFERRED→Marco queue (perf-debt; Pattern #6 — needs bench to justify the rewrite)] R13 LOW: make_topk_spatial_mask retains the pre-II-7 numpy/loop pattern (11ms class).
- [DEFERRED→Marco queue (perf-debt; per-step mx.array uploads, micro-opt)] R14 LOW: per-step mx.array uploads (get_seq_lens, cu_q) cacheable.
- [FIXED — doc note: TQ pads block table with 0 (fail-safe vs -1); reads bounded by seq_lens] R15 LOW: block-table pad sentinel inconsistency (-1 paged vs 0 TQ).

## Resolved pre-sweep
- Plain-paged decode gather+sdpa candidate: DECLINED (0.96-0.97x, bit-exact parity; legacy cost is append/build).
- Conv C=16 divergence: investigated; isolated probe correct both dtypes; gate stays (resolved-divergence).

## Loop status
Pass 1 fixes in progress. After all fixes: full suite + stressed + fresh pass 2.

PASS1-REGRESSION: order-dependent failure of test_mixed_dtype_routes_mfa (and flaky
test_kvcache_k_new_paged_succeeds) root-caused — NOT a batch test leak.  Latent
mixed-dtype kernel-input bug (§AA.5.x class): eval_gpu derives dtype_code from q
ALONE (csrc/mfa_attention.cpp:111-114), never validates K/V dtypes.  f32 q + f16
K/V → f32 kernel reinterprets f16 buffers (silent garbage, max_err ~15 vs SDPA;
NaN only when the Metal buffer pool recycles dirty allocations → order/allocation-
dependent).  Two gates fixed Python-side (no csrc changes): (1) flash_attention
dispatch now casts K/V to q.dtype before any backend (attention.py ~499, also
covers the SDPA-fallback mixed-dtype NaN); (2) paged-append casts k_new/v_new to
pool dtype before _mfa_scatter_kv_cpp (raw byte scatter of f32 rows into f16 pool
wrote reinterpreted halves incl. NaN/inf bit patterns).  test_mixed_dtype_routes_mfa
strengthened with a cast-SDPA ground-truth assert (finiteness alone passed garbage).
Remaining same-class gates (direct flash_attention_paged / varlen with mixed-dtype
user inputs) not audited — flag for pass 2. [FIXED]

F1 note: fused V-TQ CORRECT at bits 2/3/4 vs DE-ROTATED ground truth (2e-4); first probe missed the wrapper's output de-rotation (attention.py:7183). GT recipe: O_gt = WHT(sdpa(q_rot, K_rot_deq, V_rot_deq)).

## Pass-1 additional kernel find (exposed by F-batch adversarial tests)
- [FIXED+locked] D-TOPK CRITICAL: topk bisect threshold kernel grid mis-specified — grid.x=N (threads) launched only N/256 threadgroups, so only the first 8 query rows per head were written; the rest read STALE pool memory (benign zeros usually, exposed as out-of-range by adversarial pool state). Promoted AUTO-default kernel selected top-K for ~8/N rows. Fixed grid.x=N*256 at production + test sites; strengthened per-row range assertion. Validated vs ground-truth topk attention (rows 8+ now correct). Suite contamination root-caused (was the post-restart 'flaky test').

## PASS 2 (fresh re-audit, 2026-06-14)
- [CLEAN] Grid-spec class sweep (agent ad6025f): all 9 production mx.fast.metal_kernel dispatches verified correct (indexing mode ↔ grid spec ↔ output coverage). topk grid-undercount was isolated. topk_stream_v5, cider pass1/pass2, tq_decode K/V, conv im2col/matmul2d ×2 — all CORRECT + stale-read-protected. masks/lcsa/turboquant have no metal_kernel (II-7 LCSA build is pure MLX ops). C++ MTL::Size path out of scope (separate semantics).
- [CLEAN/PASS] Regression + general re-audit (agent a315d85): all 10 pass-1 fixes live-verified PASS (not over-broad). svdquant/quantize/dispatch_policy/runtime/fwd-bwd-consistency CLEAN. No v2.37.0-class short-circuit. ONE LOW [FIXED-doc]: flash_attention_gna native path forward-only (no vjp) → mx.grad raises (loud, documented) → added autograd docstring note (MFA_DISABLE_GNA_NATIVE=1 routes to differentiable sparse path). svdquant rank>min(M,K) mis-reports memory_bytes only (report-only, self-corrects in forward).
- [FINDINGS] dtype/§AA.5.x multi-gate sweep (agent a6258510):
  - Class A (dtype-from-q reinterpret) — 6 expert entries uncovered (A1 paged-STEEL, A2 varlen-STEEL, A3 paged-varlen-fused, A4 TQ, A5 sage, A6 GNA-native). [FIXED] loud-raise guard `_assert_kv_dtype_matches_q` at all 6 Python entries (Rule 8; expert APIs where dtype mismatch is a caller bug; uint8 packed pools skipped) + graceful q→pool cast at the 2 flash_attention_kvcache paged sites (the legitimate fp32-q/fp16-pool decode flow that test_kvcache_k_new_paged_succeeds exercised — was finite-but-wrong, shape/finite assert too weak to catch). flash_attention main path + gather/sparse/packed already covered (pass-1).
  - Class B (window qL_off anchor) — [FIXED-consistency] fwd/bwd DISAGREED for non-causal windowed N<S (forward 0-anchored, backward oracle S-N-anchored). VERIFIED real (fwd vs bwd 3.19). First attempted forward→S-N fix but it exposed a latent Apple-Metal pipeline-cache abort when JIT-compiling the small-N non-causal windowed variant late in the full suite (bare abort(), no MLX error, N=256 variant unaffected). Reverted forward (back to documented 0-anchor); fixed the inconsistency by making the BACKWARD oracle match the forward's anchor (`(causal and N<S)?S-N:0`). Decode users use causal=True → S-N (correct). Locks: TestB1WindowConsistency (fwd matches anchor + fwd/bwd agree, causal+non-causal). MARCO-QUEUE: (a) true position-based non-causal windows need the small-N-variant Metal abort root-caused first; (b) the small-N non-causal windowed late-dispatch Metal abort is a latent Apple-Metal resource limit worth a standalone investigation.
  - Class C (sparse block-mask geometry) — [CLEAN] fully covered; no geometry-drift consumer beyond the 3 pass-1 D4 sites (V2-V5 exclude masks at eligibility; parameterized sparse threads explicit block_tile; V34 NAX arch-independent).

## PASS 2 verdict: NOT zero-finding (Class A 6 entries + Class B inconsistency = real material). Pass 3 required.

## PASS 3 (fresh re-audit, 2026-06-14)
- [CLEAN/PASS] Regression + C++ eval_gpu deep sweep (agent ac921b2): all 3 pass-2 fixes verified PASS (dtype guard not over-broad, operand lists correct, uint8-skip right; kvcache q→pool casts correct direction + output restore; windowed backward q_off bit-identical to forward kernel formula at all causal/window/N<S combos; D2 softcap-in-window-backward intact). C++ eval_gpu: EVERY primitive cache-key complete + overflow-safe (int64 offsets) + is_equivalent complete (12 primitives + conv tabulated). TQ cache-key omitting tq_bits/packed_D CORRECT (runtime branches, one kernel serves all bit-widths post III-2). V6 fwd omitting scale CORRECT (fixed 1/sqrt(D)). ONE LOW [FIXED]: test_kvcache_k_new_paged_succeeds asserted finite-not-correct → strengthened to single-token SDPA oracle compare (F4 class).
- [FINDINGS] Numerical edge + backward correctness (agent a2a5b8f, 13 empirical probes):
  - F1 MEDIUM [FIXED+locked]: flash_attention_topk(mask=...) reference path -inf'd a fully-masked tile → softmax NaN. Zeroed dead rows (II-6 contract). Default mask=None path was already safe (-1e4).
  - F2 MEDIUM [FIXED+locked]: lcsa_nax.sparse_attention_dispatch SDPA+bias branch NaN'd empty rows while the NAX kernel branch zeroed them (branches DISAGREED). Both now zero (II-6 contract). tests/test_phase3_iii4_empty_row.py.
  - Shared root cause: -inf bias row → mx.fast.scaled_dot_product_attention NaNs; sparse Metal kernels emit zeros. Fix aligns the 2 remaining bias-expansion paths to the zeros contract.
  - [CLEAN, measured]: NaN-input PROPAGATES on every fwd path (Rule 8 clean, no silent clamp); fp16 std12 no Inf (max|out|≤54); odd dims fall back to SDPA; zero-length varlen segment clean. EVERY differentiable path correct at unit AND std-8: V34 bwd D64 causal+noncausal (fp16/bf16; std8-bf16 "blowup" PROVEN bf16-precision via fp32-GT triangulation), alibi/window/softcap+window (D2 holds)/softcap, sparse bwd ×3, GQA/MQA bit-exact + H_kv shape, rope/packed-QKV/packed-KV bit-exact, GNA forward-only grad raises loudly.

## PASS 3 verdict: 2 MEDIUM (F1/F2 empty-row, same root cause) + 1 LOW (test gap) — all FIXED. Backward + numerical surfaces CLEAN. Pass 4 required (pass 3 found material).

## PASS 4 (F1/F2-class §AA.5.x sweep, 2026-06-14)
Swept EVERY -inf bias-expansion → SDPA site for the empty-row class:
- [FIXED] _sparse_fallback_sdpa (no-ext fallback) — unsanitized -inf bias; added row-active zeroing (one more F1/F2 sibling, CI/no-GPU only).
- [CLEAN] causal masks (7 sites): never empty a row (diagonal key q always present).
- [CLEAN] windowed SDPA fallback (attention.py:709): diagonal key q always in-window → never empty.
- [CLEAN] topk_stream (215): k_count>=1 always scatters >=1 zero; also Approach-5 declined/dead code.
- [CLEAN] _sparse_fallback_sdpa_perhead (M5 main): already sanitized (II-6 _get_sanitized_bias).
- [CLEAN, probed] sparse BACKWARD with a fully-masked tile: dQ/dK/dV no NaN (II-6 sanitization + custom vjp handle it); forward zeros the masked tile.
Empty-row class CLOSED. Suite 1478 passed +2 skipped (x2).

## PASS 4 verdict: 1 sibling fixed; class fully closed. A fresh zero-finding full pass (pass 5) is needed to declare the fixed point.

## PASS 5 (convergence-confirmation, 2026-06-14)
- [CLEAN/PASS] Job-1 regression: ALL pass 1-4 fixes verified by RUNNING (empty-row zeroing doesn't zero live rows; 6 dtype guards no-false-raise; windowed causal bwd rel 0.0019; topk grid full-row correct; suite 1478). 
- [FIXED+locked] P5-1 CRITICAL (pre-existing, found via active grad-probing): mx.grad through flash_attention(return_lse=True) gave corrupt/NaN gradients (the raw mfa_forward_with_lse C++ Primitive's 2-output vjp — the exact path _make_mfa_custom exists to bypass). Confirmed wrong for causal AND non-causal (err 3.7→2682, NaN at large N/D). Fix: _make_mfa_custom_lse custom_function — real (O,L) forward, SDPA-vjp backward on dO (dL ignored, standard). Post-fix bit-exact to SDPA-vjp (0.0) all shapes/dtypes. tests/test_phase3_iii4_dispatch_guards.py::TestP5ReturnLseBackward (8 cells). Test gap closed (no test gradded a return_lse output before).
- [CLEAN, probed] beyond P5-1: NaN/empty-reduction (topk_stream dead, cider guarded); dtype (gqa_decode_cider loud-fails at compile, unreached); grid-spec (9 kernels re-confirmed); fwd/bwd (chunked_prefill/speculative/splitfuse/shared_prefix all compose flash_attention, no separate vjp); cache-key/id/silent-except clean.

## PASS 5 verdict: 1 CRITICAL (P5-1) fixed. Pass 6 required.

## PASS 6 (exhaustive gradient-probe sweep, 2026-06-14)
Empirically gradded ALL 17 feature×option combos (fp16+bf16, N=512 + N=4096/D=128) vs independent references; inventoried all 11 custom_function vjps.
- [CLEAN] every differentiable path matches its reference within fp16/bf16 floor; P5-1 return_lse confirmed FIXED (bit-exact 0.0); double-grad clean; GQA dK/dV H_kv-shaped; sage + GNA-native raise loudly (no silent wrong grad). All 11 vjps verified forward↔backward consistent (the P5-1 mismatch class does not recur).
- [FIXED] combo-1 LOW (loud, not silent): full [B,H,Nq,Nkv] fp32 attn_bias + fp16 q crashed at the SDPA fallback (mask dtype promotion) while the native per-KV kernel handled it. Cast bias→q.dtype in the fallback. Verified no-crash, bit-exact, grad clean.

## PASS 6 verdict: gradient-probe class EXHAUSTED clean; 1 LOW usability fixed. Pass 7 (final convergence confirmation) required.

## PASS 7 (TERMINATION-DECISION convergence audit, 2026-06-14)
Job 1 regression (RAN): P5-1 PASS (grad through flash_attention(return_lse=True,causal=True)[0].sum() bit-exact to SDPA-vjp rel 0.0000 fp16 AND bf16, L returned finite, fwd rel ~1e-3). combo-1 PASS (fp32 [B,H,Nq,Nkv] bias + fp16 q: no crash, fp32-bias==fp16-bias rel 0.0000, vs SDPA-GT rel 0.0000). Full suite x2: 1485 passed + 2 skipped, exit 0 both runs, NO intermittent flake.
Job 2 fresh sweep of least-touched surfaces:
- [FINDING — MEDIUM, real silent-failure] `mlx_mfa/svdquant/quantize.py:180` `_replace_layers`: `isinstance(child, dict)` matches BEFORE `isinstance(child, nn.Module)` (line 198), but **nn.Module IS a dict subclass** (`issubclass(nn.Linear, dict)==True`). A model with a DIRECT `nn.Linear` attribute (`self.fc1 = nn.Linear(...)` — the most common structure) is descended-into as a container; its `.items()` are weight/bias arrays (not Modules) → the Linear is NEVER replaced. `quantize_model` returns `stats={'layers':[],'overall_compression':1.0}` and the model runs UNQUANTIZED while reporting success (Rule 8 silent default-on-no-match). Repro: `class N(nn.Module): def __init__(s): super().__init__(); s.fc1=nn.Linear(512,1024)` → `quantize_model(N())` → 0 layers, fc1 still type Linear. Works for `nn.Sequential` (.layers list branch) and nested submodules (dict branch recurses to grandchildren) — which is exactly why all tests pass (every test uses nn.Sequential). Test gap: no test quantizes a direct-attribute model. Expert/opt-in API (svdquant), forward-only, not in default attention path → MEDIUM not CRITICAL, but actively misleads (silent no-op).
- [CLEAN, live] mlx_lm shim semantics: sinks→fallback, array-mask→fallback, GQA→native STEEL (rel 7e-4 vs SDPA, no fallback), unsupported-D→fallback, return contract = single array (matches original); quantized-cache dequant uses correct `mx.dequantize(w,scales,biases,group_size,bits,dtype=)` signature; window R1 fix intact.
- [CLEAN, live] external_cache offload→onload bit-exact across fp16/bf16/fp32 (zero-copy mx.array store, dtype+length preserved).
- [CLEAN, live] conv3d MPP + legacy: BOTH match mx.conv_general at norm_rmse 2e-4 (pad=1 production envelope) and explicit cross-corr at 4e-4 (pad=0); bf16 loud-fails outside MPP envelope (KD-7, correct). (Earlier 0.19 pad=0-vs-conv_general was a test-reference-convention artifact, not a kernel bug — confirmed via cross-corr.)
- [CLEAN, live] V34 backward (env-gated, D=64 causal, M5): dQ/dK/dV rel 7e-4 vs SDPA-vjp, no NaN.
- [CLEAN] __init__ public surface: all 101 __all__ resolve, all 33 lazy targets resolve, hooks install cleanly; pyproject/CMake/check_venv no version skew, no -ffast-math/-Ofast numerics flags; bare excepts (attention.py 889/1524, build tooling) are capability-probe/warmup graceful-degrade (Rule-8 safe); id()-caches all shape+dtype-keyed with strong-ref ABA guard.

## PASS 7 verdict: NOT zero-finding — 1 MEDIUM silent-failure (svdquant direct-attribute no-op). Audit loop continues: fix quantize.py:180 ordering (test `isinstance(child, nn.Module)` before the dict/list container branches, or exclude nn.Module from the dict branch), add a direct-attribute test, then pass 8.

## PASS 7 (final-surface sweep, 2026-06-14)
- [CLEAN/PASS] Job-1 regression: P5-1 bit-exact (0.0) fp16+bf16 at N=4096/D=128; combo-1 no-crash bit-exact; suite 1485 x2 (no flake this session).
- [FIXED+locked] F7-1 MEDIUM (pre-existing, Rule-8 silent no-op): svdquant `_replace_layers` (quantize.py:180) tested isinstance(child, dict) BEFORE isinstance(child, nn.Module) — but nn.Module IS a dict subclass, so a direct nn.Linear attribute (self.fc1=...) was descended into as a container and NEVER replaced; quantize_model reported success (compression 1.0, 0 layers) while running unquantized. All tests passed because they use nn.Sequential. Fix: nn.Module branch first. Verified: direct-attr model now quantizes 2 layers (fc1→SVDQuantLinear, fwd reflects 4-bit), Sequential regression preserved. tests/test_svdquant.py::test_quantize_direct_attribute_model.
- [CLEAN, probed] mlx_lm shim fallback completeness (sinks/array-mask/GQA/unsupported-D all correct); external_cache offload→onload bit-exact (fp16/bf16/fp32); conv3d MPP+legacy match conv_general (2e-4); V34 backward rel 7e-4; __init__ surface (101 __all__ + 33 lazy resolve); pyproject/CMake no numerics flags; bare excepts are capability-probe graceful-degrade (Rule-8 safe).

## PASS 7 verdict: 1 MEDIUM (F7-1 svdquant silent no-op) fixed. Pass 8 required.

## PASS 8 (mask constructors + svdquant forward, 2026-06-14)
- [CLEAN/PASS] F7-1 regression (direct-attr quantize 2 layers, compression 3.03); svdquant FORWARD numerics correct (rel-err monotonic in rank 0.067→0.060; manual reconstruction rel 0.001, correct sign/transpose/scale); suite 1486.
- [FIXED+locked] F8-1 MEDIUM (pre-existing, same class as R6/R11): make_axial_temporal_mask (masks.py:771) computed per-tile spatial range as `% pHW` of first/last token only — wrong when a tile spans >=pHW tokens or crosses a frame boundary (modulo wraps → inverted range → DROPPED active blocks). Fixed: true min/max of (token%pHW), over-approx to full [0,pHW-1] on wrap. Verified 0 dropped blocks across power-of-2 AND non-pow2 grids (3x3/5x5/6x6). tests/test_attention.py::TestAxialMasks::test_temporal_mask_nonpow2_grid (the only prior test used H=W=8/pHW=64 which divides the tiles, hiding it).
- [CLEAN, probed] ALL other ~18 mask constructors built + compared vs token-level reference (over-approximation correct, boundary off-by-ones, non-divisible sizes): CLEAN. mx.quantize/dequantize per-block roundtrip benign; _quant_runtime.py absent.

## PASS 8 verdict: 1 MEDIUM (F8-1) fixed; mask-constructor family now FULLY swept (R6+R11+F8-1 were its 3 bugs, all fixed; every other constructor verified). Pass 9 required.

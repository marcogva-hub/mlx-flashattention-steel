# Session Log — mlx-mfa-v2
Shared Claude Code + Codex CLI. Append-only.
Headers tagged `[CLAUDE]` / `[CODEX]`. Protocol version: 1.2.
Archive: devnotes/archive/

> Rotated 2026-06-12 (Rule 1c, Phase II close): entries 2026-04-06 →
> 2026-05-xx moved to `archive/2026-04_to_2026-05_pre-phase2-close.md`.
> Phase II summary: `docs/v50/campaign-2026-06/phase2/PHASE-II-CLOSE.md`.

## [2026-06-12 08:31] [CLAUDE] Phase II Sprint II-5: Deep Literature Autoresearch — int8 DECLINE FALSIFIED (2.00x gate pass)
STATUS: COMPLETE

### Plan
- Objective: exhaustive 2025-2026 technique survey (7+ families, web primary sources), filter vs Pattern #6, probe survivors, report
- Files to modify: csrc/mpp_int8_bench.mm (cider-form variant), docs/.../phase2/sprint-II-5-report.md
- Dependencies impacted: none (bench binding + docs only)

### Changes
- `csrc/mpp_int8_bench.mm:1-130,180-230` — cider-form (16,32,16 full-coop) int8 variant tried first; run_bench flops param; header records falsification [HIGH] [VERIFIED]
- `docs/v50/campaign-2026-06/phase2/sprint-II-5-report.md` — 28 techniques, 21 declined w/ reason, 3 probes executed, dispositions [HIGH] [VERIFIED]

### Dependency & regression check
- Callers verified: mpp_int8_microbench() binding unchanged signature; bench-only
- Test coverage: full suite 1380 passed post-rebuild

### Tech cost
- None on production paths (diagnostic binding only)

### Validation
- Ran: standalone probes (int8 GEMM bit-exact vs CPU; conv2d impulse test; cider bench_sdpa 102 configs) + .venv/bin/python -m pytest tests/ -q
- Validated: int8 264.9 TOPS vs fp16 132.6 TF = 2.00x in-repo (kill gate >=1.3 PASS); conv2d deterministic + convention identified; cider decode 1.0-1.24x (CONFIRMED-NARROW); suite 1380 passed

### Git
- `c480c51` (probe) + report commit follows; branch master

### Key findings
- II-2 false negative root cause: full-coop int8 requires M,N,K in {16,32}; II-2 probed only 64x64x128 [VERIFIED]
- NEW: MPP convolution2d primitive in 26.4 SDK — implemented, deterministic, centered xcorr NHWC/HWIO; multi-TG tiling unresolved (follow-up queued); may supersede fused-im2col XL candidate [VERIFIED]
- Marco decision items: int8 kernel sprint sequencing (revived), cider GQA-decode port (1.0-1.24x narrow window)

---
## [2026-06-12 10:05] [CLAUDE] Phase II Sprint II-6: numerics deep audit — CRITICAL V34 fused dKdV corruption found+fixed
STATUS: COMPLETE

### Plan
- Objective: numerics audit (accumulation, LSE, mixed precision, determinism, guards, quant)
- Files: csrc/mfa_v6_nax_primitive.cpp, mlx_mfa/attention.py, 2 new test files, docs

### Changes
- `csrc/mfa_v6_nax_primitive.cpp` — BK%32 guard in compile_v34_backward_pipeline (all 8 bwd Primitives); fused default BK 16→32 [HIGH] [VERIFIED]
- `mlx_mfa/attention.py` — _v34_backward_vjp auto→split; _get_sparse_row_active + _get_sanitized_bias (cached) + all-False-row zero fixup [HIGH] [VERIFIED]
- tests: test_phase2_ii6_v34_bwd_paired_mma.py (7), test_phase2_ii6_sparse_allfalse_rows.py (4, subprocess-isolated); II-0 fixtures 0.1→1.0 scale

### Dependency & regression check
- Callers: all V34 bwd Primitives flow through guarded helper; sparse fixup local to _sparse_fallback_sdpa_perhead
- Test coverage: +11 locks; suite 1391 passed x6 consecutive

### Validation
- Ran: empirical battery + magnitude sweeps + effective-P/L/S extraction probes + bench (median-30) + pytest x6
- Validated: dK/dV per-element max-err 22-130 → 0.004-0.008 at unit scale; promotion re-benched on split: 2.15x/2.61x/2.67x vs SDPA-vjp (II-0 headline preserved); sparse all-false rows now zeros (contract)

### Git
- `d76cb6e` + sparse-fix commit + report commit; branch master

### Key findings
- v2.39.1 fused BK=16 was numerically invalid (paired 16x32x16 MMA needs TK even); its 1.01-1.12x claim WITHDRAWN; II-0 gate missed it due to 0.1-scale fixtures [VERIFIED]
- M5 sparse fallback NaN-on-all-false-rows semantics regression vs kernel contract — fixed [VERIFIED]
- OPEN (II-7/II-8): Metal pool stale-value sensitivity in 3 kernels (repro in report) [VERIFIED flake, mechanism UNCERTAIN]

---
## [2026-06-12 11:20] [CLAUDE] Phase II Sprint II-7: profiling-driven hunt — LCSA 15.4x + conv3d hook coverage
STATUS: COMPLETE

### Plan
- Objective: profile 4 representative harnesses, optimize empirical hot spots, document floors
- Files: benchmarks/profile_ii7_harnesses.py (new), mlx_mfa/masks.py, mlx_mfa/_auto_hooks.py

### Changes
- `mlx_mfa/masks.py` make_lcsa_mask — numpy CPU path (GPU->CPU copies + Python pooling loop + CPU einsum) → on-GPU MLX (15.4x: 11.19→0.73ms; full LCSA loop 3.3x) [HIGH] [VERIFIED]
- `mlx_mfa/_auto_hooks.py` — mx.conv3d now patched (mlx.nn.Conv3d bypassed the conv_general-only hook; 0-engagement telemetry proof); delegates to _patched_conv_general [HIGH] [VERIFIED]
- `benchmarks/profile_ii7_harnesses.py` — 4 harnesses with build/eval split

### Dependency & regression check
- Callers: make_lcsa_mask public API unchanged (return type mx bool mask); hook install/uninstall symmetric
- Test coverage: 57 LCSA tests + suite 1391 passed x3

### Validation
- Ran: harness suite before/after + cProfile + decode ladder + pytest x3
- Validated: full-loop after-state monotonically non-worse on all 4 harnesses; LCSA mask 15.4x; conv3d routing parity max_err 0.008

### Git
- perf(lcsa)+fix(hooks) commit + report commit; branch master

### Key findings
- Decode floor quantified: TQ attend kernel 14x dense SDPA (kernel-bound; feeds II-5 decode ledger; _v_pool_fp16 50MB kept when tq_v=True) [VERIFIED]
- Non-causal D=64 bwd: V34-split opt-in 1.88x, unit-scale errs 4e-4..2e-3 — promotion decision data for Marco [VERIFIED]

---
## [2026-06-12 12:05] [CLAUDE] Phase II Sprint II-8 + PHASE-II-CLOSE: exhaustion met
STATUS: COMPLETE

### Plan
- Objective: exhaustion meta-sweep (fresh dispatch+numerics+profiling passes) -> PHASE-II-CLOSE.md
- Files: mlx_mfa/attention.py (carve-out fwd fix), PHASE-II-CLOSE.md

### Changes
- `mlx_mfa/attention.py` _make_mfa_custom — carve-out forward now Apple SDPA (bit-identical) + 1-elem sentinel L; VJP recomputes V34 (O,L) pair [HIGH] [VERIFIED]
- `docs/.../PHASE-II-CLOSE.md` — phase ledger + lessons + Marco decision queue

### Validation
- Ran: 7-cell interleaved dispatch sweep + numerics battery + suite x2
- Validated: zero inversions (0.998-1.056); fwd inversion 1.19x -> 1.00-1.03x; grad cells 2.06/2.57/2.58x (>= promotion floor); battery 10/10; 1391 passed x2

### Git
- inversion-fix commit + close commit; branch master; pushed

### Phase II closed. Exhaustion criterion met after one in-round finding (fixed, clean re-pass).

---
## [2026-06-12 13:30] [CLAUDE] Sprint II-2R Phase R.0+R.1: int8 contradiction RESOLVED — char-vs-int8_t artifact; R.2 gate GO (QK+PV variant)
STATUS: IN_PROGRESS

### Plan
- Objective: reconcile II-2/II-5 int8 contradiction; conditional XL build
- Files: csrc/mpp_int8_bench.mm (char->int8_t), sprint-II-2R-reconciliation.md

### Changes
- `csrc/mpp_int8_bench.mm` — all legacy variants char->int8_t; header rewritten to reconciled story [HIGH] [VERIFIED]
- `docs/.../sprint-II-2R-reconciliation.md` — verdict + compile matrix + R.1 gate evidence + R.2 plan

### Validation
- Ran: dtype x dims x form compile matrix (13 variants); corrected in-repo probe; V34-vs-SDPA + quant-cost bench; accuracy simulation; suite
- Validated: int8 IS implemented (II-2 probe used `char` != int8_t — type-spelling artifact); full-coop (16,32,16) = 2.00x, device-tensor 64x64x128 compiles but 0.995x (no int8 MMA mode); QK-only declined by accounting (1.01-1.13x net); QK+PV GO (1.13-1.37x ceiling, cos 0.99995 at unit scale); 1391 passed

### Git
- `289e6f6` + gate-evidence commit; branch master; pushed

### Resume command (R.2 build, next session)
- Read docs/v50/campaign-2026-06/phase2/sprint-II-2R-reconciliation.md R.2 plan
- Template: MFA_V34_DUMP_SOURCE=1 dump of v6_nax_forward (D=128) as the kernel skeleton; BaseNAXFrag::mma is already the (16,32,16) paired form — add int8_t/int32 variant
- Env: .venv/bin/python; rebuild via CMAKE_ARGS="-DPython_EXECUTABLE=$PWD/.venv/bin/python" pip install --no-build-isolation -e .

### Pitfalls for R.2
- ONLY the full-coop register form has the 2.00x (device-tensor int8 = fp16 speed) — do not use .load()/device-tensor matmuls for QK/PV
- `char` != `int8_t` in MSL templates (the II-2 artifact — use int8_t everywhere)
- coop-coop dims must be {16,32} with >= one 32; (16,16,16) rejected
- P-quant per-row for PV-int8: fold V zero-point via row-sum identity (P rows sum to 1 AFTER softmax — the zp term = zp_s weighted sum, needs the same P @ ones accounting as the simulation)

---
## [2026-06-12 15:10] [CLAUDE] Sprint II-2R Phase R.2: int8 attention kernel BUILT + DECLINED at kill gate (0.92x kernel / 0.80x net)
STATUS: COMPLETE

### Plan
- Objective: full Sage-NAX int8 attention build per R.1 GO; promote/decline on attention-level bench
- Files: benchmarks/probes/sage_int8_proto.mm (new), reconciliation report extended

### Changes
- Prototype kernel: D=128 online-softmax fwd, full-coop (16,32,16) int8 QK; 3 PV variants benched (int8/int8-chained/fp16) + 2 structural opts (packed loads, persistent dests) — all measured, best 12.22ms vs SDPA 11.25 [HIGH] [VERIFIED]

### Validation
- Ran: per-step correctness vs fp32 CPU reference (cos up to 0.999958); ablation decomposition (QK 1.69 / softmax 2.4 / PV 8.2 ms); kernel timing medians; full suite
- Validated: kill gate (>=1.10x net) MISSED by ~40% -> DECLINED with evidence; 1391 passed

### Git
- decline commit; branch master; pushed

### Key findings
- The 2.00x int8 MMA advantage is consumed by cooperative-tensor API lifecycle tax (dest cycles ~10us, elementwise staging) — dtype/shape/transpose-invariant [VERIFIED by ablation]
- Packed 4-byte loads REGRESSED (byte loads fine on M5); persistent coop dests REGRESSED (register pressure, v2.39.0 class) [VERIFIED]
- Marco-gated residual: V34-generator-integrated int8 projects 1.11-1.33x at N=8192 only — dedicated-sprint scale

---
## [2026-06-12 17:20] [CLAUDE] Sprint II-9: conv3d MPP convolution2d path PROMOTED (2.30-2.51x, no hand-fused kernel needed)
STATUS: COMPLETE

### Plan
- Objective: eliminate im2col materialization (II-4's 62% lever); MPP primitive first per Pattern #6
- Files: csrc/mfa_conv_nax.cpp (conv3d_mpp_source/dispatch + gated branch), report

### Changes
- `csrc/mfa_conv_nax.cpp` — MPP convolution2d path: kT-accumulated conv2d, float coop dest, occupancy-aware tiles, default-on within envelope (fp16, B1, k3^3, s1, d1, pad1, HW%8, C>=32&%16); opt-out MFA_DISABLE_CONV3D_MPP [HIGH] [VERIFIED]

### Validation
- Ran: 8-variant tiling sweep + ccv production-code confirmation (web) + prototype vs CPU ref + production parity grid + edge sweep + 3-session bench + suite x2
- Validated: tiling semantics resolved (tile desc + sliced dest + set_offsets=source-window + coop dest); fp16-floor parity; 2.30-2.51x headline cells (91-96% of the 2.6x ceiling); fallbacks diff 0.0; KD-7 gate intact; 1391 passed x2

### Git
- promotion commit + report; branch master; pushed

### Key findings
- C=16 WRONG through the primitive (0.17-0.31 err; C>=32 exact) — undocumented constraint, gated [VERIFIED]
- half coop dest = fp16 accumulation -> failed 1e-5 parity bars; float dest required [VERIFIED]
- MPP impl lists bf16 conv variants — potential KD-7 lift for the envelope (S probe, ledger) [VERIFIED declared, implementation UNCERTAIN]
- Fused-im2col XL ledger item RETIRED (superseded by primitive promotion)

---
## [2026-06-12 18:30] [CLAUDE] Sprint II-10: refined Approach-5 top-K BUILT + DECLINED (0.75-0.89x); Approach 5 closed permanently
STATUS: COMPLETE

- PASS-1 built matmul-grade (fp16 MPP mma + owner-lane register heaps + TGM staging): 6.94ms at audit shape — kill gate (8ms) passed; score-set parity vs CPU [VERIFIED]
- Full composed path 12.81ms vs Architecture B 11.34 (0.89x); N=2048: 4.04 vs 3.03 (0.75x) — DECLINED [VERIFIED]
- Blueprint under-modeled scatter floor (2.26ms vs 0.5-1 projected) + selection tax; ArchB's threshold-elementwise-bias is structurally tight
- mx.argpartition measured 35ms (sort-grade) — primitive route dead [VERIFIED]
- Artifact: benchmarks/probes/topk_approach5_pass1.mm; second negative closes Approach 5
- Ran: probes + component benches | Validated: above | Git: decline commit, pushed

---
## [2026-06-12 19:40] [CLAUDE] Sprint II-11: cider GQA-decode ported + benched; auto-dispatch DECLINED, expert API shipped
STATUS: COMPLETE

- Port: mlx_mfa/gqa_decode_cider.py (MIT attribution; 2-pass, runtime N/strides — no per-token recompile) [VERIFIED]
- Correctness: FP-floor vs SDPA across GQA 1..32/MQA/dtypes (6 locks) [VERIFIED]
- Post-port grid (3 sessions): consistent wins ONLY at factor>=8 S=32K (1.06-1.17x); rest ties/loses — window narrower than II-5 in-cider (launch overhead) [VERIFIED]
- Decision: no auto-dispatch (sliver vs maintenance, per sprint license); expert API tier 3; Marco flag for gated dispatch + the paged/TQ transplant follow-up
- Ran: grid bench + suite | Validated: 1397 passed | Git: commit + push

---
## [2026-06-12 21:30] [CLAUDE] II-8 addendum: gate#9 programmatic (3rd site FIXED), TK=1 closed (parity->decline), determinism classified, pool residual = fixed-point blocker
STATUS: COMPLETE (items 1,3,4 clean/closed; item 2 partially — fixed point NOT declared)

- Item 1: THIRD Pattern-#9 site (V34 forward MFA_V6_V34_BK unguarded) found+guarded x2 sites; tests/test_phase2_ii8_gate9_parity.py enumerates all paired-MMA sites; release-audit gate #9 programmatic [VERIFIED]
- Item 3: odd-TK tail in dense fused generator (BOTH paired loops — dP was a latent 2nd path); correct at noise floor + adversarial mags; fused-BK16 == split +-0.2% -> DECLINED; v2.39.1 win confirmed corrupt-math artifact [VERIFIED]
- Item 4: run-to-run determinism contract HOLDS; batch-invariance = feature -> Marco backlog [VERIFIED]
- Item 2: production vector closed (II-6); 52 directed repro rounds clean; stress canary (opt-in MFA_POOL_STRESS=1) makes residual reproducible ~1/6 IN-SUITE only; root-cause = dedicated session; FIXED POINT NOT DECLARED [VERIFIED flake, mechanism UNCERTAIN]
- Suite: 1399 passed + 1 skipped x3 | Git: commit + push

---
## [2026-06-12 22:50] [CLAUDE] Sprints II-12 + II-13: non-causal D=64 bwd PROMOTED (1.72-2.01x); hook-coverage audit COMPLETE (zero gaps, Pattern #8 structural)
STATUS: COMPLETE

- II-12: both carve-out predicates widened; three-axis at hardened bar (std-12 errors == established causal floor); fwd bit-SDPA 1.023 interleaved; GQA H_kv; bench 1.72/1.94/2.01x; contract tests + ii12_* perf-claim registry; +5 locks [VERIFIED]
- II-12 bonus: first-line carve-out now mirrors the default-scale gate (non-default scale no longer detours via STEEL fwd) [VERIFIED]
- II-13: 11 entry points enumerated+classified; zero remaining gaps on M5; M1-gated SDPA hook = flagged-unbenched ledger item; telemetry engagement tests + completeness registry + anti-silence check (5 tests) [VERIFIED]
- Suite: 1409 passed + 1 skipped x2 | Git: commits + push

---
## [2026-06-12 14:16] [CLAUDE] Phase II-14: pool-residual root cause + structural class fix + Phase II fixed-point declaration
STATUS: COMPLETE

### Plan
- Objective: make the "buffer-pool stale-value residual" reliably reproducible, find the exact mechanism, fix the CLASS structurally, drive stress canaries to zero (>=30 consecutive), declare the Phase II exhaustion fixed point.
- Files to modify: csrc/mfa/v6_nax/NAAttentionKernel.cpp, csrc/mfa_v6_nax_primitive.cpp, new tripwire test, PHASE-II-CLOSE.md, sprint report.
- Dependencies impacted: 4 sparse V34 backward generators (fused dKdV, dV, dK, dQ) + their 4 dispatch sites.

### Changes
- `csrc/mfa/v6_nax/NAAttentionKernel.cpp` — all 4 sparse backward generators: replaced data-dependent `if (!tile_active) continue;` (around live cooperative accumulators) with compacted active-tile list in threadgroup memory + uniform counted loop; dQ-sparse K/V rebased per active tile (was incremental advance) [HIGH] [VERIFIED]
- `csrc/mfa_v6_nax_primitive.cpp` — 4 loud host guards: active-list capacity ceil(qL/BQ) (resp. ceil(kL/BK) for dQ) > 1024 throws (Rule 8) [HIGH] [VERIFIED]
- `tests/test_v50_sprint_5e_ii14_pool_tripwire.py` — NEW permanent stress-gated tripwire (victim config, raw-partials bitwise, x5 amplification) [HIGH] [VERIFIED]
- `docs/v50/campaign-2026-06/phase2/sprint-II-14-report.md` — NEW sprint report (repro, mechanism, fix, ladder) [VERIFIED]
- `docs/v50/campaign-2026-06/phase2/PHASE-II-CLOSE.md` — II-2R/II-9..II-14 ledger rows, fixed-point declaration, refreshed Marco-gated decision queue, lessons #6-#7 [VERIFIED]

### Root cause (key finding)
- NOT buffer-pool reuse: divergent values are .clear() zeros (content-independent), single-lane fragments, random (head, simdgroup, K-tile) regions. Mechanism = data-dependent branch inside loop carrying cooperative-tensor accumulator state; fires even with all-true masks (~2/5 standalone); SUPPRESSED in full-suite context (47 clean stressed suite runs were misleading) [VERIFIED empirically; compiler-level attribution DEDUCED]

### Dependency & regression check
- Callers verified: 4 sparse dispatch sites in mfa_v6_nax_primitive.cpp (only callers of the 4 generators); attention.py routing unchanged.
- Test coverage: covered — 29 sparse-backward tests + new tripwire + canary; existing public sparse tests pass.

### Tech cost
- Common path (dense): zero by construction (untouched). Sparse: one single-thread <=1024-entry scan + 1 barrier per TG; loop trips = active count. Skip benefit re-measured intact (tridiag 7-15x faster than all-true).

### Validation
- Ran: fused determinism script x60; /tmp/ii14_split_determinism.py (8 configs x30); tripwire standalone x30; canary standalone x30; MFA_POOL_STRESS=1 pytest tests/ x3; default pytest x2; /tmp/ii14_perf_spot.py; /tmp/ii14_freshpass.py (meta-sweep)
- Validated: 0/60 + 0/29x8 nondet, bitwise sparse==dense at all-true, 30/30 + 30/30 canaries, 1411 passed x3 stressed (+x2 default pending at log time, confirmed before commit), fresh pass zero-finding (fwd 1.005-1.035, grads 2.52/1.89x, conv 2.31/1.73x)

### Git
- committed this entry's changes on `master` (sha in commit below); pushed

### Fixed point
- Phase II exhaustion fixed point DECLARED MET (all four II-8 addendum mandatories closed + zero-finding fresh pass). Remaining work = Marco-gated decision queue in PHASE-II-CLOSE.md.

---
## [2026-06-12 17:55] [CLAUDE] Phase III-1: KD-7 bf16 conv lift — PROMOTED default-on
STATUS: COMPLETE

### Plan
- Objective: II-2R-style probe of the bf16 MPP convolution2d variant, then promote (widen the II-9 gate) or decline.
- Files: csrc/mfa_conv_nax.cpp, mlx_mfa/_auto_hooks.py, new lock tests, perf-claims registry + doc.

### Changes
- `csrc/mfa_conv_nax.cpp` — conv3d_mpp_source dtype-parameterized; dtype in cache name; MPP gate fp16∨bf16; loud bf16-legacy-path guard [HIGH] [VERIFIED]
- `mlx_mfa/_auto_hooks.py` — bf16 admitted iff _conv3d_bf16_mpp_eligible (mirrors C++ gate + env opt-out); non-MPP bf16 falls back upstream bit-identically [HIGH] [VERIFIED]
- `tests/test_phase3_iii1_conv_bf16.py` — 6 locks [VERIFIED]
- `tests/test_release_notes_perf_claims.py` — conv claim kind (telemetry reachability); ii9_* fp16 §Z-row gap closed + iii1_* bf16 row [VERIFIED]
- `docs/PERF_CLAIMS.md` + `tests/test_perf_claims_doc_sync.py` — 2 rows; grammar ii\d+ → i{2,3}\d+ [VERIFIED]

### Key results
- Probe: bf16 variant IMPLEMENTED (rel ≤0.9%, 99.9-100% bitmatch vs mx.conv3d bf16) [VERIFIED]
- Bench (public path, 3 sessions): 2.43x / 2.66x / 2.62x / 1.40x vs pre-lift bf16 (Apple fallback) [VERIFIED]
- Fresh finding queued for III-4: C=16 correct in BOTH dtypes via isolated probe — contradicts II-9's production-path measurement; C>=32 gate left untouched [probe VERIFIED]

### Dependency & regression check
- Callers: conv3d_mpp_dispatch (1 site), hook wrapper (conv_general + conv3d delegate). fp16 path bitwise-unchanged (lock).
- Test coverage: covered (6 locks + 2 claim params + 69 existing conv tests).

### Validation
- Ran: probe script, 3x3-session bench pre/post, MFA_HOOK_TELEMETRY=on engagement checks, full suite
- Validated: suite 1417 passed + 2 skipped; engagement + fallback telemetry-verified; claims REACHABLE

### Git
- committed below; branch master

---
## [2026-06-12 19:10] [CLAUDE] Phase III-2: paged/TQ decode — §AA.5 FULL_INVERSION, PROMOTED + fused 2/4-bit corruption fix
STATUS: COMPLETE

### Plan
- Objective: close the II-7 decode floor (fused TQ attend = 14x dense); cider transplant only if §AA.5 confirms.
- Files: mlx_mfa/tq_decode.py (new), mlx_mfa/inference.py, csrc/mfa_steel_paged_varlen_tq_fwd.cpp, locks, claims.

### Changes
- `mlx_mfa/tq_decode.py` — NEW: K-dequant (2/3/4-bit) + V-gather elementwise kernels + tq_decode_attend; config-tuple kernel caches [HIGH] [VERIFIED]
- `mlx_mfa/inference.py` — step() N_q=1 default-routes to new path; opt-out MFA_DISABLE_TQ_DECODE_SDPA=1; N_q>1 stays fused [HIGH] [VERIFIED]
- `csrc/mfa_steel_paged_varlen_tq_fwd.cpp` — FIX: K+V dequant emitted 3-bit bit-planar extraction unconditionally; tq_bits=2/4 silently wrong since the kernel landed (0.147-0.150 unit-scale vs ground truth; ~49 at std 8); runtime bit-width branches added [HIGH] [VERIFIED]
- `tests/test_phase3_iii2_tq_decode.py` — 11 locks incl. ground-truth arbitration of BOTH paths [VERIFIED]
- claims registry + PERF_CLAIMS.md — iii2_tq_paged_decode_step_default [VERIFIED]

### Key results
- §AA.5 FULL_INVERSION: dequant+sdpa beats fused 7.6x BEFORE kernels; with kernels attend 13.8x/22.1x, step 5.99x/14.42x (S=4K/16K); gap to dense floor 22.8x → 1.66x [VERIFIED]
- tq_v=True semantics: new path reads always-maintained fp16 V pool — more accurate than packed-V (documented; V-quant-noise lock bar) [VERIFIED]

### Validation
- Ran: §AA.5 probe, component decomposition, bits×magnitude matrix vs Python ground truth, determinism x10, full validation script, full suite
- Validated: suite 1428 passed + 2 skipped; fused now ground-truth-exact at 2/3/4 bits; claim REACHABLE

### Git
- committed below; branch master

---
## [2026-06-12 19:50] [CLAUDE] Phase III-3: v2.51.0 tagged release (PyPI + GH)
STATUS: COMPLETE

- Change: 3-SoT bump 2.50.1→2.51.0 + CHANGELOG [2.51.0] (both Unreleased sections folded + Phase II late + Phase III items + 3 Reproduce snippets) + README header [VERIFIED]
- Audit: /mlx-mfa-release-audit GREEN (7 checks; 1 advisory = claim-id naming, substance satisfied); gate#9 programmatic pass; check_venv pass; stressed suite 1431 passed [VERIFIED]
- Ran: python -m build; twine check+upload; gh release create with artifacts | Validated: https://pypi.org/project/mlx-mfa/2.51.0/ + GH release tag v2.51.0 live
- Git: abdaa8d (release commit), tag v2.51.0, pushed
---
## [2026-06-12 18:08] [CLAUDE] Fix: III-4 doc-vs-code audit corrections (14-item batch, docs-only)
STATUS: COMPLETE

### Plan
- Objective: apply verified doc-vs-code audit fixes post-v2.51.0 (V34 D=64 default-on, MPP conv3d default, TQ decode SDPA route, NAX sparse dispatch, KD-7 lifted)
- Files to modify: README.md, ENV_VARS.md, CLAUDE.md, docs/{HOOK_TELEMETRY,TRAINING_QUICKSTART,INVENTORY,INDEX,PERF_CLAIMS}.md, docs/v50/known-debt-v2.50.md
- Dependencies impacted: none (docs only; no .py/.cpp touched per task constraint)

### Changes
- `README.md:16` — v2.50 highlights retitled "(shipped 2026-05)" [HIGH] [VERIFIED]
- `README.md:40-46` — v2.39.1 section retitled "(historical)" + default-on note; L108 stale "env unset preserves v2.36.1" reworded [HIGH] [VERIFIED]
- `README.md:116` — conv snippet ~1.6× → 2.3-2.5× fp16 / 1.4-2.7× bf16 via MPP [HIGH] [VERIFIED per task spec]
- `README.md:423-428` — Conv3D NAX section: MPP-default lead-in; legacy figures flagged non-default (MFA_DISABLE_CONV3D_MPP=1) [HIGH] [VERIFIED]
- `README.md:507-519` — sparse M5+ section rewritten: NAX dispatcher default since v2.36.1; bias-expansion kept as fallback note [HIGH] [VERIFIED]
- `ENV_VARS.md:5,63-64,77-78,98,107-108,118-119` — V34 enable/disable rows corrected (D=64 default-on v2.51.0); new rows MFA_DISABLE_CONV3D_MPP / MFA_DISABLE_TQ_DECODE_SDPA / MFA_DISABLE_AUTO_HOOKS / MFA_V34BWDF_DUMP_SOURCE+PATH; §104 "all gated behind" fixed; cross-ref to docs/v6-nax/env-vars.md [HIGH] [VERIFIED]
- `docs/HOOK_TELEMETRY.md:4,105-107,114` — conv_general+conv3d hooked; fallback-reason strings updated (KD-7 III-1); KD-7 lifted [HIGH] [VERIFIED per task spec]
- `docs/TRAINING_QUICKSTART.md:8-18,27,45-56,90-92,95-99,134-176,179-180,190` — Status → DEFAULT-ON D=64 (1.7-2.7×), opt-out documented; SHIP_OPT_IN/"2.2-2.4× slower"/carve-out opt-in language fixed throughout; causal "not supported" bullet lifted [HIGH] [VERIFIED]
- `docs/INVENTORY.md:3` — header → 2.51.0, tables flagged 2026-05-13 snapshot [HIGH] [VERIFIED]
- `docs/INDEX.md:3-5` — top note: repo-root-relative links, campaign-2026-06 not yet indexed [HIGH] [VERIFIED]
- `docs/PERF_CLAIMS.md:17` — header v2.50.1 → v2.51.0 (only change in file) [HIGH] [VERIFIED]
- `CLAUDE.md:283` — Current status → v2.51.0, 1429 tests; table flagged historical [HIGH] [VERIFIED]
- `docs/v50/known-debt-v2.50.md:244-249,303` — KD-7 marked LIFTED v2.51.0 (Sprint III-1), original text kept as history [HIGH] [VERIFIED]

### Dependency & regression check
- Callers verified: n/a (docs); tests/test_release_notes_perf_claims.py intentionally untouched per task spec
- Test coverage: tests/test_perf_claims_doc_sync.py covers PERF_CLAIMS.md sync; no automated coverage for README/ENV_VARS prose (gap flagged, pre-existing)

### Tech cost
- None (documentation only)

### Validation
- Ran: `.venv/bin/pytest tests/test_perf_claims_doc_sync.py -q`
- Validated: 4 passed in 0.07s

### Git
- WIP — uncommitted; branch master.  NOTE: pre-existing uncommitted changes in csrc/mfa_steel_paged_varlen_tq_fwd.cpp, mlx_mfa/{__init__,attention,integrations/mlx_lm}.py + new iii4-findings-ledger.md were already in the tree before this session; not touched here.
---
## [2026-06-13 01:53] [CLAUDE] Phase III-4: complete interrupted fix batch + root-cause order-dependent failure
STATUS: COMPLETE

### Plan
- Objective: finish the interrupted III-4 audit-fix batch (R6/R11/DOC-11 verification), root-cause the order-dependent test_mixed_dtype_routes_mfa failure, update ledger, green suite x2.
- Files to modify: mlx_mfa/attention.py, tests/test_attention.py, docs/v50/campaign-2026-06/phase3/iii4-findings-ledger.md
- Dependencies impacted: flash_attention dispatch (all backends), flash_attention_kvcache paged-append.

### Changes
- `mlx_mfa/attention.py:~499` — PASS1-REGRESSION FIX: cast K/V to q.dtype BEFORE dispatch (eval_gpu keys kernel dtype on q alone, csrc/mfa_attention.cpp:111-114; f32 kernel reinterpreted f16 K/V buffers → silent garbage max_err ~15, NaN when buffer pool dirty) [HIGH] [VERIFIED — poisoned-pool repro: max_err 15.6 → 7.5e-4]
- `mlx_mfa/attention.py:~2073` — paged-append: cast k_new/v_new to pool dtype before _mfa_scatter_kv_cpp (raw byte scatter of f32 into f16 pool wrote reinterpreted halves incl. NaN/inf patterns) [HIGH] [VERIFIED — single-key append now bit-exact vs V]
- `tests/test_attention.py:test_mixed_dtype_routes_mfa` — strengthened with cast-SDPA ground-truth assert (finiteness alone passed garbage) [HIGH] [VERIFIED]
- `docs/v50/.../iii4-findings-ledger.md` — D5,D11,D12,D14,D15,D17,D18,R2-R8,R11,R12,DOC-11 → [FIXED]; PASS1-REGRESSION entry added [VERIFIED — all markers + tests inspected]
- NOTE: R6/R11 masks.py fixes + regression tests and DOC-11 repointing were ALREADY present (previous agent got further than its snapshot indicated); verified complete, no re-edit needed.

### Dependency & regression check
- Callers verified: all flash_attention backends see uniform dtypes post-cast (mixed-dtype MFA routing decision preserved); paged-append scatter+fallback both covered.
- Test coverage: gap flagged — direct flash_attention_paged / varlen entries with mixed-dtype user inputs not audited (same class); deferred to pass 2 (ledger note).

### Tech cost
- Lazy astype on mixed-dtype inputs only — no cost on the common uniform-dtype path.

### Validation
- Ran: `.venv/bin/pytest tests/ -q` twice
- Validated: run1 1435 passed/2 skipped/0 failed; run2 1435 passed/2 skipped/0 failed. Repro scripts confirmed numerical correctness (7.5e-4 vs SDPA; bit-exact paged append).

### Git
- WIP — uncommitted; branch master

---
## [2026-06-14 04:40] [CLAUDE] Phase III-4 pass 1: fresh-eyes whole-repo audit — 5 agents, 66 findings, all dispositioned
STATUS: COMPLETE

### Plan
- Objective: fresh-eyes whole-repo audit (re-derive from code, not prior reports), fix/promote/decline each with evidence, repeat until zero-finding.
- 5 parallel agents: Python dispatch core (18), Python runtime (15), C++ kernels (1), tests/benchmarks (17), docs (15).

### Findings dispositioned (66 total; highlights)
- D-TOPK CRITICAL [VERIFIED+locked]: topk bisect threshold kernel grid mis-specified (grid.x=N threads → only N/256 threadgroups → only first 8 query rows/head written, rest stale pool). Promoted AUTO-default kernel was selecting top-K for ~8/N rows. grid.x=N*256 fix + per-row range assert. Exposed by the F-batch adversarial tests (the post-restart "flaky test").
- D1 CRITICAL: backend="sdpa" early return dropped softcap/alibi/window/return_lse — gated to plain case.
- R1 CRITICAL [VERIFIED repro]: patch_mlx_lm windowed decode attended only to key 0 (causal=False + window, qL_off only applies causal) — force causal in windowed decode.
- mixed-dtype silent corruption [VERIFIED, resume agent]: kernel dtype from q alone reinterpreted f16 K/V as f32 — cast K/V to q.dtype.
- D4 HIGH: M3/M4 D=128 sparse mask geometry (Python BK=16 vs C++ BK=32) — base config when has_block_mask (fwd+bwdDQ+bwdDKV).
- D7 MEDIUM [VERIFIED real, fwd 0.67/grads 1.1 at N=100]: bias-expansion re-tiled non-divisible-N masks — _expansion_tile + 6 sites.
- D16 MEDIUM [VERIFIED real, dV RMSE 0.506]: native sparse backward OR-downsample contaminated grads — bt>=64 gate.
- D2/D3/D5/D6/D8/D9/D10/D11/D12/D13/D14/D15/D17/D18 + R2-R8/R12/R15: feature-combo raises, softcap+window grads, paged batched causal, force_kernel, carve-out guards, RoPE table verify, hybrid cache demotion/multi-seq, silent-catch cleanups.
- TQ pool int64 offsets (CXX-1); 13 doc-staleness fixes; F1-F17 test retrofit (bf16 V34 GT, LCSA unit-scale, V-TQ GT, GNA fixture) — NO new kernel corruption found at unit scale.
- F8 + R13/R14 DEFERRED→Marco queue (hardware-coverage / perf-debt).

### Validation
- Ran: full suite x several, MFA_POOL_STRESS=1 x2, per-fix targeted probes + new lock files (iii4 d7/d16/dispatch_guards), rebuilt C++
- Validated: default 1470 passed + 2 skipped (x3), stressed 1472 (x2). One non-reproducing cold-run failure not recurring in 8+ subsequent runs incl. stressed (tracked).

### Git
- pass-1 commit below; branch master. Pass 2 (fresh re-audit) pending.

---
## [2026-06-14 05:30] [CLAUDE] Phase III-4 pass 2: fresh-eyes correctness re-audit (no file edits)
STATUS: COMPLETE

### Plan
- Objective: independent re-derivation re-audit. Job 1 = regression-check pass-1's own fixes; Job 2 = sweep svdquant/quantize/dispatch_policy/runtime + fwd-bwd consistency.
- Files to modify: NONE (audit-only per task).

### Findings (no fixes applied; report-only)
- Job 1: ALL pass-1 fixes re-verified PASS — return_lse/attn_bias/alibi/softcap raises (not over-broad; common paths + new guards live-tested), _expansion_tile D7 precedence (5 mask cases live-tested correct), force_kernel D8 (backend=mfa max_err 0.001 vs SDPA = real kernel, lru_cache key includes force_kernel), _rope_tables_match_base10000 D13 (no false-neg for legit base-10000 1D tables; 3D tables correctly route to STEEL), mixed-dtype cast (no-op same-dtype, live-confirmed), kv_cache R2/R3/R4 (tombstone set+checked+cleared symmetrically).
- Job 2 CLEAN: svdquant linear/quantize (SVD math consistent fwd↔calib; idempotence guard correct; LOW: rank>min(M,K) leaves self.rank inconsistent w/ array shape, report-only), quantize.py per-block/smooth_k correct, dispatch_policy no v2.37.0-class short-circuit (V34 D=64 carve-out reachable via public auto path; conv MPP + TQ decode are separate hook/runtime paths default-on), runtime.py backend resolution sound.
- Fwd/bwd (D2 class): alibi/sage-sparse/windowed+softcap backward oracles all differentiate the SAME feature-applied function — consistent.
- LOW (pre-existing, documented, LOUD-fail not silent): flash_attention_gna native path (D=128/3D/f16) has no MFAGNAForward::vjp — mx.grad raises rather than falling back to sparse-path gradient. Documented forward-only; not a pass-1 regression.

### Validation
- Ran: live import + path probes (.venv python): 7 common paths OK+finite, 3 new guards raise correctly, D8 mfa==sdpa max_err 0.001, _expansion_tile 5 cases match expected.
- Validated: zero defensible CRITICAL/HIGH findings; pass-1 batch confirmed correct.

### Git
- not applicable (audit-only, no edits); branch master

---
## [2026-06-14 06:30] [CLAUDE] Phase III-4 pass 2: fresh re-audit — grid-spec clean, regression clean, dtype+window classes found+fixed
STATUS: COMPLETE

### Findings (3 agents)
- Grid-spec class sweep: CLEAN — topk grid-undercount was isolated; all 9 production mx.fast.metal_kernel dispatches correct (indexing↔grid↔coverage).
- Regression + general re-audit: all 10 pass-1 fixes live-verified PASS (not over-broad); svdquant/quantize/dispatch_policy/runtime/fwd-bwd-consistency CLEAN. One LOW [FIXED-doc]: flash_attention_gna native forward-only grad note.
- dtype/§AA.5.x multi-gate: Class A (6 dtype-reinterpret entries) [FIXED — loud guard + kvcache cast]; Class B (window fwd/bwd anchor inconsistency) [FIXED via backward-matches-forward 0-anchor; forward→S-N attempt reverted after exposing a latent Apple-Metal small-N-windowed late-dispatch abort]; Class C (sparse geometry) CLEAN.

### Changes
- `mlx_mfa/attention.py` — `_assert_kv_dtype_matches_q` guard at 6 expert entries (paged/varlen/paged-varlen/TQ/sage/gna); q→pool cast at 2 kvcache paged sites; windowed backward oracle anchor matched to forward (`(causal and N<S)?S-N:0`); flash_attention_gna autograd docstring note [VERIFIED]
- `csrc/mfa_attention.cpp` — qL_off window-inclusive change applied then REVERTED (caused Metal abort); net no C++ change vs pass-1
- `tests/test_phase3_iii4_dispatch_guards.py` — TestB1WindowConsistency (fwd anchor + fwd/bwd agree, ±causal); `tests/test_attention.py` window refs unchanged from pass-1 (causal-conditional)

### Validation
- Ran: per-finding probes (B1 fwd-vs-bwd 3.19 confirmed; A1 paged mismatch repro; D8/D13 holds), full suite x5
- Validated: 1474 passed + 2 skipped (4/5 runs; 1 intermittent pre-existing verbose-capture flake, passes in isolation). No Metal abort with N=256 B1 variant.

### Git
- pass-2 commit below; branch master. Pass 3 required (pass 2 found real material).

---
## [2026-06-14 07:30] [CLAUDE] Phase III-4 pass 3: correctness audit — numerical edge cases + backward at adversarial scale
STATUS: COMPLETE

### Plan
- Objective: CORRECTNESS audit (Job 1 numerical edge cases; Job 2 every differentiable path vs SDPA-vjp at std 1 AND std 8). Areas passes 1-2 under-covered.
- Files to modify: NONE (audit-only per task). Probes in /tmp only.

### Findings (report-only, repro-confirmed)
- F1 [VERIFIED repro] MEDIUM: flash_attention_topk(mask=...) ref path (attention.py:3741) — a fully-masked query tile → mx.where(mask,scores,-inf) → softmax(all -inf) → all-NaN rows. Repro: q,k,v (1,1,64,64) f16, mask (2,2) bool tile0=False, topk_ratio=0.5 → rows 0..31 all NaN (32/32). Default path (mask=None, NAX bias, -1e4) is SAFE (never empties a row). Inconsistent with II-6 sparse all-False-row→zeros contract. This is the advertised FlashVSR LCSA use case (locality mask + top-K).
- F2 [VERIFIED repro] MEDIUM (conf HIGH): sparse_attention_dispatch (lcsa_nax.py:266 _bool_mask_to_float_bias) — fully-masked tile produces NaN on the SDPA+bias branch but ZEROS on the NAX-kernel branch. Same input, density-dependent result, one NaN. Repro: (1,4,1024,128) f16 BT=16, bm[:,:,3,:]=False → density=2.0 SDPA+bias rows all NaN; density=0.0 NAX rows all zero. Default threshold 1.01 routes NAX (safe); SDPA+bias reached when caller passes density_threshold=0.02 (M1/M3) or precomputed_bias.
- ROOT CAUSE shared: NaN inherited from mx.fast.scaled_dot_product_attention itself on an all--inf row (verified: raw SDPA NaNs). mlx-mfa's -inf bias-expansion paths feed it; dedicated sparse kernels emit zeros. Suggested fix: clamp empty rows post-SDPA (mx.where(rowsum(mask)==-inf-equiv, 0, out)) OR use a large-finite-negative bias (-1e4 / -3e4) like the topk default path, to match the II-6 zeros contract across all bias-expansion paths.

### Verified CLEAN (with numbers)
- Job 1 edge cases CLEAN: NaN-input propagation (Rule 8) — flash_attention/causal/sparse/topk-NAX/sage/GNA-native all PROPAGATE injected NaN (none silently zero/clamp). fp16 overflow std=12 — no Inf on flash_attention/topk/sparse/sage/GNA (max |out| ≤ 54, max-subtract holds). Odd head dims (48/80/96/100/130) fall back cleanly to SDPA, finite. Zero-length varlen segment [32,0,48] (correct [B,H,total,D] layout) — OK, finite, both STEEL and f32 split-concat (first probe used wrong [total,H,D] layout — false alarm). Windowed empty cases (causal (8,0),(0,0); non-causal (0,0)) finite. GNA self-window (1,1,1) finite (structurally never empty).
- Job 2 backward CLEAN vs independent SDPA-vjp / manual ref at std 1 AND std 8:
  - V34 native backward (MFA_ENABLE_V34_BACKWARD=1, D=64 qL=2048, kernel CONFIRMED engaging — fwd matches): unit fp16 dQ 0.0016/bf16 0.0056; std8 fp16 ≤0.048. std8 bf16 "HIGH" (dQ 0.32) PROVEN to be bf16 precision not kernel error — V34-vs-fp32GT (0.225) is CLOSER than bf16-SDPA-vs-fp32GT (0.371).
  - alibi (≤0.013), plain window(64,0) (≤0.078 std8), softcap+window D2 class (≤0.0046), softcap-only (0.0000) — all match manual ref; D2 fix holds (same feature-applied fn fwd↔bwd).
  - sparse backward all 3: sdpa/sdpa_sparse/steel_sparse ≤0.046 std8.
  - GQA/MQA (Hkv=1,2): bit-exact, dK/dV shape correctly = H_kv.
  - rope (base-10000 interleaved): bit-exact 0.0000. packed QKV [B,H,N,3,D] + KV [B,H,N,2,D]: bit-exact 0.0000.
- GNA-native autograd: forward-only (no vjp) → mx.grad raises loudly (pass-2 confirmed; MFA_DISABLE_GNA_NATIVE=1 routes differentiable sparse). Not a silent-grad bug.

### Validation
- Ran: 9 /tmp probe scripts (.venv python, M5 Max): job1_empty/edge/varlen0b/gna_sage_ovf/lcsa2/nan_input; job2_ref/v34native/v34_bf16check/features/sparse_bwd/rope_packed/packed2; sdpa_baseline.
- Validated: F1 (32/32 NaN rows), F2 (NaN vs zero by density), shared root cause (raw SDPA NaN), all CLEAN paths with measured rel-errors above.

### Git
- not applicable (audit-only, no repo edits); branch master. Pass 3 found 2 real MEDIUM (empty-row contract gap). Recommend a pass-3 FIX cycle for F1/F2 then re-audit.

---
## [2026-06-14 07:30] [CLAUDE] Phase III-4 pass 3: empty-row contract fix + backward/numerical clean
STATUS: COMPLETE

- F1/F2 [FIXED+locked]: flash_attention_topk(mask) + lcsa_nax.sparse_attention_dispatch SDPA+bias branch NaN'd fully-masked query rows (the NAX branch zeroed them — inconsistent). Both aligned to the II-6 empty-row→zeros contract. tests/test_phase3_iii4_empty_row.py (4 locks). [VERIFIED]
- C++ eval_gpu deep sweep (12 primitives + conv): cache-key complete + overflow-safe + is_equivalent complete; CLEAN.
- All 3 pass-2 fixes regression-verified PASS; test_kvcache_k_new_paged_succeeds strengthened to SDPA oracle (F4).
- Numerical/backward sweep (13 empirical probes): NaN-propagation Rule-8 clean; fp16 overflow safe; EVERY differentiable path correct at unit+std8 (V34/alibi/window/softcap+window/sparse×3/GQA/rope/packed); std8-bf16 V34 "blowup" proven bf16-precision not kernel bug.
- Ran: per-finding probes + new locks + full suite x3 | Validated: 1478 passed + 2 skipped (x3 stable)
- Git: pass-3 commit below; pass 4 required (pass 3 found material).

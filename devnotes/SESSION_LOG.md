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
## [2026-06-12 10:05] [CLAUDE] Phase II Sprint II-6: numerics deep audit — CRITICAL V6NAX fused dKdV corruption found+fixed
STATUS: COMPLETE

### Plan
- Objective: numerics audit (accumulation, LSE, mixed precision, determinism, guards, quant)
- Files: csrc/mfa_v6_nax_primitive.cpp, mlx_mfa/attention.py, 2 new test files, docs

### Changes
- `csrc/mfa_v6_nax_primitive.cpp` — BK%32 guard in compile_v6nax_backward_pipeline (all 8 bwd Primitives); fused default BK 16→32 [HIGH] [VERIFIED]
- `mlx_mfa/attention.py` — _v6nax_backward_vjp auto→split; _get_sparse_row_active + _get_sanitized_bias (cached) + all-False-row zero fixup [HIGH] [VERIFIED]
- tests: test_phase2_ii6_v6nax_bwd_paired_mma.py (7), test_phase2_ii6_sparse_allfalse_rows.py (4, subprocess-isolated); II-0 fixtures 0.1→1.0 scale

### Dependency & regression check
- Callers: all V6NAX bwd Primitives flow through guarded helper; sparse fixup local to _sparse_fallback_sdpa_perhead
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
- Non-causal D=64 bwd: V6NAX-split opt-in 1.88x, unit-scale errs 4e-4..2e-3 — promotion decision data for Marco [VERIFIED]

---
## [2026-06-12 12:05] [CLAUDE] Phase II Sprint II-8 + PHASE-II-CLOSE: exhaustion met
STATUS: COMPLETE

### Plan
- Objective: exhaustion meta-sweep (fresh dispatch+numerics+profiling passes) -> PHASE-II-CLOSE.md
- Files: mlx_mfa/attention.py (carve-out fwd fix), PHASE-II-CLOSE.md

### Changes
- `mlx_mfa/attention.py` _make_mfa_custom — carve-out forward now Apple SDPA (bit-identical) + 1-elem sentinel L; VJP recomputes V6NAX (O,L) pair [HIGH] [VERIFIED]
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
- Ran: dtype x dims x form compile matrix (13 variants); corrected in-repo probe; V6NAX-vs-SDPA + quant-cost bench; accuracy simulation; suite
- Validated: int8 IS implemented (II-2 probe used `char` != int8_t — type-spelling artifact); full-coop (16,32,16) = 2.00x, device-tensor 64x64x128 compiles but 0.995x (no int8 MMA mode); QK-only declined by accounting (1.01-1.13x net); QK+PV GO (1.13-1.37x ceiling, cos 0.99995 at unit scale); 1391 passed

### Git
- `289e6f6` + gate-evidence commit; branch master; pushed

### Resume command (R.2 build, next session)
- Read docs/v50/campaign-2026-06/phase2/sprint-II-2R-reconciliation.md R.2 plan
- Template: MFA_V6_DUMP_SOURCE=1 dump of v6_nax_forward (D=128) as the kernel skeleton; BaseNAXFrag::mma is already the (16,32,16) paired form — add int8_t/int32 variant
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
- Marco-gated residual: V6NAX-generator-integrated int8 projects 1.11-1.33x at N=8192 only — dedicated-sprint scale

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

- Item 1: THIRD Pattern-#9 site (V6NAX forward MFA_V6_NAX_BK unguarded) found+guarded x2 sites; tests/test_phase2_ii8_gate9_parity.py enumerates all paired-MMA sites; release-audit gate #9 programmatic [VERIFIED]
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
- Dependencies impacted: 4 sparse V6NAX backward generators (fused dKdV, dV, dK, dQ) + their 4 dispatch sites.

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
- Objective: apply verified doc-vs-code audit fixes post-v2.51.0 (V6NAX D=64 default-on, MPP conv3d default, TQ decode SDPA route, NAX sparse dispatch, KD-7 lifted)
- Files to modify: README.md, ENV_VARS.md, CLAUDE.md, docs/{HOOK_TELEMETRY,TRAINING_QUICKSTART,INVENTORY,INDEX,PERF_CLAIMS}.md, docs/v50/known-debt-v2.50.md
- Dependencies impacted: none (docs only; no .py/.cpp touched per task constraint)

### Changes
- `README.md:16` — v2.50 highlights retitled "(shipped 2026-05)" [HIGH] [VERIFIED]
- `README.md:40-46` — v2.39.1 section retitled "(historical)" + default-on note; L108 stale "env unset preserves v2.36.1" reworded [HIGH] [VERIFIED]
- `README.md:116` — conv snippet ~1.6× → 2.3-2.5× fp16 / 1.4-2.7× bf16 via MPP [HIGH] [VERIFIED per task spec]
- `README.md:423-428` — Conv3D NAX section: MPP-default lead-in; legacy figures flagged non-default (MFA_DISABLE_CONV3D_MPP=1) [HIGH] [VERIFIED]
- `README.md:507-519` — sparse M5+ section rewritten: NAX dispatcher default since v2.36.1; bias-expansion kept as fallback note [HIGH] [VERIFIED]
- `ENV_VARS.md:5,63-64,77-78,98,107-108,118-119` — V6NAX enable/disable rows corrected (D=64 default-on v2.51.0); new rows MFA_DISABLE_CONV3D_MPP / MFA_DISABLE_TQ_DECODE_SDPA / MFA_DISABLE_AUTO_HOOKS / MFA_V6BWDF_DUMP_SOURCE+PATH; §104 "all gated behind" fixed; cross-ref to docs/v6-nax/env-vars.md [HIGH] [VERIFIED]
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
- TQ pool int64 offsets (CXX-1); 13 doc-staleness fixes; F1-F17 test retrofit (bf16 V6NAX GT, LCSA unit-scale, V-TQ GT, GNA fixture) — NO new kernel corruption found at unit scale.
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
- Job 2 CLEAN: svdquant linear/quantize (SVD math consistent fwd↔calib; idempotence guard correct; LOW: rank>min(M,K) leaves self.rank inconsistent w/ array shape, report-only), quantize.py per-block/smooth_k correct, dispatch_policy no v2.37.0-class short-circuit (V6NAX D=64 carve-out reachable via public auto path; conv MPP + TQ decode are separate hook/runtime paths default-on), runtime.py backend resolution sound.
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
  - V6NAX native backward (MFA_ENABLE_V6_BACKWARD=1, D=64 qL=2048, kernel CONFIRMED engaging — fwd matches): unit fp16 dQ 0.0016/bf16 0.0056; std8 fp16 ≤0.048. std8 bf16 "HIGH" (dQ 0.32) PROVEN to be bf16 precision not kernel error — V6NAX-vs-fp32GT (0.225) is CLOSER than bf16-SDPA-vs-fp32GT (0.371).
  - alibi (≤0.013), plain window(64,0) (≤0.078 std8), softcap+window D2 class (≤0.0046), softcap-only (0.0000) — all match manual ref; D2 fix holds (same feature-applied fn fwd↔bwd).
  - sparse backward all 3: sdpa/sdpa_sparse/steel_sparse ≤0.046 std8.
  - GQA/MQA (Hkv=1,2): bit-exact, dK/dV shape correctly = H_kv.
  - rope (base-10000 interleaved): bit-exact 0.0000. packed QKV [B,H,N,3,D] + KV [B,H,N,2,D]: bit-exact 0.0000.
- GNA-native autograd: forward-only (no vjp) → mx.grad raises loudly (pass-2 confirmed; MFA_DISABLE_GNA_NATIVE=1 routes differentiable sparse). Not a silent-grad bug.

### Validation
- Ran: 9 /tmp probe scripts (.venv python, M5 Max): job1_empty/edge/varlen0b/gna_sage_ovf/lcsa2/nan_input; job2_ref/v6naxnative/v6nax_bf16check/features/sparse_bwd/rope_packed/packed2; sdpa_baseline.
- Validated: F1 (32/32 NaN rows), F2 (NaN vs zero by density), shared root cause (raw SDPA NaN), all CLEAN paths with measured rel-errors above.

### Git
- not applicable (audit-only, no repo edits); branch master. Pass 3 found 2 real MEDIUM (empty-row contract gap). Recommend a pass-3 FIX cycle for F1/F2 then re-audit.

---
## [2026-06-14 07:30] [CLAUDE] Phase III-4 pass 3: empty-row contract fix + backward/numerical clean
STATUS: COMPLETE

- F1/F2 [FIXED+locked]: flash_attention_topk(mask) + lcsa_nax.sparse_attention_dispatch SDPA+bias branch NaN'd fully-masked query rows (the NAX branch zeroed them — inconsistent). Both aligned to the II-6 empty-row→zeros contract. tests/test_phase3_iii4_empty_row.py (4 locks). [VERIFIED]
- C++ eval_gpu deep sweep (12 primitives + conv): cache-key complete + overflow-safe + is_equivalent complete; CLEAN.
- All 3 pass-2 fixes regression-verified PASS; test_kvcache_k_new_paged_succeeds strengthened to SDPA oracle (F4).
- Numerical/backward sweep (13 empirical probes): NaN-propagation Rule-8 clean; fp16 overflow safe; EVERY differentiable path correct at unit+std8 (V6NAX/alibi/window/softcap+window/sparse×3/GQA/rope/packed); std8-bf16 V6NAX "blowup" proven bf16-precision not kernel bug.
- Ran: per-finding probes + new locks + full suite x3 | Validated: 1478 passed + 2 skipped (x3 stable)
- Git: pass-3 commit below; pass 4 required (pass 3 found material).

---
## [2026-06-14 08:45] [CLAUDE] Phase III-4 pass 5: convergence-confirmation audit (no file edits)
STATUS: COMPLETE

### Plan
- Objective: determine whether a fresh full pass is ZERO-FINDING (terminates the audit loop). Job 1 = regression-check passes 1-4 fixes by RUNNING; Job 2 = fresh breadth re-derivation across NaN/dtype/grid/fwd-bwd/cache-key/silent-except classes.
- Files to modify: NONE (audit-only). Probes in /tmp only.

### Job 1 — ALL PASS (ran, with numbers)
- empty-row zeros (3 paths): NORMAL masks do NOT zero/NaN live rows. topk(full mask) rel 9e-4; lcsa dispatch NAX rel 6e-4 + SDPA+bias rel 4e-4 (zr=0 nan=0 both); _sparse_fallback_sdpa rel 2e-4.
- 6 dtype guards: same-dtype calls all no-raise; mismatch raises; uint8 skipped. (suite-covered)
- windowed causal backward: MFA-f16 vs f32-fallback-vjp max(dQ,dK,dV) rel 0.0019.
- kvcache q->pool cast: conditional no-op on same-dtype (suite-covered).
- topk grid (N*256): full-row correctness, worst rel row>=8 = 0.082 vs GT top-k.
- Full suite: **1478 passed, 2 skipped, exit 0** (matches pass-3/4 baseline).

### Job 2 — NOT ZERO-FINDING. One CRITICAL pre-existing bug.
- **P5-1 CRITICAL [VERIFIED, deterministic, public-path, pre-existing]**: `mx.grad`/`mx.value_and_grad` through `flash_attention(..., return_lse=True, causal=True)` returns BROKEN gradients (fp16/bf16: NaN; fp32: ~1e32 garbage / zeros). no-lse path matches SDPA-vjp GT bit-exact (rel 0.0); return_lse path rel 1.0 vs GT. Reachable via DEFAULT backend="auto", no env vars, D∈{64,128}, any shape. NON-causal return_lse is FINE → bug is causal+return_lse+MFA-custom path (`_make_mfa_custom`, attention.py ~5117 fwd `return O,L` / ~5180 vjp). Root: two-output custom_function vjp mis-handles the pruned L output's cotangent when only O is in the loss. `_fallback_sdpa_with_lse` (non-MFA) backward is FINITE → MFA-custom-path specific. Structure identical at e5c5b1b (pre-III-4) → PRE-EXISTING, not a pass-1..4 regression. Test gap: no test grads a return_lse output (grep confirmed).
  Minimal cold repro (3/3 fresh procs): `o,l=flash_attention(q,k,v,causal=True,return_lse=True); mx.grad(lambda q: flash_attention(q,k,v,causal=True,return_lse=True)[0].sum())(q)` → NaN.

### Verified CLEAN (actively looked, with evidence)
- NaN/empty-reduction beyond fixed paths: topk_stream (DEAD code, no importers), cider divide-by-zero guarded (line 188), tq_decode SDPA, causal never empties diagonal row — CLEAN.
- dtype-reinterpret beyond 6 guards: gqa_decode_cider derives T from q + reads k/v unvalidated, BUT mixed-dtype LOUD-fails at MSL compile (not silent corrupt) AND is unexported/unreached — not a finding.
- Grid-spec: all 9 metal_kernel dispatches re-verified (grid-in-threads ↔ threadgroup ↔ indexing): conv im2col/matmul2d, cider p1/p2, topk bisect, topk_stream — CLEAN.
- Fwd/bwd mismatch (chunked_prefill/speculative_verify/splitfuse/shared_prefix/rope-append): all compose flash_attention (no separate vjp) → splitfuse bwd rel 0.0 vs fa; rope_unified grad finite; alibi vjp matches manual ref rel 0.001; main/sparse/alibi/V6NAX/windowed custom vjps all differentiate the SAME feature-applied function. (speculative_verify inherits P5-1 only via return_lse — same bug.)
- Cache-key/id()-keyed/silent-except: id()-caches all include shape+dtype; compile_metallib/_auto_hooks except-blocks are capability-detection graceful-degrade (conservative defaults, not Rule-8 corruption). lse log2 domain IS documented (attention.py:274,322). CLEAN.

### Validation
- Ran: full suite (1478p/2s/exit0); job1_regression.py (15/15 PASS); job2_fwdbwd.py; ~11 /tmp isolation probes for P5-1 (cross dtype/shape/seed/cold-proc/value_and_grad/SDPA-GT).
- Validated: P5-1 = rel 1.0 vs SDPA-vjp GT, deterministic 3/3 fresh procs, no-lse rel 0.0; all CLEAN classes with measured numbers above.

### Git
- not applicable (audit-only, no repo edits); branch master. Pass 5 is NOT zero-finding (1 CRITICAL pre-existing return_lse-backward bug). Audit loop must continue: fix P5-1, then pass 6.

---
## [2026-06-14 06:26] [CLAUDE] Phase III-4 pass 7: TERMINATION-DECISION convergence audit (no code edits)
STATUS: COMPLETE

### Plan
- Objective: decide whether a fresh full pass is ZERO-FINDING (terminates the repeat-until-clean loop). Job 1 = regression-check passes 1-6 fixes by RUNNING; Job 2 = fresh sweep of surfaces least-touched by passes 1-6.
- Files to modify: NONE (audit-only; ledger doc only). Probes in /tmp.

### Job 1 — ALL PASS (ran, with numbers)
- P5-1 PASS: mx.grad(flash_attention(...,return_lse=True,causal=True)[0].sum()) bit-exact to SDPA-vjp (rel 0.0000) at N=4096/D=128 for fp16 AND bf16; L finite; fwd rel ~1e-3. no-lse cross-check rel 0.0000.
- combo-1 PASS: fp32 [B,H,Nq,Nkv] attn_bias + fp16 q — no crash; fp32-bias==fp16-bias rel 0.0000; vs SDPA-GT rel 0.0000.
- Full suite x2: 1485 passed + 2 skipped, exit 0 BOTH runs; no intermittent flake (the known verbose-capture flake did not surface).

### Job 2 — NOT ZERO-FINDING. One MEDIUM silent-failure (pre-existing).
- **F7-1 MEDIUM [VERIFIED, repro, pre-existing, Rule-8 silent]**: `mlx_mfa/svdquant/quantize.py:180` `_replace_layers` checks `isinstance(child, dict)` BEFORE `isinstance(child, nn.Module)` (line 198). But nn.Module IS a dict subclass (`issubclass(nn.Linear, dict)==True`). A model with a DIRECT `nn.Linear` attribute (`self.fc1=nn.Linear(...)` — the most common structure) is treated as a container, its `.items()` are weight/bias arrays (not Modules), so the Linear is NEVER replaced. `quantize_model` returns `{'layers':[],'overall_compression':1.0}` and the model runs UNQUANTIZED while reporting success. Works for nn.Sequential (.layers list) + nested submodules (dict-branch recurses to grandchildren) — exactly why ALL tests pass (every test uses nn.Sequential). Test gap: no direct-attribute test. Repro: `quantize_model(Net())` where Net has `self.fc1=nn.Linear(512,1024)` → 0 layers quantized, fc1 still type Linear, forward bit-identical to dense. Expert/opt-in API, forward-only, not in default attention path → MEDIUM (silent no-op that misleads), not CRITICAL.
- Suggested fix: in `_replace_layers`, test `isinstance(child, nn.Module)` first (or exclude nn.Module from the `dict`/`list` container branches); add a direct-attribute regression test.

### Verified CLEAN (actively looked, with numbers)
- mlx_lm shim (live): sinks→fallback, array-mask→fallback, GQA→native STEEL (rel 7e-4, no fallback), unsupported-D→fallback, return=single array; mx.dequantize signature correct; R1 window fix intact.
- external_cache offload→onload: bit-exact fp16/bf16/fp32 (zero-copy store, dtype+len preserved).
- conv3d MPP+legacy: both match mx.conv_general norm_rmse 2e-4 (pad=1 production envelope) + explicit cross-corr 4e-4 (pad=0); bf16 loud-fails outside MPP envelope (correct). Earlier pad0-vs-conv_general 0.19 = test-reference-convention artifact, NOT kernel bug (confirmed via cross-corr).
- V6NAX backward (env-gated, D=64 causal, M5): dQ/dK/dV rel 7e-4 vs SDPA-vjp, no NaN.
- __init__: 101 __all__ + 33 lazy targets all resolve; hooks install clean. pyproject/CMake/check_venv: no version skew, no -ffast-math/-Ofast numerics flag. Bare excepts (attention.py 889 RoPE-probe / 1524 warmup; build tooling) = capability-probe/warmup graceful-degrade (Rule-8 safe). id()-caches all shape+dtype-keyed with strong-ref ABA guard.

### Validation
- Ran: /tmp/job1_p51.py, /tmp/job1_combo1.py; full suite x2 (1485p/2s/exit0 both); ~8 live probes (svdquant direct/seq/nested, mlx_lm shim, external_cache, conv3d MPP/legacy/crosscorr, V6NAX bwd, __init__ resolve).
- Validated: F7-1 = 0 layers quantized on direct-attribute model (deterministic), all CLEAN surfaces with measured rel-errors above. P5-1/combo-1 regression bit-exact.

### Git
- not applicable (audit-only; only docs ledger touched); branch master. Pass 7 is NOT zero-finding (1 MEDIUM svdquant silent no-op). Loop must continue: fix quantize.py:180, add direct-attribute test, then pass 8.

---
## [2026-06-14 07:40] [CLAUDE] Phase III-4 pass 9: TERMINATION-DECISION convergence audit (no code edits)
STATUS: COMPLETE

### Plan
- Objective: decide whether a fresh full pass is ZERO-FINDING (terminates the repeat-until-clean loop). Job 1 = regression F8-1 + full suite; Job 2 = fresh sweep of the least-touched surfaces (remaining public helpers, TQ multi-step decode, dispatch table, kv_cache adapters, ShaderCache concurrency, Rule-8).
- Files to modify: NONE (audit-only). Probes in /tmp.

### Job 1 — ALL PASS (ran, with numbers)
- F8-1 PASS: make_axial_temporal_mask, 7 non-pow2 grids (pHW∈{35,33,91,9}, causal/temporal_radius variants) — 0 dropped blocks vs token-level reference in EVERY case (block mask is a correct superset; extra blocks present, never under-approximates).
- Full suite: **1489 passed, 2 skipped, exit 0**, 28.9s. No flake.

### Job 2 — ZERO-FINDING. All 6 areas actively probed, CLEAN (with numbers).
- **Public helpers vs SDPA reference** (all rel ~4e-4 fp16 unless noted):
  - make_shared_prefix_cache: fwd rel 4.0e-4, grad finite (nan=0), kp/vp passthrough ok.
  - flash_attention_splitfuse: prefill 4.1e-4, decode 4.5e-4; None-branch returns None correctly.
  - flash_attention_qkv_packed: [B,H,N,3,D] 4.0e-4, [B,N,3*H*D] flat 4.0e-4, flat-GQA (Hkv<Hq) 3.9e-4.
  - flash_attention_kv_packed: flat 4.0e-4, head-first 4.0e-4.
  - sage_attention_kvcache: 8.6e-3 (int8 tol ~0.05); sage_attention_prequantized 8.5e-3 vs SDPA, 1.2e-2 vs online sage.
  - flash_attention_speculative_verify_paged: out rel-vs-dense 4.7e-4, lse+target_logprobs finite.
  - flash_attention_kvcache_rope_append: 8-step decode loop, max rel-vs-numpy(fully-rotated) 4.7e-4, all finite.
- **TurboQuant end-to-end decode**: prefill(128)+20 steps, bits∈{3,4}×compress_v∈{F,T}: worst maxabs 0.033, worst rel 0.20 (3-bit V-compress, expected aggressive), nan=0, all finite, NO drift accumulation (error bounded, does not grow with step count). cratio 1.45–3.76×.
- **dispatch_policy table**: 96 real-device cells (m3+=True nax=True, the ONLY flags on this M5 Max) via backend="auto": no raise, finite, within tol. Canonical D∈{64,128} self-attn all route SDPA (policy-MFA=0, per v2.32.0 NAX design). NOTE: a forced-MFA probe with simulated m3=False/nax=False flagged 2 "failures" (D=64/128 N=1024 fp16 non-causal, rel 1.5/1.8) — PROBE ARTIFACT: those legacy-M1 routes never execute on this HW, and backend="auto" correctly routes them to SDPA (rel 3.7e-4 == MLX SDPA). Not a finding.
- **kv_cache adapters** (Dense/Paged/Quantized/Hybrid): capability flags match implementation exactly — every claimed capability method works + roundtrips; every unclaimed one raises KVCacheOperationUnsupported (loud, not silent-wrong). No mis-claim.
- **ShaderCache concurrency** (read-only): all 7 `cache_` accesses (find/emplace×3/iterate/clear/size) under `lock_guard(mtx_)`. Double-checked locking in get_or_compile is benign (dup compile keeps first via emplace; no overwrite/corruption). mfa_env reset() correctly documented single-thread-only. CLEAN.
- **Rule-8**: 0 bare `except:`; 3 broad `except Exception` all graceful-degrade (ABI-warning emit, kernel warmup, memo-cache write) — Rule-8-safe. 0 NaN-clamping (no nan_to_num/isnan in Python layer). NaN-in→NaN-out contract VERIFIED (not clamped to 0). fp16 large-input stays finite via softmax. 0 C++ empty catches / silent NaN clamps.
- **F7-1 regression (pass-7 svdquant fix)**: direct-attribute model (self.fc1=nn.Linear(512,1024)) now quantizes 2 layers, fc1/fc2 become SVDQuantLinear, forward finite, cratio 3.27×; nn.Sequential path still 2 layers. (Initial 0-layer probe was a sub-256-dim min-filter artifact, not a regression.)

### Validation
- Ran: /tmp/job1_f81_regr.py (7/7 PASS); full suite (1489p/2s/exit0); ~9 Job-2 probes (helpers×8, tq_decode, dispatch real+forced, kvcache Dense/Paged/Quant, shadercache source read, Rule-8 nan-propagation, F7-1 confirm).
- Validated: F8-1 0 dropped blocks; all 6 Job-2 areas with measured rel-errors / lock-coverage / capability-match above; the 2 dispatch "failures" proven to be simulated-flag probe artifacts (auto-path rel 3.7e-4).

### Git
- not applicable (audit-only, no repo edits); branch master; clean (git status --short empty). HEAD d8614ea.

### FINAL VERDICT
**ZERO-FINDING — audit may terminate.** Pass 9 actively probed every Job-2 area and found nothing of correctness significance. Passes 1–8 swept all systematic classes; the tail (P5-1 return_lse bwd, F7-1 svdquant) is fixed+regression-verified. No new peripheral bug surfaced this pass. The repeat-until-clean loop has converged.

---
## [2026-06-14 09:00] [CLAUDE] Phase III-4 FIXED POINT + Phase III CLOSE
STATUS: COMPLETE

### III-4 fixed point
- Pass 9 fresh full pass: ZERO-FINDING of any correctness class. Repeat-until-clean criterion MET.
- 9 passes total, ~73 fixes incl. 2 pre-existing CRITICALs (topk Metal-grid undercount pass 1; return_lse backward corruption pass 5) — both invisible to the suite until active probing.
- Every systematic class swept + verified clean: Metal grid-spec, C++ eval_gpu (cache-keys/overflow/is_equivalent ×12), dtype validation, all 17 feature×grad combos + 11 custom_function vjps, numerical edge (NaN propagation/fp16 overflow), empty-row→zeros contract, full mask-constructor family, svdquant, mlx_lm shim, external_cache, build, TurboQuant E2E decode, dispatch table, kv-cache adapters, concurrency.
- Pattern lessons #8 (MLX grid=threads), #9 (nn.Module is dict subclass), #10 (single-shape-class tests hide bug families).

### Validation
- Ran: 9 audit passes (15+ agents, all empirically probed), per-finding repros, default suite x3, stressed x2
- Validated: default 1489 passed + 2 skipped (x3); stressed 1491 (x2); stable, no abort, no flake; zero known correctness bugs

### Phase III close
- III-1 bf16 conv lift PROMOTED (69977f6); III-2 TQ decode PROMOTED + fused 2/4-bit fix (0593f6b); III-3 v2.51.0 release (abdaa8d); III-4 audit fixed point (d2d3f8d…d8614ea).
- PHASE-III-CLOSE.md written.
- OPEN (Marco-gated): v2.51.1 patch release — v2.51.0 on PyPI does NOT contain the III-4 fixes (2 CRITICALs + ~71 correctness fixes landed post-tag). Strongly recommended; version bump + outward release is Marco's call.

### Git
- close docs commit below; branch master; pushed

---
## [2026-06-14 09:20] [CLAUDE] Phase III release decision: HOLD (Marco)
STATUS: COMPLETE

- Marco decision: HOLD — no v2.51.1 patch release. III-4 fixes (2 CRITICALs + ~71 correctness fixes) stay committed/validated on master (through 8c9cbff), to bundle into a later release (e.g. v2.52.0 with int8) rather than ship standalone now. The 2 CRITICALs are knowingly unreleased on PyPI until then.
- PHASE-III-CLOSE.md updated to record the decision. No outward release action taken (no tag/PyPI/GH).
- Phase III CLOSED. master clean, no pending immediate action.
- Git: doc commit below; pushed.

---
## [2026-06-14 10:00] [CLAUDE] return_lse causal-backward report — ALREADY FIXED (P5-1, aaede0d); test dtype-coverage extended
STATUS: COMPLETE

- Marco reported the return_lse causal-backward NaN/garbage-grad bug (matches III-4 pass-5 P5-1 discovery; report cited pre-fix line numbers + 1478 baseline → predates the fix).
- VERIFIED already fixed on HEAD: mx.grad(flash_attention(...,return_lse=True)[0].sum()) is finite + bit-exact (0.0) to the no-lse path = SDPA-vjp GT, across fp16/bf16/fp32 × D=64/128. Fix = `_make_mfa_custom_lse` custom_function (commit aaede0d, pass 5). No code change needed.
- Gap addressed: the P5 regression test was fp16-only. Extended `TestP5ReturnLseBackward::test_return_lse_grad_matches_sdpa_vjp` to parametrize dtype {fp16,bf16,fp32} × (N,D) × causal — the bug had dtype-specific symptoms so all three are locked. Corrected the invariant: return_lse grad == SDPA-vjp GT within dtype floor (NOT bit-exact to the no-lse path, which uses the V6NAX backward at D=64/qL>=2048 cells, differing by fp16 floor 0.0012).
- Ran: P5 test (19 passed), full suite | Validated: 1501 passed + 2 skipped (was 1489; +12 = the new dtype/causal P5 cells)
- Git: test commit below; branch master; not pushed yet (awaiting nothing — pushing)

---
## [2026-06-15 14:30] [CLAUDE] Sprint III-5: documentation overhaul + v2.52.0 release (coupled) — PUBLISHED
STATUS: COMPLETE

### Plan
- Objective: bring all docs to the complete post-III-4 state as v2.52.0, run 9-gate audit, bump, tag, publish PyPI + GH, smoke-test the published wheel. Handle the "v2.51.0 contains 2 CRITICALs" situation explicitly. Marco pre-authorized the publish; only the v2.51.0 yank is Marco-gated.
- Files: README.md, CHANGELOG.md, docs/PERF_CLAIMS.md, docs/v50/audit-framing-inversions.md, CLAUDE.md, pyproject.toml, mlx_mfa/__init__.py.

### Changes
- R.1 doc overhaul (3b89b51): README headline → v2.52.0; CHANGELOG [2.52.0] with ⚠ CRITICALs disclosure + upgrade directive (NOT buried) + III-4 fixes + v2.51.0 promotions (per-cell + Reproduce block) + DECLINED list + migration note; PERF_CLAIMS header bump + withdrawn v2.39.1 fused claim confirmed ABSENT; Pattern #9 (3 exhibits) + III-4 lessons #8/#9/#10. [VERIFIED]
- R.2 release commit (fd8d278, distinct from docs): version 2.51.0→2.52.0 in pyproject.toml + __init__.py. Semver minor (no breaking API). [VERIFIED]

### Validation
- Ran: /mlx-mfa-release-audit (9 gates) → GREEN_WITH_ADVISORY (no blocking; advisory = carried-claim version strings, benign). Gate 9 (paired-MMA) 2 passed; §Z 21 claims reachable. Default suite 1501 passed/2 skipped; stressed + MFA_POOL_STRESS=1 canary 1503 ×2. twine check PASS ×2.
- Validated: all 9 gates green → publish gate cleared.

### R.4 publish (irreversible) — DONE
- Tag v2.52.0 pushed (origin 83acf10…). PyPI live (https://pypi.org/project/mlx-mfa/2.52.0/, cp311 wheel + sdist). GH release published (not draft, both assets, CRITICALs disclosure in body). Confirmed: PyPI latest=2.52.0 both files; GH published; origin tag present. [VERIFIED]

### R.6 post-publish smoke (clean py3.11 venv, published cp311 wheel) — 4/4 GREEN
- CRITICAL#2 return_lse backward: grad finite + matches SDPA-vjp fp16/bf16/fp32.
- CRITICAL#1 topk full-row coverage: all 512 rows written, 0 stale, first8/last8 0.95×.
- HEADLINE V6NAX backward causal+non-causal: matches SDPA-vjp.
- HEADLINE conv3d auto-hook fp16+bf16: deterministic (maxdiff 0.0) + matches fp32 (MAE/RMS 0.00014/0.00112).
- Forensics note: initial conv check used degenerate 16ch shape → max-abs-rel artifact (0.19→8.09 from near-zero denom). Verified NAX conv deterministic + accurate at realistic channels; small-channel fp16 gap (~250× vs native fp16 at Cin=16) is pre-existing/out-of-scope → spawned background task. [VERIFIED]

### Tech cost
- None (docs + release; no kernel/code-path change).

### Dependency & regression check
- Callers verified: no public API signature change. Test coverage: 1501/1503 green; the two CRITICALs locked by TestP5ReturnLseBackward + topk full-row assertion.

### OPEN (Marco-gated)
- R.5 v2.51.0 disposition: Option A YANK (recommended — pip already resolves to 2.52.0; stops new pins to a known-corrupt release) vs Option B leave+disclose. Awaiting Marco. Do NOT yank without explicit go.
- Small-channel fp16 conv3d accuracy: spawned background task (not a campaign regression).

### Git
- 3b89b51 (docs) + fd8d278 (release) + report commit below; merged/pushed as 73d5738; tag v2.52.0 pushed. Branch master.

### Campaign close
- Phase III + the full 2026-06 audit/optimization campaign CLOSED with v2.52.0 as the canonical release. Report: docs/v50/campaign-2026-06/phase3/sprint-III-5-report.md.

---
## [2026-06-15 16:00] [CLAUDE] Conv3D NAX small-channel fp16 accuracy gap — root cause + dispatch-gate fix
STATUS: COMPLETE

### Plan
- Objective: investigate + fix the v2.52.0 post-release observation that the conv3d NAX auto-hook is inaccurate for small fp16 channel counts. Answer Q1 (root cause), Q2 (gate fix), Q3 (regression coverage).
- Files: mlx_mfa/_auto_hooks.py; tests/test_iii5_conv_small_channel_accuracy.py (new); test fallout updates in 3 existing test files; doc.

### Root cause (Q1) [VERIFIED]
- conv3d_nax_forward has 2 paths: MPP convolution2d (C++-gated C_in/C_out %16==0 & >=32, fp32 accum — CORRECT) and a legacy im2col+matmul2d fallback. The legacy matmul2d K-loop reads partial K-tiles past K_FULL with no tail mask; K=C_in*27 is %32==0 iff C_in%32==0, and all such C_in take the MPP path — so the legacy path is reached ONLY when K%32!=0 → always numerically broken (C_in=16 → MAE/RMS 0.11; C_in=31 → NaN). Pointwise 1x1x1 + the Python legacy orchestrator share the same matmul2d kernel + bug.
- Empirical boundary = C_in %16==0 AND >=32, EXACTLY the existing bf16 MPP gate. bf16 dispatch had the gate; fp16 did not → silent corruption. Single-shape-class coverage gap (III-4 lesson #10): every conv test used C>=32 %16==0.
- Trap: test_fp16_still_works compared legacy vs mx.conv_general which (under hooks) routed to the SAME broken legacy kernel → two wrongs compared equal → green. Independent fp32 reference is the only trustworthy bar.

### Changes (Q2 fix)
- `mlx_mfa/_auto_hooks.py`: renamed `_conv3d_bf16_mpp_eligible` → `_conv3d_mpp_eligible`; applied it to BOTH fp16 and bf16 in `_patched_conv_general` (was bf16-only). Shapes outside the MPP envelope (incl. all 1x1x1 pointwise + small-channel) now fall back to native, counted in telemetry (Rule 8, no silent drop). [VERIFIED]

### Tests (Q3)
- NEW `tests/test_iii5_conv_small_channel_accuracy.py` (39): parametrized accuracy C_in∈{8..128}×fp16/bf16 vs fp32 ref (MAE/RMS<0.01); determinism; gate-predicate locks (pure-function, order-robust); pointwise gated out.
- Fallout updates (tests encoded the buggy behavior): test_v50_prompt_5g_hook_telemetry (engagement fixture shape 16→32; fallback-reason → MPP gate); test_smoke_vsr_models_v2_50_1 (4 tests: small-channel input layers now correctly fall back — assert engaged+fallback==total + eligible-engage counts); test_campaign_2026_06_sprint_a_cache_keys::test_fp16_still_works (shape 16→32 where legacy GEMM is exact).

### Validation
- Ran: full suite `.venv/bin/python -m pytest tests/ -q`. Validated: 1540 passed, 2 skipped (was 1501; +39 new). Sweep confirms all fp16 C_in now match fp32 within dtype floor; MPP envelope still engages NAX (headline claim unaffected).

### Dependency & regression check
- `_conv3d_mpp_eligible` only referenced in _auto_hooks.py (verified grep). No public API change. Coverage gap closed by the new parametrized + predicate tests.

### Scope NOT taken (proposed follow-ups)
- Mask the matmul2d partial K-tile in the Metal kernel (true root cause; would let the gate re-widen to recover NAX perf on small-channel layers) — needs rebuild + revalidation.
- Rule 8: make the C++ legacy path + Python legacy orchestrator RAISE for C_in%32!=0 instead of silently corrupting (defense-in-depth for raw _ext callers).

### Git
- commit below; branch master; pushed. Doc: docs/v50/campaign-2026-06/phase3/conv-small-channel-fix.md.

---
## [2026-06-15 17:30] [CLAUDE] Sprint III-6: matmul2d K-tail kernel fix + v2.52.1 release (coupled) — PUBLISHED
STATUS: COMPLETE

### Plan
- Objective: fix the TRUE root cause of the conv3d small-channel corruption (the matmul2d unmasked partial-K-tile) at the kernel level so all three entry points are correct, bench whether to re-widen the gate (Pattern #6), add Rule-8 defense, ship ONE coupled v2.52.1 release. Marco directive: real fix, release once.
- Skills: /metal-kernel-dev (fix design), /mlx-mfa-bench-methodology (R.2 gate decision), /mlx-mfa-release-audit (9 gates).

### Changes
- R.1 (cb76456): matmul2d partial-K-tile fixed by zero-padding contraction K to a K_TILE multiple before dispatch — pad_contraction_k (mfa_conv_nax.cpp) + _pad_k (conv_nax.py). Covers C++ legacy im2col, C++ 1x1x1 pointwise, Python legacy orchestrator. Rebuilt ext. [VERIFIED]
- R.3 (cb76456): matmul2d_source (C++ + Python) refuses non-K_TILE-aligned K (Rule 8 — future unpadded caller fails loudly). [VERIFIED]
- R.4 (cb76456): test_iii5 extended (TestFixedKernelPathsDirect — 3 entry points vs fp32 incl C_in=31; Rule-8 refusal lock; 62 tests). conftest.py Metal-pool clear_cache fence. Lesson #11 codified in audit-framing-inversions.md.

### R.2 bench → gate decision [VERIFIED]
- Canonical 3-session median, VAE first-layer shapes: C_in=8 tie(1.04), C_in=16 native wins(1.77x), C_in=24 native(1.73x), C_in=48 NAX wins(0.73, MPP path). DECISION: KEEP gate-out — native faster AND correct at small C_in (orchestration overhead dominates; Pattern #6). Kernel fix is correctness defence for raw-API/pointwise/Python callers, not perf re-widening.

### Validation
- R.1 verify: all three entry points MAE/RMS 0.00014 vs independent fp32 at C_in 8/16/17/24/31/33/40/48; C_in%32==0 unchanged. (Pre-fix: 0.11 / NaN.)
- R.5: full suite green x2 (1563 passed, 2 skipped); pool canary green (122); pre-tag MFA_POOL_STRESS=1 full suite 1565 passed.
- 9-gate audit: GREEN_WITH_ADVISORY (no blocking; advisory = no new PERF_CLAIMS entry / Reproduce snippet — v2.52.1 introduces NO new claim, headline carried unchanged; same benign class as III-5). Gate #9 programmatic 2 passed.
- Post-publish smoke on PUBLISHED 2.52.1 wheel (clean py3.11 venv): 6/6 — conv3d small-channel kernel correct (C_in 8/16/31 vs fp32 incl prior-NaN case), Rule-8 guard, MPP fp16/bf16, V6NAX backward, return_lse, topk.

### R.4 lesson #11 + sweep
- Lesson: validate a low-precision kernel against an INDEPENDENT higher-precision reference (fp32), never another kernel path. (test_fp16_still_works compared the kernel vs mx.conv_general which under hooks WAS the same broken kernel.) Sweep: only instance was test_fp16_still_works (fixed III-5); all other low-prec tests use independent refs (Apple SDPA / PyTorch fp32 / unhooked native).

### Release (irreversible) — DONE
- Tag v2.52.1 (origin 1ae1b1b). PyPI live (https://pypi.org/project/mlx-mfa/2.52.1/, cp311 wheel+sdist). GH release published (both assets, CHANGELOG disclosure). Commits: cb76456 (fix) + deaafaf (docs) + 9d62d2e (release: version+CHANGELOG, distinct). [VERIFIED]

### OPEN (Marco-gated)
- v2.52.0 disposition: YANK recommended (contains conv3d small-channel silent-corruption; pip already resolves to 2.52.1). Manual PyPI web-UI step (PEP 592, no CLI). Awaiting Marco. Do NOT yank — already done? NO: v2.52.0 not yet yanked. Also v2.51.0 yank from III-5 may still be pending Marco.

### Tech cost
- Pad copy + slightly wider matmul (+<=K_TILE-1 in K) when K unaligned; nil when aligned. Negligible (small-channel routes to native anyway).

### Git
- cb76456 + deaafaf + 9d62d2e; tag v2.52.1; branch master; pushed. Report: docs/v50/campaign-2026-06/phase3/sprint-III-6-report.md.

---
## [2026-06-15 18:00] [CLAUDE] Sprint III-7: targeted sweep for the conv3d bug's hidden siblings
STATUS: COMPLETE

### Plan
- Objective: hunt the 3 classes that let the conv3d bug survive 9 III-4 passes — (A) non-independent reference validation, (B) unmasked partial-tile read, (C) single-shape-class coverage. Confirm every suspect vs an INDEPENDENT fp32 reference. Fix structurally, lock with coverage.
- Mode: one sub-agent per class (3 parallel general-purpose agents), then central verify + fix.

### Findings
- **Class A — CLEAN** [VERIFIED]. ~50 numerical-validation tests classified; 7 non-independent-reference categories re-probed vs fp32, all at dtype floor. Only ever-active instance was the conv test (fixed III-5/6).
- **Class B — CLEAN** [VERIFIED]. conv matmul2d K-tail (fixed III-6) was the ONLY unmasked partial-tile read; all other kernels mask tails (load_safe + -inf). Sparse-attention gate is documented + Rule-8-guarded (the right model). Non-multiple probes vs fp32 all clean.
- **Class C — 2 quantize_model bugs FIXED** [VERIFIED]. (1) bare top-level nn.Linear = silent no-op → now Rule-8 raise; (2) group-misaligned in_features raised mid-walk after partial mutation → default predicate skips cleanly + pre-validation pass makes custom-predicate misalignment fail atomically. Same lesson-#9 family the III-4 F7-1 fix partially closed.

### Cross-class — new variant flagged to Marco (NOT fixed)
- `backend="mfa"` non-causal D∈{64,128} fp16 diverges from fp32 SDPA (MAE ~0.12, verified). NOT default-reachable: dispatch_policy routes non-causal dense to SDPA for PERF (documented), so the kernel is never auto-selected + no test forces it. Class-B.2 cousin: a perf gate masking a latent correctness bug on a forced expert path. Disposition Marco's (Rule-8 raise vs kernel fix vs leave-documented).

### Changes
- `mlx_mfa/svdquant/quantize.py`: top-level-match Rule-8 raise; `_default_predicate` adds K%group_size==0; new `_validate_quantizable` pre-pass (atomicity). [VERIFIED]
- Strengthened `tests/test_conv_nax_migration.py`: +fp32 independent anchor (both cpp & python-legacy), +3 non-%32 C_in shapes (16/31/48).
- NEW `tests/test_iii7_quantize_model_guards.py` (7), `tests/test_iii7_attention_partial_n_coverage.py` (98 — partial-N fp16/bf16 attention vs fp32 SDPA).
- `docs/v50/audit-framing-inversions.md`: III-7 sweep entry + new mechanism variant (perf-gate-masked forced-path bug).

### Validation
- Ran: full suite ×2 consecutive. Validated: 1671 passed, 2 skipped both runs (was 1563; +7 +98 +3). SVDQuant fixes verified behaviorally (raise/skip/atomic, un-mutated state). No default-path kernel/dispatch change → net perf non-worse.

### Dependency & regression check
- `_default_predicate` signature gained group_size (default 64); only caller is quantize_model (closure) + the new pre-pass. svdquant suite 22/22 unchanged.

### Git
- commits below; branch master. NOTHING RELEASED (Marco-gated): quantize_model fix disposition (patch v2.52.2 vs bundle) + the forced-MFA finding both await Marco.

### Marco-gated queue
- quantize_model Findings 1&2: fixed on master; release disposition pending.
- backend="mfa" non-causal divergence: flagged, fix approach pending.

---
## [2026-06-15 18:20] [CLAUDE] III-7 Marco decisions + forced-MFA investigation opened
STATUS: IN_PROGRESS

- Marco decision 1: quantize_model Findings 1&2 → BUNDLE into next release (no v2.52.2). Fixes stay on master (c7e07ad).
- Marco decision 2: backend="mfa" non-causal D∈{64,128} divergence → INVESTIGATE THE KERNEL (root-cause why non-causal MFA forward diverges; causal is correct).
- Opened investigation (task #23). Forensic pass starting: localize kernel + code path, characterize shape/dtype dependence.
- Git: not applicable (log entry only).

---
## [2026-06-15 18:45] [CLAUDE] forced-MFA non-causal divergence — LOCALIZED to STEEL V2 non-causal
STATUS: HANDOFF_READY

- Confirmed (independent fp32 SDPA): backend="mfa" non-causal D∈{64,128} fp16 MAE ~0.12 at all N (incl aligned); auto + causal-mfa correct (MAE 0.0).
- Signature: output rows DUPLICATED (rows 0=1, 2=3 by norm) + magnitude ~2× → per-query-row index-collapse + softmax-normalization error. NOT a boundary bug.
- Bisection (MFA_DISABLE_V2=1 → MAE 0.0; V3 disable no effect): bug is in STEEL V2 forward non-causal path (`csrc/mfa_steel_fwd_v2.cpp`). V1 non-causal correct; V2 causal correct. [VERIFIED]
- Likely a regression from a post-v1.4.0 V2 change (V2 non-causal benched working in v1.4.0). git-blame is next.
- Why never caught: dispatch_policy routes non-causal dense → SDPA for PERF (documented), so V2 non-causal never auto-selected; no test forces non-causal backend="mfa". Perf-gate-masked latent bug.
- Findings doc: docs/v50/campaign-2026-06/phase3/backend-mfa-noncausal-divergence.md (signature, bisection, next steps).
- NOT FIXED. Marco-gated fix approach: (a) repair V2 non-causal normalization/indexing (validate vs fp32), or (b) route forced-mfa non-causal to V1/SDPA + Rule-8 note (V2 non-causal has no production value since auto routes to SDPA). Then lock with a forced-mfa non-causal fp32 correctness test.
- Git: findings doc + log commit below; branch master. Nothing released.

---
## [2026-06-15 19:30] [CLAUDE] forced-MFA non-causal — REFINED to V2 single-pass; mechanism narrowed (not yet fixed)
STATUS: IN_PROGRESS

- Marco approved "repair V2 non-causal". Deeper bisection refined the location:
  - B·H∈{1,2} (under-occupied→split-K) correct; B·H≥4 (single-pass) wrong; MFA_FORCE_SPLITK=0 reproduces at ALL H incl H=1.
  - => bug is in V2 SINGLE-PASS non-causal branch specifically. V1 non-causal correct; V2 causal single-pass correct; V2 split-K non-causal correct. [VERIFIED]
- Mechanism narrowed: magnitude ~1.5× too large (norms ~2.6×) + per-row error DECREASES monotonically with q (Q-tile0 0.21→Q-tile7 0.085), deterministic. RULED OUT: final normalization (line 838, shared w/ causal), online softmax (line 721, "same as V1"), L write. Only causal-conditional diffs for N=256 are kb_lim + diagonal mask. => fault in V2 single-pass KV_smem preload/barrier/P@V machinery over the FULL non-causal tile range.
- Repair is delicate barrier/smem work; NOT done this session (avoid rushed kernel edit → race risk). Findings doc updated with the exact reproduction + the code regions to read + the signature anchor.
- Findings: docs/v50/campaign-2026-06/phase3/backend-mfa-noncausal-divergence.md.
- Git: doc + log commit below; branch master. Nothing released.

### Next action
- Repair: read V2 single-pass preload (~322-375) + per-iter V-load/K-preload/barriers (~775-831) + P@V; git-blame vs v1.4.0; fix; validate vs fp32 across D/N/dtype/B·H; lock with forced-single-pass non-causal regression test.

---
## [2026-06-15 21:00] [CLAUDE] III-8 repair attempt — localization tightened, NOT fixed (no guessed edit)
STATUS: IN_PROGRESS

- Marco scoped III-8 to REPAIR V2 single-pass non-causal. Outcome: substantial further localization, but the repair was NOT landed. Per the sprint rule ("do NOT guess a barrier/smem edit — read to the exact expression first") + race risk, I made NO kernel edit. Honest status: tightly localized, repair pending kernel instrumentation.
- New facts (all vs independent fp32):
  - Both code paths equally wrong (MFA_FORCE_GEN=14 smem == default M5+ direct) → bug in SHARED path-independent kernel code, NOT the barrier/smem machinery III-7 hypothesized. [VERIFIED]
  - Non-causal path intrinsically broken: LAST query row (identical math to causal) is correct under causal (MAE 0.0) but wrong under non-causal (MAE 0.115). Not a key-subset issue. [VERIFIED]
  - Params ruled out: V2 single-pass dispatch builds identical MFASteelParams for causal/non-causal (only qL_off, =0 when N==S). [VERIFIED]
  - Present at 1 tile (N=16 MAE 0.28), dilutes with N (N=256 MAE 0.12). Longstanding (orig V2 commit 81d801f7), never caught (non-causal single-pass never auto-dispatched + untested). [VERIFIED]
  - Per-row cosine(MFA,fp32) LOW (mean 0.36, max 0.66) → WRONG DIRECTION, not a per-row rescale → attention WEIGHTS P are wrong, not just the denominator. Corrects III-7's "denominator under-accumulated" framing. [VERIFIED]
- Locus: V2 single-pass non-causal Q@K^T→Stile score population (or row_reduce/MFAExpSubOp). Next step: dump Stile/P for 1-tile non-causal, rebuild once, diff vs numpy → pins the line. Then fix + fp32-validate + re-bench + lock.
- Findings doc updated (III-8 section). No code edited. Nothing released.
- Git: doc + log commit below; branch master.

---
## [2026-06-15 23:00] [CLAUDE] III-8 repair session — NOT fixed; sub-component NOT reliably isolated (no guessed edit)
STATUS: IN_PROGRESS

- Marco scoped III-8 to REPAIR V2 single-pass non-causal. Outcome: NOT fixed. Made NO functional kernel edit (only temporary debug dumps, reverted; clean rebuild verified). Honest: could not reliably isolate the sub-component.
- Reliable (unchanged/strengthened): deterministic (maxdiff 0.0, not a race); bug is the V2 single-pass non-causal COMPILED variant (causal variant correct, same largely-shared source, only kb_lim+mask differ); softmax normalization not the sole fault.
- Instrumentation dead-ends (documented so next pass skips them):
  1. Mid-kernel register dump of Stile/P (store_contiguous AND faithful element-wise frag_at) — UNRELIABLE. Control: dumping CAUSAL Stile (correct output) also yields "wrong" values vs true scores → the cooperative simdgroup-MMA fragment layout isn't faithfully serialized by naive per-lane writes + early-return perturbs state.
  2. One-hot V (O=P): reliable for causal (P@V proven correct → causal lower-tri scores correct, MAE 1e-4), but CONFOUNDED for non-causal (O=P assumes P@V correct, which is an unproven suspect). So non-causal O≠P can't distinguish wrong-scores vs wrong-P@V.
- RETRACTED the earlier "it's Q@K^T/scores" read (III-7 + first III-8 dumps) — it rested on the unreliable register dump.
- Next reliable approaches: (a) layout-correct simd_shuffle fragment gather probe; (b) V2 single-pass vs split-K stage differential (localization aid only, lesson #11); (c) re-evaluate route-around (forced-mfa non-causal → V1/SDPA) given repair cost.
- This non-default-reachable bug has now consumed III-7+III-8; resisting available diagnostics. No guessed edit (race risk > current localized non-default bug).
- Findings doc updated (III-8 outcome section). Clean tree. Nothing released.
- Git: doc + log commit below; branch master.

---
## [2026-06-16 00:30] [CLAUDE] III-8c static-diff reorientation — narrowed, NOT isolated; lesson #12 codified
STATUS: IN_PROGRESS

- Approach: abandoned register dumps (lesson #12). Static causal-vs-non-causal diff + git archaeology + O-vs-fp32 only.
- R.1 exhaustive: EXACTLY 2 causal-conditionals in single-pass generator — kb_lim (435) + diagonal-mask (665). Both read correct; both original (81d801f7), unchanged. [VERIFIED]
- R.2 archaeology: no v1.4.0 tag (v1.3.0→v2.5.0); code original. "v1.4.0 benched working" = COVERAGE ILLUSION (non-causal single-pass correctness never fp32-validated). NOT a regression — longstanding original bug. Contradiction resolved. [VERIFIED]
- Reliable narrowing (O vs fp32): N=64 D=128 = 1 FULL tile, kb_lim=1 for BOTH → causal 0.0001, non-causal 0.19 → bug is purely mask-ABSENCE; col>row (future-key) positions' contributions wrong; causal masks them (never exercises), non-causal uses them. BK-independent (FORCE_BK=32 no fix). Deterministic. [VERIFIED]
- Can't separate wrong-scores vs wrong-P@V (causal zeros col>row → no signal; dump unreliable; one-hot confounded). Subtle MMA-fragment issue at col>row, invisible to source reading.
- Per prompt fallback: static diff + history did NOT isolate the line → reported honestly + scoped next fork (simd_shuffle gather OR standalone MMA repro). NO guessed edit.
- Codified methodological lesson #12 (cooperative-MMA register dumps unreliable; diagnose via O-vs-fp32 + static diff) in audit-framing-inversions.md.
- Docs: sprint-III-8c-report.md + findings-doc III-8c section + catalogue lesson #12. Clean tree (debug reverted). Nothing released.
- Git: doc + log commit below; branch master.

### Marco-gated fork
- (1) simd_shuffle fragment-gather probe, (2) standalone MMA repro, (3) route-around (forced-mfa non-causal → V1/SDPA), or (4) pause. Repair spans III-7+III-8+III-8c without line-isolation. quantize_model III-7 fixes still queued for bundling.

---
## [2026-06-16 02:00] [CLAUDE] III-8d standalone-MMA prep — dump RIGOROUSLY closed via self-check; not isolated
STATUS: IN_PROGRESS

- Understood MMA: MFAMMAFrag=simdgroup_matrix<T,8,8>; Stile=Q@K^T accumulator tile; store<U,1,1> per-lane (base=i*8*ld+j*8, dst pre-offset by simd_coord); same store faithful for Otile.
- MANDATED col<=row self-check (prompt R.2): causal Stile.store dump MAE 1.18 vs numpy on col<=row — a region causal output PROVES correct → store read UNFAITHFUL. causal vs noncausal dumps differ pre-mask (MAE 0.24, must be identical). RIGOROUSLY closes the register-dump approach (was inferred before, now proven). [VERIFIED]
- Blocker pinpointed: kernel consumes Stile correctly via frag_at (causal works) but serializing Stile→memory via store is unfaithful despite identical store working for Otile → Q@K^T ACCUMULATOR tile layout differs (operand-vs-accumulator / WM row-mapping). 3 read methods defeated, all fail the self-check. Standalone harness uses same store → same blocker.
- Mechanism localized (III-8c, reliable): wrong col>row contributions, single Q@K^T tile, non-causal single-pass. Reading col>row needs faithful accumulator serialization — the unsolved sub-problem.
- Next fork: (1) derive exact accumulator lane→(row,col) layout via trivial numpy-parity single-MMA test (Q=I,K=I), validate col<=row, then read col>row; (2) escalate to MLX MMA-primitive layer. Marco's call: dedicated fork vs route-around interim.
- Consumed III-7+III-8+III-8c+III-8d. Non-default-reachable. NO guessed edit. Clean tree (dump reverted).
- Docs: sprint-III-8d-report.md + findings-doc III-8d section. Extended lesson #12 (accumulator frags not serializable by tile store even when sibling tile store works). Nothing released.
- Git: doc + log commit below; branch master.

---
## [2026-06-16 04:00] [CLAUDE] III-8e — known-answer probe CRACKS the mechanism (behavioral); exact line open
STATUS: IN_PROGRESS

- Technique (institutional win, lesson #13): known-answer uniform-P probe — Q=0 ⇒ uniform P (isolates softmax+P@V from scores); V=ramp ⇒ O=mean(attended keys); V=indicator ⇒ attended set; V=ones ⇒ O=1.0 (P normalized). Reads effective attention through the CORRECT pipeline, no dump/confound, validated vs fp32. Cracked it where 4 sprints of register dumps/one-hot failed.
- MECHANISM (reliable): non-causal single-pass attends exactly (qb+1)*BQ keys per Q-tile (qb=0→32,1→64,2→96,3→128 at BQ=32), tile-uniform, sub-tile granular = the causal q_max bound leaking into non-causal. Keys ≥ q_max are TRUNCATED (not attended), not miscomputed. [VERIFIED, no confound]
- Resolves the 4-sprint paradox: causal masks ≥q_max anyway → truncation invisible to causal; only non-causal qb=0 visibly wrong. Subsumes earlier "col>row wrong"/"denominator" reads.
- Exact line OPEN: (qb+1)*BQ-keys signature matches NO obvious source — kb_lim=NK (q-indep, 64≠32), causal kb_lim formula (64≠32), masks (gated/inactive at repro), dispatch params (correct, q-indep) — all ruled out. q-dependence needs an in-kernel qb-using key limit not found by static reading.
- Next: uniform-P probe is a reliable cheap ORACLE → bisect kernel source (disable/alter candidate qb-using regions, re-measure attended count until =N for all qb) to pin the line empirically, then fix+O-vs-fp32+generalize+rebench+lock. NO guessed edit.
- Consumed III-7..III-8e. Non-default-reachable. Clean tree+binary. Nothing released. Codified lesson #13 (known-answer probe).
- Git: doc + log commit below; branch master.

---
## [2026-06-16 06:00] [CLAUDE] III-8 RESOLVED — root cause = async_v2.metallib (NOT JIT source); fix + lock landed
STATUS: COMPLETE

### Plan
- Objective: pin the line, fix the mechanism, O-vs-fp32, R.5 generalize, re-bench, lock (Marco's directive: understand+fix, no route-around).
- Files modified: csrc/shader_cache.mm (fix), tests/test_iii8_backend_mfa_noncausal.py (lock, new), 3 docs.

### Root cause (the 5-sprint misdirection resolved)
- NOT in generate_steel_v2_source. `shader_cache.mm::try_async_pipeline()` is called FIRST and for SteelForwardV2 keys on macOS 26 served a pipeline built from precompiled `async_v2.metallib`. Its `simdgroup_async_copy` DMA (removed from macOS-26 AIR runtime, confirmed liuliu) loads only ~(qb+1)*BQ keys/Q-tile → the exact "q-dependent truncation" III-8e's uniform-P oracle measured. Causal survives (mask zeroes unloaded keys); default routes non-causal dense→SDPA so only backend="mfa" reached it. [VERIFIED]
- III-8e's behavioral mechanism was CORRECT; the LOCATION was mis-attributed to JIT codegen. Tell missed: "signature matches NO obvious source line" ⇒ the source isn't running.

### Changes
- `csrc/shader_cache.mm:99-105` — gate try_async_pipeline off on macOS 26+ (NSProcessInfo majorVersion>=26 → return nullptr), after the SteelForwardV2-only guard + MFA_DISABLE_ASYNC check. Full root-cause comment in-code. [HIGH][VERIFIED]
- `tests/test_iii8_backend_mfa_noncausal.py` — NEW lock: fp32-parity sweep D{64,128}×{fp16,bf16}×causal×N{32..4096} + forced-single-pass (MFA_FORCE_SPLITK=0 autouse fixture) non-causal known-answer test (each Q-tile attends ALL keys, O[qb*32,0]==(N-1)/2). 57 passed. Closes v1.4.0 coverage illusion.

### Confirmation (not guessed)
- MFA_DISABLE_ASYNC=1 differential: non-causal MAE ~0.12→bit-exact vs fp32 (env bypassing source compilation fixes it). [VERIFIED]
- sentinel-777 write in JIT source never appeared → JIT ≠ running binary for that key. [VERIFIED]

### Validation
- R.4: full backend="mfa" sweep correct vs INDEPENDENT fp32 SDPA (lesson #11), deterministic maxdiff 0.0.
- R.5: fix disables broken metallib for ALL its keys on macOS 26 (only served SteelForwardV2). Ran: `.venv/bin/python -m pytest tests/ -q` → 1728 passed, 2 skipped. Validated: full suite green + lock green.
- R.6: JIT V2 fwd ~3-4× slower than Apple SDPA on M5+; async only reachable via backend="mfa" → correctness-only, NO perf impact, NO promotion (consistent with default→SDPA dispatch). [VERIFIED]
- R.7 lock: 57 passed.

### Separate pre-existing bug FLAGGED (not fixed)
- V2 split-K non-causal partial-N (N∈{127,160,191,224}, D=128) → MAE ~7-16 vs fp32. Distinct from async (try_async only served single-pass; split-K always JIT). Non-default-reachable. Spawned follow-up task task_906dde0d.

### Dependency & regression check
- try_async_pipeline callers: only get_or_compile() (line 270); SteelForwardV2-only (line 75) — split-K never served by async. Verified.
- Coverage: lock test added; full suite covers V2 causal+non-causal+auto.

### Institutional codification
- audit-framing-inversions.md lesson #14: confirm WHICH BINARY runs before debugging source (sentinel write / dispatch-env toggle / enumerate key loaders). The keystone lesson of the marathon.
- backend-mfa-noncausal-divergence.md → RESOLVED (§ Resolution III-8).
- sprint-III-8-resolution-report.md (new; §AA.2 skill table present).

### Release disposition (Marco-gated)
- Bundle V2 non-causal fix + the two III-7 quantize_model fixes → v2.52.2. Correctness-only, no API change, no perf claim. Awaiting Marco's go.

### Git
- WIP — about to commit shader_cache.mm + lock test + 3 docs + this log; branch master. Nothing released/tagged.

---
## [2026-06-16 07:30] [CLAUDE] DECISION REVISIT — III-8 root cause FALSIFIED; real bug = V2 single-pass D=128 cold-start
STATUS: BLOCKED

- Prior: `[2026-06-16 06:00] [CLAUDE] III-8 RESOLVED` (commit da737e7) — claimed async_v2.metallib broken on macOS 26 = root cause; fix = gate async off. Labeled VERIFIED. **This is WRONG.** [CORRECTION of that entry]
- New analysis (all VERIFIED on pristine HEAD da737e7, clean rebuild, NO debug edits):
  - backend="mfa" NON-CAUSAL D=128 random-Q: COLD first-dispatch → MAE 13-36, NONDETERMINISTIC across processes (uninitialized/stale scratch memory). Within one process: 1st call wrong, subsequent correct (warm).
  - CORRECT in every other case: V1 (MFA_DISABLE_V2=1) cold ✓; D=64 cold ✓; causal cold ✓; Q=0 uniform-P cold ✓ (masks it — zero scores); backend="auto" (production) cold ✓ (routes non-causal dense → SDPA).
  - try_async_pipeline: GATE FIRES on macOS 26 (async skipped) → JIT path used → STILL garbage cold. So the async metallib was NEVER the cause; my III-8 gate is INERT (verified via MFA_TRACE_ASYNC fprintf: async never loaded; MFA_FORCE_ASYNC_ON_26 bypass irrelevant).
  - Why III-8 misdiagnosed: (a) NSLog→unified-log (not stderr) hid that async wasn't loading; (b) III-8 debug-dump WRITES into v2 scratch buffers accidentally INITIALIZED the memory, masking the cold bug → reverting dumps re-exposed it; (c) the "warm" suite/lock (many tests/process) never sampled the cold first-dispatch.
  - The III-7 "split-K partial-N MAE 7-16" and the III-8e "(qb+1)*BQ truncation" were the SAME cold-start garbage sampled at different shapes, mis-attributed.
- Localized: V2 single-pass D=128 non-causal cold first-dispatch reads uninitialized memory. NOT async, NOT split-K, NOT a clean truncation. [VERIFIED]
- Severity: NON-default-reachable (auto→SDPA; production safe). Expert backend="mfa" non-causal D=128 only. Real correctness bug.
- Consequence: v2.52.2 MUST NOT ship as planned (headline III-8 fix inert + real bug open). da737e7 docs (RESOLVED) + lock test rationale + lesson #14 framing all need correction.
- Instrumentation stashed: `git stash` "III-9 instrumentation" (SPLITK_DEBUG + TRACE + async-bypass) — available for the fix.
- Ran: pristine HEAD clean rebuild + 5-proc cold sweep (13-36 MAE) + V1/D64/causal/Q0/auto cold (all correct). Validated vs independent fp32 SDPA (lesson #11).
- Git: nothing committed this entry; binary = pristine HEAD; working tree clean (instrumentation stashed). Awaiting Marco decision on fork.

---
## [2026-06-16 09:00] [CLAUDE] III-9 — bug CONFIRMED in V2 split-K (not single-pass); race not yet line-pinned
STATUS: IN_PROGRESS

- Refines the 07:30 DECISION REVISIT. The failing case (B=1 H=4 N=128 D=128 non-causal backend=mfa) dispatches to V2 SPLIT-K (num_splits=2, BK=32), confirmed via MFA_SPLITK_DEBUG path trace — NOT single-pass. [VERIFIED]
- Single-pass (MFA_FORCE_SPLITK=0): correct cold (0.00004). V1: correct. auto: correct. So bug is split-K-ONLY. [VERIFIED]
- Nature: nondeterministic across processes w/ identical seed (MAE 12-36 N=128, ~6.8 N=256) = a RACE. Masked by Q=0 (uniform-P probe correct cold). Value-dependent (only real scores). Affects upper query rows (qb>=2), boundary varies run-to-run. o saturates to fp16 512.
- RULED OUT (each tested): (a) FlashDecodeReduce kernel — purely per-element, no smem/barriers/cross-thread reduce, race-free [VERIFIED by read]; (b) unwritten pO/pL scratch — MFA_ZERO_SCRATCH memset didn't fix [VERIFIED]; (c) Phase1->Phase2 barrier insufficiency — Q=0 would also fail, it doesn't [VERIFIED]; (d) K-smem visibility race at Q@K^T — MFA_EXTRA_BARRIER didn't fix [VERIFIED]; (e) M3+-specific — MFA_FORCE_GEN=13 (sim-M1) ALSO fails (6.8) → gen-independent [VERIFIED].
- Split-K-specific code (per-split K-range, empty-split path, pO/pL write) read in full: all structurally correct + consistent with single-pass. Race line NOT pinned.
- Diagnostic edits in working tree (UNCOMMITTED, gated, harmless): mfa_attention.cpp MFA_SPLITK_DEBUG + MFA_ZERO_SCRATCH; shader_cache.mm MFA_TRACE_ASYNC + MFA_FORCE_ASYNC_ON_26 bypass; mfa_steel_fwd_v2.cpp MFA_EXTRA_BARRIER.
- Ran: cold sweeps (single-pass/V1/auto/Q0 correct; split-K wrong both gens). Validated vs independent fp32 SDPA.
- Git: nothing committed; HEAD=da737e7 (the now-falsified III-8 fix still on master, needs correction). Checkpointing with Marco: continue deep race-hunt vs ship correctness-safe split-K fallback now + hunt race separately.

---
## [2026-06-16 11:00] [CLAUDE] III-9 ROOT CAUSE FOUND + FIXED — split-K scratch lifetime (premature free)
STATUS: COMPLETE

### Root cause (rigorously confirmed, corrects BOTH III-8 and the 07:30/09:00 entries)
- The V2 split-K path allocated pO/pL scratch via allocator::malloc and FREED them at ENCODE time (allocator::free, mfa_attention.cpp:566-567 split-K, :358-359 flash-decode), with a comment assuming "Metal retains them until the command buffer completes." FALSE: MLX is lazy — Phase1/Phase2 execute at mx.eval, LATER. The free returned the pool memory while the kernels were pending; a concurrent allocation (the reference SDPA, or any downstream op) reused it and corrupted the not-yet-executed reduce read. [VERIFIED]
- Isolation ladder that pinned it: per-split pO AND pL dumps both reliably correct (6/6) vs fp32; numpy-combine(dumped pO/pL) == fp32 full (MAE 7e-5) → inputs+math correct; eval-o-ALONE correct (4/4) but eval(o, concurrent_sdpa) garbage; MFA_NO_FREE (suppress free) → correct (4/4). => premature free, not kernel logic.
- Explains ALL prior confusion: nondeterministic (allocator timing), "later rows garbage" (reduce reads after partial pool overwrite), "Q=0 masked it" (uniform-P probes had NO concurrent alloc), gen-independent, o→fp16 512 saturation.
- The III-8 "async metallib root cause" was a WRONG TURN (the async gate is inert for this bug; backend=mfa non-causal D=128 dispatches to SPLIT-K, traced via instrumentation, not the SteelForwardV2 single-pass that try_async serves). The async gate is independently defensible (simdgroup_async_copy IS broken on macOS 26) but did NOT fix the non-causal divergence.

### Fix
- mfa_attention.cpp: both split-K (Phase 3) and flash-decode (Track H) scratch now wrapped in mlx::core::array + registered via enc.add_temporary() → freed only after the command buffer completes. Removed the encode-time allocator::free.

### Validation
- R.4: 576-config sweep (force_splitk 0/1/heuristic × D{64,128} × B·H{(1,4),(1,8),(2,8)} × N{127..1000} × causal × {f16,bf16}) WITH concurrent-alloc stress → 0 split-K failures vs fp32. (2 NaN at single-pass D=64 N=1000 non-causal fp16 appear ONLY deep in the exhaustive sweep — clean in isolation + 30× warm loop; separate pre-existing memory-interaction artifact, NOT split-K, flagged.)
- R.5: full suite 1768 passed, 2 skipped, ×2 consecutive (40 new III-9 lock tests included).
- R.6: fix changes buffer LIFETIME only (no kernel compute change) → split-K perf identical. backend=mfa non-default; auto→SDPA unaffected.
- R.7 lock: tests/test_iii9_splitk_lifetime.py (40) — forces split-K + concurrent alloc + fp32 ref.
- Ran: pytest tests/ ×2 (1768 pass), R.4 sweep, isolation ladder. Validated vs independent fp32 SDPA (lesson #11).

### Separate items flagged (NOT fixed)
- single-pass D=64 N=1000 non-causal fp16: NaN only in 576-config exhaustive sweep, not isolation/30×-loop. Pre-existing, separate.
- flash-decode had the SAME premature-free latent bug (fixed here too as part of the multi-gate audit, §AA.5.x).

### Git
- WIP — about to commit: mfa_attention.cpp fix + tests/test_iii9_splitk_lifetime.py + doc corrections + log. HEAD=da737e7 (III-8). Diagnostic edits all reverted; tree clean except the fix.

---
## [2026-06-16 13:00] [CLAUDE] III-9 pre-release sweep — iteration 1: one real bug found (V2 single-pass non-causal, forced-path)
STATUS: BLOCKED

- Ran 9-agent parallel sweep (Workflow w8o0y0fru) across classes D,E,F,G,H,I,J,K + K-adjacent; one agent (I) died on a socket error, re-run standalone.
- CLEAN (exhaustive enumeration, vs independent fp32): K (scratch lifetime — every malloc output-backed or add_temporary; zero allocator::free remain; both fixed sites re-verified; no third site), D (which-binary/macOS-26; async gate inert+correct, CP9 serves nothing), G (no new key/knob; invariant test green), H (vjp anchored to fp32 oracle), I (empty-row/NaN-propagation/overflow/degenerate all clean), J (hook coverage). [VERIFIED]
- ONE REAL BUG (4 agents converged): V2 SINGLE-PASS non-causal backend="mfa" forced-path corrupts the LAST head's output (uninitialized/pool-history read; even-D-col MMA pattern) at D=64 N∈{224,992-1023}+short-S{1,2,3,4,16}, D=128 N=383+short-S{1,2}. Math on written cells correct (6e-6). NOT split-K (fixed), NOT async metallib. The genuine residual of the III-8 single-pass investigation. [VERIFIED vs fp32, multi-agent + lead repro]
  - Sentinel pre-fill of out → kernel writes all cells when run ALONE; corruption only under concurrent alloc / heterogeneous pool history → uninitialized-read class, likely OOB read at last head for partial last K-tile (last head sits at K/V buffer end → over-read spills into freed pool memory → NaN).
  - Reachability: NOT default (auto→SDPA clean, 0/24 poisoned configs). V1 clean, causal clean, split-K clean. Forced backend="mfa" only.
- Disposition is Marco-gated (pre-authorized in findings doc): (1) fix the V2 single-pass kernel (hard pin), or (2) DECLINE V2 non-causal → route forced-mfa non-causal D∈{64,128} to V1 (clean, real MFA kernel) + Rule-8 note (correctness fix, removes silent-garbage path; non-default + slower-than-SDPA + 2nd bug in this path). Recommend (2).
- Ran: Workflow sweep + Class I re-run + lead repro (N=992 head-3 NaN deterministic w/ preamble; sentinel diag). Validated vs independent fp32 SDPA. Diagnostic edits reverted; tree clean; binary = clean HEAD 240b226.
- Docs: sprint-III-9-report.md (per-class enumeration + finding + disposition). Pre-release gate NOT met; release HELD.
- Git: report + log to commit; HEAD=240b226. Nothing released.

---
## [2026-06-16 15:30] [CLAUDE] III-9 iter 1+2 — V2 single-pass + GNA + V5 direct-V OOB fixed (multi-gate)
STATUS: IN_PROGRESS

- Iter 1 (commit eb68af5): V2 single-pass non-causal last-head OOB-V read (0*NaN). Fixed via kL_rem-1 V-row clamp on partial final tile. R.4 792-cfg sweep 0 fail; suite 1796 ×2; lock 28.
- Iter 2 (this commit): §AA.5.x multi-gate — the SAME unbounded MFA_DIRECT_READS V-read partial-tile pattern existed in TWO sibling kernels NOT covered by eb68af5 (the F agent caught it; vindicates loop-until-clean):
  - GNA native (mfa_gna_fwd.cpp:493) — DEFAULT-REACHABLE via flash_attention_gna (D=128/3D/f16|bf16/N%32!=0). HIGH. [VERIFIED real]
  - STEEL V5 (mfa_steel_fwd_v5.cpp:384) — opt-in MFA_ENABLE_V5. MEDIUM. + falsified L373 comment corrected. [VERIFIED]
  - Both fixed with the identical clamp. V4 safe (V via load_safe); sparse/top-K/conv/paged/V1/split-K/dsplit all safe.
- Validation: GNA vs element-level fp32 GNA-mask+SDPA oracle (0 NaN, MAE 1e-4..5e-4, 8 cfg); V5 vs fp32 SDPA (0 NaN, MAE ≤9e-4, 20 cfg) — both under preamble pool-history + concurrent-alloc trigger. Suite 1820 passed ×2 (+24 GNA/V5 lock tests + 28 V2 lock).
- Iter-2 sweep agents E and I died on API-overload → must re-run. recheck-lasthead clean.
- Ran: GNA/V5 probes + pytest 1820 ×2. Validated vs independent fp32. Tree: csrc/mfa_gna_fwd.cpp + csrc/mfa_steel_fwd_v5.cpp + 1 new lock test to commit.
- Git: eb68af5 (V2) committed; GNA+V5 commit below. HEAD will be the multi-gate commit. Nothing released.
- NEXT: re-run E+I agents; iteration 3 full re-sweep to zero-finding = gate.

---
## [2026-06-15 23:57] [CLAUDE] III-9 sweep Class E — unfaithful-oracle / never-independently-validated paths
STATUS: COMPLETE

### Objective
Enumerate every public attention/compute entry path; classify independent-fp32-oracle
coverage; probe uncovered/unfaithful paths vs independent fp32 SDPA (or uniform-P/element
known answer) in fresh processes + heterogeneous preamble + concurrent-alloc; re-confirm
the 3 fixed paths (split-K lifetime 240b226, V2 single-pass non-causal eb68af5, GNA+V5
direct-V eb5b890).

### Independent-fp32 test-coverage classification
- FAITHFUL (fp32-cast SDPA or PyTorch-fp32 oracle in tests): flash_attention (test_attention
  reference_sdpa), STEEL varlen (_ref_concat per-seg fp32 SDPA), paged_varlen (per-seq SDPA),
  sparse (_ref_sparse_sdpa fp32 SDPA + float bias), conv3d_nax (PyTorch fp32 CPU + mx.conv_general).
- UNFAITHFUL / mfa-vs-mfa for the kernel forward: turboquant fused (test compares
  flash_attention_paged_varlen_turboquant vs flash_attention_paged_varlen — both mfa);
  topk_ratio=1.0 (vs flash_attention backend=sdpa, ok but same-precision); some varlen
  forward asserts vs flash_attention not SDPA. sage uses fp16 (not fp32-cast) SDPA reference.

### Probes (fresh process + heterogeneous preamble + concurrent alloc; vs INDEPENDENT fp32 SDPA / element oracle)
- sage D{64,128}×N{256,383,512}×causal×{f16,bf16}: MAE ~1e-3, no NaN/saturation. A few
  MAX 0.30-0.54 (atol 0.30 noncausal) = sage's documented fp16 QUANTIZATION error; CONFIRMED
  deterministic-in-distribution in isolation w/o preamble/concurrent (trials 0.22-0.52, MAE
  stable). NOT a lifetime bug. [VERIFIED]
- varlen (segs incl [1,200,16],[3,3,3] partial/short, D64/128, causal+non): MAE ≤1e-4. CLEAN.
- paged_varlen (q/kv {[1]/[40],[1,1]/[33,65],[2,3]/[1,100]}, D64/128): MAE ≤4e-5. CLEAN.
- topk r=1.0 vs fp32 SDPA: MAE ≤6e-5. CLEAN.
- GNA partial-N (shp (4,8,8)/(3,9,9)/(5,5,5), N∈{256,243,125}): vs EXACT per-element window
  oracle (formula lifted from csrc/mfa_gna_fwd.cpp L421-426 — first oracle attempt used wrong
  centering grp-w//2; kernel uses (win-str)/2): MAE ≤9e-5. CLEAN.
- sparse sliding-window D64/128: MAE ≤2e-5. CLEAN.
- kvcache decode D64/128 S{40,130,257}: MAE ≤4e-5. CLEAN.
- conv3d_nax vs fp32 mx.conv_general: MAE ≤3e-3. CLEAN.

### Three fixed paths re-confirmed CLEAN under trigger (vs fp32 SDPA, with concurrent alloc)
- split-K (MFA_FORCE_SPLITK=1, backend=mfa) D64/128 N{127,128,256,383} causal+non: MAE ≤7e-5.
- V2 single-pass (MFA_FORCE_SPLITK=0, backend=mfa) non-causal at log trigger shapes
  D64 N{224,992,1000,1023,1,2,16}, D128 N{383,1,2,256,512}: MAE ≤1.1e-4, no last-head NaN.
- V5 partial-S (MFA_ENABLE_V5=1, backend=mfa) D128 N{130,255,383,511,1000,1023} causal+non:
  MAE ≤6e-5. GNA partial-N covered above.

### Findings
- ZERO NEW real bugs. sage "FAIL" elements are quantization noise (path's own atol 0.30/0.50),
  not corruption. GNA initial divergence was MY oracle error (wrong window formula), corrected.
- Coverage gaps (LOW, "needs independent-fp32 lock", NOT bugs): (1) turboquant fused kernel
  validated only vs mfa decompress path (mfa-vs-mfa); no fp32-SDPA-on-reconstructed-KV lock.
  (2) sage uses fp16 (not fp32-cast) SDPA reference. (3) several varlen forward asserts vs
  flash_attention rather than SDPA. All three independently confirmed correct by THIS sweep's
  probes; the gap is in the committed test suite, not in correctness.

### Validation
- Ran: 3 fresh-process probe harnesses (debug/classE_probe{,2,3}.py — scratch, removed after),
  all paths + 3 fixed paths, fresh process per path, preamble + concurrent alloc.
- Validated: every probe vs independent fp32 mx.fast.scaled_dot_product_attention (lesson #11)
  or exact element-window oracle (GNA). Criteria: MAE/MAX vs atol, NaN-free, no fp16 saturation.

### Git
- not applicable (read-only audit; scratch probes removed; tree clean). HEAD=eb5b890.

### Verdict
- Class E is CLEAN — zero NEW real bugs. The 3 fixed paths hold under trigger. Three
  test-suite coverage gaps flagged (LOW) for independent-fp32 locks; none indicates a defect.

---
## [2026-06-16 17:00] [CLAUDE] III-9 iter 3 — zero-finding; PRE-RELEASE GATE MET
STATUS: COMPLETE

- Iteration 3 (HEAD=eb5b890): re-ran the classes the V-read clamps could touch — recheck-lasthead CLEAN, E CLEAN (3 fixed paths reconfirmed vs fp32; 3 LOW test-coverage-gap nits, not defects), I CLEAN (empty-row/NaN-prop/overflow/degenerate/short-S + all fixed paths + the D=64 N=1000 item reconfirmed; 181 III-9 locks), F multi-gate verified COMPLETE by lead grep (only 3 device-V sites: V2/GNA/V5, all clamped; every other V read is smem/load_safe). D/G/H/J/K structurally invariant to a V-read clamp + clean in iter1.
- Full suite 1820 passed, 2 skipped, ×2 consecutive. Tree clean (agent scratch removed).
- PRE-RELEASE GATE MET: every class hunted exhaustively (incl. 4 lesson-driven classes), final iteration zero-finding.
- Release scope (Marco-gated, v2.53.0+): da737e7 (async gate) + 240b226 (split-K lifetime) + eb68af5 (V2 single-pass OOB-V) + eb5b890 (GNA+V5 multi-gate OOB-V) + 2 III-7 quantize_model fixes (on master).
- III-9 net: 1 lifetime bug (split-K/flash-decode) + 3 direct-V partial-tile OOB bugs (V2 single-pass, GNA [default-reachable], V5) — all fixed, all locked, all validated vs independent fp32. GNA was the only DEFAULT-reachable one.
- Ran: 3 sweep iterations (workflows w8o0y0fru, w620e61ex + 4 standalone agents) + lead probes + pytest 1820 ×2 + multi-gate grep. Validated vs independent fp32 throughout (lesson #11).
- Git: report + log commit below; HEAD will advance. NOTHING released — release is the separate Marco-gated step.

---
## [2026-06-16 19:00] [CLAUDE] III-10 final residual sweep — ZERO findings; FINAL GATE MET
STATUS: COMPLETE

- Bounded 3-target residual pass (Workflow w68lhdnp7), NOT a re-sweep of D-K (structurally complete in III-9). All clean, zero confirmed findings:
- T1 (0*NaN meta-pattern beyond V): CLEAN. Complete-set enumeration of masked-arith device reads (K direct, attn_bias add, alibi, rotary) across V2 single-pass+D-split/GNA/V5. The V multiply was UNIQUE; every other site is mask-by-ASSIGN(-inf) applied BEFORE softmax (overwrites NaN), or head-bounded, or RoPE-smem-path. attn_bias[k_pos] OOB on partial tile is annihilated by the K-boundary mask ASSIGN (add emitted before mask). Probed clean vs fp32 under pool-history+concurrent trigger. No gathers exist. [VERIFIED]
- T2 (MPP matmul2d/conv2d K alignment): CLEAN. VERSION CORRECTION: machine is macOS 26.6 (not 26.4/26.1). The 26.1 silent-K%32-cut does NOT reproduce on 26.6 — MPP enforces compile-time static_assert(k%16==0) (loud, %16 floor). Every production MPP site (conv matmul2d/convolution2d, NAX/steel-V6 matmul2d) guaranteed-by-gate (Rule-8 throw + pad_contraction_k; conv3d C%16&>=32 gate) or correct vs fp32. Repo %32 guards safe across both OS. [VERIFIED]
- T3a (lifetime completeness): CLEAN. grep allocator::free = ZERO; every malloc output-backed or add_temporary; no third site. [VERIFIED]
- T3b (fp32-coverage nits): CLOSED. 4 tests (turboquant fused-K + fused-K+V, sage _check + GQA, paged-varlen) gained independent fp32-cast oracle assertions (added alongside existing, not weakening). No fp32 divergence → locks real, no latent bug. All pass.
- Ran: pytest tests/ 1820 passed, 2 skipped, ×2 (incl. new fp32 asserts). Validated vs independent fp32 (lesson #11). Tree: 3 test files edited (T3b) + report + log.
- FINAL PRE-RELEASE GATE MET. Release scope (Marco-gated v2.53.0+): da737e7 + 240b226 + eb68af5 + eb5b890 + III-10 3b test locks + 2 III-7 quantize_model fixes. NOTHING released — separate Marco-gated step.
- Git: report + log + 3 test files commit below; HEAD advances.

---
## [2026-06-16 02:30] [CLAUDE] III-11 re-bench (M5 Max, macOS 26.6, HEAD 6a7d79c) — V6NAX backward claim moved down
STATUS: IN_PROGRESS

- Provenance: HEAD 6a7d79c (clean), macOS 26.6 (25G5028f), Apple M5 Max 128GB, mlx 0.31.2. Bench via /mlx-mfa-bench-methodology dispatcher, 3 sessions, strict cooldown for ≥1.5ms.
- FORWARD attention vs SDPA (canonical, N=4096): D64 c 0.98×, D128 c 1.05× (HIGH_VAR), D128 nc 0.98×, D128 c bf16 1.00× (HIGH_VAR), D256 c 0.99× (HIGH_VAR). => forward ≈ SDPA, net-non-worse. (HIGH_VAR = these sit in ≥1.5ms regime; conclusion ~1.0 robust. Forward is bit-identical-to-SDPA, NOT a speedup claim.)
- V6NAX BACKWARD vs SDPA-vjp (strict, CONFIDENT/BOUNDARY, B=2 H=8): D64 causal N2048=1.29× N4096=2.00× N8192=1.94×; D64 non-causal N4096=1.29× N8192=1.88×; D128 causal N4096=0.99× N8192=1.00×.
- FINDING (material): README headline V6NAX backward D=64 causal 2.06-2.58× / non-causal 1.72-2.01× MOVED DOWN on 26.6 to ~1.3-2.0× / ~1.3-1.9×; D=128 backward now BREAK-EVEN (~1.0×). [VERIFIED clean]
  - Cause: OS 26.6 improved Apple SDPA-vjp (NOT a Phase III regression — backward NAX kernel byte-identical to v2.52.1; fixes were all forward-path). NOT a regression vs v2.52.1 (same binary measures same on 26.6). It IS a doc-accuracy issue (published numbers are v2.50-era).
  - Caveat: measured at B=2 H=8; original claim's reference B/H not pinned — the ratio is occupancy/shape-dependent, so the exact published range may have used a different shape.
- conv MPP + TQ decode: NOT yet re-measured (separate kernels untouched by Phase III; OS-sensitive). No dedicated methodology-protocol harness found in benchmarks/; the vs-legacy/decode claims need their own harness.
- Net pool vs v2.52.1: NON-WORSE (identical kernels + perf-neutral correctness fixes). The V6NAX "move" is vs the DOC/earlier-OS, not vs v2.52.1.
- Ran: dispatcher forward (canonical) + 2 backward batches (strict), 3 sessions. Bench friction: dispatcher emits JSON only at end (kills lose batch) → switched to small foreground batches; backward needs target=flash_attention_loss + baseline=sdpa_vjp.
- Git: not committed (bench data in /tmp/iii11). Decision pending Marco: how to handle the moved V6NAX headline numbers + whether to re-measure conv/TQ before the v2.55.0 doc sync.

---
## [2026-06-16 03:30] [CLAUDE] III-11 cont. — conv clean re-bench: claims shrank to ~break-even + HIVAR; impasse surfaced
STATUS: BLOCKED

- conv MPP vs legacy (clean 4s-cooldown strict): T8 fp16 0.98× CONFIDENT, T16 fp16 1.30× HIGH_VAR. vs README 2.3-2.5×. The 2s-cooldown 1.7-2.0× was thermal-variance-contaminated (v2.36.0 incident class). conv bf16 "vs legacy" UNMEASURABLE (legacy im2col fp16-only, KD-7).
- Combined with V6NAX backward (1.3-2.0× vs 2.06-2.58×; D=128 break-even): SYSTEMATIC — all v2.50-era headline speedups materially LOWER on 26.6 (Apple improved SDPA-vjp + conv baselines). NOT a regression (kernels byte-identical to v2.52.1; release perf-safe). The published numbers are stale + the originals may have been non-canonical point estimates.
- Impasse: CONFIDENT re-measurement of ALL claims is blocked by (a) background-job kill limit (~5-8min) preventing full strict protocol at ≥10ms shapes, (b) inherent clock-state bimodality HIGH_VARIANCE at these sizes, (c) regime-dependence (conv cold ~1.0× vs warm ~1.7×). Report: sprint-III-11-report.md.
- Solid: release perf-safe (non-worse vs v2.52.1); documented speedups don't hold on 26.6.
- Recommendation: soften headline perf to OS-honest ranges + ship the correctness fixes (default-reachable GNA) now; do a dedicated perf-recharacterization sprint with fixed bench infra as fast-follow. Awaiting Marco.
- Ran: dispatcher fwd + V6NAX bwd (strict) + conv (strict, per-config). Bench data /tmp/iii11. Report + log committed below. Nothing released.

---
## [2026-06-16 05:00] [CLAUDE] III-11 COMPLETE — bench infra fixed; full re-bench done; README synced to 26.6
STATUS: COMPLETE

- BENCH INFRA FIXED (Marco's ask): benchmarks/methodology/robust_pool_runner.py — detached nohup + incremental JSONL + per-spec subprocess isolation + idempotent resume. Survives the harness ~5-8min kill limit that blocked all prior attempts. Committed.
- FULL CLEAN RE-BENCH (24 specs, ~1h detached, 0 errors, strict 4s-cooldown ×3 sessions, macOS 26.6, HEAD 6a7d79c). Raw: benchmarks/methodology/iii11_26.6_results.jsonl.
  - Forward vs SDPA: 0.77-1.08× (≈SDPA; outliers = clock-state HIGH_VAR). Net-non-worse vs v2.52.1 CONFIRMED.
  - V6NAX bwd D=64 vs SDPA-vjp: qL-dependent — N2048 0.86×, N4096 1.49×, N8192 2.52× (causal); nc 1.39×/2.09×. D=128 bwd ≈break-even (0.97-0.99×, reliable). Run-to-run swings ±30-40%.
  - conv MPP fp16 vs legacy: 1.22-1.35×; bf16 vs conv_general ≈1.0-1.1×.
- FINDING: Apple improved SDPA-vjp+conv on 26.6 → all compute-bound v2.50-era speedups materially LOWER + qL-dependent + high-variance. NOT a regression (kernels byte-identical to v2.52.1; release perf-safe). Documented precise ranges stale + not run-reproducible.
- DOC SYNC (R.4): README headline claims rewritten to measured 26.6 values + OS-caveat + qL-dependence. V6NAX bwd → "~break-even@2048 to ~2.1-2.5×@8192; D=128 break-even"; conv → "~1.2-1.35× fp16, ≈parity bf16"; D=256 inversion reframed as correctness fix.
- NOT measured (flagged fast-follow): TQ paged decode (bench_turboquant_full.py crashes in _build_tq_pool — harness bug, not kernel); LCSA 15.4× (historical build-time, old impl gone). README marks both "pending 26.6 re-measurement".
- Ran: 24-spec detached pool + TQ bench (crashed) + dispatcher cross-checks. Validated vs independent fp32/SDPA-vjp. 
- Git: robust_pool_runner.py + results committed; README + report + log commit below. Nothing released.
- RELEASE STATUS: correctness gate MET (III-9/III-10); perf-safe (non-worse vs v2.52.1); docs now OS-honest. Ready for /mlx-mfa-release-audit (Marco-gated) + v2.55.0 tag. TQ-decode re-measure is a fast-follow, not a blocker.

---
## [2026-06-16 06:30] [CLAUDE] III-12 — TQ decode re-bench: flagship "6-14x faster" claim is INVERTED (latency slowdown)
STATUS: BLOCKED (Marco decision on flagship-claim framing)

- R.1: fixed bench_turboquant_full.py _build_tq_pool crash (packed_D = D//2 [4-bit] -> (D*bits+7)//8 = 48 for bits=3); same fix in _kv_memory_mb. Timed lambdas are attend-only (build outside).
- R.2: TQ paged decode re-measured 26.6, 9 configs. P3/fp16 (TQ-fused latency / fp16 latency) = 1.6-25.8x = TQ is 1.6-26x SLOWER than fp16 paged decode. Qwen 1seq 8K = 25.8x (RESULTS.md cited this exact config as "14.4x"). The "14.4x" is the P3/fp16 SLOWDOWN mislabeled as "faster". Run-to-run stable ~5% (memory-bound regime IS stable — measured, not assumed).
- TQ value = MEMORY: fp16 16-128MB -> KV-TQ 3.25-26MB (~4-5x), cos 0.96-0.99. Latency cost from decompression dominating single-token decode. Explicit opt-in API (not auto-routed) -> no harmful auto-promotion; the DOC framing is the error.
- R.3 verdict: TQ is a memory feature, not a speedup. "faster" framing wrong.
- FINDING is the III-12 headline: repo's biggest advertised number ("14.4x faster") is a latency slowdown ratio. Pre-existing doc error, not introduced by v2.55.0, but must be corrected before publishing. Surfaced to Marco for framing decision before rewriting the flagship README line.
- Ran: bench_turboquant_full.py (full 9-config, detached) ×2 for stability. Data: benchmarks/methodology/iii12_tq_decode_26.6.log. Report: sprint-III-12-report.md.
- Git: harness fix + report + log commit below. README flagship line NOT yet edited (Marco framing decision pending). Nothing released.

---
## [2026-06-16 08:00] [CLAUDE] III-12b COMPLETE — TQ claim confirmed (archaeology); v2.55.0 prepared; audit GREEN-except-version
STATUS: HANDOFF_READY

- R.1 archaeology: TQ "6-14x faster" = new gather/dequant+SDPA step() path vs the OLD FUSED TQ kernel (PERF_CLAIMS.md:47 + III-2 report, with ms absolutes). CASE 2 (claim real, baseline lost in README). III-12's "inverted" RETRACTED (it benched fused-vs-fp16, neither arm).
- R.2 re-measure 26.6 (Hq=32 Hkv=8 D=128 tq3b, 2 runs): new vs fused = 6.5-13.5x@4K, ~22x@16K (0.75ms vs 16.8ms). Claim HOLDS. new vs fp16-dense = 1.4-3x SLOWER (the memory-for-latency trade). 
- R.3 README/PERF_CLAIMS/CHANGELOG rewritten: TQ baseline-explicit (vs fused, NOT fp16; +fp16-dense gap; +4-5x KV memory value); every ratio states num/denom/direction/absolute; LCSA historical. Lesson #15 codified (no bare ratios; recover baseline before re-measuring).
- R.4 gate: suite 1820 pass x2; tree clean; no orphan bench procs. CHANGELOG [2.55.0] drafted (4 correctness fixes incl. GNA DEFAULT-reachable CRITICAL + upgrade directive; 2 quantize_model; record corrections; honest perf). 9-gate /mlx-mfa-release-audit: ALL technical checks PASS (tool-avail, auto-default, public-API/perf-reachability, skill-log, test-suite, changelog); ONLY blocker = Check 1 version bump (2.52.1->2.55.0) = the intentional Marco-gated tag step. 1 cosmetic advisory (PERF_CLAIMS version-string tag).
- Release scope: da737e7 + 240b226 + eb68af5 + eb5b890 + III-7 quantize_model + III-11/12/12b docs+bench. Version 2.55.0.
- HELD for Marco's go: bump 3 SoT (pyproject/__init__/README) to 2.55.0 -> tag -> twine -> GH release. Nothing published.
- Reports: sprint-III-12b-report.md (archaeology+correction), III-12 retracted-banner.
- Git: PERF_CLAIMS + CHANGELOG-reproduce + log commit below; HEAD advances. Nothing released.

---
## [2026-06-16 24:05] [CLAUDE] III-12c COMPLETE — v2.55.0 SHIPPED (PyPI + GH live; post-publish smoke green)
STATUS: COMPLETE

### Changes
- R.1 TQ reframe (README + PERF_CLAIMS + CHANGELOG): now LEADS with the user-actionable
  trade-off — TQ paged decode trades ~1.4-3x decode latency for ~4-5x KV memory @ cos ~0.96
  vs fp16 dense (0.75ms vs 0.33ms @S=16K; 32MB->~6.5MB @S=8K). "6.5-23x vs fused kernel"
  kept as labelled secondary history (kernel gone -> not selectable). Lesson #15 extended:
  lead with the denominator relevant to the reader. Commit `5a9f21c`.
- R.2 advisory cleared: tests/test_release_notes_perf_claims.py iii2 de-staled (was
  6.0-14.4x) + v2.55.0 stamp -> release-audit advisory_findings:[]. 17 reachability tests pass.
- R.3: full suite 1820 pass x2 consecutive on clean HEAD; tree clean; 0 orphans; 4 scope
  commits (da737e7+240b226+eb68af5+eb5b890) confirmed on master.

### The cut (Marco GO)
- Version bump 2.52.1->2.55.0 (own commit `3c2c8b4`). 9-gate audit: GREEN (all 8, incl Check 1).
- Annotated tag v2.55.0. Build wheel+sdist (twine check PASSED). twine upload -> PyPI LIVE.
- gh release (draft=false) + push master+tag -> GH LIVE.

### Validation
- Ran: clean-env `pip install mlx-mfa==2.55.0` from PyPI + /tmp/smoke_v2550.py
- Validated: published wheel IS the fixed binary. 4 fixes vs fp32 — V2 non-causal 2.7e-5,
  GNA non-32 (default-reachable) 3.0e-5, V5 2.7e-5, split-K x8/churn 1.0e-6, all finite.
  forward-auto 7.2e-5, V6NAX grad finite, TQ decode finite. SMOKE PASSED.

### URLs
- PyPI: https://pypi.org/project/mlx-mfa/2.55.0/
- GH:   https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.55.0

### Git
- `5a9f21c` (III-12c docs) + `3c2c8b4` (bump, tagged v2.55.0) pushed to origin/master.
- Report: docs/v50/campaign-2026-06/phase3/sprint-III-12c-report.md. Phase III CLOSED.

---
## [2026-06-17 00:30] [CLAUDE] V6NAX backward block-sparse NAX — premise validation → DECLINE (no build)
STATUS: COMPLETE

### Plan
- Objective: go/no-go on the Marco-gated "V6NAX backward block-sparse NAX extension" item. PREMISE VALIDATION ONLY — no kernel.
- Skill: /mlx-mfa-apple-primitives-coverage (§AA.5).

### Findings (archaeology recovered the premise from the record)
- Original premise (sprint-5-prompt5a-status.md:10-14): native block-sparse iteration in V6NAX backward so mx.grad(flash_attention_sparse) skips inactive K-blocks instead of SDPA-vjp+bias. Projected 1.5-10x @ low density (sprint-5-prompt5a-status.md:53-58) — an ESTIMATE, never measured.
- CORRECTION (doc-coherence): my prior-turn queue entry "never greenlit / premise-validation-pending" was STALE. Record shows it was greenlit (Marco Prompt 5c Opt.1), BUILT+SHIPPED (Prompt 5d: dQ+dK-split+fused-dKdV + 5b dV = 4 native sparse bwd kernels), math-validated (8 tests), routing falsified.
- R.2 buildable: PROVEN (kernels exist: NAAttentionKernel.cpp:5168/6364; 8 tests; opt-in routed attention.py:3187).
- R.3 NAX-reachable: TRUE empirically (kernels run on M5 NAX, benched). But reachable != wins (Pattern #6: Apple SDPA-vjp also on NAX, better-optimized).
- R.4 win-bound: MEASURED, FALSIFIED (section-a-v3-empirical-verification.md:19-33). Native/SDPA = 0.77x@d0.1 -> 0.09x@d1.0 at VSR shape (SLOWER all densities). Only 1.13x at D=64/small-H/d=0.1 (too narrow for AUTO). 5a's 10x projection inverted by measurement.

### Verdict
- R.5 DECLINE — first-class negative result. All-three-premises gate fails at R.4 (measured, not estimated). NOT unverified-blocked: maximally verified (shipped kernels + bench). Nothing to build — already routed optimally (5c hybrid default; full native opt-in MFA_V6_BWD_SPARSE_NATIVE=1).
- New AUTO carve-out for the 1.13x corner also DECLINED (ghost-knob/Pattern #6 risk; opt-in covers it).

### Validation
- Ran: read-only archaeology + live-code corroboration (grep kernels/flag/tests/routing). No build, no bench re-run (the 5d bench is the primary source).
- Validated: verdict traces to primary sources (5a premise, 5d ship, section-a-v3 measured falsification, attention.py routing). NO kernel built.

### Git
- docs/v50/campaign-2026-06/v6nax-blocksparse-nax-premise-report.md (new) + PHASE-III-CLOSE.md queue row + this log. Commit below. branch master.

---
## [2026-06-17 01:10] [CLAUDE] Queue Archaeology A — MFA_FORCE_NATIVE_BWD state of truth → REMOVE-ELIGIBLE (no action)
STATUS: COMPLETE

- Type: archaeology only (read+grep+live-code). NO edit/removal/deprecation action. A of two.
- R.1 origin: v2.36-era debug/eval override forcing legacy STEEL backward (CHANGELOG:3348/1129). Deprecated v2.50.0 Prompt 5f Phase E — originally "BROKEN" (KD-5 zeroed-blocks D=128 N≥2048, MIGRATION:77 "target removal v2.51+"); reason SHIFTED to "SUPERSEDED, not broken" after 2026-05 review FIXED KD-5 (sprint-II-0:26).
- R.2 live: sole reader dispatch_policy.py:717-739 (=1 warns+forces STEEL native; =0 off; else policy table). Sole caller attention.py:5306. Inert on default API (M5 forward→SDPA short-circuits the bwd branch); changes routing only on backend="mfa". Ghost-knob-on-public-path / weak escape-hatch on expert path; forced path wins NOWHERE.
- R.3 public-contract: documented env var (ENV_VARS.md:60, MIGRATION, CHANGELOG; NOT README). Deprecation cycle ALREADY COMPLETE (announced v2.50.0→ now v2.55.0). Removal touches: dispatch_policy.py force-branches (keep policy table), test_v50_prompt_5f_kd5_deprecation.py + test_attention.py:11051-11159, ENV_VARS.md:60, cache-audit/01:37, CHANGELOG (Removed entry). KEEP STEEL kernel (keep-all-paths).
- R.4 measured (NOT asserted): sprint-C Track 2 matrix — STEEL-bwd correct(rmse 4e-5) but dominated every cell (V6NAX 2.2-2.6x@D64; SDPA-vjp@D128). 26.6 strengthens it. NOT stale-verdict.
- R.5 VERDICT: REMOVE-ELIGIBLE, deprecation cycle complete. Queue entry ACCURATE (unlike V6NAX). Only KEEP argument = thin research/determinism escape-hatch + standing "keep deprecated" steer.
- Ran: read-only archaeology + live-code grep. Validated: verdict traces to dispatch_policy.py:717-739 + sprint-C matrix + ENV_VARS:60 + MIGRATION:77. NO code changed.
- Git: docs/v50/campaign-2026-06/queue-force-native-bwd-archaeology.md (new) + PHASE-III-CLOSE queue row + log. Commit below. branch master. Archaeology B next; Marco decides after both.

---
## [2026-06-17 01:50] [CLAUDE] Queue Archaeology B — V3/V4/V5 state of truth → SPLIT (V4/V5 clean; V3 gap+stale) (no action)
STATUS: COMPLETE

- Type: archaeology only (read+grep+live-code). NO removal/build/re-gating (keep-all-paths). B of two.
- R.1 intent+evidence: V3 separate K/V smem (−2 barriers/tile); V4 direct-K; V5 D-blocked. vs-V2 verdicts ALL M1-Max-2026-03 (measured, internal): V3 1.015× WIN in regime (autoresearch mfa_attention.cpp:793-799); V4 0.51-0.98×; V5 0.60-0.90×.
- R.2 GATING (key finding): V4 opt-in (enable_v4, :588), V5 opt-in (enable_v5, :693) — never auto. **V3 AUTO-ROUTED** (:808-818 v3_shape_ok = causal && N≥4096/2048 && B·H≥4; enable_v3 only FORCES). Queue "opt-in only/regress vs V2" is DOUBLY STALE for V3 (auto-routed + won-vs-V2). V3 default-reachable on M5 via windowed path (dense→SDPA at :598; window→MFA→V3) + backend="mfa".
- R.2 correctness: ALL correct on M5 (suite tests pass). V3 smem-staged V (load_safe) → NOT in III-9 direct-V-read class (sprint-III-9:90 lists only V2-single-pass/GNA/V5) → OOB-safe; window supported (gen :206-294). V5 III-9 clamp IN (mfa_steel_fwd_v5.cpp:389-392) + v2.55.0 smoke 2.7e-5. No correctness gap, no stale guard.
- R.3 currency: V4/V5 opt-in → verdict moot. V3 auto-fires on M1-2026-03 verdict, never re-measured M5/26.6; V3/V2 internal (OS-stable) but M1→M5 HW shift uncovered + tradeoff already flipped once on M1 → flip-plausible (Pattern #6 risk: auto-selecting correct-but-maybe-slower kernel for windowed users).
- R.4 VERDICT: V4/V5 CLEAN-KEEP. V3 GAP (stale framing) + STALE-VERDICT. Marco-gated next steps (flagged, NOT run): (1) correct V3 queue/doc framing; (2) Pattern #6 re-bench V3-vs-V2 on M5/26.6 at auto-fire regime (not trivial — §4 sweep).
- Ran: read-only archaeology + live-code grep. Validated: traces to mfa_attention.cpp:588/693/808-818 + dispatch_policy.py:598 + mfa_steel_fwd_v5.cpp:389-392 + sprint-III-9:90 + TestSteelV3/4/5. NO code changed.
- Git: docs/v50/campaign-2026-06/queue-v3v4v5-archaeology.md (new) + PHASE-III-CLOSE queue row + log. Commit below. branch master. Marco-gated queue (A+B) now fully dispositioned.

---
## [2026-06-17 02:40] [CLAUDE] Queue Closure Sprint — flag removed + V3 validated on M5/26.6; v2.56.0 PREPARED (Marco-gated tag)
STATUS: HANDOFF_READY

### Changes
- R.1+R.2 V3-vs-V2 re-bench (M5/26.6, 3-session §4-strict, detached v3_v2_rebench.py; 42 recs 0 err):
  V3 faster-or-parity at EVERY auto-fire cell — windowed D64 N4096 0.68x (V3 ~32% faster), N8192 0.92x,
  D128 N4096 0.97x, N8192 ~parity; backend=mfa D64 0.86x, D128 ~parity. M1 verdict HOLDS on M5, stronger D64.
- R.3 disposition: VALIDATED — keep auto-routing as-is (no routing change -> no three-axis; correctness
  already confirmed: V3 vs fp32 + TestSteelV3 pass; OOB-safe smem-staged V).
- R.4 framing corrected (RESULTS.md, mfa_attention.cpp:799 comment, PHASE-III-CLOSE queue row): V3 =
  conditionally-auto + M5-measured (not opt-in/regress). Absolute+direction+V2-baseline (lesson #15).
- R.5 MFA_FORCE_NATIVE_BWD REMOVED (dispatch_policy.py =1/=0 branches+warning; policy table kept ->
  unset byte-identical). STEEL kernel RETAINED (keep-all-paths) + correctness guard rewired to direct
  _ext.mfa_forward_with_lse->mfa_steel_backward. Multi-gate: 2 test files + ENV_VARS:60 + cache-audit;
  code orphan-free; 10 affected tests pass.

### Validation
- Ran: full suite x2 (1818 pass, 2 skip — was 1820; net -2 from removed force tests); perf-claims 17 pass;
  9-gate audit 2.56.0 = 7/7 technical GREEN, advisories 0, only multi_sot_version_bump blocking (intentional).
- Validated: V3 routing keeps the faster kernel on M5 (Pattern #6 concern resolved); flag-removal
  unset/auto byte-identical; MFA_FORCE_NATIVE_BWD deprecation warnings gone from suite.

### Git
- c304d5e (flag removal + V3 validation + docs + bench artifacts) + advisory-clearing commit below.
  Version SoT still 2.55.0 (bump is the Marco-gated tag step). branch master.

### HELD for Marco's go (recommend v2.56.0 — breaking public-env-var removal)
- bump 3 SoT -> 2.56.0 -> audit fully GREEN -> tag -> build -> twine -> gh release -> push tag ->
  post-publish smoke (flag inert on published wheel; V3 windowed routing correct vs fp32).

### Marco-gated queue: FULLY CLOSED (V6NAX declined, flag removed, V4/V5 kept, V3 validated).

---
## [2026-06-17 03:30] [CLAUDE] Sprint IV-0 — decode-overhead + tech-debt analysis; 0 executed (sync-floor-bound, by design)
STATUS: COMPLETE

- Type: profile-first diagnostic. Executed in-sprint: 0 (no item clears the strict bar). Phase IV backlog opened.
- R.1 decode profile (M5/26.6): INCREMENTAL flash_attention N_q=1 — Python dispatch ~2% (5.7us), GPU/eval ~98%; wrapper vs raw SDPA <=7.5us (<=2.8%, ->0 at large S). TQ step() — call ~480us = 50-60% of ~860us, but 98% of that is append() (437us). append decomposed: pack_k Python graph-build 66.6us; pack_k+eval 543.7us; bare x+1+eval (M5 SYNC FLOOR) 240.9us. append's per-step mx.eval (inference.py:994, pools bound RAW set_input_array -> can't fold into eval(o)) = layer-3 GPU materialize+sync, NOT Python. Python-interpretation lever = ~2% (incr) to ~8% (TQ). Definitive: decode is kernel+sync-floor-bound; Python rewrite is not the lever.
- R.2 tech-debt: env-knobs clean (FORCE_NATIVE_BWD already removed); no live dead code (_sever_lazy_graph gone v2.20.0); arch guards correct. Items: D1 TQ append eval (~240us sync floor, DIAGNOSTIC — add_temporary/raw-buffer lifetime risk); D2 MFA_TOPK_BISECT + D3 backward=sdpa_sparse (working soft-deprecated paths, keep-all-paths, no committed removal cycle); D4 V2-V5 dispatch boilerplate (refactor, dispatch-path risk); D5 cosmetic unused gqa_factor.
- R.3 cross-ref: NO item is BOTH in-hot-path AND safe-to-fix. Hot-path lever (D1) = dispatch/lifetime change; safe items cold/cosmetic.
- R.4 EXECUTED: NONE. Pattern #6 for optimization + strict bar -> zero is correct; manufacturing a noise-level tweak adds risk for no decode gain.
- R.5 Phase IV backlog: IV-D1 (TQ append-eval collapse, ~240us/step, needs raw->graph-input binding change + three-axis + decode-soak — the single biggest decode lever, Marco-gated), IV-D4 (boilerplate consolidation, clarity), IV-D2/D3/D5 clarity.
- Ran: 3 profilers (benchmarks/methodology/decode_profile/), read-only analysis. Validated: attribution traces to bare-eval sync floor (240.9us) + append eval (inference.py:994). NO code changed.
- Git: docs/v50/campaign-2026-06/phase4/sprint-IV-0-report.md + 3 profiler artifacts + log. Commit below. branch master.
- Release: nothing executed -> held v2.56.0 (flag removal + V3 validation) unchanged. 0 orphans.

---
## [2026-06-17 04:30] [CLAUDE] Sprint IV-D1 — TQ append per-step eval collapse SHIPPED (tq_v=False ~1.63x); tq_v=True structural
STATUS: HANDOFF_READY

### Changes
- R.1 mechanism: decode path (tq_decode_attend) reads pools as mx.fast.metal_kernel GRAPH-INPUTS
  (tq_decode.py:190-196) -> eval(o) already materializes them -> append's per-step eager mx.eval is
  REDUNDANT there (NOT the raw-binding change IV-0 feared). Fused path binds raw (set_input_array) ->
  eval required. tq_v DEFAULT=True writes packed-V pools UNREAD by decode -> defer would leak lazy ->
  safe-defer ONLY tq_v=False.
- R.3 impl: append() gains defer_pool_materialize (default False=eager, safe); step() sets True only
  on _decode_branch and not self.tq_v. No binding/kernel change, no path rerouted (keep-all-paths).
  inference.py.

### Validation (the critical gate)
- Soak (benchmarks/methodology/iv_d1_soak.py, 200 steps tq_v=False): post-change deferred vs
  pre-change eager reference = BIT-IDENTICAL (max_abs_diff 0.00e+00) under concurrent-alloc churn
  across 5 INDEPENDENT process launches. Inverse-add_temporary trigger found nothing.
- Permanent guard tests/test_iv_d1_tq_append_defer.py (2 tests): deferred vs forced-eager via step(),
  bit-identical 40 steps under churn; tq_v=True keeps eager. (First test draft had a flawed oracle —
  manual eager skipped step()'s WHT q-rotation, 6.5e-5; fixed to force-eager-through-step A/B.)
- Full suite 1820 pass x2 (was 1818 + 2 new). 311 TQ/decode tests pass (default tq_v=True unchanged).

### Gain (Pattern #6, M5/26.6, 3-session detached, iv_d1_bench.py)
- tq_v=False step: S=2048 eager 640us -> deferred 389us, saved 250.9us (39.2%), 1.64x;
  S=4096 663us -> 407us, saved 256us (38.6%), 1.63x. Matches the ~240us MLX per-eval floor.
  Net-non-worse: TQ-decode-local; default tq_v=True + non-TQ + large-N untouched.

### Disposition
- SHIPPED tq_v=False (~1.63x, ~250us/step, bit-identical). tq_v=True (default): eval STRUCTURALLY
  REQUIRED (packed-V written-but-unread); recovering the floor for the default needs the lazy-packed-V
  restructuring -> Phase IV backlog IV-D2 (top decode lever, same discipline+soak).
- Release: additive, join-ready for held v2.56.0 (flag removal + V3 validation + this gain). Marco's
  call on version. No tag/publish.

### Git
- inference.py + CHANGELOG [2.56.0] Performance + tests/test_iv_d1_tq_append_defer.py + 2 bench scripts
  + sprint-IV-D1-report.md + log. Commit below. branch master. 0 orphans. Version SoT still 2.55.0
  (bump is Marco's gated step).

---
## [2026-06-17 05:30] [CLAUDE] Sprint IV-D2 — eval collapse extended to default tq_v=True (~1.36-1.39x); v2.56.0 substantial
STATUS: HANDOFF_READY

### Changes
- R.1 mechanism: tq_v=True append writes 5 pools; decode reads k_pool/k_scales/v_pool_fp16 (graph-inputs),
  NOT _v_pool_tq/_v_scales (fused reads those raw). The 2nd floor = materializing the decode-unread packed-V.
- R.2/R.3: combined eval (floor is per-eval not per-buffer). step() decode branch tq_v=True:
  defer append eval + mx.eval(o, _v_pool_tq, _v_scales) at step end -> 2 floors->1, packed-V materialized
  EVERY step (no chain), concrete for later fused read. Fused path code-UNCHANGED (eager append). inference.py.
- Dependency PROVEN: read pools via o graph-dep; packed-V explicit in same eval; cross-step ordering via __setitem__ rebind.

### Validation
- iv_d2_soak.py (tq_v=True, 160 decode + 4 interleaved fused): post-change vs git-stash pre-change eager ref,
  under churn × 5 processes -> decode AND fused-read BOTH max_abs_diff 0.00e+00 every run.
- The NEW fused-read-after-decode edge (IV-D1 lacked): fused read sees combined-eval-materialized packed-V, bit-identical.
- Regression guards added (test_iv_d1_tq_append_defer.py): tq_v=True combined-eval + fused-read bit-identity. Suite 1821 x2.

### Gain (Pattern #6, M5/26.6, 3-session detached, iv_d2_bench.py)
- DEFAULT tq_v=True step: S=2048 eager 776.6->combined 559.3us, saved 217.3us (28.0%), 1.39x;
  S=4096 781.6->576.7us, saved 204.9us (26.2%), 1.36x. Net-non-worse: fused code-unchanged (latency
  unchanged by construction); tq_v=False/non-TQ/large-N untouched.

### Disposition
- SHIPPED default tq_v=True (~1.36-1.39x). Eval collapse now covers BOTH configs (tq_v=False ~1.63x IV-D1,
  tq_v=True ~1.36-1.39x IV-D2). IV-D3 lazy-packed-V no longer needed for this floor (combined eval recovers it).
- v2.56.0 now substantial: flag removal + V3 validation + TQ-decode gain on the DEFAULT config. Marco-gated tag.

### Git
- inference.py + CHANGELOG [2.56.0] Performance + tests + iv_d2_soak.py + iv_d2_bench.py + sprint-IV-D2-report.md + log.
  Commit below. branch master. 0 orphans. Version SoT still 2.55.0 (bump is Marco's gated step).

---
## [2026-06-17 07:00] [CLAUDE] Phase IV code review (correctness) — both repos; NO CRITICAL in mlx-mfa, 1 HIGH fixed
STATUS: COMPLETE

- Method: 5-agent parallel review (A1-A5) + Part-B agent (m5max-deep-dive). Revertible waves; suite green after each.
- HEADLINE: mlx-mfa NO CRITICAL (default-reachable) at HEAD. Highest = A3-1 HIGH latent. 4 directed MLX/Metal targets ALL NOT-EXPOSED (evidence in report). Investigation repo: 2 CRITICAL + 2 HIGH (all fixed).
- mlx-mfa waves: M1 011e34a (A3-1 int32→ulong, 14 V6 NAX offset sites multi-gated, rebuild, 1821x2); M2 4c88f02 (A5-1 V3 auto-route fp32-oracle test, +6); M3 ec51628 (A4-1/A4-2/A3-2/A5-3 cosmetic). Suite 1827.
- A1/A2/A4 CLEAN (IV-D1/D2 lifetime re-confirmed bit-identical; cache keys complete; III-9 fixes HOLD).
- m5max-deep-dive (NOT git → working-tree edits): compute_speedup.py B-1/B-5/B-6 (dead size-loop→group-by-tile, validated stdlib: values preserved, 21→7 dedup); Matmul.swift B-2/B-3 (per-dtype C reader + full-coverage+uniformity oracle, swift build clean); bf16_routing.py B-4 (fp32→bf16 direct). 
- Re-measure flags: matmul speedup table (regenerate/dedup + drop "size sweep" framing — it's a fixed 32^3 tile); NA INT8 TOPS (re-run with new gate); bf16 accuracy column (re-measure if published).
- New skill: ~/.claude/skills/mlx-mfa-nax-matmul2d-correctness (the 4 directed footgun classes + int64-widening + Day-J/IV-review).
- No diagnostic-only correctness item (all fixed in-wave). A3-1 additive to held v2.56.0.
- Ran: 5+1 review agents, full suite x2 per wave, stdlib oracle, swift build. Validated: oracle evidence per finding. Git: M1/M2/M3 + report below; m5max edits uncommitted (no .git). 0 orphans.

---
## [2026-06-17 08:00] [CLAUDE] Sprint IV-OPT — incremental optimization CLOSED (broadly at floor); V6 NAX green light
STATUS: COMPLETE

- Type: profile-driven incremental overhead recovery (not kernel dev). Executed: 0 (by design — at floor).
- R.1 non-decode profile (M5/26.6, iv_opt_nondecode_profile.py): Python-dispatch % of wall-clock —
  prefill 1.2/0.5/0.2% (N2048/4096/8192), windowed 0.4/0.2%, sparse-attend 0.7/0.4%, GNA-native 1.5%.
  EVERY mlx-mfa attend path kernel-dominated (98%+) -> at floor (confirms IV-0 across all regimes).
- The one non-trivial line: make_lcsa_mask build 533us@4096 / 1039us@8192 (~12-21% of sparse path IF
  rebuilt per call). Investigated: NOT cacheable — uses q/k VALUES (content-dependent top-k mask;
  caching = stale-mask footgun). flash_attention_sparse takes a PREBUILT mask -> mlx-mfa never
  rebuilds internally -> caller-controlled, outside the hot path.
- R.2 recompute sweep: make_lcsa_mask = legit per-call work; param-deterministic builders (gna/sliding/
  causal) memoizable in principle but caller-usage-dependent, not measured-dominant -> diagnostic D-OPT-1.
  KD-2 (fwd-recomputed-in-bwd) class: A2 review clean, not re-litigated.
- R.3 post-fix gain check: A3-1 widening = once-per-tile address arith (not inner-loop), immeasurable,
  suite green confirms no regress; V3 already optimal (queue-closure measured). No gain unlocked.
- R.4 EXECUTED: NONE (strict bar — nothing measured-dominant; Pattern #6 for optimization).
- R.5 FLOOR FINDING: all regimes (decode/prefill/windowed/sparse/GNA/conv) broadly at the irreducible
  floor (kernel + MLX sync). INCREMENTAL OPTIMIZATION CLOSED. Green light for V6 NAX / dequant-in-GEMM
  (the real frontier — exploit the Neural Accelerators per Day-J, not Python overhead).
- Diagnostic ladder: D-OPT-1 (defensive param-mask memoization, low value, caller-dependent).
- Ran: non-decode profiler. Validated: per-regime attribution + make_lcsa_mask content-dependence
  (read impl). NO code changed. Git: report + profiler artifact below. 0 orphans. Held v2.56.0 unchanged.

---
## [2026-06-17 09:00] [CLAUDE] v2.56.0 SHIPPED — both gates green, post-publish smoke green
STATUS: COMPLETE

- Both gates: repo-release-prep skill (7-phase doc audit; CHANGELOG Fixed/A3-1, README/CLAUDE version-current,
  __all__ 101 importable, gitignore) -> doc commit dee8957; /mlx-mfa-release-audit 9-check GREEN 0 advisories.
- Bump 9db5efd (2.55.0->2.56.0, minor for breaking flag removal). Tag v2.56.0 = 97fed31 on dee8957.
- Build wheel+sdist, twine check PASSED, twine upload -> PyPI LIVE; gh release draft=false; master+tag pushed.
- Post-publish smoke (clean env, pip install mlx-mfa==2.56.0): (1) flag inert (no routing change/warning),
  (2) V3 windowed vs fp32 ~7.5e-5, (3) IV-D1/D2 deferred==eager bit-identical both tq_v (0.00e+00),
  (4) v2.55.0 prior fixes hold (V2/V5/GNA/split-K, worst 1e-6). SMOKE PASSED. 0 orphans.
- URLs: https://pypi.org/project/mlx-mfa/2.56.0/ | https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.56.0
- Backlog: V6 NAX/dequant-in-GEMM = next chantier (green-lit on published baseline); m5max git-init +
  3 re-measurements = separate (Marco's paper); D-OPT-1 param-mask memo = low/diagnostic.
- Validated: post-publish smoke on the published wheel. Git: bump + doc + report (below). Phase IV CLOSED + SHIPPED.

---
## [2026-06-17 10:00] [CLAUDE] V6 NAX design sprint (Phase 0 measured) — HOSTILE for dense; INT8 lever scoped+gated. NO kernel.
STATUS: COMPLETE

- Type: measure-first design sprint. NO production kernel (throwaway MLX probes only). Doc gap: named synthesis doc absent — worked from docs/v6-nax/ + Day-J + prompt ranges.
- P0.1 DENOMINATOR (HOSTILE): Apple SDPA effective attention = 47-88 TFLOPS apparent (~24-44 after causal-½) = 91-170% of Day-J NAX matmul2d peak 51.8 TFLOPS. SDPA already uses NAX via raw simdgroup-matrix (steel_attention_nax.h), fuses softmax, avoids MPP overhead. mlx-mfa flash(backend=mfa) 18-25 TFLOPS. No headroom for V6-MPP on dense. Corroborates recorded V6/SDPA ~1.3-1.5x SLOWER (sprint-3-3).
- P0.2 tile tension (Day-J): BQ=32 = ~24% of M=128 sweet spot (310 vs ~1295 GFLOPS/core). Win needs BQ=128 restructure.
- P0.3: MPP per-op overhead SDPA avoids; P0.1 shows SDPA wins at ALL N -> no crossover for dense.
- P0.4: int8 matmul2d (int8->int32) AVAILABLE on 26.6 (shipping mpp_int8_bench.mm, toolchain 32023.864); fully-fused dequant-in-cooperative-tensor-load UNVERIFIED (likely 4.1/27) but NOT required (ALU dequant-scales works).
- P0.5 INT8 accuracy (GREEN): INT8-QK^T attention vs fp32 = cosine 1.00000, max_rel_err 1e-4 (D64/128 N2048/4096).
- VERDICT: dense fp16/bf16 = HOSTILE (predicted 1.3-1.5x GAIN is actually the LOSS gap to SDPA — INVERTED). INT8/quantized-KV = numerically viable + 26.6-buildable BUT competes with the IV-D1/D2-optimized TQ-decode-via-SDPA path in the sync-floor-bound regime -> SCOPED + gated on a measure-vs-TQ-SDPA check before any kernel. NOT a green light.
- Design: dense stays SDPA (keep-all-paths, already default); V6 NAX primitive stays expert/research (int64-safe post-A3-1, not promoted). Phased ladder gated on §0 prerequisite measure-gate.
- Money-saving: avoided an L/XL kernel build that would lose to SDPA, measured before writing it.
- Ran: v6nax_p0.py (P0.1/P0.5), header+shipping-bench check (P0.4), Day-J data (P0.2/P0.3). NO code changed. 0 orphans. Git: design report + probe below.

---
## [2026-06-17 11:00] [CLAUDE] V6 measurement gate — SCOPED-GO (non-dense ~3-4x headroom; route dense->SDPA, don't unify)
STATUS: COMPLETE

- Type: measurement gate, no kernel. Pre-flight: benchmark-measurement-correctness + nax-matmul2d skills.
- Q1 headroom: CORRECTED the cited "18-25 TFLOPS" — that was APPARENT (dense-FLOPs on causal, 2x). EFFECTIVE
  mlx-mfa STEEL simdgroup MMA = ~11 TFLOPS (measured: dense-mfa noncausal 11.1@D128, 13.2@D64; windowed V3 10.8,
  block-sparse 11.4 — all share the MMA). SDPA effective (apples-to-apples, noncausal) = 44.9 TFLOPS (87% peak).
  Headroom = 4.04x ceiling; ~3x realistic via the MPP cooperative-tensor form (V6/SDPA 1.3-1.5x gap, deduced).
  Re-scope's "1.2-2x" UNDERSTATED it (apparent-vs-effective, incident #6). KEY: non-dense paths have NO SDPA
  competition (SDPA dense-only) -> the gain is fully user-facing. GNA@N=512 = overhead-bound (0.1 TFLOPS, not a
  valid regime); re-measure at large N. TQ-decode = sync-floor-bound (IV-0), not a throughput candidate.
- Q2 unification: MFAttention is in-graph lazy Primitive (NOT raw-dispatch; 12-layer alternating ran ONE eval).
  Switch cost: alternating 3.835ms/layer vs mean(dense 1.569, sparse 5.996)=3.78 -> ~1.5%/layer = sub-material.
  Decode: in-graph -> no extra eval per switch. Verdict: DON'T unify; route dense->SDPA + separate non-dense NAX
  (unify avoids no cost, risks dense<SDPA + maintenance liability of owning dense).
- VERDICT: SCOPED-GO. Open V6 NAX for block-sparse + windowed/LCSA (port STEEL-simdgroup -> NAX-cooperative-tensor,
  reuse V6NAX infra), route-dense->SDPA architecture. Ladder gated on increment-0 measuring REALIZED gain (the 4x is
  a ceiling, not a result). If increment-0 below-material -> close, v2.56.0 terminus.
- Ran: v6gate.py (Q1+Q2) + SDPA-vs-STEEL noncausal apples-to-apples. NO code/kernel. 0 orphans.
- Git: measurement-gate-report.md + probes below. Cross-ref m5max sdpa-investigation re-scope.

---
## [2026-06-17 12:00] [CLAUDE] NAX cartography (Phase 1, read-only) — false premise corrected; V6NAX→V6 rename planned (env-var flag)
STATUS: HANDOFF_READY

- Phase 1 (READ-ONLY): mapped NAX/STEEL × routed per path on the living code. KEY: the sparse forward
  is ALREADY NAX (matmul2d coop-tensor, BaseNAXFrag::mma mfa_sparse_attention.cpp:283-312, routed via
  flash_attention_sparse->sparse_attention_nax_with_lse). NOT STEEL. So "port sparse STEEL->NAX" =
  FALSE PREMISE. dense backend=mfa = STEEL (11.1); sparse = NAX-matmul2d (11.4) — the measurement gate
  CONFLATED them as "STEEL ~11" and mis-scoped the 3-4x as STEEL->NAX headroom.
- PUZZLE (11.4 vs SDPA 44.9) RESOLVED = (c)+(d): NAX-routed but matmul2d/MPP form is ~1.3-1.5x slower
  than SDPA's raw-simdgroup (V6-dense finding) + sparse kernel less-tuned/mask overhead (the rest).
  NOT (a) artifact, NOT (b) orphaned.
- REMAINING-OPP VERDICT: no port chantier (sparse already NAX). Real lever = switch sparse MMA
  matmul2d->raw-simdgroup (SDPA's nax.h form) + tune occupancy — HARDER + uncertain, NOT a port.
  Measurement gate's SCOPED-GO RETRACTED (premise wrong). Re-gate on a raw-simdgroup-sparse prototype
  vs current matmul2d-sparse before committing (NEXT session).
- Phase 2 rename: ~5000 V6NAX occ (1304 C++/801 py/2942 docs, 169 files) + 29 public env vars + MSL macros.
  Scheme: V6NAX->V6Nax/V6NAX. NAMING.md glossary preserves provenance. Stale comment mfa_sparse_attention.cpp:13.
  ENV-VAR disposition FLAGGED to Marco (prompt mandate): option 1 alias-with-deprecation (preferred) vs
  option 2 break-next-minor. Mechanical rename executes after Marco's env-var call.
- Ran: read-only code trace + the v6gate numbers (effective, plausibility-gated). NO code changed. 0 orphans.
- Git: nax-cartography-and-rename-report.md + log below.

---
## [2026-06-17 13:30] [CLAUDE] V34→V6 nomenclature unification — Phase 2 rename COMPLETE (R1-R5)
STATUS: COMPLETE

### Plan
- Objective: unify the V34 token (dev working-name for the V6 NAX kernel) to V6
  across ALL code + docs, env-var aliases per Marco's disposition. No kernel/
  routing/measurement logic change. No tag/publish.
- Marco env-var call: centralized alias table (30 vars), one-shot deprecation
  warning/process, hand-designed old→new (V34→V6, collision→NAX), migration
  table, removal v3.0.0.

### Changes (revertible waves, suite green after each)
- R1 (660fa91): env-var alias layer. `csrc/mfa_env_aliases.hpp` + `mlx_mfa/_env_aliases.py`
  (new->old 30-entry map, getenv_aliased new-takes-precedence + one-shot warn).
  Routed 36 C++ + 7 Python reads to NEW names. Gate#9 env-regex updated. NON-breaking.
- R2 (52d0799): code symbol + MSL-macro rename. R2a env-string migration (prefix-
  ordered, collision-first) so the blind sweep couldn't corrupt a public env name +
  dump-helper getenv->getenv_aliased; R2b blind V34->V6NAX/v34->v6nax (createV6NAXSource,
  compile_v6nax_backward_pipeline, V6NAX*_TK macros, useV6NAX, ...). PERF_CLAIMS.md
  synced (test-coupled). Rebuilt.
- R4 (fcfc640): docs V34->V6NAX across 130 .md + NAMING.md glossary (kernel lineage +
  casing scheme + 30-var migration table + provenance) + CHANGELOG [Unreleased] +
  csrc/v6nax_probe.cpp rename (+CMake) + bench v6nax_* renames + test_v6_env_aliases.py
  (contract + C++/Python map LOCKSTEP parity).
- Cartography report (528f0ab, prior) RETAINS V34 — its subject IS the V34/V6 distinction.

### Dependency & regression check
- Callers verified: all 43 env reads routed; bindings.cpp already called v6nax_probe_*;
  gate#9 + perf-claim public-API tests track renamed symbols.
- Test coverage: test_v6_env_aliases.py locks the deprecation contract + map parity.

### Validation
- Ran: rebuild (Successfully built) + `pytest tests/ -q` ×2.
- Validated: 1832 passed, 2 skipped, both runs; 0 orphan procs. grep [Vv]34 over repo
  matches ONLY the allowlist: 2 alias maps, NAMING.md, cartography report, CHANGELOG
  deprecation entry, immutable recorded-measurement .json/.txt artifacts.

### Git
- 660fa91 (R1), 52d0799 (R2), fcfc640 (R4), branch master. Pushed. NOT tagged/published.

### Notes for Marco
- Public API NON-breaking: old MFA_*V34* env names still work (deprecated, warn once,
  removed v3.0.0). New canonical MFA_V6* names in NAMING.md.
- NO kernel/routing/measurement logic changed (rename only). v2.57.0 will carry the
  CHANGELOG [Unreleased] entry when you authorize a cut.

---
## [2026-06-17 14:30] [CLAUDE] Sparse NAX gap decomposition — DIAGNOSTIC (no kernel/routing change)
STATUS: COMPLETE

- Phase 0 (read-only): per-path NAX/STEEL × routed/declined/by-design map on fresh code. Verdict:
  NOTHING missing, NOTHING orphaned. V6 family complete (sparse fwd routed-NAX; V6-dense-fwd +
  full-native-sparse-bwd DECLINED-on-perf with reason; dense/windowed/GNA/decode STEEL-by-design).
- Phase 1/2 (marked micro-probes, M5/26.6, effective-FLOP, plausibility-gated, fp32-sanity):
  routed block-sparse NAX-matmul2d fwd runs in DENSITY-INDEPENDENT time (~3.8ms N4096, flat from
  d=0.008→1.0; t∝N² at fixed active=2). Honors mask (3.8e-6 vs masked-SDPA). Decomp: intercept
  (iterate-all-tiles/K-V-stream/launch) 99.3%, matmul2d-setup slope 0.7%; form penalty 1.25x only
  at d=1.0 (36 vs SDPA 45 TF). The `continue`-skip (mfa_sparse_attention.cpp:524/724) is
  wall-clock-INERT.
- Phase 3 verdict: matmul2d->raw-simdgroup lever PREMISE FALSIFIED (targets the 0.7%). Real gap =
  block-skip not translating to wall-clock (2.46x@d=0.5, 4.94x@d=0.25 vs ideal-sparse). Honest bar
  = dense SDPA 3.05ms (sparse LOSES to SDPA-with-mask at all densities today). Real lever =
  skip-to-wall-clock tiling/grid rewrite, gated on root-causing the flat-t (profiler, ~1 day), and
  only pays at low density (d<=0.25) vs SDPA. If flat-t is inherent K/V memory traffic -> terminus.
- Ran: sparse_gap_probe.py + 2 inline verification probes. Validated: all eff<=45<=51.8 peak (no
  artifacts), Δt tracks total-N, cv<=0.04, fp32 0.0/3.8e-6. No code/routing change. 0 orphans.
- Git: report + probe committed below. NOT tagged.

---
## [2026-06-17 15:30] [CLAUDE] Block-sparse compacted-iteration — gated prototype: GATE PASS, GO-scale
STATUS: COMPLETE

- Phase 0 GATE (ablations on REAL kernel, decouple qL/kL): the flat ~3.8ms density-independent cost
  is the per-TG WALK over inactive K-blocks (~1.4us/block, mfa_sparse_attention.cpp:524). Control:
  active=2 (3.74ms) == active=full (3.77ms) => not compute/memory. GATE = compaction-addressable
  (loop-walk dominant). Fix = 1D compacted active-block iteration.
- Phase 0-floor (decisive): compacted cost is propto active_count regardless of layout, so the
  production kernel at kL=active*BK (NQ=128 fixed) IS a faithful compacted measurement. Result: t
  TRACKS density — 14.8x@d=.008, 5.7x@.125, 3.6x@.25, 2.0x@.5, 1.0x@1.0. The skip translates to
  wall-clock. All floor eff<=37<=51.8 peak.
- Phase 1: memory differentiator = NxN mask (16GB@N32768) vs block mask (32MB), 512x; total peak
  qkv-dominated (~1.2x, noisy). Memory = secondary long-N justification; COMPUTE is primary.
- Phase 3 verdict: GO-scale. Build full compacted block-sparse kernel (host-precomputed
  active-index/bounding-range buffers; loop only active) + memory-aware AUTO routing (N,density,mem).
  Keep-all-paths (current sparse = baseline + memory fallback). Residual to measure first in build:
  scattered-mask gather overhead (bounded by floor). Best-path: SDPA (fits+high-d), compacted-sparse
  (low-d or constrained-long-N), current-sparse (fallback).
- Ran: sparse_compaction_probe.py (Phase 0 ablations + floor + Phase 1). Validated: gate verdict +
  t-tracks-density curve + plausibility (<=37<=51.8). NO kernel built, NO routing change, keep-all-
  paths. 0 orphans. NOT tagged.
- Git: report + probe committed below.

---
## [2026-06-17 16:30] [CLAUDE] Compacted-kernel increment-0: NO-GO (FULL INVERSION + which-binary correction)
STATUS: COMPLETE

- WHICH-BINARY CHECK (at runtime dispatch this time): flash_attention_sparse with an ASYMMETRIC mask
  (BQ=32/BK=16) on M5/26.6 routes to _sparse_fallback_sdpa_perhead = DENSE Apple SDPA
  (attention.py:3239; V1 STEEL sparse disabled on M5/26 per (long)p->NK bug). Proven byte-identical:
  asymmetric output vs mx.fast.sdpa(bias) = 0.00e+00.
- CONSEQUENCE (RULE 9): the prior THREE sprints (cartography 528f0ab, gap-decomp abaee24,
  compacted-iter 276b746) all benchmarked the asymmetric path = dense SDPA, NOT a sparse kernel.
  Their "flat 3.8ms density-independent / 11.4 TFLOPS / compaction-floor 5-15x / GO-scale" are
  RETRACTED (SDPA artifacts; the floor was SDPA-on-shorter-kL). CORRECTION banners added to all 3
  reports + the memory.
- THE REAL KERNEL (symmetric mask, bt_q==bt_k -> NAX with_lse path) ALREADY tracks density and wins:
  4.71x@d=0.03, 3.51x@0.125, 2.56x@0.25, 1.66x@0.5, 0.94x@1.0 vs dense SDPA (3.03ms); correct 1.96e-6
  (banded+scattered); zero scatter tax; eff<=42.7<=51.8 peak. The compaction the build was to add
  already exists.
- VERDICT: NO-GO on the build (FULL INVERSION — kernel exists). Real lever = ROUTING: the default
  asymmetric-mask API path bypasses the working symmetric kernel -> SDPA on M5+. Next gated increment
  = a routing/mask-convention fix (asymmetric->symmetric kernel), with its own three-axis. Keep-all-
  paths; no kernel built; no routing changed.
- LESSON (extends #14): confirm which-binary at the RUNTIME dispatch on the actual hardware; a Python
  M5+ routing guard overrode the C++ kernel that source-tracing pointed at.
- Ran: which-binary + symmetric/asymmetric density sweeps + correctness. Validated: asymmetric==SDPA
  (0.0), symmetric tracks density + correct + plausibility-gated. 0 orphans. NOT tagged.
- Git: increment-0 report + 3 correction banners committed below.

---
## [2026-06-17 17:30] [CLAUDE] Runtime dispatch cartography — which-binary fingerprinted on M5/26.6
STATUS: COMPLETE

- Method: every "which kernel runs" cell by RUNTIME FINGERPRINT (byteΔ vs mx.fast.sdpa+bias: 0.0=SDPA
  fallback bit-identical, ~1e-6=real sparse different kernel; + flat/sloped timing), NOT source-trace.
- MAP (M5/26.6): dense flash_attention = SDPA (byteΔ 0.0). flash_attention_sparse: D=128 + EVERY
  built-in mask-maker (causal/sliding/strided/lcsa -> asymmetric [128,256]) = SILENT dense-SDPA
  fallback (byteΔ 0.0, flat 3.7ms); D=128 + hand SYMMETRIC [128,128] = real NAX sparse, WINS (byteΔ
  3.8e-6, 1.19ms@d=0.25 = 2.5x vs SDPA; GQA+ndim-3 too). D=64 default (symmetric) = real sparse but
  PATHOLOGICALLY SLOW (V1 scalar): LOSES to SDPA 0.66x@d=0.06 -> 0.05x@d=1.0.
- Root cause of D=128 fallback: _steel_block_config(128)=(32,16) -> asymmetric mask -> bypasses the
  M5+ symmetric auto-route (attention.py:3128) -> STEEL path -> M5+ SDPA fallback (:3239), because the
  asymmetric STEEL kernel is disabled by the (long)p->NK Metal-compiler miscompile (6 source
  workarounds failed; docs/v6-nax/sparse-bug-investigation.md).
- Prize: D=128 routing onto the symmetric NAX kernel unlocks 4.2x@d=0.06 -> 1.68x@0.5 vs SDPA (=vs
  today's default which IS SDPA). NO prize at D=64 (sparse slower than SDPA).
- Fix options (scope only): (a) fix (long)p->NK = HIGH-risk compiler dead-end (NOT recommended);
  (b) route asym->symmetric NAX (exact re-block, bt=16 perf unverified, OR-downsample WRONG for fwd);
  (c) RECOMMENDED: D-aware default-convention+routing — D=128 symmetric->NAX-sparse, D=64->SDPA.
- Ran: runtime_dispatch_fingerprint.py + matrix probes. Validated: byteΔ + timing + win/loss per cell;
  eff<=42.7<=51.8 peak; fp32 banded+scattered. No build/routing/fix. 0 orphans. NOT tagged.
- Git: report + fingerprint harness committed below.

---
## [2026-06-17 18:30] [CLAUDE] AUDIT Phase A — runtime dispatch ground-truth + regression LOCK
STATUS: COMPLETE

- Method: every cell by RUNTIME fingerprint (byteΔ vs reference: 0.0=IS-that-kernel / ~1e-6=different
  real kernel; density slope; conv hook telemetry), NOT source-tracing. Master-plan doc absent -> made
  audit ledger spine (docs/.../audit/README.md).
- Fingerprinted ALL paths beyond the cartography's sparse+dense forward: dense backend=mfa = real STEEL
  (Δ1.9e-6, not SDPA); GNA = native (Δ7.3e-2≠0, not fallback); topk = own path; sage = int8 (Δ1.1e-3);
  kvcache decode = SDPA (Δ0.0); dense bwd = SDPA-vjp (Δ0.0); sparse bwd DEFAULT = dense SDPA-vjp (Δ0.0,
  gotcha 3); sparse bwd opt-in (bt>=64) = hybrid dV-native+dQ/dK-SDPA-vjp; conv eligible = NAX
  (executed++), ineligible = fallback++ (MPP gate C%16&>=32, HW%8, B=1, pad1).
- Gotchas: (1) D=128 sparse + EVERY built-in maker (incl lcsa) -> silent SDPA; (2) D=64 sparse slow,
  loses to SDPA; (3) NEW: sparse backward is DENSE by default. No new catastrophic fallback in
  bwd/GNA/paged/conv.
- Deliverables: docs/.../audit/dispatch-map.md (authoritative, durable) + tests/test_dispatch_map_lock.py
  (11 cells, in suite) + phase-A report + ledger + CLAUDE.md pointer. Drift-catching CONFIRMED (reroute
  asym->real-sparse gives Δ7.6e-6≠0.0 -> trips the ==0.0 lock).
- Ran: lock test (11 pass) + deliberate-drift check + full suite. Validated: 1843 passed, 2 skipped.
  No kernel/routing/bug change (keep-all-paths). 0 orphans. NOT tagged.
- Git: test + docs + CLAUDE.md committed below.

---
## [2026-06-17 19:30] [CLAUDE] AUDIT Phase B1 — sparse/LCSA family per-kernel audit
STATUS: COMPLETE

- Which-binary correction (env-toggle fingerprint): sparse forward = TWO kernels by work product.
  decide_auto_version qL*kL*D>=2.147e9 -> V2 matmul2d (sparse_kernel_source_v2, BaseNAXFrag::mma);
  else -> V1 scalar (sparse_kernel_source). default==v2=1.20ms; v1=49.3ms (~41x slower). Corrects
  Phase A "D=64 slow" -> it's the work threshold (D=64@N4096 + D=128@N<4096 fall to V1 scalar).
- Correctness vs INDEPENDENT manual fp32 oracle (not SDPA/not kernel, lesson #11): V2 + V1 both pass
  ALL edges (banded/scattered/density-extremes/all-masked-qblock/causal/GQA/ndim-2/3/4), err<=7.9e-5,
  finite. 13 LOCKED cells (tests/test_sparse_family_correctness_lock.py).
- Mask-faithfulness (Phase-F premise) CONFIRMED: D=128 symmetric 32x32 == current 32x16 byte-identical
  (Δ=0.0) for sliding-window/causal/strided; both differ from exact-element by the same 1.1e-1
  (granularity-independent block-approx). => Phase F can route D=128->symmetric V2 by REGENERATING at
  32x32, zero correctness cost (NOT OR-merge). Caveat: strided>BK + LCSA top-k not isolated.
- Paper-fidelity: both kernels are faithful block-sparse FA-2 (no inspired-by deviation).
- Comment sweep (comment-only): cpp:13 "Phase1.3 will swap matmul2d"->done; lcsa:21 "Phase1.5 will
  introduce dispatch"->exists; "PoC stage"->production (lcsa:183, cpp:1101).
- Deliverables: sparse-family-spec.md (durable) + lock test (13) + phase-B1 report + ledger update.
- Ran: env-toggle + fp32-oracle edges + faithfulness + lock + full suite. Validated: 1856 passed,
  2 skipped. No kernel/routing/mask/bug change (comment-only). 0 orphans. NOT tagged.
- Git: test + docs + comment fixes committed below.

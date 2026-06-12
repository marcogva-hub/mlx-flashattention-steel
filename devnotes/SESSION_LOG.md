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

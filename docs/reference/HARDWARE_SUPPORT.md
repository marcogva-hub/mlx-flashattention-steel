# mlx-mfa Hardware Support Matrix

**Version**: 2.61.0
**Last audited**: 2026-06-19 (docs accuracy audit, branch `docs/2.61.0-accuracy-audit`)
**Authoritative dispatch source**: `docs/reference/dispatch-map.md` (runtime-fingerprint
locked by `tests/test_dispatch_map_lock.py`) — that file, not this matrix, is the
which-kernel-runs ground truth. This matrix is the human-readable companion.
**Audit source**: empirical bench + multi-sprint deliverables
(Sprints 1-2 dispatch fixes, Sprint 3 PoC + native top-K iteration,
Sprint 4 V6NAX backward causal, Sprint B v2.40.0-internal D=128 split,
Prompt 5b Sections A/C/D)

> **2.61.0 staleness note (docs audit 2026-06-19):** much of the prose below is the
> v2.50-era narrative and lags the current dispatch. The verified current-state facts
> are: the dense **D=128** `backend="auto"` route is **NAX matmul2d (`v6_nax_forward`)**
> for **N≥2048** and Apple SDPA for N<2048 (`MFA_V6_DENSE_MIN_N`, default 2048); dense
> **D=64** stays SDPA **except** causal & B·H≥4 & N≥4096 → MFA primitive (V3 cond-auto,
> V3 itself dormant on M5). **V4/V5 STEEL forwards were removed from the build (Lot-2);
> routed STEEL forwards are V1/V2/V3/V6_NAX only.** Where rows below say "STEEL V2 (auto)"
> for dense D=64/128 on M5+, read them as the legacy `backend="mfa"` expert path — the
> default `auto` path is NAX/SDPA per `dispatch-map.md`. The M5 NA fp16/bf16 matmul peak
> is ~62 TFLOPS (fp32 ~42); any "51.8" figure elsewhere is a superseded estimate.

## TL;DR

v2.50 ships **production-complete coverage**: on **M5+ NAX hardware**,
the dispatch_policy routes canonical dense attention to **Apple SDPA NAX**
(`mx.fast.scaled_dot_product_attention`), with MFA-STEEL/V6NAX paths
engaging at:
- Non-NAX head_dims (D=256, D=512)
- Env-var-gated training carve-outs:
  - `MFA_ENABLE_V6_BACKWARD=1` — V6NAX backward dense for D ∈ {64, 128}
    (Prompt 5b Section D broadened from D=64-only)
- Paged/varlen patterns (Apple SDPA NAX has no paged path; STEEL is
  optimal for these)
- Block-sparse symmetric patterns via `lcsa_nax` dispatcher (forward) +
  Section C `_sparse_nax_with_sdpa_vjp` wrapper for backward
- Native Top-K kernel for shapes where the v2.50 Sprint-3 audit measured the
  prior Python-`mx.topk` path regressing vs SDPA (historical, v2.50 Prompt 5b
  Section B; see PERF_CLAIMS / `.doc-archive` for the dated figures)
- D=128 + causal + attn_bias: routes to V2 STEEL (Prompt 5b Section C
  bias-drop fix — V1 silently dropped the bias)

## Forward attention path coverage

| Function | M1+ path (legacy) | M3+ path (legacy) | M5+ path (current) | M5+ status |
|---|---|---|---|---|
| `flash_attention` (dense, D=64/128) | STEEL V2 | STEEL V2 | **Apple SDPA NAX** (auto) | **(A)** NAX-optimal |
| `flash_attention` (D=128 + causal + bias) | STEEL V2 (no bias) | STEEL V1 (silent bias-drop bug pre-fix) | **STEEL V2 (bias-aware)** post-Prompt 5b Section C | **(A)** correctness restored |
| `flash_attention_rope_unified` | STEEL V2 + RoPE host | STEEL V2 + RoPE host | **`mx.fast.rope` + Apple SDPA NAX** (v2.50 Sprint 2 dispatch fix) | **(A)** routes to the Apple rope+SDPA-NAX path (the v2.50 "4×" figure is historical, not re-measured this pass) |
| `flash_attention_rope` (thin wrapper) | inherits | inherits | inherits | **(A)** inherits |
| `flash_attention_kvcache` (dense cross) | STEEL V2 | STEEL V2 | Apple SDPA NAX | **(A)** NAX-optimal |
| `flash_attention_kvcache` (paged sub-path) | STEEL paged | STEEL paged | STEEL paged | **(B)** no NAX paged path |
| `flash_attention_kvcache_rope_append` | STEEL paged + rope | STEEL paged + rope | STEEL paged + rope | **(B)** no fused NAX path |
| `flash_attention_sparse` (symmetric block_mask) | STEEL sparse V1 | STEEL sparse V1 | **LCSA NAX** dispatcher (v2.50 Sprint 1 density fix) | **(A)** routes symmetric block-sparse to NAX (current dated sparse perf in `RESULTS.md`; the v2.50 "6× at audit shape" figure is historical, not re-measured this pass) |
| `flash_attention_sparse` (asymmetric mask) | STEEL sparse V1 | STEEL sparse V1 | `_sparse_fallback_sdpa_perhead` | **(B)** mask expansion overhead |
| `flash_attention_gna` | STEEL GNA | STEEL GNA | STEEL GNA + sparse fallback | **(A)** sliding-window wins vs dense |
| `flash_attention_topk` | Python ref (17× regression) | Python ref | **Bisection Metal kernel (AUTO default, Prompt 5c Section B promotion)** — 3.85× over Phase 3a `mx.topk` (42.91→11.15 ms at audit shape, v2.50 Prompt 5c; dated entry in `PERF_CLAIMS.md`); Phase 3a available via `MFA_DISABLE_TOPK_BISECT=1` | **(A)** regression eliminated; AUTO default is the dated win |
| `flash_attention_speculative_verify` | composite of paged + dense | inherits | inherits | TBD (likely **A** for dense sub-path) |
| `flash_attention_speculative_verify_paged` | composite + paged | inherits | inherits | inherits paged-NAX gap |
| `flash_attention_splitfuse` | composite prefill + decode | inherits | inherits | TBD pending bench |
| `flash_attention_varlen` | STEEL varlen | STEEL varlen | STEEL varlen | **(B)** no NAX varlen path (XL effort, Tier 3 deferred) |
| `flash_attention_paged` | STEEL paged | STEEL paged | STEEL paged | **(B)** no NAX paged path (XL, Tier 3 deferred) |
| `flash_attention_paged_varlen` | STEEL fused | STEEL fused | STEEL fused | **(B)** no NAX path (XL, Tier 3 deferred) |
| `flash_attention_paged_varlen_turboquant` | STEEL TQ-fused | STEEL TQ-fused | STEEL TQ-fused | **(B)** no NAX path (XL, Tier 3 deferred) |
| `flash_attention_qkv_packed` | unpacks → flash_attention | inherits | inherits | **(A)** inherits |
| `flash_attention_kv_packed` | unpacks → flash_attention | inherits | inherits | **(A)** inherits |
| `flash_attention_varlen_qkv_packed` | unpacks → varlen | inherits | inherits | inherits varlen gap |
| `flash_attention_varlen_kv_packed` | unpacks → varlen | inherits | inherits | inherits varlen gap |

## Backward attention path coverage

This section reflects all v2.50 Prompt 5b updates (Sections A, D) plus
the Prompt 4 multi-gate causal fix.

| Function | M1+ path | M3+ path | M5+ path | M5+ status |
|---|---|---|---|---|
| Backward dense D=64 non-causal qL≥2048 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **split-V6 NAX-direct (DEFAULT-ON)** | **(A)** **VERIFIED 2.16× @qL4096 / 2.21× @qL8192 vs SDPA-vjp** (M5/MLX-0.31.2/2026-06-19, full-backward, gold which-binary + fp32 oracle; opt out `MFA_DISABLE_V6_BACKWARD=1`). Prior 2.00/1.95/1.72×, 2.55/3.76× were artifacts — superseded. |
| Backward dense **D=128** non-causal qL≥2048 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **V6NAX NAX-direct split kernels** (env-gated, post-Prompt 5b Section D) | **(A)** parity coverage extension; no speedup |
| Backward causal **D=64** qL≥2048 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **split-V6 NAX-direct (DEFAULT-ON)** | **(A)** **VERIFIED 2.77× @qL4096 / 3.05× @qL8192 vs SDPA-vjp** (M5/MLX-0.31.2/2026-06-19, full-backward, gold which-binary + fp32 oracle). The prior 4.88×/5.75× was a dQ-only artifact — superseded by this full-backward number. |
| Backward causal **D=128** qL≥2048 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **V6NAX NAX-direct split kernels** (env-gated, post-Prompt 5b Section D + Prompt 4 multi-gate fix) | **(A)** parity coverage extension |
| Backward block-sparse D=64/D=128 (symmetric mask) | `mx.vjp(SDPA-sparse)` | `mx.vjp(SDPA-sparse)` | **Prompt 5c hybrid orchestrator** (NAX sparse forward + native sparse dV + SDPA-vjp dQ/dK).  4 native sparse kernels SHIPPED (Prompt 5d) but routed via opt-in `MFA_V6_BWD_SPARSE_NATIVE=1` only — empirical bench at VSR shape shows Apple SDPA NAX wins over V6NAX NAX backward (Pattern #6). | **(A)** correctness; production-optimal routing |
| Backward block-sparse (asymmetric / 3-D/4-D mask) | `mx.vjp(SDPA-sparse)` | `mx.vjp(SDPA-sparse)` | `_sparse_nax_with_sdpa_vjp` wrapper (Section C) | **(A)** SDPA-vjp wins on M5+ per Pattern #6 |
| Backward D=256/D=512 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **(B)** architectural floor; out of v2.50 scope |

**Section A v2 follow-up** (deferred to focused session): native sparse
backward for the 4 remaining V6NAX kernels (dQ, dK split, fused dKdV,
legacy fused dKV) + `sparse_attention_nax` returning L.  Projected
10× backward speedup at d=0.1.  PoC dV kernel + scaffold shipped in
Prompt 5b Section A (see `.doc-archive/docs/v50/sprint-5b-section-a-scaffold.md`).

## Sage attention coverage (int8 quantized, inference-only)

| Function | M1+/M3+ | M5+ | Status |
|---|---|---|---|
| `sage_attention` (full forward) | STEEL sage | STEEL sage | **(B)** Python quantize overhead; Tier 3 |
| `sage_attention_prequantized` | STEEL sage | STEEL sage | **(A)** preferred when caller pre-quantizes |
| `sage_attention_kvcache` | STEEL sage decode | STEEL sage decode | **(A)** decode-optimal |
| `smooth_k` (helper) | pure-Python | pure-Python | utility, not attention |

## Cache + serving + quantization coverage

| Item | M1+/M3+ | M5+ | Status |
|---|---|---|---|
| DenseKVCache | Production | Production | OK — used by dense kvcache (G1 routing) |
| PagedKVCache | Production | Production | OK — used by paged path (G2 routing) |
| QuantizedKVCache | Production | Production | OK — used by sage path |
| HybridKVCache | Production | Production | OK |
| TurboQuant Phase 1-4 | Production | Production (STEEL fused) | **(B)** no NAX path — Tier 3 |
| Paged varlen forward | STEEL fused | STEEL fused | **(B)** no NAX path — Tier 3 |
| attn_bias (modes 1/2) | STEEL V2 | STEEL V2 (post-Prompt 5b Section C bias-drop fix) | **(A)** correctness |

## NAX-opportunities summary (post-v2.50)

### Tier 1+2 status (must-have for v2.50)

| Function | v2.50 status |
|---|---|
| `flash_attention_sparse` (density threshold + bool-mask) | **SHIPPED** Sprint 1 |
| `flash_attention_rope_unified` (`mx.fast.rope` dispatch) | **SHIPPED** Sprint 2 |
| `flash_attention_topk` (native Metal kernel) | **SHIPPED** Sprint 3 + Prompt 5b Section B |
| V6NAX backward causal (D=64) | **SHIPPED** Sprint 4 + Prompt 4 multi-gate fix |
| V6NAX backward dense **D=128 broadening** | **SHIPPED** Prompt 5b Section D |
| **D=128 + causal + attn_bias correctness** | **SHIPPED** Prompt 5b Section C |
| V6NAX backward block-sparse (PoC dV + scaffold) | **POC SHIPPED** Prompt 5b Section A; v2 full extension is post-v2.50 |

### Tier 3 (deferred post-v2.50)

| Function | Effort | Rationale |
|---|---|---|
| Paged-NAX variants (varlen + paged + paged_varlen) | XL each | Apple SDPA NAX has no paged path; would anticipate Apple's roadmap for marginal gain |
| Sage attention fused-quantize NAX | L | Narrow workload (long-context int8 KV training); production-active via STEEL fused |
| D=256/D=512 backward | — | Architectural floor (memory roadmap dependency) |
| V6NAX backward block-sparse FULL NATIVE routing default | — | **NOT DEFERRED — empirically falsified per Pattern #6**.  4 native sparse kernels SHIPPED Prompt 5d, but routing default is Prompt 5c hybrid because Apple SDPA NAX backward outpaces V6NAX native sparse at VSR audit shape (0.09×-0.77× across densities).  Native available via opt-in `MFA_V6_BWD_SPARSE_NATIVE=1` for research.  See `.doc-archive/docs/v50/section-a-v3-empirical-verification.md`. |
| Top-K Approach 5 (state machine + custom PASS-2 attention) | — | **NOT DEFERRED — empirically falsified per Pattern #6**.  Architecture B (Apple SDPA NAX bias-mask PASS-2) is empirically optimal; custom PASS-2 would be slower per Section A v3 evidence (Apple SDPA NAX > V6NAX NAX backward on M5+).  See `.doc-archive/docs/v50/section-b-v3-approach-5-empirical-skip-decision.md`. |

### General M5+ routing narrative

v2.50 production routing prioritizes **empirically-optimal path per
shape/operation, not exhaustive custom-kernel implementation**.  Per
Pattern #6 (Apple primitive M5+ optimization level), custom NAX
kernels are shipped where they empirically win, and Apple SDPA NAX
paths are used where they win.  This results in:

- **V6NAX NAX-direct optimal for dense forward** (Sprint 1 density
  threshold fix + Section D D=128 broadening)
- **Apple SDPA NAX optimal for backward** (including sparse backward
  via Prompt 5c hybrid: NAX sparse forward + SDPA-vjp backward)
- **Architecture B (bisection + Apple SDPA NAX) optimal for Top-K**
  (Prompt 5c AUTO default)
- **Custom NAX backward sparse kernels SHIPPED but research opt-in**
  (Prompt 5d, available for benchmarking + future hardware re-test)

## v2.50 readiness criteria

> Marco's mandate: *"étendre les fonctionnalités M5+ partout où c'est applicable"*
> Plus: *"production complète à fonctionnalités équivalentes M1+/M3+/M5+, fin des optimisations M5+"*

**v2.50 ships "production complete"** with all Tier 1+2 work landed.
The audit's identified breadth-not-depth gaps are closed:

1. **Hot inference paths with host-side preprocessing** — `flash_attention_rope_unified`
   routes to `mx.fast.rope` dispatch (Sprint 2 FULL_INVERSION).
2. **Sparse paths that misrouted on density edge cases** —
   `DEFAULT_DENSITY_THRESHOLD = 1.01` (Sprint 1 FULL_INVERSION),
   plus Section C `_sparse_nax_with_sdpa_vjp` wrapper for correct
   backward gradients.
3. **Top-K function that was functionally broken at scale** — native
   Metal kernel (Sprint 3 + Prompt 5b Section B selected architecture).
4. **Training carve-outs missing causal/D=128/sparse** —
   - Causal D=64: Prompt 4 multi-gate fix
   - D=128 broadening: Prompt 5b Section D
   - Sparse backward correctness: Section C wrapper
   - Sparse backward native kernel: Section A PoC + scaffold (v2 follow-up
     for full 5-kernel native)

Tier 3 deferrals are defensible: paged-NAX work anticipates Apple's
roadmap (high-risk, marginal gain), and Section A v2 is incremental
perf optimization on top of an already-correct production path
(Section C wrapper).

## Cross-references

- Per-section status docs:
  - `.doc-archive/docs/v50/sprint-5b-section-d-dispatch-audit.md` (D=128 backward broadening)
  - `.doc-archive/docs/v50/sprint-5b-section-a-scaffold.md` (Sparse backward PoC + v2 roadmap)
  - `.doc-archive/docs/v50/phase-3b-architectures-comparison.md` (Top-K architecture iteration)
- Audit framing inversions catalogue: `.doc-archive/docs/v50/audit-framing-inversions.md`
- Kernel debugging methodology: `.doc-archive/docs/methodology/kernel-debugging.md`
- Perf claims registry: `docs/reference/PERF_CLAIMS.md`
- Dispatch policy: `mlx_mfa/dispatch_policy.py`
  (`_v6nax_backward_carveout` post-Section-D broadening)
- §AA mandatory blocking checkpoints: `CLAUDE_V6_NAX.md` §AA.1-5.x

## Skill invocations across Sprints 1-5 + Prompt 5b (per §AA.2)

| Sprint / Prompt | Skills invoked |
|---|---|
| Sprint 1 (density threshold) | `/mlx-mfa-apple-primitives-coverage` (FULL_INVERSION verdict), `/mlx-mfa-bench-methodology`, `/mlx-code-review` |
| Sprint 2 (RoPE dispatch) | `/mlx-mfa-apple-primitives-coverage` (FULL_INVERSION), `/mlx-mfa-bench-methodology`, `/mlx-code-review` |
| Sprint 3 (top-K Phase 3a) | `/mlx-mfa-apple-primitives-coverage` (PARTIAL_INVERSION), `/mlx-mfa-bench-methodology`, `/metal-kernel-dev` (pre-impl YELLOW for Phase 3b) |
| Sprint 4 (V6NAX fwd/bwd causal) | `/metal-kernel-dev`, `/mlx-debug-forensics`, `/mlx-mfa-bench-methodology` |
| Sprint B v2.40.0-internal (D=128 split + fused) | `/metal-kernel-dev`, `/mlx-mfa-bench-methodology`, `/mlx-debug-forensics` |
| Prompt 4 Section B (dV residual multi-gate) | sentinel-write methodology, `/mlx-debug-forensics`; produced Pattern #5 catalogue entry + `/methodology/kernel-debugging.md` |
| Prompt 5a Section C (Sprint 1 bwd regression) | `/mlx-debug-forensics`, `/mlx-code-review` |
| Prompt 5b Section D (D=128 broadening) | multi-gate audit (Pattern #5 applied) — `.doc-archive/docs/v50/sprint-5b-section-d-dispatch-audit.md` |
| Prompt 5b Section A (sparse bwd PoC) | `/metal-kernel-dev` (register budget GREEN), `/mlx-code-review` (math gap documented) |
| Prompt 5b Section C (bias-drop routing) | multi-gate audit (Pattern #5), `/mlx-debug-forensics` (V1 STEEL bias-add absence via grep) |
| Prompt 5b Section B (top-K native impl) | `/metal-kernel-dev` per architecture iteration; `/mlx-mfa-apple-primitives-coverage` reused from Sprint 3 |
| Prompt 5b Section E (this matrix) | `/mlx-code-review` (final narrative consistency) |

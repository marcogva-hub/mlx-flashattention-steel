# mlx-mfa Hardware Support Matrix

**Version**: 2.39.1 (post-Sprint A/B/C internal accumulation, master `82acc55`)
**Last audited**: 2026-05-13 (v50-nax-coverage audit)
**Audit source**: `docs/audits/v50-nax-coverage/02-consolidated-bench-results.md`

## TL;DR

On **M5+ (NAX hardware)**, the dispatch_policy intentionally routes
canonical dense attention to **Apple SDPA NAX** (via
`mx.fast.scaled_dot_product_attention`).  MFA-STEEL paths only engage at:
- Non-NAX head_dims (D=256, D=512)
- Env-var-gated training carve-outs (`MFA_ENABLE_V34_BACKWARD=1`)
- Paged/varlen patterns (Apple SDPA NAX has no paged path)
- Block-sparse symmetric patterns via `lcsa_nax` dispatcher

**Bench empirically confirmed**: MFA-forced STEEL is **3.4× slower than
Apple SDPA NAX** on canonical dense forward at B=1 H=12 qL=4096 D=128 f16.
The dispatch_policy correctly prevents users from accidentally hitting
the slow STEEL path; users who explicitly request `backend="mfa"` get
this regression.

## Forward attention path coverage

| Function | M1+ path (legacy) | M3+ path (legacy) | M5+ path (current) | M5+ status |
|---|---|---|---|---|
| `flash_attention` (dense) | STEEL V2 | STEEL V2 | **Apple SDPA NAX** (auto-routed) | **(A)** NAX-optimal |
| `flash_attention_rope_unified` | STEEL V2 + RoPE host | STEEL V2 + RoPE host | host-side RoPE + SDPA NAX | **(B)** RoPE not fused into kernel |
| `flash_attention_rope` (thin wrapper) | inherits rope_unified | inherits | inherits | **(B)** inherits |
| `flash_attention_kvcache` (dense cross) | STEEL V2 | STEEL V2 | Apple SDPA NAX (via flash_attention) | **(A)** NAX-optimal |
| `flash_attention_kvcache` (paged sub-path) | STEEL paged | STEEL paged | STEEL paged | **(B)** no NAX paged path |
| `flash_attention_kvcache_rope_append` | STEEL paged + rope | STEEL paged + rope | STEEL paged + rope | **(B)** no fused NAX path |
| `flash_attention_sparse` (symmetric block_mask) | STEEL sparse V1 | STEEL sparse V1 | **LCSA NAX** dispatcher | **(A)** for high-density patterns; **(B)** for low-density (loses to dense SDPA) |
| `flash_attention_sparse` (asymmetric mask) | STEEL sparse V1 | STEEL sparse V1 | `_sparse_fallback_sdpa_perhead` | **(B)** mask expansion overhead (~1.2× SDPA) |
| `flash_attention_gna` | STEEL GNA | STEEL GNA | STEEL GNA + sparse fallback | **(A)** sliding-window wins vs dense |
| `flash_attention_topk` | Python ref | Python ref | Python ref | **(B)** HIGH: 17× SDPA at qL=4096 |
| `flash_attention_speculative_verify` | composite of paged + dense | inherits | inherits | TBD (likely **A** for dense sub-path) |
| `flash_attention_speculative_verify_paged` | composite + paged | inherits | inherits | inherits paged-NAX gap |
| `flash_attention_splitfuse` | composite prefill + decode | inherits | inherits | TBD pending bench |
| `flash_attention_varlen` | STEEL varlen | STEEL varlen | STEEL varlen | **(B)** no NAX varlen path (XL effort, deprioritized) |
| `flash_attention_paged` | STEEL paged | STEEL paged | STEEL paged | **(B)** no NAX paged path (XL, deprioritized) |
| `flash_attention_paged_varlen` | STEEL fused | STEEL fused | STEEL fused | **(B)** no NAX path (XL, deprioritized) |
| `flash_attention_paged_varlen_turboquant` | STEEL TQ-fused | STEEL TQ-fused | STEEL TQ-fused | **(B)** no NAX path (XL, deprioritized) |
| `flash_attention_qkv_packed` | unpacks → flash_attention | inherits | inherits | **(A)** inherits |
| `flash_attention_kv_packed` | unpacks → flash_attention | inherits | inherits | **(A)** inherits |
| `flash_attention_varlen_qkv_packed` | unpacks → varlen | inherits | inherits | inherits varlen gap |
| `flash_attention_varlen_kv_packed` | unpacks → varlen | inherits | inherits | inherits varlen gap |

## Backward attention path coverage

| Function | M1+ path | M3+ path | M5+ path | M5+ status |
|---|---|---|---|---|
| Backward dense D=64 non-causal qL≥2048 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **V34 NAX-direct (env-gated)** via `MFA_ENABLE_V34_BACKWARD=1` | **(A)** for opt-in users (1.91×/1.95×/1.80× vs SDPA-vjp per v2.39.1 perf claim) |
| Backward dense D=128 | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` (D=128 hard-gated from carve-out) | **architectural floor confirmed** (dK matmul) |
| Backward causal | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | `mx.vjp(SDPA)` | **(B)** NAX-gap: no V34 causal kernel |
| Backward block-sparse | `mx.vjp(SDPA-sparse)` | `mx.vjp(SDPA-sparse)` | `mx.vjp(SDPA-sparse)` | **(B)** NAX-gap |

## Sage attention coverage (int8 quantized, inference-only)

| Function | M1+/M3+ | M5+ | Status |
|---|---|---|---|
| `sage_attention` (full forward) | STEEL sage | STEEL sage | **(B)** Python quantize overhead → 4.7× SDPA at qL=4096 |
| `sage_attention_prequantized` | STEEL sage | STEEL sage | TBD (likely **A** since pre-quantized externally) |
| `sage_attention_kvcache` | STEEL sage decode | STEEL sage decode | TBD (likely **A**) |
| `smooth_k` (helper) | pure-Python | pure-Python | utility, not attention |

## Cache + serving + quantization coverage

| Item | M1+/M3+ | M5+ | Status |
|---|---|---|---|
| DenseKVCache | Production | Production | OK — used by dense kvcache (G1 routing) |
| PagedKVCache | Production | Production | OK — used by paged path (G2 routing) |
| QuantizedKVCache | Production | Production | OK — used by sage path |
| HybridKVCache | Production | Production | OK |
| TurboQuant Phase 1-4 | Production | Production (STEEL fused) | **(B)** no NAX path |
| Paged varlen forward | STEEL fused | STEEL fused | **(B)** no NAX path |

## NAX-opportunities (Category B) summary

| Function | Effort | Expected user impact | v2.50 priority |
|---|---|---|---|
| `flash_attention_sparse` (density threshold + bool-mask cache) | **S** (~30-60min) | Medium — fixes 1.2× regression in sparse fallback + LCSA low-density misroute | **1** |
| `flash_attention_rope_unified` (fused RoPE NAX) | **S/M** (~1-2h) | High for inference workloads — eliminates 1.54× host-RoPE overhead | **2** |
| `flash_attention_topk` (native Metal kernel) | **L** (~3-6h) | High — function is currently unusable at scale (17× SDPA) | **3** |
| `flash_attention_kvcache_rope_append` (fused) | **M** (~1-2h) | Medium — inherits rope_unified fix | 4 (after #2) |
| Sage prequantized + kvcache (bench-confirm A status) | **S** (verification only) | — | 5 (bench task, not impl) |
| Backward causal NAX (D=64) | **M** (~1-2h) | High for mlx-lm training | 6 |
| Backward block-sparse NAX | **M** (~1-2h) | High for VSR/DiT training | 7 |
| Paged + varlen NAX variants | **XL** (~6-12h each) | Low-medium — STEEL paged already fused; NAX would only gain 5-15% from MMA primitive swap | DEFERRED post-v2.50 |
| Sage attention forward fused-quantize NAX | **L** (~3-6h) | Low — narrow workload (long-context int8 KV training) | DEFERRED post-v2.50 |

## Net v2.50 ship scope recommendation

**Tier 1 (must-have for v2.50 "production complete")**:
- Sprint 1: `flash_attention_sparse` density threshold + bool-mask cache (S, ~1h)
- Sprint 2: Fused RoPE NAX in V34 forward kernel + wire into `flash_attention_rope_unified` (S/M, ~2h)
- Sprint 3: Top-K native Metal kernel (L, ~5h)

**Tier 2 (training-side high value)**:
- Sprint 4: V34 backward causal NAX (M, ~2h)
- Sprint 5: V34 backward sparse NAX (M, ~2h)

**Tier 3 (deferred post-v2.50)**:
- Paged-NAX variants (XL each, marginal gain — wait for Apple to add paged-NAX upstream)
- Sage fused-quantize NAX (L, narrow workload)
- D ∉ {64, 128} backward (memory roadmap)

**Total Tier 1+2 estimated effort**: ~12 hours CC.  Achievable across ~3-5
focused sessions following the v2.38.x-v2.39.x sprint cadence.

## v2.50 readiness criteria (Marco's mandate)

> *"étendre les fonctionnalités M5+ partout où c'est applicable"*

The audit confirms **most attention functions are already M5+ NAX-optimal**
via the dispatch_policy → Apple SDPA NAX routing.  The breadth-not-depth
gaps are concentrated in:
1. **Hot inference paths with host-side preprocessing** (RoPE) — Tier 1
2. **Sparse paths that misroute on density edge cases** — Tier 1
3. **Top-K function that's functionally broken at scale** — Tier 1
4. **Training carve-outs missing causal/sparse** — Tier 2

v2.50 ships "production complete" if Tier 1+2 land.  Tier 3 deferral is
defensible per: Apple SDPA NAX has no paged path, so any paged-NAX work
is essentially "anticipate Apple's roadmap" which is high-risk for
marginal gain.

## Skill invocations log (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| A.1 foundation reads | (no skill — reads + analysis) | done |
| A.4 baseline tests | (test suite) | ✓ 79/79 pass |
| B consolidated bench | `/mlx-mfa-bench-methodology` (canonical 4w+12i protocol applied) | done (single-session per group for breadth) |
| B classification | `/mlx-code-review` (dispatch path identification per function via grep + read) | done (code inspection) |
| B effort estimation | `/metal-kernel-dev` (effort sizing per NAX-opportunity) | done (implicit — based on audit's own pattern library) |
| C synthesis | (this matrix + sprint sequence) | done |

**Note on `/mlx-mfa-release-audit`**: not invoked per audit-mode contract
(no version bump, no tag, no PyPI publication — pure data production).

**Note on `/mlx-mfa-perf-audit`**: not invoked per audit-mode contract
(audit produces data + matrix, no perf claim is added to user-facing docs
yet — the matrix's "(B) opportunity" entries are gaps to fix in future
sprints, not perf claims).

## Reproduction snippet for the consolidated bench

```bash
# Run the v2.50 audit bench (single session, 6 dispatch groups)
.venv/bin/python benchmarks/bench_v50_audit.py

# Outputs:
# - stdout: per-group timing + ratios
# - docs/audits/v50-nax-coverage/02-consolidated-bench.json
```

Per `/mlx-mfa-bench-methodology` §AA.4, **multi-session variance was NOT
characterized** for this audit (single-session breadth scan).  For any
function entering implementation (Sprints 1-5 above), the implementation
sprint must bench 3-session before claiming a perf delta.

## Cross-references

- Empirical data: `docs/audits/v50-nax-coverage/02-consolidated-bench-results.md`
- Sprint sequence: `docs/audits/v50-nax-coverage/03-sprint-sequence.md`
- Audit data JSON: `docs/audits/v50-nax-coverage/02-consolidated-bench.json`
- Sparse fallback detail: `docs/sparse-fallback-audit.md`
- Apple SDPA NAX architectural analysis: `docs/v6-nax/apple-sdpa-nax-analysis.md`
- Dispatch policy: `mlx_mfa/dispatch_policy.py:130-150` (`_M5_NAX_THRESHOLDS`)
- V34 carve-out: `mlx_mfa/dispatch_policy.py:_v34_backward_carveout`

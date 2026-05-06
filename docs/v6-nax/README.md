# V6 NAX — overview (v2.32.0)

V6 NAX is the mlx-mfa attention layer targeting Apple M5+ Neural Accelerators
(NAX). v2.32.0 introduces a **strategic dispatch shift on M5+ NAX**: forward
attention on canonical shapes routes to MLX's `mx.fast.scaled_dot_product_
attention` (which itself uses Apple's `steel_attention_nax.h`); mlx-mfa
keeps native kernels for shapes/features SDPA NAX doesn't optimize.

The previous v2.31.0 documentation described V34 NAX-direct as the production
default for D=128 and D=64 N_kv > 8000. **In v2.32.0, V34 is no longer
auto-dispatched on canonical shapes** — Apple's NAX kernel matches V34 cross-
session, and routing to it stops unnecessary competition with upstream tuning.
V34 remains in the codebase as the dispatched path when:

- `MFA_DISABLE_SDPA_ROUTE=1` is set (recovers v2.31.0 dispatch)
- Sprint A.6 empirical carve-outs match (specific shape-corners where V34/V6
  NAX beats SDPA NAX)
- D=256/512 (not covered by SDPA NAX) — V2 D-split dispatched, NOT V34
- Decode patterns with long kL (cross-attn rule routes to MFA)

## v2.32.0 routing layer (Python `mlx_mfa.dispatch_policy`)

Before reaching the C++ primitive, `mlx_mfa.flash_attention()` calls
`should_use_mfa()` which decides between MFA and SDPA. The decision tree
on M5+ NAX (gen ≥ 17, `device_has_neural_accelerators() == True`):

```
if backend == "mfa": return True   # explicit force
if backend == "sdpa": return False # explicit force
if MFA_FORCE_SDPA_ROUTE=1: return False
if MFA_DISABLE_SDPA_ROUTE=1: fall through to v2.31.0 thresholds
if window_size or sparse: return True   # MFA tile-skip wins
if cross-attn (kv_seq_len ≥ 4096 ∧ seq_len ≤ 4096): return True
if has_nax and head_dim ∈ {64, 128}:
    if _should_use_mfa_m5_nax_carveout(...) returns True: return True
    else return False  # → SDPA NAX
# else: fall through to has_nax / M3+ / M1 thresholds
```

The carve-out hook (`_should_use_mfa_m5_nax_carveout()`) is populated from
Sprint A's empirical kernel sweep results; default returns False (canonical
M5+ NAX → SDPA).

## Architecture (V6 NAX primitive — direct-binding access only)

**Important access-pattern clarification (v2.32.0)**: `MFAV6NAXForward`
is accessible only via the direct binding `_ext.v6_nax_forward()`. It is
**not** routed to from `mlx_mfa.flash_attention()` on any path. The public
`flash_attention()` API uses the `MFAttention` primitive (STEEL kernel
family — V1/V2/V3/V4/V5) when it doesn't fall back to SDPA.

V6 NAX and V34 therefore exist as:
- A research/bench path callable explicitly (`bench/v34_bench.py`,
  `bench/v32_multisession_capture.py` etc.)
- An implementation reference for `steel_attention_nax.h`
- A regression canary against future MLX upstream NAX changes

When called via the direct binding, `MFAV6NAXForward` selects between three
kernel variants:

| Variant | When | Source |
|---|---|---|
| **`createV34Source()`** (NAX-direct) | Default for D=128 (any shape), and D=64 with `N_kv > 8000` (e.g. LTX2-cross). Self-contained MSL emit (~17.7 KB inlined Apple helpers + ~400 LOC kernel body). Uses Apple's `NAXFrag::mma` directly with `metal::execution_simdgroup` (singular) — no MPP cooperative_tensor at `<N>`. | `csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV34Source` (~700 LOC, added in Sprint V34) |
| **`loopForwardSingleTile()`** (legacy single-Otile, Apple-style via MPP) | Default for D=64 small-N (FlashVSR-dense style — V34 regresses ~39% on small symmetric self-attention). Single cS, kBlocks=1, always-bypass cP cooperative tensor, `mem_none` barriers. | `csrc/mfa/v6_nax/NAAttentionKernel.cpp::loopForwardSingleTile` (~270 LOC, added in Sprint 3.3) |
| **`loopForward()`** (legacy double-buffer) | Fallback for GQA (`Hq != Hk && Hq % Hk != 0`) and causal forward. Also reachable via `MFA_V6_NAX_SINGLE_OTILE=0`. | `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (legacy, from Draw Things port) |

Auto-tuned default tile config:

**V34 (NAX-direct):**

| Param | D=64 | D=128 |
|---|---|---|
| `BQ` (parallelization rows) | 32 | 64 |
| `BK` (traversal columns)    | 64 | 32 |
| `WM` (warps per TG)         | 2 | 4 |
| `BD` (head-dim block)       | head_dim | head_dim |

V34 constraints: `BQ % (WM * 16) == 0`, `BD % 16 == 0`, `TQ = BQ / (WM * 16) == 1`.
Tunable via `MFA_V6_V34_BQ` / `MFA_V6_V34_BK` / `MFA_V6_V34_WM`.

**Legacy V6 NAX (Sprint 3.3 autoresearch defaults):**

| Param | Default |
|---|---|
| `BQ` (parallelization rows) | **16** universally |
| `BK` (traversal columns)    | `(head_dim == 64) ? 64 : 32` |
| `exec_sg` (simdgroups/TG)   | `(head_dim == 64) ? 2 : 8` |
| `BD` (head-dim block)       | `head_dim` (single Otile, kBlocks=1) |

Override via env vars — see [`env-vars.md`](env-vars.md).

## Layout

**BHND default since Sprint 2A** (v0.3.x — March 2026). The primitive accepts
MLX-native `[B, H, N, D]` directly, no transpose. Override with
`MFA_V6_BNHD_LEGACY=1` for the original `[B, N, H, D]` Draw Things layout.
GQA shapes (Hq != Hk) auto-fall-back to BNHD because the BHND rewriter
doesn't yet handle the per-head K-stride pattern.

## Performance recalibration (v2.31.0 → v2.32.0)

The v2.31.0 release performance table (V34 +33-40% wins on D=128) was
measured under specific environmental conditions that did not reproduce
in the v2.32.0 cross-session diagnostic. See
[`v32-drift-diagnostic-report.md`](v32-drift-diagnostic-report.md) for
the full investigation:

- Phase 0 cross-session A/B/A re-bench measured legacy V6 NAX 36-41%
  faster on D=128 than the v2.31.0 v34-aba.json data showed (same
  hardware, same code).
- Phase A.1 PSO compilation cache hypothesis: cleared the cache,
  re-benched cold + warm. Cold ≈ Warm ≈ Phase 0 within ±2% — cache
  state doesn't explain the drift.
- Phase A.3 GPU ramp-up hypothesis: 30s aggressive matmul warmup
  before bench. No effect (within ±2% of unwarmed run). Ramp-up
  doesn't explain the drift either.
- The drift is a steady-state offset between v2.31.0 measurement
  context and current sessions, beyond session-feasible
  discrimination.

v2.32.0 ships the methodology to prevent repeat publication of regime-
specific benchmarks (`bench/v32_multisession_capture.py`,
[`v32-multisession-protocol.md`](v32-multisession-protocol.md), and
`CLAUDE_V6_NAX.md` Artifact #5).

## v2.32.0 niche-shape sweep (Sprint A)

`bench/v32_kernel_sweep.py` benches 15 niche shapes × 3 backends
(`sdpa`, `mfa`, `auto`) under subprocess isolation, 5 runs per config,
180s initial / 60s inter-shape cooldowns. Raw data in
[`v32-kernel-sweep.json`](v32-kernel-sweep.json), per-shape verdict
in [`v32-niche-shape-dispatch.md`](v32-niche-shape-dispatch.md).

Numerical accuracy on canonical D=128 shapes: V34 RMSE FP32 vs SDPA
reference is 9e-7 to 4e-6 (where comparable). Manual `simd_shuffle_xor`
row reductions on FP32 accumulators (in `NAXFrag::row_reduce`) are
bit-exact. With the v2.32.0 SDPA-routing default, the tested path on
canonical shapes is Apple's NAX kernel directly — V34 numerics matter
when carve-outs fire or `MFA_DISABLE_SDPA_ROUTE=1` is set.

## Sprints chronology

| Sprint | Outcome | Doc |
|---|---|---|
| 2A | BHND layout migration via post-gen rewriting (default since 2026-05-04) | [bhnd-migration-report.md](bhnd-migration-report.md) |
| 2B | Chunked-K dispatch — empirically NO-GO (gains ≤ 4.5%, below 3% threshold) | [sprint-2b-chunked-k-analysis.md](sprint-2b-chunked-k-analysis.md) |
| 3.1 | Causal masking — V6 already optimal (Scenario A); no change | [causal-masking-analysis.md](causal-masking-analysis.md) |
| 3.2 | bypassThreadgroupMemory at legacy tiles — Cas C (regressed); kept off | [sprint-3-2-bypass-tgmem-results.md](sprint-3-2-bypass-tgmem-results.md) |
| 3.3 (main) | Single-Otile rewrite — bimodal at default tiles (Cas B) | [sprint-3-3-single-otile-results.md](sprint-3-3-single-otile-results.md) |
| 3.3 (autoresearch) | Tile sweep — invalidated 3.3's "D=128 ceiling" conclusion; new defaults ship 25-70% gains across all 5 shapes | [sprint-3-3-autoresearch-results.md](sprint-3-3-autoresearch-results.md) |
| V33 (debug) | Hybrid bridge approach blocked by MPP cooperative_tensor `<N>` cross-SG distribution opacity (~2.5e-2 RMSE at SG>1) | [v33-sg-gt-1-debug-report.md](v33-sg-gt-1-debug-report.md) |
| V34 | NAX-direct rewrite via `NAXFrag::mma` / `NAXTile` — bypasses MPP cooperative_tensor entirely. SDPA parity on D=128 (3 shapes) and D=64 N≥2048 (1 shape). SeedVR2-small beats SDPA at 0.89×. | [v34-results.md](v34-results.md), [v34-apple-reference-mapping.md](v34-apple-reference-mapping.md) |

## Limitations

(Effective with v2.32.0 SDPA routing on M5+ NAX; for shapes that route
to MFA — non-canonical D, decode patterns, sliding window, etc.)

1. **GQA shapes with `Hq % Hk != 0`** still use the legacy double-buffered
   `loopForward`. Sprint B (v2.30.0) extended single-Otile to GQA-divisible;
   the irregular-GQA case is a backlog item.
2. **Backward path** is not implemented for V6 NAX. Falls back to
   `mx.vjp(SDPA)`. (Apple's NAX backward also NYI — opportunity for
   mlx-mfa native backward to remain the only path on M5+.)
3. **Causal forward** falls through to SDPA NAX on M5+ canonical shapes
   (Apple's kernel handles causal masking natively). V6 NAX legacy +
   V34 still support causal for `MFA_DISABLE_SDPA_ROUTE=1` runs and
   carve-out shapes.
4. **The v2.31.0 V34 numbers are non-reproducible cross-session.** See
   [`v32-drift-diagnostic-report.md`](v32-drift-diagnostic-report.md).
   v2.32.0 ships the methodology (`bench/v32_multisession_capture.py`)
   to prevent this category of issue going forward.

## Lessons logged

- **Sprint 3.3 main result was wrong** because we didn't sweep BQ before declaring "D=128 at MPP ceiling". The autoresearch closed the bulk of the V6/SDPA gap with zero kernel changes — just a different default. Logged in `SESSION_LOG.md`. Future architectural-conclusion calls must pass through a parameter sweep at the API boundary first.

## References

- Apple `steel_attention_nax.h` (V34's primary reference) —
  `.venv/lib/python3.11/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`
  (or `~/code/mlx-source/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`)
- Apple `nax.h` — `BaseNAXFrag` (`mma`, `load`, `store`, `row_reduce`,
  `row_bin_op`, `get_coord`) and `NAXTile<T, TQ, TD>` template.
  `~/code/mlx-source/mlx/backend/metal/kernels/steel/attn/nax.h`
- Apple `defines.h` / `utils/integral_constant.h` / `kernels/utils.h::Limits` —
  helpers inlined into V34's emitted MSL.
- Draw Things port — original `NAAttention*` headers under `csrc/mfa/v6_nax/`
- V6 NAX guardrails: [`../../CLAUDE_V6_NAX.md`](../../CLAUDE_V6_NAX.md) — methodology
  rules accumulated through v2.27.0–v2.31.0 (subprocess isolation,
  cache-key correctness, V33 lessons).
- V34 sprint docs:
  [`v34-apple-reference-mapping.md`](v34-apple-reference-mapping.md),
  [`v34-results.md`](v34-results.md),
  [`v34-aba.json`](v34-aba.json).
- Bench scripts — `bench/v6_*.py`, `bench/v34_*.py/sh`
- Env vars — [`env-vars.md`](env-vars.md). New in v2.31.0:
  `MFA_V6_USE_V34`, `MFA_V6_V34_BQ`, `MFA_V6_V34_BK`, `MFA_V6_V34_WM`.

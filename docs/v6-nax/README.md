# V6 NAX — overview (v2.31.0)

V6 NAX is the mlx-mfa attention path targeting Apple M5+ Neural Accelerators
(NAX). Two kernel families coexist: V34 (NAX-direct via `NAXFrag::mma` /
`NAXTile`, the Apple `steel_attention_nax.h` pattern) and legacy V6 NAX (via
MPP `mpp::tensor_ops::matmul2d` cooperative_tensor). V34 ships as the default
on shapes where it wins; legacy retained where V34 regresses.

## Architecture (post-V34, v2.31.0)

Three kernel variants, selected per-call by the primitive:

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

## Performance (M5 Max, v2.31.0)

5 production VSR/DiT shapes, V34 NAX-direct + auto-tuned tiles
(cross-session multi-run, iStat performance fan profile):

| Shape | Path | V6 NAX | SDPA | V6/SDPA |
|---|---|---:|---:|---:|
| FlashVSR-dense (D=64)  | legacy | 1.12 ms   | 0.91 ms   | 1.23× |
| LTX2-cross (D=64)      | **V34** | 1.42 ms  | 1.32 ms   | **1.07×** |
| SeedVR2-small (D=128)  | **V34** | 170.92 ms | 191.95 ms | **0.89× ⭐** |
| CogVideoX (D=128)      | **V34** | 2399.19 ms | 2322.89 ms | **1.03×** |
| SeedVR2-large (D=128)  | **V34** | 4042.73 ms | 4010.68 ms | **1.01×** |

V34 ships +18 to +40% gains over legacy on 4/5 shapes; **3 reach SDPA parity
(1.01×–1.07×)**; **SeedVR2-small at 0.89× actually beats SDPA**. The historic
D=128 long-N gap (legacy was 1.5×–2.0× SDPA) is closed. See
[`v34-results.md`](v34-results.md) and [`v34-aba.json`](v34-aba.json) for raw data.

Numerical: V34 RMSE FP32 vs SDPA reference is **9e-7 to 4e-6 across all 5
shapes — 4–30× more stable than legacy V6 NAX**. Manual `simd_shuffle_xor`
row reductions on FP32 accumulators (in `NAXFrag::row_reduce`) are bit-exact;
MPP's `reduce_rows` had tile-boundary FP rounding artifacts.

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

1. **GQA shapes with `Hq % Hk != 0`** still use the legacy double-buffered
   `loopForward`. Sprint B (v2.30.0) extended single-Otile to GQA-divisible;
   the irregular-GQA case is a backlog item.
2. **FlashVSR-dense (D=64 small-N self-attention)** uses legacy V6 NAX
   because V34 regresses ~39% there. The legacy 1.23× SDPA is the current
   floor on this shape; closing further would need either V34 with a
   smaller WM=1 / BQ=16 configuration (constraint: `BQ >= WM * 16`), or
   a different kernel structure for small-N dense.
3. **Backward path** is not implemented for V6 NAX. Falls back to
   `mx.vjp(SDPA)`.
4. **Causal path** uses `loopForwardSingleCausal` (legacy MPP). V34 not
   yet ported to causal — Apple's reference at
   `steel_attention_nax.h:278-303` shows the masking pattern; mechanical
   port for a future sprint.
5. **V34 doesn't yet write `lse`** (logsumexp output). Backward via
   `mx.vjp(SDPA)` doesn't need it, but any user reading L from V34
   output would get uninitialized data.

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

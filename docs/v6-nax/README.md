# V6 NAX — overview (v2.29.0)

V6 NAX is the mlx-mfa attention path targeting Apple M5+ Neural Accelerators
(NAX) via Metal Performance Primitives (`mpp::tensor_ops::matmul2d`). It's the
fastest path on M5+ for the dense self/cross-attention production shapes
shipped in mlx-mfa.

## Architecture (post-Sprint 3.3)

Two kernel variants, selected per-call by the primitive:

| Variant | When | Source |
|---|---|---|
| **`loopForwardSingleTile()`** (Apple-style) | Default for non-GQA (`Hq == Hk`). Single cS, kBlocks=1, always-bypass cP cooperative tensor, `mem_none` barriers, K-loop step BK. | `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (~270 LOC, added in Sprint 3.3) |
| **`loopForward()`** (legacy double-buffer) | Fallback for GQA (`Hq != Hk`) — the BHND rewriter doesn't yet handle per-head K-stride for single-Otile. Also reachable via `MFA_V6_NAX_SINGLE_OTILE=0`. | `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (legacy, from Draw Things port) |

Auto-tuned default tile config (Sprint 3.3 autoresearch on M5 Max):

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

## Performance (M5 Max)

5 production VSR/DiT shapes, single-Otile + auto-tuned tiles:

| Shape | V6 NAX | SDPA | V6/SDPA |
|---|---:|---:|---:|
| FlashVSR-dense (D=64)  | 1.11 ms  | 0.91 ms  | 1.22× |
| LTX2-cross (D=64)      | 1.59 ms  | 1.33 ms  | 1.20× |
| SeedVR2-small (D=128)  | 276 ms   | 185 ms   | 1.49× |
| CogVideoX (D=128)      | 3060 ms  | 2275 ms  | 1.35× |
| SeedVR2-large (D=128)  | 8392 ms  | 4067 ms  | 2.06× |

Numerical: SeedVR2-large RMSE 5.79e-5 → 2.93e-6 (20× more stable) under
the new defaults — single-buffer commits each row reduction before the
next K-tile overwrites, eliminating cross-tile FP16↔FP32 rounding error.

## Sprints chronology

| Sprint | Outcome | Doc |
|---|---|---|
| 2A | BHND layout migration via post-gen rewriting (default since 2026-05-04) | [bhnd-migration-report.md](bhnd-migration-report.md) |
| 2B | Chunked-K dispatch — empirically NO-GO (gains ≤ 4.5%, below 3% threshold) | [sprint-2b-chunked-k-analysis.md](sprint-2b-chunked-k-analysis.md) |
| 3.1 | Causal masking — V6 already optimal (Scenario A); no change | [causal-masking-analysis.md](causal-masking-analysis.md) |
| 3.2 | bypassThreadgroupMemory at legacy tiles — Cas C (regressed); kept off | [sprint-3-2-bypass-tgmem-results.md](sprint-3-2-bypass-tgmem-results.md) |
| 3.3 (main) | Single-Otile rewrite — bimodal at default tiles (Cas B) | [sprint-3-3-single-otile-results.md](sprint-3-3-single-otile-results.md) |
| 3.3 (autoresearch) | Tile sweep — invalidated 3.3's "D=128 ceiling" conclusion; new defaults ship 25-70% gains across all 5 shapes | [sprint-3-3-autoresearch-results.md](sprint-3-3-autoresearch-results.md) |

## Limitations

1. **GQA shapes** (Hq != Hk) use the legacy double-buffered `loopForward`. Porting BHND rewriter to handle per-head K-stride for single-Otile is a backlog item (~30 min).
2. **D=128 long-N residual gap**: ~1.35-2.06× SDPA. Cause unconfirmed post-autoresearch — likely intrinsic MPP overhead vs Apple's NAXFrag::mma. Closing it would require API switch (NAXFrag, like Apple's `steel_attention_nax.h`).
3. **Backward path** is not implemented for V6 NAX. Falls back to `mx.vjp(SDPA)`.
4. **Causal path** uses `loopForwardSingleCausal` which has not been rewritten in single-Otile style. Still the legacy double-buffer kernel, but Sprint 3.1 verified it already implements all three Apple-style causal-skip optimizations.

## Lessons logged

- **Sprint 3.3 main result was wrong** because we didn't sweep BQ before declaring "D=128 at MPP ceiling". The autoresearch closed the bulk of the V6/SDPA gap with zero kernel changes — just a different default. Logged in `SESSION_LOG.md`. Future architectural-conclusion calls must pass through a parameter sweep at the API boundary first.

## References

- Apple `steel_attention_nax.h` — `.venv/lib/python3.11/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`
- Draw Things port — original `NAAttention*` headers under `csrc/mfa/v6_nax/`
- Bench scripts — `bench/v6_*.py`
- Env vars — [`env-vars.md`](env-vars.md)

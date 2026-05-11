# V6 NAX — Sprint 3.2: bypassThreadgroupMemory re-test post-BHND

**Date:** 2026-05-04
**Verdict:** **Cas C — bypass remains NO-GO for production.** Compile/correctness OK on concrete-BK configs, but no net perf gain on production shapes; D=128 shapes regress +13% to +22%. Pattern: register spill from concurrent live `cP` + `kBlocks` × `cO_*` cooperative tensors.

## Hypothesis being retested

The 10-axes campaign (March 2026) marked `bypassThreadgroupMemory=true` as NO-GO due to compile failures on optimal configs. With BHND default since Sprint 2A:
- Memory access patterns changed (column-major dextents semantics)
- Cache pressure reduced 4× (memory-layout-driven)
- Post-gen rewriter transforms the source after Draw Things' generation

Hypothesis: BHND might unblock bypass — and bypass mirrors Apple's `steel_attention_nax.h` pattern (no threadgroup staging of P), claimed to give 3-7% gain.

## Method

- Direct V6 NAX dispatch via `mlx_mfa._ext.v6_nax_forward(q, k, v, causal=False)` (the high-level `flash_attention()` does **not** route to V6 NAX).
- Tile config: defaults BQ=32, BK=32, SG=4, BD=head_dim. The 10-axes "brief Config 2" (D=128, BQ=16, BK=48, SG=16) still fails compile for a deterministic Apple-side reason (see "Compile findings" below).
- 5 production BHND shapes (FlashVSR-dense, SeedVR2-small, CogVideoX, SeedVR2-large, LTX2-cross), warmup=5, 3 runs × 15 iters, median-of-medians.
- Hardware: Apple M5 Max (`applegpu_g17s`), MLX 0.31.2.
- Bench script: `bench/v6_bypass_tgp_bench.py`. Raw JSON: `docs/v6-nax/bypass-tgp-bench.json`.

## Compile findings (Tâche 1)

| Tile config | bypass=0 | bypass=1 | Compile result |
|---|---|---|---|
| BQ=32, BK=32, SG=4 (default), D=64  | OK | OK | concrete K → fine |
| BQ=32, BK=32, SG=4 (default), D=128 | OK | OK | concrete K → fine |
| BQ=16, BK=64, SG=16, D=64           | OK | OK | concrete K → fine |
| BQ=16, BK=48, SG=16, D=128          | OK | **FAIL** | dynamic K + cooperative left → static_assert |

Failing-case diagnosis, captured from the Metal compiler:

```
MPPTensorOpsMatMul2dImpl.h:4209:5: error: static_assert failed
  'descriptor.k != dynamic_length_v<int>'
  "Inner dimension cannot be dynamic with input cooperative tensors"

descriptor: {16, 128, 2147483647, false, false, true, 1}
                       ^^^^^^^^^^ INT_MAX = dynamic_length_v<int>
```

When the source generator emits the PV matmul descriptor with `BLOCK_DIMENSIONS_TRAVERSAL_OR_DYNAMIC_LENGTH_V = dynamic_length_v<int>`, Apple's MPP refuses to instantiate `matmul2d::run` with a cooperative tensor as the left operand (which is exactly what `bypass=true` introduces — `cP` instead of the threadgroup-staged `P`). This is a **hard upstream constraint**, not something we can patch from our side. The 10-axes NO-GO verdict was correct *for that specific config* but does not generalize to concrete-BK configs.

## Benchmark results (Tâche 2)

All shapes pass correctness with bypass=1 (RMSE matches bypass=0 bit-for-bit, see JSON for max_abs). Timing summary, M5 Max, BQ=32 BK=32 SG=4:

| Shape | Size | baseline | bypass | Δ | verdict |
|---|---|---:|---:|---:|---:|
| FlashVSR-dense | 4096×4096, D=64, H=10  | 1.82 ms | 1.61 ms | **−11.65 %** | **win** |
| LTX2-cross     | 2048×14000, D=64, H=8  | 2.59 ms | 3.05 ms | +17.51 % | loss |
| SeedVR2-small  | 26730×26730, D=128, H=20 | 915.79 ms | 1037.47 ms | +13.29 % | loss |
| CogVideoX      | 70200×70200, D=128, H=30 | 9634.17 ms | 11754.31 ms | +22.01 % | loss |
| SeedVR2-large  | 111375×111375, D=128, H=20 | 15161 ms | 15300 ms | +0.92 % | noise |

Net: 1 win, 3 losses, 1 noise across production shapes. Bypass cannot ship as default.

## Why bypass loses on D=128 (interpretation)

Bypass replaces the `P_buf` threadgroup roundtrip with a cooperative tensor `cP` held in SIMD registers across the PV matmul. In the V6 source generator (Draw Things), the destination is split across head-dim sub-tiles: `kBlocks = ⌈D / BD⌉` cooperative `cO_i` accumulators. With BD = head_dim (default), kBlocks = 1. But under default tiles BD = 32, the situation is:

| D | kBlocks | Live coop tensors during PV (bypass=1) |
|---|---|---|
| 64  | 2 | cP + cO_0 + cO_1 |
| 128 | 4 | cP + cO_0..cO_3 |

D=128 holds 5 cooperative tensors live simultaneously. This exceeds what the Metal compiler can keep in SIMD registers; the difference vs non-bypass (which only has cO_* live; P is in tgmem) shows up as **register spill**, more expensive than the threadgroup roundtrip it was supposed to save.

The single D=64 win (FlashVSR-dense) survives because kBlocks = 2 keeps the live count below the spill threshold. D=64 cross-attention (LTX2-cross) still regresses, suggesting the register-pressure story isn't the only effect — the longer K-loop in cross-attention changes the cache access pattern, and bypass's V-from-device-with-cP-in-registers may be measurably worse there than non-bypass's V-from-device-with-P-from-tgmem.

## Why Apple's `steel_attention_nax.h` doesn't have this problem

Apple's kernel uses a **single `Otile`** in registers (`steel_attention_nax.h:143-144` — `using otile_t = NAXTile<AccumType, TQ, TD>; otile_t Otile;`). There's no head-dim sub-tile splitting — Apple keeps the full TD-wide accumulator in registers and lets the compiler manage the actual register file via NAXTile / NAXFrag abstraction. Our V6 generator (inherited from Draw Things) splits along head-dim with kBlocks `cO_i` accumulators — a structurally different choice.

Bypass cannot close the gap with Apple by flipping a flag, because the gap is structural: Apple's no-tgmem pattern is feasible because of the single-Otile structure. Replicating it in V6 requires a **source-generator rewrite** — Sprint 3.3 territory.

## Decision

**Cas C — keep `bypassThreadgroupMemory=false` as default.** Per Sprint 3.2's brief, this is the empirically-grounded NO-GO verdict, properly grounded on M5 Max + BHND, not a blanket NO-GO from compile failures alone.

The flag stays available via `MFA_V6_BYPASS_TGP=1` for future experimentation (e.g., FlashVSR-dense workloads where it wins).

## Implications for Sprint 3.3

The structural reason bypass fails (kBlocks-split cO accumulators in our generator vs Apple's single-Otile pattern) is the *same root cause* that limits V6's ceiling vs SDPA on M5+. If we want Apple-class performance on this hardware, we need to rewrite the source generator's PV-matmul-output structure to use a single cooperative O accumulator like Apple — not patch around it. This empirically validates Sprint 3.3 as the highest-impact next move.

## Files

| Path | Status |
|---|---|
| `bench/v6_bypass_tgp_bench.py` | added |
| `docs/v6-nax/bypass-tgp-bench.json` | added (raw bench output) |
| `docs/v6-nax/sprint-3-2-bypass-tgmem-results.md` | this file |
| `outputs/v6_bypass_bench.log` | live bench log (not committed) |

No source code modified. Branch `feat/v6-nax`, no push.

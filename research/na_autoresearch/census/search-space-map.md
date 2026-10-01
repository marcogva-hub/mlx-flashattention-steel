# V6-NAX autoresearch — search-space map (Volet 2 census, read at source)

7-agent read-at-source census (raw in `census_raw.json` / `journal.jsonl`). Cell counts are
coarse-grid screening estimates per family.

## Knob-driven axes (env-swept, NO rebuild) — the night's material
All read via `getenv`/`getenv_aliased` per dispatch (→ new cache key → Metal JIT recompile, no C++
rebuild). **Dense tile axes** (`mfa_v6_nax_primitive.cpp`):
| knob | default | domain (source) | note |
|---|---|---|---|
| `MFA_V6_NAX_BQ` | D64 32 / D128 64 | `BQ %(WM*16)==0` (:368) | **TQ>1 OK** — see correction below |
| `MFA_V6_NAX_BK` | 32 (all D) | mult of 32 (:384) | **BK=128 = M5 doctrine target, UNTESTED** (#1 axis) |
| `MFA_V6_NAX_WM` | D64 2 / D128 4 | `32*WM ≤ 1024` (:91) | D128 WM=4 already = 2×2 doctrine |
| `MFA_V6_NAX_D_SUBTILE` | head_dim | pinned to head_dim for D<256 | **0 DOF** at D64/128 (opens at D≥256) |
Orthogonal correctness-neutral perf knobs: `MFA_V6_RELAXED_PRECISION` {0,1}, `MFA_V6_UNROLL_MODE`
{full,none,2,4}, `MFA_V6_MAX_THREADS` (occupancy buckets), `MFA_V6_FORCE_DYNAMIC_K` {0,1}. Routing gates:
`MFA_V6_DENSE_MIN_N` (2048), `MFA_DISABLE_V6_DENSE`.
**GHOST knobs (DO NOT sweep)**: `MFA_V6_BLOCK_R/C`, `MFA_V6_EXEC_SG`, `MFA_V6_BYPASS_TGP` — vestigial
post-F-3 (dead simdgroup path; runtime-fingerprint-proven not to steer the NAX kernel). The old
"BQ=16 wins / BQ=64 catastrophic 4×" note describes THIS dead path and is **false** for the NAX kernel.

### ⚠ RULE-16 correction (census label refuted at source + fp32 oracle)
Two agents reported a `TQ==1` static_assert restricting tiles to `BQ==WM*16` (3 pairs). **Refuted**:
`NAAttentionKernel.cpp:2809` is `const int TQ = BQ/(WM*kU); // expected = 1 …` followed by `(void)TQ;`
— a discarded computation with a comment, **not an enforced assert**. Empirical: 180/180 configs incl.
TQ=2 ran correct+engaged (raised=0); TQ=2 `(128·32·4)` → cos_vs_fp32 = 0.999998. **Valid domain =
`BQ%(WM*16)==0`** (broader than doctrine-safe). Trusting the label would have under-searched dense.

## Codegen axes (need a source-generator variant + recompile) — Phase-1+, MAP ONLY
8 axes in `createV6NAXSource` (`NAAttentionKernel.cpp`): (1) simdgroup layout (WM×1 band vs 2×2/2D
cooperative), (2) **Morton/Z-order grid walk — currently NOT implemented for dense fwd** (plain tgid;
3rd M5-doctrine element, open question), (3) K/V buffering (single vs double/prefetch — currently
single), (4) barrier placement/count (3 barriers, mem_none), (5) MMA fragment grouping + unroll,
(6) D-subtile replay-vs-stage (D≥256 only), (7) `kU=16` MMA width (hardcoded), (8) tgmem bypass.
All `CODEGEN_REBUILD`. None env-swept.

## Families (coarse screening cell counts)
| family | cells | terminal | notes / open questions |
|---|---|---|---|
| **dense fwd D64/D128** | 108 (doctrine) / broader empirical | `v6_nax_forward` (NAX matmul2d) | prod route D128-only; D64 needs force path; BK=128 untested |
| sparse BT32 + causal | 288 | `v6nax_sparse` (separate JIT) | WM pinned 1 (perf, not correctness); BT pinned 32; doctrine not applied |
| backward D64 + decode | 750 | 3-kernel split bwd (D64); decode → **STEEL V2 (no own NAX kernel)** | bwd BK=32/WM=2; decode carveout is parity-by-identity |
| varlen + GNA + conv3d | 202 | NAX matmul2d / MPP conv | varlen PINS BQ32/BK32/WM2 via Python; Morton IS on for single-Otile |
| frontiers D256/D512/int8/bf16 | 480 | **SDPA (permanent gates)** | D256/D512 → SDPA (`999_999` thresholds); NO int8 NAX kernel (§AA.5 kill-gate passed); all M1/M2 optima unverified on M5 |
| **total coarse** | **~1,828** | | |

## Prioritized night ordering (payoff × cost, screening)
1. **dense D64/D128 tiles** (cheap, prod-critical, BK=128 untested) — NIGHT-1 priority.
2. dense orthogonal knobs (RELAXED/UNROLL/MAX_THREADS) — fills NIGHT-1.
3. sparse BT32 (needs executor) — NIGHT-2.
4. backward D64 (default-on, high traffic; needs executor).
5. frontiers (D256/D512/int8) — highest-risk, re-test the "permanent SDPA" M1/M2 gates on M5.

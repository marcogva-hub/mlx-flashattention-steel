# Campaign 2026-06 — Sprint C report: autonomous discovery

**Date**: 2026-06-12 · **Status**: COMPLETE · Commits: `acab95f` (Tracks 1+2),
Track 0 docs, Track 6 refactor, this report.

## Headline

What prior sessions missed, found and shipped this sprint:

1. **D=256 causal dispatch inversion on M5** (Track 2): the M4-era
   promote (1.58-1.68×) inverts — MFA V2-dsplit is 0.69-0.73× SDPA
   across a 9-cell shape grid.  Promoted: M5+ routes D=256 causal to
   SDPA → **1.38× speedup** on the auto path at the probe shape, output
   diff 0.0.  (Audit-framing-inversions class #1/#6; M1/M4-era verdict,
   never re-benched on M5 until now.)
2. **6 hot-path wins** (Track 1): paged-RoPE via mx.fast.rope (kills
   per-decode-step mx.compile churn), TQ decode 3-sync elimination
   (bit-identical outputs, ~2%), block-table caching (numpy round-trip
   per token → per block-fill), rope-cache bound, conv padding memo.
3. **Track 6 CacheKey layer shipped** (Marco-authorized): all 11 C++
   key structs migrated to tie()-derived ==/hash — the C1/C6 omission
   class is structurally impossible now; static invariant test green
   throughout; perf-neutral (<1%).
4. **Track 0**: 83-knob ledger; 1 new ghost (`MFA_TOPK_BISECT`), 2
   stale-documented corrected, 10 undocumented knobs documented, v2.30
   EXEC_SG sweep results formally invalidated.

## Track 2 — M5 dispatch re-bench matrix (key rows)

Forward (B=1 H=8 fp16, ms, 3-block median):

| Shape | SDPA | auto | V2 | V3 | V4 | V5 | Verdict |
|---|---|---|---|---|---|---|---|
| D64 N4096 c | 0.54 | 0.57 | 1.54 | 1.52 | 1.58 | 1.73 | SDPA-optimal HOLDS (M5 numbers fresh) |
| D128 N4096 c | 1.06 | 1.06 | 3.55 | 3.56 | 3.31 | 3.65 | HOLDS; V4>V2 at D=128 now (M1-era V-ordering inverted, academic — all ≥3× behind SDPA) |
| D256 N4096 c | 4.62 | 6.46→**4.67** | 6.43 | — | — | — | **INVERTED → FIXED** |
| D256 N4096 nc | 3.53 | 3.54 | 11.94 | — | — | — | HOLDS (already SDPA) |
| D512 N4096 c | 6.97 | 7.13 | n/a | — | — | — | HOLDS (SDPA-conservative correct) |

V3/V4/V5 conclusion: all remain 3-4× behind Apple SDPA NAX on M5 at
D≤128 — the M1-era "experimental, not promoted" verdict holds with
fresh M5 evidence; the V-variant relative ordering shifted (V4 now best
at D=128) but is academic.

## Track 2 — `MFA_FORCE_NATIVE_BWD` matrix (Marco-gated; AWAITING SIGN-OFF)

Causal fp16 backward, ms (B=1 H=8):

| Cell | SDPA-vjp (default) | V34 (opt-in env) | STEEL-bwd¹ | Winner |
|---|---|---|---|---|
| D64 N2048 | 2.98 | **1.37 (2.2×)** | 2.68¹ | V34 |
| D64 N4096 | 11.53 | **4.50 (2.6×)** | 8.36¹ | V34 |
| D64 N8192 | 44.54 | **17.47 (2.5×)** | 31.30¹ | V34 |
| D128 N2048 | **3.32** | 5.71 | 6.56¹ | SDPA-vjp |
| D128 N4096 | **12.44** | 24.09 | 23.07¹ | SDPA-vjp |
| D128 N8192 | **49.05** | 106.3 | 88.57¹ | SDPA-vjp |

¹ STEEL-bwd is UNREACHABLE via the public API on M5 (forward routes to
SDPA → the env var never engages); measured via backend="mfa".  Post-KD-5
fix it is CORRECT everywhere (rmse 4e-5 incl. the previously-zeroed
D=128 N≥2048 cells) and 1.12-1.42× faster than SDPA-bwd at D=64 on the
forced path — but dominated by V34 there and by SDPA-vjp at D=128.

**Recommendation for Marco** (defaults UNCHANGED pending sign-off):
1. **Auto-promote V34 backward at D=64 causal** (qL≥2048, fp16/bf16,
   M5+): 2.2-2.6× training speedup, currently opt-in.  This is the
   single largest unexploited win in the repo.
2. **Keep `MFA_FORCE_NATIVE_BWD` deprecated**: STEEL backward is now
   correct (KD-5 fixed) but dominated at every cell by V34 or SDPA-vjp;
   the deprecation rationale shifts from "broken" to "superseded".

## Tracks 3+4 — survey verdicts

6 of 8 literature families died at the applicability filter (full
table + citations in the survey record): FA-3 warp-specialization (no
Metal TMA analogue; async_copy removed in macOS 26), FlashDecoding++
(Apple sdpa_vector_2pass covers dense decode to D=256), vAttention (no
Metal page remap), NSA (model-training method), online-softmax numerics
(repo already fp32-accum + base-2 LSE; historical bugs were convention,
not accuracy), 3D-Winograd conv (shipped implicit-GEMM already at
63-87% NAX peak on 5/6 shapes).

MLX 0.31.2 surface re-check: `sinks` param + mxfp4/mxfp8 quant modes
are new; **no inversion deletions available** (paged/varlen/TQ/backward
custom kernels remain Apple-uncovered; mlx-lm sinks fallback confirmed
optimal).  Fused-backward kernels: still ZERO in mlx.metallib.

Surviving prototype-worthy candidates (Marco-gated kernel sprints, per
§AA.5 institutional workflow):

| Candidate | Premise state | Gate |
|---|---|---|
| **A. Sage-NAX int8 attention** | MPP int8×int8→int32 matmul2d VERIFIED in headers; Draw Things v2 reference impl exists; theoretical 1.3× (QK-only) to 1.7× (both GEMMs) vs SDPA NAX; repo sage path currently ~7-9× from int8 ceiling | 30-min microbench REQUIRES the repo's MSL4 compile path (MPP headers cannot load through mx.fast.metal_kernel — verified this sprint); kill threshold int8 < 1.3× fp16 sustained |
| **B. Top-K streaming kernel (Approach 5)** | §AA.5 CONFIRMATION verdict already on record; design + register budget GREEN (phase-3b doc); remaining gap 3.5× vs dense SDPA | ~6h scoped kernel sprint |
| **C. Conv3D small-K retune** (up3_resnet, 41% peak) | Gap VERIFIED; fixability UNCERTAIN (may be a small-K utilization floor) | Phase-1.1 microbench grid restricted to K≤4096 |

## Promoted / declined ledger

Promoted (each three-axis validated, committed):
D=256 M5 dispatch fix (1.38×) · paged-RoPE fast path · TQ 3-sync
elimination (~2%, bit-identical) · block-table cache · rope-cache bound
· conv padding memo · Track 6 tie() refactor (perf-neutral).

Declined with evidence: dense-decode early-exit (3.4µs/call measured;
Rule-8 erosion), TQ searchsorted (no primitive), dummy-L alloc /
env-read-per-closure / varlen per-seq dispatch / GNA mask construction
(micro or cold on M5), V3/V4/V5 promotion (3-4× behind SDPA, fresh M5
numbers), all 6 dead literature families (table above).

Contradiction resolved: a sub-agent probe claimed MFA 1.42× at D=256
causal (one shape) — did not reproduce under 3-block-median; the 9-cell
grid at 0.69-0.73× is authoritative.

## Rollbacks

None final.  The TQ sync removal was provisionally implicated by one
post-rebuild timing transient (+13%), re-measured 2×2 → actually ~2%
faster; kept.

## Track 5 — fixed point

Post-change sweep: 3× consecutive full-suite runs green (1366); fresh
cache-site grep shows no new cache constructs beyond the tie-migrated
set; the static invariant test now structurally guards all 11 C++ keys;
all surveyed candidates carry dispositions.  **Zero new actionable
candidates remain that are not Marco-gated.**  Fixed point reached for
the autonomous scope.

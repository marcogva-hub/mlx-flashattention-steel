# Phase IV — Code Review (Correctness Pass), both repos

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High (5-agent parallel review + Part-B agent)
**Repos:** mlx-mfa-v2 @ `b1d81bf` (primary) + m5max-deep-dive (investigation).
macOS 26.6 (25G5028f), Apple M5 Max, mlx 0.31.2, Metal Toolchain 32023.

## Headline

**mlx-mfa: NO CRITICAL (default-reachable) correctness bug at HEAD.** Highest-severity material is
**one HIGH latent** int32-overflow in the legacy/MPP V6 NAX attention path (V6NAX-off + multi-GB
shapes only) — fixed. The 4 directed MLX/Metal target classes (matmul2d type-matching, `vec<>`
namespace, K%32, 128-byte align) are **all NOT-EXPOSED** in mlx-mfa, with evidence. The
investigation repo had **2 CRITICAL** published-number/methodology bugs + 2 HIGH — all fixed.

## Consolidated findings

| ID | repo | sev | status | one-line | disposition |
|---|---|---|---|---|---|
| A1 (hot-path) | mlx-mfa | — | CLEAN | routing/mask/IV-D1-D2 eval-collapse all correct vs fp32; bit-identical deferral re-confirmed | — |
| A2 (runtime) | mlx-mfa | — | CLEAN | cache keys complete; vjp unchanged; IV-D1/D2 lifetime safe (graph-input dep) | — |
| **A3-1** | mlx-mfa | **HIGH** | CONFIRMED | legacy/MPP V6 NAX device offsets `tgid.z*batch_stride` etc. are 32-bit → wrap >2³² (silent wrong-mem); latent (V6NAX-off + multi-GB / >4M varlen tokens) | **FIXED (Wave M1)** |
| A3-2 | mlx-mfa | LOW | SUSPECTED | `mpp_int8_bench.mm` accumulator `"int"` vs `"int32_t"` (compiles; int32_t==int on Apple) | FIXED (M3) |
| A3 targets 1–4 | mlx-mfa | — | NOT-EXPOSED | matmul2d type-match (Q=K=V single prec, dests float, non-const inputs); no bare `vec<`/`bfloat16_t`; K∈{16,32,64,128} %16 or dynamic_length; no host MTLTensor (no 128-byte-align surface) | — (evidence in A3) |
| A4 (STEEL/dispatch) | mlx-mfa | — | CLEAN | III-9 lifetime + OOB clamps + KernelKey completeness + Pattern-#9 parity all HOLD | — |
| A4-1/A4-2 | mlx-mfa | LOW | CONFIRMED | V3 generator stale comments (BK 64/32→32/16, TGP bytes) + false "sparse" capability claim | FIXED (M3) |
| **A5-1** | mlx-mfa | **MEDIUM** | CONFIRMED | V3 conditional-auto (production-reachable) had NO independent-oracle correctness test (only forced-V3 sub-threshold vs V2) | **FIXED (Wave M2)** |
| A5-2/A5-3 | mlx-mfa | LOW | CONFIRMED | IV-D perf claims not in §Z registry (internal A/B — no-action); stale v2.50 MIGRATION lines | A5-3 FIXED (M3); A5-2 no-action |
| **B-1** | m5max | **CRITICAL** | CONFIRMED | `compute_speedup.py` `for size` loop filters hardcoded `M==32&K==32` (matches all rows; CSV is fixed 32³ tile) → 3 identical duplicate rows; implies a size sweep the data lacks | **FIXED + validated** |
| **B-2** | m5max | **CRITICAL** | CONFIRMED | `Matmul.swift` DIAG binds C as `UInt16` for the INT8 (int32) path → reads half the tile / splits accumulators | **FIXED (swift build clean)** |
| B-3 | m5max | HIGH | CONFIRMED | NA coverage gate checks "non-zero" not correctness (the Day-I partial-tile hole) | FIXED (full-coverage + uniformity oracle) |
| B-4 | m5max | HIGH | CONFIRMED | `bf16_routing.py` bf16 input is fp32→fp16→bf16 (double mantissa loss); biases the accuracy column (throughput unaffected) | FIXED (cast fp32→bf16 directly) |
| B-5 | m5max | MEDIUM | CONFIRMED | unguarded metric-column access + bare `except: pass` silently drops an LLM row | FIXED (existence-check + loud skip) |
| B-6 | m5max | LOW | CONFIRMED | `glob[0]`/`glob[-1]` file selection (lexical, not newest) | FIXED (`_newest` by mtime) |

## Wave-by-wave fix record (mlx-mfa — revertible commits, suite green after each)

| Wave | commit | what | suite |
|---|---|---|---|
| M1 | `011e34a` | A3-1: widen ALL 14 V6 NAX device-offset multiplies uint→ulong (multi-gate §AA.5.x: fwd/bwd/L/mask/varlen). Rebuilt. | 1821 ×2 |
| M2 | `4c88f02` | A5-1: V3 conditional-auto correctness test vs independent fp32 (windowed-causal at the auto regime, GQA, backend=mfa) | +6 tests |
| M3 | `ec51628` | A4-1/A4-2/A3-2/A5-3: stale V3 comments, false sparse claim, bench int32_t, MIGRATION superseded banner | 1827 |

**Three-axis for the M1 dispatch-path change:** output sanity = the fp32-oracle suite tests pass
(1821 ×2); path entered = pure arithmetic widening (no routing/codegen-structure change); edges =
normal shapes pass, the >2³² edge is the fix target (not directly testable at multi-GB — correct
by construction, matching the V6NAX path's existing 64-bit arithmetic + the conv path's <2³¹ guard).

## Investigation-repo fixes (m5max-deep-dive — NOT a git repo: working-tree edits, no commit)

- `analysis/compute_speedup.py` — B-1/B-5/B-6. **Validated** (stdlib oracle, pandas unavailable):
  the 7 (dtype,backend) groups' new group-by-tile median == old `M==32&K==32` median **exactly**
  for all 7 → speedup VALUES preserved; the old `for size` loop emitted each row **3×** (21→7).
- `benchmarks/micro/swift_metal/.../Matmul.swift` — B-2/B-3. **`swift build` clean.** Per-dtype C
  reader (int32 for int8) + full-coverage + uniformity gate (uniform inputs ⇒ uniform fully-written
  tile; catches the Day-I partial-tile class). Needs an M5 bench *run* to exercise the gate.
- `benchmarks/meso/bf16_routing.py` — B-4. bf16 input now cast fp32→bf16 directly (syntax OK).

### Published numbers flagged for re-measurement
1. **Matmul M1→M5 speedup table** (`results/speedup_table.csv` → paper figures): VALUES are correct
   (stdlib oracle), but **regenerate to dedup** (21→7 rows) and **correct any "1024/2048/4096 size
   sweep" framing** — the matmul bench is a fixed **32×32×32 tile**, not a size sweep.
2. **NA INT8 TOPS** (97.1 TOPS / 1.876×): the coverage gate that validates it was blind on the
   upper int32 half + non-zero-only. **Re-run the NA INT8 bench with the new gate** to confirm
   full-coverage+uniformity before trusting the published INT8 peak.
3. **Any bf16 ACCURACY figure** sourced from `bf16_routing.csv rel_err`: was biased by the
   fp32→fp16→bf16 double cast — re-measure with the direct cast. (The bf16≈fp16 THROUGHPUT claim is
   unaffected.)

## New skill

`/mlx-mfa-nax-matmul2d-correctness` (created) — encodes the 4 directed MLX/Metal matmul2d/NAX
footgun classes as a pre-kernel checklist (type-matching, `device const T*` propagation, int8
`int8_t`/`int32_t`, `vec<>`/`bf16` include, K%16/dynamic_length, 128-byte align, device-offset
int64 widening). Pays forward to V6 NAX work.

## Diagnostic-only ladder (feeds the optimization pass) — NONE

All confirmed findings were fixable in-wave (HIGH widening + MEDIUM test + LOW cosmetics + the
investigation-repo methodology). No correctness item was deferred as diagnostic-only. The
optimization pass inherits the IV-0 structural ledger (decode is sync-floor-bound; IV-D1/D2 closed
the eval-floor lever) — not a correctness backlog.

## Release disposition

The A3-1 widening (Wave M1) is a real latent-correctness fix → additive to the held **v2.56.0**
scope. M2/M3 are test+docs. Investigation-repo fixes are local (no release). Marco-gated tag.

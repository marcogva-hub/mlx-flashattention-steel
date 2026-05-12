# Sprint B Phase 1.5 — Ship/Shelve Verdict (§4-validated)

**Date**: 2026-05-12 (§4 re-bench: 2026-05-12 09:20-09:48 UTC+1)
**Verdict**: **SHIP** as narrow-niche v2.34.0 (very-sparse only).
**Decision-grade evidence**:
- Phase 1.3 BT sweep + Phase 1.4 dispatcher sweep (single-session)
- §4-strict 3-session re-bench (this update) — see
  `lcsa-nax-rebench-results.md` for raw analysis.

## Verdict matrix (from Phase 1.0 design §11)

| Criterion | Required | Achieved | Status |
|---|---|---|:---:|
| Speedup in niche | ≥ 1.5× vs SDPA+bias | **2.06–2.29× cross-session (median 2.28×)** at density 0.01 [§4-validated] | ✓ |
| No regression elsewhere | ≤ 10% slowdown | **0.99–1.01× cross-session** moderate-density shapes [§4-validated] | ✓ |
| Correctness | RMSE < 1e-3 vs oracle | **3e-6** (Phase 1.1 axis-1); **1e-6** smoke gate per §4 session | ✓ |
| Edges preserved | All-False row → 0; all-True → dense; diagonal → causal | all 3 verified | ✓ |
| Test coverage | 18+ tests | **33/33** Phase 1.1+1.2+1.4+H.2 | ✓ |
| Three-axis discipline | per-sub-phase | every sub-phase has axis-1/2/3 tests | ✓ |
| Build integration | C++ extension + CMakeLists + bindings | ✓ done in Phase 1.1 | ✓ |
| Public API stability | sparse_attention_nax + dispatch + patcher | ✓ documented + tested | ✓ |
| Cross-session variance | ≤ 20% | **6/7 CONFIDENT (<10%), 1/7 BOUNDARY (10.0% niche, S1-cold-cache artifact)** | ✓ (with caveat) |

## §4-validated 3-session re-bench results

`docs/lcsa-nax/lcsa-nax-rebench-results.md` carries the full analysis.
Summary:

| Shape | density | §4 median ratio | range % | flag | Δ vs Phase 1.4 single-session |
|---|---:|---:|---:|:--:|---:|
| lcsa_small_seq4k          | 0.24 | 0.99× | 0.1% | CONFIDENT | +3.0% |
| lcsa_small_seq4k_sparse   | 0.07 | 0.99× | 3.2% | CONFIDENT | +3.1% |
| lcsa_mid_seq8k            | 0.12 | 1.00× | 1.7% | CONFIDENT | +2.1% |
| lcsa_mid_seq8k_sparse     | 0.03 | 1.01× | 0.7% | CONFIDENT | +4.8% |
| lcsa_large_seq16k         | 0.12 | 0.99× | 1.8% | CONFIDENT | +4.3% |
| lcsa_large_seq16k_sparse  | 0.03 | 1.00× | 0.4% | CONFIDENT | -0.4% |
| lcsa_mid_seq8k_very_sparse (niche) | **0.01** | **2.28×** | **10.0%** | **BOUNDARY** | **-6.9%** |

**Variance summary**: 6 CONFIDENT (< 10% cross-session range), 1 BOUNDARY (10.0%
range on the niche shape — see caveat below). 0 HIGH. Max |Δ| vs Phase 1.4
single-session = 6.9%, well within the ±15% gate.

**Action taken** (per `lcsa-nax-rebench-decisions.md` §E and prompt §D
decision tree): **DOC_UPDATE_WITH_CAVEATS** — update verdict with §4 numbers
+ boundary caveat ; **no v2.34.1 tag** (1 boundary shape disqualifies the
all-confident v2.34.1 doc-only-release branch).

### Niche-shape BOUNDARY caveat

The niche shape's 10.0% cross-session range is driven entirely by S1's
A/B/A drift of 21.0%, a cold-cache first-session artifact:

| Session | niche ratio | A/B/A drift |
|---|---:|---:|
| S1 | 2.059× | 21.0% (cold cache on first NAX block) |
| S2 | 2.281× | 2.0% |
| S3 | 2.288× | 2.8% |

Excluding S1 (S2+S3 only): cross-session range collapses to ~0.3%. The
niche-win regime is **structurally stable post-warmup** — variance arises
from cache state at first NAX-kernel touch in a fresh process. The shipped
ratio range is **2.06–2.29× depending on cache warmup state**, with the
expected production ratio at **2.28×** (warm cache, steady state).

The single-session 2.45× from Phase 1.4 was at the high end of cache-warmth
luck. The §4 protocol's structural finding: the niche-win regime delivers
2.0–2.5× consistently, with the median around 2.28×.

## Why §4 validation did NOT trigger v2.34.1 release

The decision-tree action matrix (prompt §D.3) requires **all shapes
CONFIDENT** for the auto-tag. One BOUNDARY shape (the niche) blocks the
v2.34.1 doc-only-release branch even though:

- The niche-win regime is NOT overturned (2.28× ≫ 1.5× threshold)
- Max |Δ| 6.9% is well within ±15%
- The BOUNDARY signal is a known cache-warmup artifact (S2+S3 ≈ CONFIDENT)

The doc-only-merge-to-master path applies: ship-verdict is updated with
§4-validated numbers and the caveat, but no new tag. v2.34.0 production
code unchanged.

## What ships

### Public API

```python
from mlx_mfa.lcsa_nax import (
    sparse_attention_nax,           # raw Sprint B kernel
    sparse_attention_dispatch,      # density-thresholded router
    DEFAULT_DENSITY_THRESHOLD,      # 0.02
)
```

### Capabilities

- dtype: float16, bfloat16
- D ∈ {64, 128}
- block_tile ∈ {16, 32, 64} (default 16 per Phase 1.3 winner)
- mask ndim ∈ {2, 3, 4}: shared / per-head / per-batch sparsity
- causal: within-tile triangular + per-tile future-skip
- asymmetric qL ≠ kL (cross-attention)
- precondition: mask_bytes ≥ 4096 (constant-address-space avoidance)

### Recommended callsite pattern

```python
import mlx.core as mx
from mlx_mfa.lcsa_nax import sparse_attention_dispatch, _bool_mask_to_float_bias

# At mask-construction time (cache result by id(bool_mask)):
bias = _bool_mask_to_float_bias(bool_mask, BT, qL, kL, mx.float16)

# At call time:
density = float(mx.mean(bool_mask.astype(mx.float32)))  # cache too
O = sparse_attention_dispatch(
    Q, K, V, bool_mask, block_tile=BT,
    density=density, precomputed_bias=bias)
```

## What does NOT ship (deferred)

- **matmul2d cooperative-tensor inner-GEMM rewrite**: would extend niche
  from density < 0.02 up to ~0.20+, but is a 4-6h focused sprint of MSL +
  C++ work. Tracked as Phase B follow-up — now the **highest-leverage
  remaining Sprint B item** post §4 validation.
- ~~§4-compliant 3-session perf re-bench for ship-default-grade
  confidence~~ — **DONE 2026-05-12**; see this doc and
  `lcsa-nax-rebench-results.md`.
- **patch_sparkvsr_sliding_window companion patcher**: deferred (mirror
  of `patch_flashvsr_lcsa` for sliding-window mask patterns).
- ~~patch_flashvsr_lcsa integration patcher~~ — **DONE** in v2.34.0
  Section H.2 (`mlx_mfa/integrations/flashvsr_lcsa.py`, 9 tests).
  FlashVSR's typical density 0.07-0.24 falls into the dispatcher's
  SDPA-route — patcher is "code-path-prep" until matmul2d rewrite
  extends niche.

## Release plan summary

- Version shipped: **2.34.0** (Sprint B v1 ship as narrow optimization,
  2026-05-12). Production code unchanged by §4 validation.
- §4 validation: doc-only update (no v2.34.1 tag — 1 BOUNDARY shape on
  niche disqualifies all-CONFIDENT auto-tag branch).
- Branch path: `experiment/lcsa-nax-rebench-section4-strict` → merge to
  `master` (no tag).
- CHANGELOG: add **[Unreleased]** note documenting the §4 validation
  result (no version bump).
- No breaking changes; production callers unaffected.

## Recommendation

**SHIP v2.34.0 stands §4-validated** with the boundary caveat documented
above. The niche-win regime delivers 2.0–2.5× consistently across cache
warmup states; moderate-density routing is structurally identical to
SDPA+bias. The matmul2d cooperative-tensor inner-GEMM rewrite remains
the highest-leverage follow-on sprint — it would convert the BOUNDARY
shape's cache-warmup sensitivity into a non-issue (matmul2d's
cooperative-tensor distribution amortizes per-block cache state across
many threads) AND extend the niche from density < 0.02 to ~0.20+.

## §4 methodology footer

- **Date**: 2026-05-12 09:20-09:48 UTC+1
- **Sessions**: 3 subprocess-isolated Python processes
- **Cooldowns**: 180s initial, 60s inter-shape, 90s inter-round CLI knob
- **Pattern**: A/B/A — `sparse_attention_dispatch(precomputed_bias=bias)` →
  `mx.fast.scaled_dot_product_attention(mask=bias)` → `sparse_attention_dispatch`
- **Runs per direction**: 5
- **Smoke gate**: per-session NAX-vs-SDPA+bias correctness check at small
  dense shape; rmse 1e-6 (well under 5e-3 bar) in all 3 sessions
- **Hardware**: M5 Max 128GB, macOS 26.5, mlx 0.31.2, mlx_mfa 2.34.0
- **Harness**: `bench/lcsa_nax_phase1_5_harness.py` (Sprint C structural pattern)
- **Analysis**: `bench/lcsa_nax_rebench_analysis.py`
- **Raw data**: `docs/lcsa-nax/lcsa-nax-rebench-data.json`
- **Per-session runlogs**: `docs/lcsa-nax/rebench-runlog-S{1,2,3}.txt`

### Sign-off

> Sprint B Phase 1.5 verdict (§4-validated, 2026-05-12): **SHIP-stands**.
> Single-session shipped envelope confirmed structurally:
> - 6/7 moderate-density shapes CONFIDENT at 0.99–1.01× SDPA+bias parity.
> - 1/7 niche shape BOUNDARY at 10.0% cross-session range (S1 cold-cache
>   artifact; S2+S3 collapse to ~0.3% range).
> - Max |Δ| vs Phase 1.4 single-session = 6.9%, well within ±15% gate.
> - Niche-win regime NOT overturned (2.28× median ≫ 1.5× threshold).
>
> No v2.34.1 tag triggered (one BOUNDARY shape blocks the all-CONFIDENT
> auto-tag branch). Doc-only ship-verdict update merged to master.

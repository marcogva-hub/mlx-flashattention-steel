# Whole-repo review report — 2026-05 (post-v2.50.1)

**Master before**: `aa5741c` · **Master after**: 4 review commits (`9e57389`,
`92e5b44`, `6b7e30f`, `f0f7688`)
**Method**: 5 parallel review agents (Python hot-path, Python runtime, C++
NAX, C++ STEEL/dispatch, tests/benchmarks/docs) → consolidated ~30 findings →
4 implementation waves, each commit independently revertible, full suite after
every wave.  Previously-discarded conclusions re-investigated per directive
(M1-era kernel verdicts re-tested on M5 Max).

## Headline result — KD-5 root cause found and FIXED

The "STEEL backward D=128 N≥2048 zeroed-blocks bug" — carried as
deferred/deprecated debt since Prompt 5a and slated for v2.51 removal — was a
**grid/generator mismatch**: `MFASteelBwdDKV::eval_gpu` launched
`NK = ceil(S/cfg.BK)` threadgroups (cfg.BK = 32 on M3+ at D=128) while the
source generator hardcodes `BK = 16` for D > 64.  Threadgroups each processed
16 K-rows at 16-row strides — every K-row beyond NK·16 was simply never
written.  Invisible on M1 (cfg.BK already 16); exact match for the observed
"zeroed for rows ≥ 1024 at N=2048" signature.  One-expression fix; both
xfails removed; all 4 TestNativeBackwardRouting shapes now assert against
SDPA-VJP and pass.

## Test-suite truth restoration

| Metric | Before | After |
|---|---|---|
| passed | 1312 | **1346** |
| xfailed | 2 | **0** |
| xpassed (invisible regression gaps) | 32 | **0** |
| in-suite flakes (bisect/sage, chronic) | 1-2 per run | **0 across 3 consecutive runs** |

The flake cure is attributed to P1 (dispatch decision cache now keyed on the
steering env vars that tests mutate — the stale-decision contamination
vector).

## Correctness fixes (silent-wrong-results class)

| ID | Fix | Severity |
|---|---|---|
| S1 | KD-5 BK dispatch/generator mismatch (above) | CRITICAL (research path) |
| P2 | Per-head 3-D/4-D sparse mask backward used cross-head UNION (4 sites); ndim-preserving bias now; 2-D-only kernels fail loudly | HIGH |
| C1 | `scale` absent from all 9 V6NAX backward pipeline cache keys → wrong-kernel reuse across scales | HIGH (latent) |
| C5 | V6Key bit-packing collisions at production shapes (qbs = 2²⁴ at H=8 N=16384 D=128) | HIGH (latent) |
| C2 | `force_v6nax` missing from is_equivalent → LSE-domain mixup via graph dedup | HIGH (latent) |
| C4 | int32 overflow in batch strides (2 files) | MEDIUM (large shapes) |
| P1 | dispatch cache ignored env mutations | MEDIUM |
| C7 | V6NAX fusion dropped caller's custom scale (now gated) | MEDIUM (latent) |
| S2/S4 | flash-decode RoPE guard; steel-bwd sparse-input guard | LOW (latent, defensive) |
| P7/R2/R3/R4/R5 | bias-cache ABA, strided-mask ZeroDivision, cache reset metadata, mlx_lm swallow, svdquant idempotence | LOW-MED |

## Memory/perf fixes (benchmarked on M5 Max, median)

| Path | Before | After | Delta |
|---|---|---|---|
| TurboQuant paged append (per decode token) | 0.697 ms | 0.472 ms | **-32%** |
| rope decode loop (500 steps) | 124.8 ms | 89.9 ms | **-28%** |
| conv hook dispatch overhead (ineligible 2D) | 0.239 ms | 0.175 ms | **-27%** |
| flash_attention_sparse forward (2048/D64) | 5.18 ms | 4.97 ms | -4% |
| Pipeline cache leak (MTLComputePipelineState per compile race) | leaked | CFRelease'd | — |

Headline attention paths verified post-review: output diff vs SDPA = 0.0 at
all probed shapes; no dispatch regression.

## Investigated and declined (with evidence, per directive)

- **P5 rope lru churn** (`cache_seqlens` in factory key): measured closure
  recreation cost is sub-µs vs 0.25 ms/step total; the clean fix requires
  re-architecting per-call state through `mx.custom_function` primals —
  training-flow regression risk exceeds <1% gain.
- **V3/V4/V5 kernel re-enablement**: all 22 accuracy variants now PASS on
  M5 Max (markers removed — they now guard regressions).  Perf promotion was
  NOT attempted: the M1-era perf verdicts (V3 0.77-0.88× V2, V5 0.60-0.90×)
  would need a dedicated M5 bench campaign per §AA.4 before any dispatch
  change; out of scope for a correctness-first review.  Flagged as a v2.51
  candidate: re-bench V3/V4/V5 on M5 hardware.
- **Module-level env caching for live-read dispatch vars**: would break the
  documented test contract (monkeypatch must take effect immediately, per
  mfa_env.hpp design note).

## Rollbacks

None required — all 4 waves landed green on first full-suite validation
(1 pre-existing order-dependent flake observed mid-stream disappeared after
P1; identity of the flake moved between runs, confirming environmental
class, and 3 consecutive post-review runs are fully green).

## v2.51 candidates surfaced by this review

1. Re-bench V3/V4/V5 STEEL variants on M5 Max (all now pass accuracy; M1-era
   perf verdicts may invert — same pattern as audit-framing inversions #1-3).
2. Reconsider MFA_FORCE_NATIVE_BWD removal: the KD-5 fix makes STEEL backward
   correct; deprecation text updated to reflect this.
3. `v6_nax_forward` scale parameter (currently bakes 1/sqrt(D); Python-side
   gate added as interim).
4. mlx_lm quantized-cache integration test against current mlx-lm
   (version-compat markers removed; passes today).

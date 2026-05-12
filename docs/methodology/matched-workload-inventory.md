# Matched-workload-family warmup — inventory

## Problem (from v2.36.0 methodology REGRESSION)

The v2.36.0 continuous-workload protocol (256×256 FP16 matmul warmup,
50ms gap) **resolved 2/3 HIGH** shapes but **regressed 3/4 CONFIDENT**
shapes — net REGRESSION verdict.

Root cause (per `sub1ms-protocol-diagnostic.md` §"Root cause hypothesis"):
the matmul warmup's 128 KB working set competed with V2 for **M5 Max
cluster-shared L2 cache**, evicting V2's Q/K/V/mask between dispatches.

## This sprint's hypothesis

Per `sub1ms-protocol-diagnostic.md` §"Path forward" option 1:

> Match the warmup to V2's working-set profile: instead of 256×256
> matmul, use a small `sparse_attention_nax` dispatch with a different
> shape (e.g., qL=kL=512, D=64, BT=32). Same kernel family, small
> enough to keep GPU warm without colliding with the measured shape's
> working set.

**Hypothesis H_MW**: a `sparse_attention_nax` warmup using
**different D and different shape** than the measured kernel will
(a) hold GPU power state above the < 100ms downclock threshold
(per `downclock-threshold-data.json`) while (b) **not** evicting the
measured kernel's L2 cache lines — because D=64 vs D=128 produces
distinct shader instantiations, smaller Q_smem/K_smem (4 KB total
warmup smem footprint vs 32+ KB for measured), and a working set that
fits in private per-core L1 rather than colliding at the cluster L2.

## Acceptance criteria (per prompt §E.1, same as prior sprint)

| Verdict | High→CONFIDENT | CONFIDENT regressed | n_ratio≥1.2× | Action |
|---|:--:|:--:|:--:|---|
| **GREEN** | 3/3 | 0/4 | ≥6/7 | v2.36.1 V2-default flip + tag + release |
| GREEN_NARROWER | 3/3 | 0/4 | <6/7 | v2.36.1 with narrower envelope doc |
| PARTIALLY_GREEN | 2/3 | 0/4 | any | document only; no release |
| NOT_GREEN | <2/3 | 0/4 | any | option 2 (heartbeat) or option 4 (shape-aware) |
| **REGRESSION** | any | ≥1/4 | any | abandon protocol; stay v2.35.0 SHIP_OPT_IN |

## Reference anchors

- **Downclock threshold**: `<100ms idle → +146% slowdown` (empirical,
  v2.36.0 Section B characterization). Warmup gap **must be < 100ms**.
- **v2.36.0 V2-only re-bench baseline**: 4 CONFIDENT + 3 HIGH on 7
  Sprint B shapes (D=128, qL=kL ∈ {4096, 8192, 16384}). Reference data
  in `docs/lcsa-nax/lcsa-nax-v2-only-rebench-analysis.json`.
- **Three-axis rule** (`CLAUDE_V6_NAX.md` §3.5): every methodology
  change must validate (1) output sanity, (2) path entered, (3) edges
  preserved. The v2.36.0 sprint was caught by axis-3 — same gate applies.

## Out of scope

- Backward pass (V2 sparse forward only).
- Shapes not in the 7 Sprint B reference set.
- Per-shape adaptive warmup tuning (uses single fixed warmup for all 7).
- Hardware power-state API exploration (deferred to option 3).

## Deliverables checklist

- [ ] `docs/methodology/matched-workload-inventory.md` (this file)
- [ ] `docs/methodology/matched-workload-decisions.md` (DM1-DM9)
- [ ] `bench/methodology/matched_workload_harness.py`
- [ ] `bench/methodology/matched_workload_analysis.py`
- [ ] 3 session runs → `docs/methodology/matched-workload-data.json`
- [ ] `docs/methodology/matched-workload-results.md` (verdict)
- [ ] `docs/methodology/matched-workload-analysis.json`
- [ ] Per-session runlogs (M1/M2/M3 .txt)
- [ ] `CLAUDE_V6_NAX.md` §4.X amendment (caveat status update)
- [ ] `devnotes/SESSION_LOG.md` entry

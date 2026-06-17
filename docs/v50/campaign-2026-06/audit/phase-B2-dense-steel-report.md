# Audit Phase B2 — Dense STEEL Family per-variant audit (sprint report)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `aee5d70`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight: `mlx-debug-forensics`,
`benchmark-measurement-correctness`, `metal-kernel-dev`. **No kernel/routing/threshold/bug change**
(comment sweep found the family fresh — no edits needed). Durable spec: `dense-steel-family-spec.md`;
lock: `tests/test_dense_steel_family_lock.py` (14 cells). Resolves Phase-A carry-forward #1.

## (1) Per-variant specs — see `dense-steel-family-spec.md`
8 STEEL variants (V1/V2/V3/V4/V5/split-K/dsplit/flash_decode), each a **faithful FlashAttention** (no
inspired-by deviation), distinguishing features tabulated. The `backend="mfa"` path is real STEEL
(Δ=1.9e-6 vs SDPA, Phase A); default `flash_attention` → SDPA.

## (2) Correctness vs independent fp32 oracle — all edges, locked
Manual fp32 attention (not SDPA, not another variant — lesson #11). V1/V2/V3(D64+D128)/V4/V5/split-K/
dsplit/flash_decode/GQA all pass: **max_abs_err ≤ 7.7e-5, finite** → 14 locked cells.

## (3) Variant-level dispatch map — SENTINEL, resolves carry-forward #1
The variants are **byte-identical to each other (Δ=0.0)** — byte-identity cannot distinguish them
(the prompt's premise, now proven). No C++ dispatch-trace env exists. **Sentinel = env-toggle timing**:
- D64 causal N4096: default **2.73ms** vs `MFA_DISABLE_V3`(→V2) **3.13ms** (1.14×) → **V3 is default**
  for causal-large-N. [V] (D128 V3≈V2 parity masked the toggle — predicate confirms V3 there too.)
- D128 causal N512: default **0.42ms** ≠ `MFA_FORCE_V2` **0.34ms** → **V1** below v3_min_N. [V]
- D128 non-causal → **V2** (V3 needs causal). `MFA_ENABLE_V4` changes dispatch (12.29→11.54ms) → V4
  eligible but **env-gated, not default**; V5 ineligible at tested shapes. [V]
- D=256 → **dsplit**; decode (N≤4,S≥256) → **flash_decode**. [V]

Full map in the spec. Lock approach: since variants are byte-identical (no runtime fingerprint) and
timing is CI-flaky, the dispatch is locked via (a) **forced-variant correctness** (each reachable +
correct) + (b) a **source-predicate threshold lock** (v3_min_N / flash_decode-gate / m3_prefers_v1
forms) — a threshold change trips CI, forcing a deliberate map update (the KD-5/Gate-9 source-lock
pattern). Drift-catching: changing `v3_min_N` in source fails `test_v3_min_N_threshold`.

## (4) Selection-threshold audit
**No arbitrary/overflow-looking threshold** in the dense dispatch. `v3_min_N=(D==64)?4096:2048` is
documented + benchmark-derived + M5-re-validated (not arbitrary). flash_decode/occupancy/calibration
thresholds are sensible/data-driven. **Re-examined the sparse `2^31` (B1 flag): it = 4096·4096·128
(calibration shape) computed in Python (unbounded int) → NOT an overflow bug**, just a coincidental
value at 2³¹. Sparse flag downgraded to "benign calibration value."

## (5) Comment sweep
Dense STEEL comments are **fresh** (V3 re-validated 2026-06-17; V4/V5 "pending benchmarks" and the
scratch-lifetime "Phase 1/2 pending under lazy eval" are accurate). No corrections needed.

## Flagged for Phase E
- **M5-optimal vs M1–M4-legacy:** default dense → SDPA; whether ANY STEEL variant beats SDPA on M5 is
  unmeasured here (Phase A: backend=mfa is the expert path). Do not assume legacy or optimal.
- V5 reachability (ineligible at all tested shapes) — when does V5 ever fire? Phase E.

## Disposition
Dense STEEL family: **spec-verified + correctness-locked (14 cells) + dispatch-mapped (sentinel) +
thresholds-audited + comments-confirmed-fresh.** Carry-forward #1 resolved (variant dispatch mapped at
runtime). No kernel/routing/threshold/bug change. Suite green. No orphans. Not tagged. B3 (backward),
B4 (GNA/conv/topk/sage/paged) remain.

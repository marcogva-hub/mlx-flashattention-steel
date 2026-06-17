# Sprint IV-OPT — Incremental Optimization Pass

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `6e2f38d` (clean, post-correctness-review), macOS 26.6 (25G5028f),
Apple M5 Max 128GB, mlx 0.31.2.
**Type:** profile-driven incremental overhead recovery on EXISTING code (not kernel dev).
**Outcome: INCREMENTAL OPTIMIZATION CLOSED — all regimes broadly at the irreducible floor. 0
measured-dominant safe gains; 0 executed (by design). Green light for V6 NAX.**

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Non-decode perf profiling | `/mlx-mfa-bench-methodology` | call/eval per-regime attribution |

---

## R.1 — Non-decode regime profile (M5/26.6) — the deliverable

`benchmarks/methodology/opt_profile/iv_opt_nondecode_profile.py`. Per regime: `call` (Python
dispatch + lazy graph build, no GPU) vs `eval` (kernel). **Python-dispatch % = the reducible lever.**

| Regime | shape | call (Python) | eval (kernel) | **Python %** |
|---|---|---|---|---|
| prefill dense causal | B2 H8 D128 N=2048 | 7.8us | 652us | **1.2%** |
| | N=4096 | 8.1us | 1774us | **0.5%** |
| | N=8192 | 10.3us | 6050us | **0.2%** |
| windowed-causal (V3 path) | N=4096 | 11.8us | 2749us | **0.4%** |
| | N=8192 | 13.1us | 5394us | **0.2%** |
| sparse LCSA attend (prebuilt mask) | N=4096 | 14.0us | 1956us | **0.7%** |
| | N=8192 | 25.8us | 7295us | **0.4%** |
| GNA native | N=256 | 3.8us | 248us | **1.5%** |

**Every mlx-mfa attend path is kernel-dominated (Python 0.2–1.5%) — at the floor.** Same verdict
as IV-0's decode finding, now confirmed across prefill / windowed / sparse / GNA. (conv3d NAX runs
via the `mx.conv` auto-hook, not a direct export; covered by III-1 + the conv tests.)

**The one non-trivial line — sparse mask construction:** `make_lcsa_mask` = 533us @N=4096 /
1039us @N=8192 — ~12–21% of the sparse path *if a caller rebuilds it per forward*. Investigated
(R.2): **NOT a recompute-of-invariant.** `make_lcsa_mask` reads q/k VALUES (it pools the actual
tensors for the top-k spatial selection — a content-dependent dynamic mask), so it is legitimate
per-call algorithm work and is **not cacheable** (caching would return a stale mask — the
id()-cache / lesson-#11 footgun). And `flash_attention_sparse(q,k,v,block_mask,…)` takes a
**prebuilt** mask — mlx-mfa never rebuilds a mask internally per attend. So the mask cost is
caller-controlled (build-once-reuse is correct usage), outside mlx-mfa's measured hot path.

## R.2 — Recompute / redundant-alloc / invariant sweep

- **make_lcsa_mask:** content-dependent (q/k values) → legitimate per-call work, not cacheable (above).
- **Param-deterministic mask builders** (`make_gna_mask`, `make_sliding_window_mask`,
  `make_causal_block_mask` — depend only on int/shape params, not q/k values): memoizable in
  principle, BUT mlx-mfa does not build them internally per attend (the attend takes a prebuilt
  mask), and the native GNA path builds no Python mask (py 1.5%, at floor). A defensive `lru_cache`
  would only help a *caller* who rebuilds a param-only mask per call — caller-usage-dependent, NOT
  measured-dominant in any mlx-mfa-internal regime. → **diagnostic-only** (D-OPT-1).
- **KD-2 class (forward recomputed in backward):** the A2 correctness-review axis came back CLEAN on
  the vjp wrappers; no residual recompute-of-forward found. Not re-litigated.
- **Redundant alloc / graph nodes on hot paths:** the IV-0 + A1/A2/A4 review already swept these
  (clean); the call-times above (3.8–25.8us) leave no room for a material alloc/graph-node lever.

## R.3 — Post-correctness-fix gain check

- **A3-1 int64 widening:** no measurable cost. The widened multiplies are **once-per-threadgroup
  address arithmetic** (not inner-loop), on the V6 NAX path — which the auto prefill profile doesn't
  even exercise on M5 (dense → SDPA). Suite green ×2 post-rebuild confirms no correctness regression;
  a 64-bit vs 32-bit address multiply once per tile is immeasurable. No re-bench warranted (targeted
  Pattern #6, not blanket).
- **V3 validation:** the queue-closure sprint already measured V3 wins/parity on M5 at its auto-fire
  regime (N≥4096 D64 / N≥2048 D128); the thresholds match where it wins — no faster cell left on the
  table. No flip reason since; no re-bench.
- **No gain unlocked** by either correctness change.

## R.4 — Executed gains: **NONE** (per the strict bar)

No item is measured-dominant in an mlx-mfa-internal regime (everything is 98%+ kernel/eval). Per
Pattern #6 for optimization + the strict bar, executing zero is the correct outcome — a noise-level
Python tweak (<run-to-run variance) is not a gain and adds risk. Same disciplined close as IV-0.

## R.5 — Per-regime honest floor finding

| Regime | Floor verdict |
|---|---|
| decode (incremental + TQ) | AT FLOOR (IV-0; eval-floor lever closed by IV-D1/D2) |
| prefill / large-N dense | AT FLOOR (kernel 99.5–99.8%) |
| windowed-causal (V3) | AT FLOOR (kernel 99.6–99.8%) |
| sparse LCSA attend | AT FLOOR (kernel 99.3–99.6%; mask-build is caller-controlled content-dependent work) |
| GNA native | AT FLOOR (kernel 98.5%) |
| conv3d NAX | covered III-1 (auto-hook; bf16 MPP lift shipped) |

**INCREMENTAL OPTIMIZATION IS CLOSED.** The existing code is broadly at the irreducible floor
(kernel compute + MLX's per-eval sync) across every regime. There is no Python/dispatch/alloc
overhead lever left that meets the bar. This is the green light for **V6 NAX / dequant-in-GEMM** —
the real performance frontier, where the gain comes from exploiting the M5 Neural Accelerators the
Day-J characterization was built for (not from trimming already-in-the-noise orchestration).

## Diagnostic-only ladder (for Marco / future)

- **D-OPT-1** (defensive, low value): `lru_cache` the *param-deterministic* mask builders
  (`make_gna_mask`/`make_sliding_window_mask`/`make_causal_block_mask`, keyed on their int/shape
  params — NOT `make_lcsa_mask`, which is content-dependent) so a caller that rebuilds a param-only
  mask per forward gets a free cache-hit. Not measured-dominant in any mlx-mfa regime; caller-usage-
  dependent. Ship only if a real caller profile shows per-call param-mask rebuilds.
- **The frontier (separate chantier): V6 NAX / dequant-in-GEMM.** The Day-J data characterized the
  M5 Neural Accelerators (FP16 51.8 / BF16 51.5 TFLOPS, INT8 ~97 TOPS); the win is moving the
  dequant into the matmul2d GEMM (not Python overhead). This is the next, larger effort.

## Release disposition

Nothing executed → nothing new to release. The held **v2.56.0** (flag removal + V3 validation +
IV-D1/D2 decode gains + A3-1 int64 fix) stands unchanged. Phase IV incremental optimization closes
here; V6 NAX is the next sprint.

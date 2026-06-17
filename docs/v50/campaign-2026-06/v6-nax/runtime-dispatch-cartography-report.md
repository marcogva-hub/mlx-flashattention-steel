# Runtime Dispatch Cartography — which-binary, fingerprinted on M5/26.6 (NO build, NO routing change)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `5a0c22c`, macOS 26.6, M5 Max 128GB, mlx 0.31.2. Pre-flight:
`benchmark-measurement-correctness`, `mlx-debug-forensics`. **Every "which kernel runs" cell is
established by RUNTIME FINGERPRINT (byte-identity vs `mx.fast.sdpa`+bias, density signature, timing),
NOT source-tracing** — the lesson of four which-artifact inversions. No kernel built, no routing
changed.

## Headline: at **D=128, every built-in sparse mask-maker silently falls to dense Apple SDPA on M5+** — the working sparse kernel (1.68–4.2× vs SDPA) is reachable only via a hand-passed symmetric mask. At **D=64**, sparse runs but is pathologically slow (LOSES to SDPA at every density).

The fingerprint method (from the inversion): a real sparse kernel and the SDPA fallback both compute
the same masked math, but the fallback is *literally* the `mx.fast.sdpa` call → **byteΔ == 0.0**
(bit-identical), while the real sparse kernel is a different kernel → **byteΔ ~1e-6** (different
rounding). Timing (flat vs sloped) corroborates.

## The verified runtime-dispatch map (M5/26.6, B2 H8 N4096 f16 non-causal)

| Entry × input | Runs (fingerprinted) | byteΔ vs SDPA | timing | evidence |
|---|---|---|---|---|
| `flash_attention` (dense) | **Apple SDPA** | **0.0** | — | byte-identical |
| `flash_attention_sparse`, **D=128**, `make_causal_block_mask` | **dense SDPA (SILENT FALLBACK)** | **0.0** | flat 3.71ms | byte-identical |
| …, D=128, `make_sliding_window_mask` | **dense SDPA (FALLBACK)** | **0.0** | flat 3.69ms | byte-identical |
| …, D=128, `make_strided_mask` | **dense SDPA (FALLBACK)** | **0.0** | flat 3.71ms | byte-identical |
| …, D=128, `make_lcsa_mask` (FlashVSR) | **dense SDPA (FALLBACK)** | (shape [128,256] asym) | — | asym→fallback |
| …, D=128, **hand symmetric [128,128]** | **real NAX sparse (WINS)** | 3.8e-6 | sloped, 1.19ms@d=0.25 | different kernel |
| …, D=128, GQA Hk=2, symmetric | real NAX sparse | 3.8e-6 | — | different kernel |
| …, D=128, ndim-3 [H,NQ,NK], symmetric | real NAX sparse | 3.8e-6 | — | different kernel |
| `flash_attention_sparse`, **D=64**, default (symmetric) | **real sparse but SLOW** | 3.8e-6 | sloped, 6.8ms@d=0.25 | different kernel |

**Root cause of the D=128 fallback:** `_steel_block_config(128) = (BQ=32, BK=16)` → the default mask
is **asymmetric [128,256]**. The M5+ symmetric auto-route (`attention.py:3128`) fires only when
`bt_q == bt_k`; asymmetric falls through to the STEEL path, which on M5+ returns
`_sparse_fallback_sdpa_perhead` (`:3239`) — dense SDPA — because the asymmetric STEEL kernel is
disabled by the `(long)p->NK` compiler bug (§ below). `make_causal_block_mask(N, 64)` is symmetric
[128,128] (BK=32), so **D=64 engages a real kernel; D=128 does not.** A symmetric mask bypasses the
`_steel` validator via the earlier auto-route — but the mask-makers never emit one for D=128.

## The two gotchas (silent on M5+)

1. **D=128 — silent SDPA fallback (loses an available win).** Every built-in mask-maker yields an
   asymmetric mask → dense SDPA. The user passes a block-sparse mask expecting sparsity and gets dense
   SDPA (correct output, full O(N²) compute + the N×N mask memory). The working sparse kernel exists
   and wins (below) but is unreachable through the public mask API.
2. **D=64 — slow sparse (loses to SDPA the other way).** The default (symmetric) mask engages a real
   sparse kernel that is **pathologically slow** (the V1 scalar path): 2.08ms@d=0.06 → 25.98ms@d=1.0,
   i.e. **0.66×→0.05× vs SDPA (1.36ms)** — a LOSS at every density. D=64 should route to SDPA for
   speed; it currently runs the slow kernel. (Why D=64 is 5.5× slower than D=128 on the same path is
   an open sub-question — DEDUCED: poor scalar-kernel occupancy at 64-wide rows — flagged.)

## Size of the prize (D=128 symmetric NAX sparse vs what the default API delivers today = SDPA)

| density | sparse (symmetric) | dense SDPA | **gain over today's default-API (SDPA)** |
|---|---|---|---|
| 0.062 | 0.72 ms | 3.04 ms | **4.21×** |
| 0.125 | 0.87 ms | 3.04 ms | **3.47×** |
| 0.250 | 1.22 ms | 3.04 ms | **2.50×** |
| 0.500 | 1.81 ms | 3.04 ms | **1.68×** |
| ~0.75 | ~3.0 ms | 3.04 ms | crossover (SDPA wins above) |

Correct: byteΔ 3.8e-6 (banded + scattered), zero scatter tax. eff ≤ 42.7 ≤ 51.8 peak. **The prize is
D=128-specific and real** — only at d ≲ 0.75. There is **no prize at D=64** (sparse kernel slower than
SDPA at all densities).

## The `(long)p->NK` bug (investigated, not fixed)

Metal-compiler miscompile (MSL 4.x, M5/gen17): the `int→long` cast of the struct field `p->NK` reads
the wrong offset (NQ_aligned=16 at offset 36 instead of NK=32 at offset 32), corrupting the asymmetric
STEEL kernel's mask address (`qb_actual = 2*row`). **6 source-level workarounds all failed** (hoist,
recompute, int-local, loop-accumulate, pointer-arith, kTilesPerTG=1) — it is a compiler bug, not a
source bug (`docs/v6-nax/sparse-bug-investigation.md`, 2026-05-02). **Verdict: fixing the asymmetric
STEEL kernel is HIGH-risk / compiler-dependent — NOT the recommended path.**

## Fix options to make the sparse win reachable at D=128 (scope only)

| Option | What | Cost / risk |
|---|---|---|
| (a) Fix `(long)p->NK` | Re-enable the asymmetric STEEL kernel | HIGH — compiler miscompile, 6 workarounds failed; needs a function-constant-NK or non-long-cast rewrite, uncertain |
| (b) Route asymmetric → symmetric NAX | Re-block [128,256]→symmetric, run the working kernel | MEDIUM — must be EXACT (split BQ=32 query-blocks → bt=16 [256,256]; OR-downsample is WRONG for forward — attends masked keys); bt=16 kernel perf unverified |
| (c) **D-aware default convention + routing** | D=128 default mask symmetric (BQ=BK=32) → engages the proven-fast NAX kernel; D=64 → route to SDPA (sparse too slow) | **LOW–MEDIUM, RECOMMENDED** — uses the measured-fast bt=32 kernel; changes the public D=128 mask shape (compat note); pairs with D-aware routing |

## Verdict

1. **Verified runtime map (above)** replaces the source-traced cartography (now known wrong about
   runtime dispatch). Ground truth: D=128 built-in-mask sparse = SDPA; D=128 symmetric = real sparse
   (wins); D=64 = real sparse (slow, loses); dense = SDPA.
2. **Reachability gap:** the D=128 sparse win (1.68–4.2×) exists but the default API bypasses it to
   SDPA; quantified above.
3. **Recommended next increment (gated, NOT this session):** a **D-aware routing + mask-convention
   fix** — D=128 → symmetric NAX sparse (the win), D=64 → SDPA (sparse loses), asymmetric-D128 either
   re-blocked exactly to symmetric or the makers changed to emit symmetric. Its own three-axis +
   Pattern #6. Do NOT fix the `(long)p->NK` compiler bug (high-risk dead end).

## Validation / discipline
- Every cell fingerprinted at the RUNTIME dispatch (byteΔ vs `mx.fast.sdpa`+bias: 0.0=fallback,
  ~1e-6=real sparse; + flat/sloped timing + win/loss vs SDPA), not source-traced. effective-FLOP,
  plausibility-gated (≤ 42.7 ≤ 51.8 peak), independent fp32 reference (banded + scattered). No build,
  no routing change, no bug fix. Cross-ref the RULE-9-corrected prior reports. No orphans. Not tagged.

# Sprint II-11 — cider GQA-Decode Port (2026-06-12)

**Status**: COMPLETE — ported + benched; **auto-dispatch DECLINED
(perf-vs-maintenance), shipped as flag-free expert API** with the
numbers flagged to Marco per the sprint's explicit license.

## R.0 — Recovered findings + license

II-5 benched cider in-repo: 70/102 wins at 1.0–1.24x (best GQA-factor
8–16, N=16–32K) vs MLX 0.31.2 SDPA; the published 1.57x does not
reproduce.  MIT (Copyright 2026 Mininglamp contributors) — attribution
in the module docstring.

## R.1 — Port

`mlx_mfa/gqa_decode_cider.py`: cider's v9 2-pass kernels (contiguous-
chunk split-KV; threadgroup y-dim = gqa_factor so each K/V chunk is
read once per GROUP; TILE=4 register tiling) ported body-style through
`mx.fast.metal_kernel` (standard MSL).  N and K/V strides are RUNTIME
params (no per-token recompile — the Phase-I churn lesson); scale a
f32 input; (D, BLOCKS) JIT-cached (3 block variants).  Public surface:
`gqa_decode_cider(q, k, v, scale)` — dense decode, N_q=1, [B,H,S,D].

## R.2 — Correctness (locks in test_phase2_ii11_gqa_decode_cider.py)

FP-floor parity vs `mx.fast.scaled_dot_product_attention` across GQA
factors 1..32 (incl. MQA and the MHA degenerate), S 512–16K, fp16/
bf16/fp32: max_err 0.00006–0.00024.  Prefill rejected loudly.

## R.3 — Post-port bench (3 sessions, medians, M5 Max, D=128, fp16)

Wins (ratio = sdpa/cider, >1 = cider faster), consistent across
sessions ONLY at:

| cell | cider ms | sdpa ms | ratio |
|---|--:|--:|--:|
| Hq32/Hkv4 (factor 8) S=32768 | 0.305 | 0.346 | **1.12–1.14x** |
| Hq32/Hkv2 (factor 16) S=32768 | 0.299 | 0.330 | **1.06–1.17x** |
| Hq32/Hkv4 S=16384 | 0.246 | 0.258 | 1.01–1.10x (noisy) |

Everything else ties or loses (0.84–1.00x): all S<=4096 cells, all
MHA/low-factor cells, factor-8 S<=16K, Hq16 cells.  The window is
NARROWER than II-5's in-cider numbers (which peaked 1.24x from N=16K):
the mx.fast.metal_kernel route adds two-launch + per-call params
overhead (~0.02-0.03 ms) that compresses the margin at the ~0.2-0.4 ms
absolute scale of decode steps.

## R.4 — Decision: auto-dispatch DECLINED; expert API shipped

The consistent win is ~0.04 ms/step at exactly (GQA-factor >= 8,
S = 32K).  Promoting would add a dispatch gate + maintenance surface
for a <=1.17x sliver that vanishes below 24K context — precisely the
case the sprint instructed to flag rather than promote.  Disposition:

- Module shipped as **expert API** (Auto-default principle tier 3):
  `from mlx_mfa.gqa_decode_cider import gqa_decode_cider` — users with
  high-GQA 32K-context serving can adopt it directly.
- NO auto-dispatch entry; `sdpa_vector_2pass` remains the decode
  default everywhere (II-1 map unchanged).
- **Marco flag**: if the serving portfolio lands on GQA>=8 32K-context
  workloads, a one-line gated dispatch can be added; the bench grid
  here is the evidence base.  Also noted: the stronger transplant
  (the same scheduling into the PAGED/TQ decode kernels, where the
  II-7 ladder shows a 14x-vs-dense kernel-bound floor and Apple has
  no kernel) remains the higher-value follow-up — it composes with
  this port's kernel structure.

Suite: 1397 passed (+6 locks).

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` | R.0 | current dispatch (Apple sdpa_vector_2pass) confirmed as baseline; port justified only for the surveyed window |
| `/metal-kernel-dev` | port | body-style metal_kernel, runtime params vs bake (churn), JIT cache keyed (D, BLOCKS) |
| `/mlx-mfa-bench-methodology` | R.3 | 3 sessions x 30-iter medians, warmed, full grid incl. losing cells |

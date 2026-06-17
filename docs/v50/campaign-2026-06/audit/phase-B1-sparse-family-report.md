# Audit Phase B1 — Sparse / LCSA Family per-kernel audit (sprint report)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `0fb1020`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight: `mlx-debug-forensics`,
`benchmark-measurement-correctness`, `mlx-mfa-nax-matmul2d-correctness`. **No kernel/routing/mask/bug
change** (comment-only corrections in scope; Phase F fixes). Durable spec: `sparse-family-spec.md`;
locks: `tests/test_sparse_family_correctness_lock.py` (13) + the Phase-A dispatch lock.

## Headline — which-binary correction (the spec foundation)

The working sparse forward is **TWO kernels selected by work product**, not one:
`decide_auto_version` routes `qL*kL*D ≥ 2_147_483_648 (=4096·4096·128)` → **V2 matmul2d**
(`sparse_kernel_source_v2`, the `BaseNAXFrag::mma` cooperative-tensor kernel), else → **V1 scalar**
(`sparse_kernel_source`, per-thread scalar dots). **Env-toggle fingerprint** (D=128 N=4096): default
**1.20 ms == v2 1.20 ms**; **v1 = 49.3 ms (~41× slower)**. This corrects Phase A's "D=64 is slow" —
it's the *work-product threshold*, not D: D=64@N=4096 (1.07e9) and D=128@N<4096 fall below → V1 scalar.

## (1) Verified specs — see `sparse-family-spec.md`
Both forward kernels are **faithful block-sparse FlashAttention-2** (matmul2d/scalar QK^T + online
softmax + mask-gated block skip + P@V) — no "inspired-by" deviation found (strict paper-fidelity). The
mask machinery, SDPA-fallback paths, and backward variants are spec'd there.

## (2) Correctness vs independent fp32 oracle — all edges, locked
Manual fp32 attention (NOT SDPA, NOT the kernel — lesson #11). All pass, finite:

| edge | V2 matmul2d | V1 scalar |
|---|---|---|
| banded / scattered | 6.5e-6 / 4.4e-6 | 8.4e-6 / 8.4e-6 |
| density→1.0 / →min | 2.2e-6 / 3.0e-5 | — |
| all-masked query-block | 8.9e-6 | 1.1e-5 |
| causal | 7.8e-5 | 7.9e-5 |
| GQA / ndim-3 / ndim-4 | 4.9e-6 / 4.4e-6 / 4.4e-6 | — |

→ 13 locked cells (`test_sparse_family_correctness_lock.py`).

## (3) Mask-faithfulness — the Phase-F premise, CONFIRMED
A D=128 **symmetric 32×32** convention is **byte-identical (Δ=0.0e+00)** to the current 32×16 for
sliding-window, causal, AND strided (both differ from the exact element pattern by the same 1.1e-1 —
the granularity-independent block-sparse approximation). **⇒ Phase F can route D=128 → symmetric V2 by
REGENERATING masks at 32×32 with zero correctness cost** (do NOT OR-merge — superset/not faithful).
Caveat: strided>BK and LCSA top-k not separately isolated ([D] likely faithful; verify in F).

## (4) Comment sweep (comment-only)
`mfa_sparse_attention.cpp:13` ("Phase 1.3 *will* swap to matmul2d" → done in V2),
`lcsa_nax.py:21` ("Phase 1.5 *will* introduce dispatch" → exists), and two "PoC stage" → "production".

## Gotchas re-confirmed (→ Phase F / KNOWN_ISSUES)
1. D=128 + built-in makers (asymmetric) → silent SDPA (loses 1.7–4.2×).
2. **Work < 2.147e9 → V1 scalar, ~41× slower than V2** (sharper than Phase A's "D=64 slow"; also hits D=128 N<4096).
3. Sparse backward dense by default.

## Disposition
Sparse/LCSA family: **spec-verified + correctness-locked (13 cells) + comments-swept.** The
which-binary (V2/V1 by work threshold) and the mask-faithfulness premise (32×32 ≡ 32×16) are the
ground Phase F's routing fix stands on. The per-kernel method (which-binary → fp32-oracle edges →
faithfulness → comment sweep → lock) is proven for B2–B4. No kernel/routing/bug change. Suite green.
No orphans. Not tagged.

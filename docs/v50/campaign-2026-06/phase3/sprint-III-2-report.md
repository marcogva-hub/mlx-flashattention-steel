# Sprint III-2 — Paged/TQ decode transplant

**Date:** 2026-06-12 · **Status:** COMPLETE — **PROMOTED default-on** +
fused-kernel correctness fix · **Version:** 2.50.1 (bump in III-3)

## Verdict

The II-7 decode floor (fused TQ attend = 14× dense) is closed without
any cider-style kernel restructure: **§AA.5 FULL_INVERSION**.  Per-step
K-dequant/V-gather (two tiny elementwise Metal kernels) + Apple
`mx.fast.scaled_dot_product_attention` replaces the fused TQ attend for
single-token decode, default-on:

| metric | S=4096 | S=16384 |
|---|--:|--:|
| attend-only (vs fused TQ kernel) | **13.8×** (4.11→0.298 ms) | **22.1×** (15.07→0.683 ms) |
| full `step()` (vs fused) | **5.99×** (4.65→0.78 ms) | **14.42×** (16.68→1.16 ms) |
| gap to dense-decode floor | 1.66× (was 22.8×) | 2.4× (was 52×) |

Bonus root-cause find: **the fused TQ kernel was silently WRONG at
tq_bits=2 and 4** — fixed in the same sprint.

## §AA.5 premise validation (FULL_INVERSION)

**Prescription** (II-11 ledger): transplant cider KV-group scheduling
into the fused TQ kernel (M effort).

**Decomposition**: fused attend 4.11 ms; candidate Python-graph
dequant + sdpa 0.539 ms (7.6×) before any kernel — the fused-dequant
premise of TurboQuant P2–P4 is inverted on M5 (Apple's NAX sdpa_vector
2-pass + a materialized fp16 K beats in-kernel dequant by ~an order).
Within the candidate, K-dequant dominated (0.52 ms graph-unpack) →
scope-corrected kernel work = two ~30-line elementwise kernels
(`mlx_mfa/tq_decode.py`, `mx.fast.metal_kernel`, block table consumed
in-kernel, no Python gather), NOT the cider transplant.

WHT is orthogonal → rotated-q·rotated-K == q·k; no de-rotation.
V reads the always-maintained fp16 pool — faster AND more accurate
than packed-V dequant (under `tq_v=True` the new path's output is the
better one; locked at a V-quant-noise bar vs fused).

## Fused-kernel bit-width fix (silent-corruption class)

Arbitration against Python ground truth (II-6 lesson: never validate
one internal path against another) showed the **fused kernel** 0.147–
0.150 max-abs wrong at unit scale for tq_bits ∈ {2, 4} (≈49 at std 8):
`csrc/mfa_steel_paged_varlen_tq_fwd.cpp` emitted the 3-bit bit-planar
extraction UNCONDITIONALLY in BOTH the K and V dequant paths —
`tq_bits` was a runtime param the MSL never branched on.  Present
since the kernel landed; every prior validation used bits=3.  Fixed
with runtime bit-width branches in both sites; fused now matches
ground truth at 2/3/4 bits (1e-4); locked by
`TestDecodePathGroundTruth::test_fused_kernel_bitwidth_fix`.

## Changes

| File | Change |
|---|---|
| `mlx_mfa/tq_decode.py` | NEW — K-dequant kernel (2/3/4-bit layouts), V-gather kernel, `tq_decode_attend()`; kernel caches keyed by config tuples |
| `mlx_mfa/inference.py` | `step()` N_q=1 routes to the new path default-on; opt-out `MFA_DISABLE_TQ_DECODE_SDPA=1`; N_q>1 keeps fused (causal offsets) |
| `csrc/mfa_steel_paged_varlen_tq_fwd.cpp` | bit-width branches in K + V dequant (silent 2/4-bit corruption fix) |
| `tests/test_phase3_iii2_tq_decode.py` | 11 locks: ground-truth parity (new path 2/3/4-bit; fused 2/4-bit fix), routing, opt-out, wht_in_kernel, N_q>1, determinism ×5, adversarial finite |
| claims registry + docs/PERF_CLAIMS.md | `iii2_tq_paged_decode_step_default` (engagement kernel-cache-verified) |

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| §AA.5 premise validation | `/mlx-mfa-apple-primitives-coverage` | FULL_INVERSION — dispatch + 2 elementwise kernels, cider transplant cancelled |
| Perf discovery (§Z) | `/mlx-mfa-perf-audit` rubric | executable claim, REACHABLE via public `step()` |
| Bench methodology | §AA.4 | 30-iter medians, warmed, public surface |
| Kernel debugging | ground-truth arbitration per `docs/methodology/kernel-debugging.md` | fused 2/4-bit bug isolated in 2 probes |

## Validation

Suite: **1428 passed, 2 skipped** (+11 III-2 locks).  TQ suite (96)
green.  Determinism 10/10.  bits×magnitude matrix vs ground truth all
≤ tolerance.  Audit-framing-inversions catalogue: this is the 4th
inversion (density threshold, rope NAX, top-K partial, now TQ decode).

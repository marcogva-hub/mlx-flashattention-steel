# Audit Phase B3 — Backward Family per-kernel audit (sprint report)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `ec228c8`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight: `mlx-debug-forensics`,
`benchmark-measurement-correctness`, `mlx-mfa-nax-matmul2d-correctness`. **No kernel/routing/threshold/
bug change** (comments confirmed fresh). Durable spec: `backward-family-spec.md`; lock:
`tests/test_backward_family_lock.py` (6 cells).

## (1) Per-backward specs — see `backward-family-spec.md`
Native backward kernels (when engaged): dQ=`v6_nax_backward_query`, dK/dV=`v6_nax_backward_kv`/fused/
split, sparse variants `*_sparse_raw` — all NAX matmul2d, recompute O/L from saved sparse-LSE
(Pattern #5), faithful 7-GEMM FA-2 backward (no inspired-by deviation).

## (2) Gradient correctness vs independent fp32 oracle — all edges, locked
Oracle = `mx.vjp` of a MANUAL pure-mlx fp32 forward (not another kernel — lesson #11), trusted via
~1e-7 agreement with SDPA-vjp on the SDPA paths (two independent impls) + an FD sign/scale check.
Every path × gradient: **err ≤ 1.2e-4** (dV highest; dQ/dK ≤ 5e-7), finite → 6 locked cells.

## (3) THE per-(path × gradient) which-binary map — completes B1's "hybrid" glimpse
Native gradients are **byte-DISTINCT** from SDPA-vjp (Δ>0; unlike B2's byte-identical STEEL variants),
so each gradient's source is fingerprinted by byte-identity:

| Path | dQ | dK | dV |
|---|---|---|---|
| dense D=128 | SDPA-vjp | SDPA-vjp | SDPA-vjp |
| **dense D=64 causal/non-causal N≥2048 (default-on)** | **NATIVE** | **NATIVE** | **NATIVE** |
| sparse DEFAULT | SDPA-vjp | SDPA-vjp | SDPA-vjp |
| **sparse opt-in `MFA_ENABLE_V6_BACKWARD` bt≥64 (hybrid)** | SDPA-vjp | SDPA-vjp | **NATIVE** |
| sparse opt-in `MFA_V6_BWD_SPARSE_NATIVE` bt≥64 | NATIVE | NATIVE | NATIVE |

**The backward is a MIX, mapped precisely** (not assumed): the sparse hybrid is native-dV-only.
Lock extended with the 6 backward cells (byteΔ-distinct → robust, not timing). Drift-catching: if
dense-D64 default-on reverted to SDPA-vjp, the "native" Δ>0 assertion fails.

## (4) Threshold audit
- dense carveout `N≥2048` floor — MEASURED (lowered 4096→2048 after v2.39.1 BK=16 parity; documented).
- sparse hybrid `bt≥64` — a CORRECTNESS gate (mask OR-downsample, III-4 D16), not perf/overflow.
- D=128 default-off — measured (D=128 V6NAX backward slower than SDPA-vjp).
No arbitrary/overflow threshold. **Phase-E carry-forward (open):** the sparse V1↔V2 `2^31` PERF
crossover validity (overflow already resolved benign in B2; the perf question stands for E).

## (5) Comment sweep
Backward comments **fresh** — "PoC" labels (sparse-native dV) accurately mark the declined-on-perf
path; "will transpose / consume lse / for now" are accurate in-call descriptions. No corrections.

## Disposition
Backward family: **spec-verified + gradient-correctness-locked (6 cells) + per-gradient which-binary
mapped + thresholds-audited + comments-fresh.** Third family done. The backward-is-a-mix truth is now
precise (native-dV-only hybrid confirmed). No kernel/routing/threshold/bug change. Suite green. No
orphans. Not tagged. **B4 (GNA/conv/topk/sage/paged, incl. the deferred GNA per-element-window
correctness) remains, then Phase C.**

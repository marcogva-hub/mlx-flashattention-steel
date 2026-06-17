# Audit Phase A — Runtime Dispatch Ground-Truth + Regression Lock (sprint report)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `8b86e74`, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight: `mlx-debug-forensics`,
`benchmark-measurement-correctness`. Harness extended from `8b86e74`
(`benchmarks/methodology/runtime_dispatch_fingerprint.py`). **No kernel change, no routing change, no
bug fix** (those are Phase F). Master-plan doc (`mlxmfa_complete_audit_master_plan.md`) was NOT present
in the repo or working tree — proceeded with the self-contained Phase-A spec; created the audit ledger
spine at `docs/v50/campaign-2026-06/audit/README.md`.

## Deliverables
1. **Authoritative dispatch map** (durable): `docs/v50/campaign-2026-06/audit/dispatch-map.md`.
2. **Regression lock** (in suite): `tests/test_dispatch_map_lock.py` — 11 cells, all RUNTIME-
   fingerprinted, asserts runtime == map; drift fails CI. Drift-catching confirmed (a reroute of the
   D=128-asymmetric cell to real-sparse gives Δ=7.6e-6 ≠ 0.0 → trips the `==0.0` assertion).
3. This report (provenance; archived in Phase D).

## What Phase A fingerprinted (beyond the cartography's sparse+dense forward)

| Path | Verdict (fingerprint) |
|---|---|
| dense `backend="mfa"` | real **STEEL** (Δ=1.9e-6 vs SDPA — NOT a silent SDPA) |
| dense `backend="auto"` | **SDPA** (Δ=0.0) |
| GNA native (D=128 3D) | **native GNA kernel** (Δ=7.3e-2 ≠ 0 vs block-bias SDPA → not a fallback) |
| topk | own path (Δ=1.9e-6 @ ratio=1.0) |
| sage | int8 sage kernel (Δ=1.1e-3) |
| kvcache decode (N_q=1) | **SDPA** (Δ=0.0) |
| dense backward | **SDPA-vjp** (dQ Δ=0.0) |
| sparse backward (default) | **dense SDPA-vjp** (dQ Δ=0.0) — sparse fwd, dense bwd |
| sparse backward (opt-in, bt≥64) | hybrid dV-native + dQ/dK SDPA-vjp (dQ Δ=0.0 by design) |
| conv3d eligible / ineligible | **NAX conv** `executed++` / `mx.conv_general` `fallback++` (hook telemetry) |

## Gotchas (classified)
1. **D=128 sparse + any built-in mask-maker → silent dense SDPA** (SILENT-FALLBACK; loses 1.7–4.2×). Known (cartography); re-confirmed for `make_lcsa_mask` too.
2. **D=64 sparse → slow, loses to SDPA** (DECLINED-perf). Known (increment-0).
3. **NEW: sparse backward is dense by default** — `mx.grad(flash_attention_sparse)` runs the real sparse forward but a *dense SDPA-vjp backward* unless `MFA_ENABLE_V6_BACKWARD=1` + bt≥64 (declined-on-perf opt-in). Routed-but-suboptimal; correctness fine.

**No new catastrophic silent-fallback in backward/GNA/paged/conv** — all route as intended
(backward=SDPA-vjp, GNA=native, decode=SDPA in the sync-floor regime, conv=NAX-when-eligible).

## Method discipline
- Every cell by RUNTIME fingerprint (byteΔ vs reference / density slope / conv hook telemetry), never
  source-tracing. byteΔ==0.0 ⇒ is-that-kernel; ~1e-6 ⇒ different real kernel. The lock test embeds the
  references in-test and asserts the documented current reality (gotchas locked as "expected,
  documented" — to catch unintentional drift, NOT to enshrine them; Phase F updates map+test together).
- M5+-gated (the guards are M5-specific; the lock skips on non-M5).
- No kernel/routing/bug change. Keep-all-paths. No orphans. Not tagged.

## Audit-ledger status (this phase)
Every path above is now **dispatch-runtime-verified + test-locked**. Per-kernel correctness/perf
(Phase B/E) and the routing fix for the gotchas (Phase F) build on this verified floor.

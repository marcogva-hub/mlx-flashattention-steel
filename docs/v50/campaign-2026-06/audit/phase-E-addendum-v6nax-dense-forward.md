# Audit E-addendum / B-gap — the dense NAX forward `v6_nax_forward` vs SDPA

**Date:** 2026-06-18 · **Executor:** Claude Opus 4.8 (1M)
**Provenance:** master, M5 Max, macOS 26.6, mlx 0.31.2. Pre-flight:
`benchmark-measurement-correctness` (PRIMARY). MEASURE + DOCUMENT only — NO routing
change (F-target recorded, not applied); no tag/publish. Discipline: lesson #11
(independent fp32 oracle), lesson #15 (ms + ratio + direction), Pattern #6 / which-
binary (byteΔ vs SDPA + call-path), plausibility-gate ≤51.8 TFLOPS, effective-FLOP
(causal = work/time factor, ×0.5), 3-replicate median.

**Why:** Marco flagged E's "dense STEEL 3–4× slower than SDPA → backend=mfa legacy on
M5" as implausible *if* STEEL used matmul2d cooperative tensors. It surfaced a gap: E
benched the simdgroup variants and never measured the ONE dense kernel on the
competitive NAX matmul2d primitive.

## Step 0 — Archaeology (confirmed on current master)
- **V1–V5 = `simdgroup_matrix`.** `mfa_steel_fwd.cpp` 10 simdgroup hits, 0 matmul2d;
  V2–V5 (`_v2..._v5.cpp`) carry 0 of either — they build on V1's simdgroup MMA core.
- **`v6_nax_forward` = NAX `matmul2d` cooperative-tensor.** `mfa_steel_fwd_v6_nax.cpp`
  10 matmul2d hits, 0 simdgroup. The ONLY dense forward on the competitive primitive.
- **Role = backward-recompute only.** Called at `attention.py:5358`
  (`_v6_fwd(q,k,v,causal,True)`) inside `_v6nax_backward_vjp`, gated by `_v6nax_eligible`
  + `MFA_ENABLE_V6_BACKWARD=1`. As a FORWARD, reachable ONLY via `_ext.v6_nax_forward`
  — **no `backend=` path routes to it.**
- **Scale limitation REAL.** Binding `v6_nax_forward(q,k,v,causal,force_v6nax=False)`
  has NO scale parameter; host bakes `p.scale = 1/sqrt(D)` (attention.py:5018 comment
  confirms). Default-scale ONLY.
- **Forward-eligible shapes (binding DC12):** D=128 any N; D=64 only Nk>8000.
- **Prior devnote verdict:** `v6-metal-profile.md` + `sprint-3-2-bypass-tgmem-results.md`
  DID profile it as a forward — found a **structural ~5–7pp ALU gap** to SDPA (v6 38–43%
  vs SDPA 45–50%), root cause = kBlocks-split cO accumulators vs Apple's single-Otile
  pattern; closing it needs a source-generator rewrite (Sprint 3.3, not done). So it was
  evaluated, NOT 3–4× — a modest structural gap. But never benched forward-vs-SDPA under
  full E-discipline / plausibility-gate on current 26.6.

## Step 1 — Standalone forward correctness (B-gap closed)
`v6_nax_forward` vs an INDEPENDENT manual fp32 oracle (lesson #11 — not SDPA, not another
kernel), default scale 1/sqrt(D). All PASS (max_err vs fp32):
| | D128 N2048 | D128 N4096 | D64 N8192 | D128 GQA Hq8Hk2 |
|---|---|---|---|---|
| non-causal | 3.5e-6 | 2.3e-6 | 1.8e-6 | 3.8e-6 |
| causal | 7.1e-5 | 4.9e-5 | 7.4e-5 | 6.0e-5 |
Faithful FA-2 forward. Default-scale constraint confirmed (binding has no scale arg;
output matches the 1/sqrt(D) oracle). Locked: `tests/test_v6_nax_forward_lock.py` (9 cells).

## Step 2 — The bench (forward; 3-replicate median; which-binary by call-path + byteΔ)
B=1 H=8 f16, default scale. Δ(NAX,SDPA)=1.9e-6 & Δ(STEEL,SDPA)=1.9e-6 (both >0 ⇒ distinct
real kernels, not SDPA fallback; timing also differs).

| shape | SDPA | v6NAX (matmul2d) | STEEL (simdgroup) | **NAX/SDPA** | STEEL/SDPA | eff TFLOPS (NAX / SDPA) |
|---|---|---|---|---|---|---|
| D128 N2048 nc | 0.85ms | 0.75ms | 1.75ms | **0.89× (NAX faster)** | 2.06× | 23 / 20 |
| D128 N2048 c  | 0.50 | 0.47 | 1.24 | **0.95×** | 2.50× | 18 / 17 |
| D128 N4096 nc | 1.64 | 1.64 | 6.23 | **1.00×** | 3.79× | 42 / 42 |
| D128 N4096 c  | 1.05 | 1.02 | 3.53 | **0.97×** | 3.37× | 34 / 33 |
| D128 N8192 nc | 5.83 | 5.69 | 24.90 | **0.98×** | 4.27× | 48 / 47 |
| D128 N8192 c  | 3.22 | 3.22 | 11.99 | **1.00×** | 3.72× | 43 / 43 |
| D64 N8192 nc  | 2.46 | 3.00 | 10.22 | 1.22× (NAX slower) | 4.16× | 46 / 56 ⚠ |
| D64 N8192 c   | 1.40 | 1.64 | 5.24 | 1.17× (NAX slower) | 3.73× | 42 / 49 |

⚠ Plausibility flag (honest, per skill #1): D64 N8192 nc SDPA = 56 TFLOPS > 51.8 ceiling
— treated as an ARTIFACT, not a discovery; the D=64 verdict rests on the FLOP-independent
wall-clock ratio (NAX 1.17–1.22× slower). All D=128 readings ≤48 TFLOPS (plausible).

**Answer as numbers:** at **D=128, v6_nax-dense MATCHES OR BEATS SDPA (0.89–1.00×)** across
N and causal/non-causal. At **D=64 it loses (1.17–1.22×)**. simdgroup STEEL loses
**2.06–4.27×** everywhere (confirms E's gap — but that gap is simdgroup-specific).

## Step 3 — "legacy" framing RESOLVED: REVISED (not blanket-legacy)
- `backend="mfa"` → **simdgroup STEEL** IS legacy on M5 (2–4× slower than SDPA) — E's
  verdict stands *for the kernel backend=mfa actually routes to*.
- BUT the implied "even matmul2d dense can't beat SDPA" reading is **WRONG**: a competitive
  dense NAX matmul2d forward (`v6_nax_forward`) exists at **parity-or-better than SDPA for
  D=128**, currently backward-recompute-only, reachable only via `_ext`, default-scale-only.
- Mechanism: the simdgroup gap is the OLD primitive; the NAX path closes it (prior devnote's
  "~5–7pp ALU gap" is, at the wall-clock level on current 26.6, parity-or-win at D=128).

## F-target (recorded, NOT applied — F decides)
**Consider exposing/routing the dense NAX matmul2d forward** (`v6_nax_forward`) as a
user-facing dense path for **D=128** (parity-or-win vs SDPA), where today `backend="auto"`
goes to SDPA and `backend="mfa"` goes to slow simdgroup STEEL. Constraints to honor:
(1) **default-scale only** — custom scale must NOT route here (gate on scale==1/sqrt(D),
else SDPA); (2) **D=64 loses** — do NOT route D=64 here (SDPA wins 1.17–1.22×);
(3) forward-eligibility D=128-any-N. Since SDPA is parity (not a loss) and is the
zero-maintenance Apple path, the win is marginal at D=128 — F weighs whether a parity
NAX dense forward is worth a new routed surface, or whether it stays an `_ext` expert path.
Note: a "competitive dense NAX forward" also re-frames the M5 dense story (Apple SDPA is
matched by our matmul2d kernel at D=128, not 3–4× ahead).

## Disposition
B-gap closed (standalone forward correctness locked, 9 cells). E-gap measured (the dense
NAX forward IS competitive at D=128 — E's "legacy" was simdgroup-only). Held Phase-F
doc-item "document backend=mfa legacy-on-M5" REVISED: backend=mfa→STEEL legacy stands, +
a competitive dense NAX path exists (backward-only today) → F-target. MEASURE + DOCUMENT
only; no routing change; suite green; no orphans; not tagged.

# V34 Backward Block-Sparse NAX — Premise Validation (go/no-go)

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Type:** PREMISE VALIDATION — no kernel built. Output = go/no-go verdict + evidence.
**Skill:** `/mlx-mfa-apple-primitives-coverage` (§AA.5 premise dispatcher).

## TL;DR — **DECLINE (premise empirically FALSIFIED). The item is already BUILT, SHIPPED, and routed optimally.**

The archaeology overturns the queue entry's framing. This was **not** "never built /
premise-validation-pending." It was greenlit (Marco's Prompt 5c Option 1 mandate), **built
and shipped** (Prompt 5d: 4 native sparse backward kernels), math-correctness-validated (8
tests), and the win premise **empirically FALSIFIED** at the audit shape (Pattern #6). No
kernel work remains; nothing to greenlight.

---

## R.1 — Archaeology: the premise, recovered from the record (cited)

**Original proposal (Prompt 5a, 2026-05-14)** — `docs/v50/sprint-5-prompt5a-status.md:10-14`:
> "extend V34 backward kernels with native block-sparse iteration so that
> `mx.grad(flash_attention_sparse(...))` would use NAX-direct backward kernels (skipping
> inactive blocks) instead of falling back to SDPA-vjp with expanded bias."

**Projected win (the thing flagged for validation)** — `sprint-5-prompt5a-status.md:53-58`:
a **projected** 1.5–10× backward speedup at density 0.1–0.5, derived purely from the
active-block-skip ratio (10× skip at d=0.1 → "~1.5 ms" projected). Explicitly a projection,
never measured. Deferred per §AA.1 as optimization-tier (correctness was already restored by
the Section C `_sparse_nax_with_sdpa_vjp` wrapper). **This projection is exactly what needed
empirical validation before a greenlight.**

**What the record then shows happened (and what my prior-turn queue entry MISSED):**
- **Prompt 5d** (`docs/v50/sprint-5d-decisions.md:5-20`) — per Marco's explicit Prompt 5c
  Option 1 mandate, **built and shipped** 4 native sparse backward kernels (dV PoC from 5b +
  dQ + dK-split + fused-dKdV at D=64/128), math-correctness validated.
- **Live-code corroboration:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:5168` (`V34BWDV_SPARSE`),
  `:6364` (`V34BWDQ_SPARSE`) + dK-split + fused-dKdV; 8 shipped tests in
  `tests/test_v50_sprint_5d_sparse_backward_native.py`; opt-in routed at
  `mlx_mfa/attention.py:3187-3192`.

**Documentation-coherence finding:** the Marco-gated queue entry I authored last turn ("Sprint
5 V34 backward block-sparse NAX extension — was premise-validation-pending; never greenlit")
was **stale**. It was based on the 5a DEFERRED doc and missed the later 5d SHIP + 5c/5d
empirical falsification. Corrected here and in the queue.

## R.2 — Buildability premise: **PROVEN BUILDABLE (built + shipped).**

Not a question — a settled fact. The block-sparse mask gates which K-blocks the existing V34
backward traversal visits (mirrors the V34 forward LCSA sparse iteration); it composes as a
clean extension (`#if V34BWD*_SPARSE` mask blocks in the 4 K-parallel/dQ generators). The
kernels exist, compile, dispatch, and pass dQ/dK/dV correctness vs SDPA-vjp (8 tests). dataflow
verdict: **buildable as a V34 extension — confirmed empirically, not by reasoning.**

## R.3 — NAX-reachability premise: **REACHABLE (kernels run on M5 NAX) — but reachable ≠ wins.**

Primary source: the 4 shipped kernels ARE V34 NAX backward kernels (cooperative-tensor MMA
path), and they execute on M5 NAX — benched live across two shapes (below). So NAX-reachability
for this exact kernel shape (dQ/dK/dV, block-sparse, D∈{64,128}, fp16/bf16) is empirically
**TRUE**. The Pattern #6 caveat: Apple's own SDPA-vjp ALSO runs on M5 NAX and is more highly
optimized — so reaching NAX does not make the custom kernel win. (Test gate
`is_m5_plus`; kernels require M5+ NAX, consistent with the macOS-26.2+/gen-≥17 NAX-activation
finding.)

## R.4 — Pattern #6 win-bound: **MEASURED (not estimated) — FALSIFIED.**

**Current path** (`mlx_mfa/attention.py:3193-3199`): `mx.grad(flash_attention_sparse)` on M5+
routes by **default** to the Prompt 5c **hybrid** (NAX sparse forward + native sparse dV +
SDPA-vjp dQ/dK); full-native (4 kernels) is opt-in `MFA_V34_BWD_SPARSE_NATIVE=1`.

**Measured** — `docs/v50/section-a-v3-empirical-verification.md:19-33` (`mx.grad(loss)(q,k,v)`):

| Density | SDPA-vjp | Full native (5d) | **Native/SDPA** |
|---|---|---|---|
| VSR shape (B1 H12 qL4096 D128) d=0.1 | 17.41 ms | 22.58 ms | **0.77× (slower)** |
| d=0.3 | 17.40 ms | 60.67 ms | 0.29× |
| d=0.5 | 16.71 ms | 98.18 ms | 0.17× |
| d=1.0 | 16.93 ms | 181.07 ms | 0.09× |
| D=64 small-H (B1 H4 qL2048 D64) d=0.1 | 1.42 ms | 1.26 ms | **1.13× (faster)** |
| D=64 small-H d≥0.3 | — | — | 0.28–0.65× (slower) |

The 5a **projection** (10× faster at d=0.1) was **inverted by measurement** (0.77× = slower) —
the canonical Pattern #6 case: the active-block-skip saving does not survive against Apple's
highly-optimized SDPA NAX backward at audit shape. The **only** winning regime is D=64 /
small-H / **d=0.1 only**, at **1.13×** — "win envelope too narrow for production AUTO routing"
(`section-a-v3:41`), and already served by the opt-in flag. Win-bound verdict: **unlikely win →
falsified at the audit shape; marginal-and-too-narrow elsewhere.**

## R.5 — Greenlight gate: **DECLINE.**

| Premise | Verdict |
|---|---|
| R.2 buildable | YES (built + shipped) |
| R.3 NAX-reachable | YES (runs on M5 NAX) |
| R.4 plausible win | **NO — empirically falsified (0.09–0.77× at VSR; 1.13× in one narrow corner)** |

Gate requires all three; R.4 fails on **measured** data (stronger than the bounded-estimate the
gate asks for). Verdict: **DECLINE — first-class negative result.** This is the opposite of
UNVERIFIED-BLOCKED: the premise is maximally verified (shipped kernels + cross-shape bench).

**Nothing to build.** The kernels already exist, are correct, and are routed optimally:
- **Default** (production): Prompt 5c hybrid — empirically optimal.
- **Opt-in** (`MFA_V34_BWD_SPARSE_NATIVE=1`): full native — for the narrow D=64/small-H/low-d
  regime, research benchmarking, and as a reference + future-hardware hedge (`section-a-v3:74-79`).

A new AUTO carve-out for the 1.13× D=64/small-H/d=0.1 corner is also DECLINED: a single density
point in a narrow shape regime does not justify a density-and-shape-gated dispatch branch
(ghost-knob / Pattern #6 risk); the opt-in flag already covers it.

**Lesson reinforced (Pattern #6 + the TQ lesson #15 cousin):** "M5 has NAX" and "the kernel
reaches NAX" do not imply a win — Apple's primitive is the bar, and it must be *measured*, not
projected. The 5a active-block-skip projection (10×) vs the 5d measurement (0.77×) is a textbook
inversion. Recover-the-record-before-acting (the freshly-applied TQ archaeology discipline)
prevented re-running an already-settled, already-falsified investigation as if it were open.

---

## Disposition

Queue item **CLOSED — DECLINE (already built + shipped + falsified-for-default; opt-in retained).**
No kernel built this sprint; the verdict is the deliverable. No primary-source gap, no probe
needed (R.3 is empirically resolved by the shipped, benched kernels).

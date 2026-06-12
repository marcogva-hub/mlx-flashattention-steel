# Phase II — Sprint II-0 report: Finalization & Origin Push

**Date**: 2026-06-12 · **Status**: COMPLETE · Starting tip `ec8d701` → pushed tip `a4f189d`

## Change 1 — V34 backward D=64 causal DEFAULT-ON ✅

Three-axis (M5 Max, 3-block median):

| Axis | Result |
|---|---|
| 1 Output | rmse vs SDPA-vjp 0.0008–0.0011 (V34 fp16 floor; tolerance 5e-3) |
| 2 Path entered | 2.14×/2.57×/2.65× at N=2048/4096/8192 (1.32/4.35/16.26 ms vs 2.83/11.20/43.18 ms); opt-out restores SDPA-vjp timing exactly |
| 3 Edges | D=64-nc, D=128-c, D=128-nc UNCHANGED (rmse 0.0, timing parity).  **GQA edge FAILED initially** → root-caused + fixed (below) → GQA re-validated: shapes correct, rmse ≤3.3e-3, **2.7× speedup benched** → kept in envelope.  MQA validated (rel-rmse ~1e-4). |

**Axis-3 discovery — GQA gradient-shape bug (latent since v2.37.0)**: V34
backward kernels emit H_q-shaped dK/dV; the orchestrator never group-summed
to H_kv.  The opt-in path returned WRONG-SHAPED gradients for GQA/MQA all
along.  Fixed (group-sum in `_v34_backward_vjp`); locked by 2 new tests.
This is exactly what the three-axis edge gate exists to catch.

Envelope shipped: D=64 · causal · qL≥2048 · fp16/bf16 · M5+ · MHA/GQA/MQA.
Opt-out `MFA_DISABLE_V34_BACKWARD=1`.  Broader envelope stays opt-in.
+10 regression tests.  Suite: **1376 passed ×3**.

## Change 2 — Deprecation reword ✅
`MFA_FORCE_NATIVE_BWD` → "superseded" at all 3 sites (warning, ENV_VARS,
KD-5 ledger); deprecation tests pass; removal stays Marco-gated.

## Change 3 — 9-gate audit ✅
Gates 1-8 green (version SoT, tools, auto-default — the promotion itself
follows it, public-API perf claims 13/13, skill log, suite 1376, CHANGELOG
entry added for gate 7, hook contract via integration smoke).  **Gate 9**
(manual checklist): the ONLY generator-side conditional override of a
cfg-derived constant in csrc is `mfa_steel_bwd.cpp:776` (the KD-5 site) and
its dispatch mirror at `mfa_attention.cpp:1774` is intact; all other
generators consume cfg values directly.  PASS.  Revertibility spot-checked.

## Change 4 — Push ✅
`aa5741c..a4f189d master -> master`; local = origin, ahead-count 0.
No tag, no PyPI (per scope).

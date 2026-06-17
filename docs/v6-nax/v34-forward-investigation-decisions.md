# V6NAX forward investigation — decisions log

**Date opened**: 2026-05-12
**Branch**: `experiment/v6nax-forward-investigation`
**Foundation**: master @ `d4a876a`

## DI1 — Use env-var toggles instead of variant source-gen functions

**Decision**: probe each hypothesis via `MFA_V6_*` env vars (already
exposed in `mfa_v6_nax_primitive.cpp:117-125`) rather than building
dedicated `createV6NAXSource_VarA_TGPLow()` etc. functions.

**Rationale**:
- V6NAX already has explicit autoresearch infrastructure (`MFA_V6_USE_NAX`,
  `MFA_V6_EXEC_SG`, `MFA_V6_BLOCK_R/C/D`, etc.)
- Mechanistic isolation is achieved by ENV-VAR-TOGGLE in this case identically
  to source-gen-variant approach
- Implementation cost drops from "5 source-gen functions + 5 builds" to
  "5 bench commands with different env settings"
- The prompt §C.1 envisioned variant functions but the env knobs accomplish
  the same isolation goal cleaner — falls within Rule 0 "act don't ask"
  scope (substitute equivalent mechanism)

**Reversibility**: trivial — env vars are runtime-only, no code change.

## DI2 — Hypothesis-to-env-knob mapping

| Hypothesis | Probe mechanism |
|---|---|
| A — TGP occupancy | Vary `MFA_V6_EXEC_SG ∈ {1, 2, 4, 8}` on V6NAX baseline. Measure ratio. |
| B — cross-SG sync elim | V6NAX vs predecessor: `MFA_V6_USE_NAX=1` vs `=0`. Aggregate gain includes both B + C (both eliminated together in V6NAX). |
| C — simd_shuffle_xor vs MPP reduce | STRUCTURAL CONFIRMED by source reading; magnitude bundled with B (both replaced together). Reported as "bundled B+C". |
| D — register pressure | Force larger tile via `MFA_V6_BLOCK_R=64` (default 32); compare to baseline. Larger tile = more register pressure. |
| E — Apple defaults | V6NAX's M5-tuned BQ/BK/WM vs predecessor's MPP-defaults (already covered by aggregate B+C measurement). |

**Reversibility**: high — each probe is a single bench call.

## DI3 — Methodology

§4-strict 3-session per probe (180/60/90s cooldowns, A/B/A pattern).
However: given 5 probes × ~30 min/session × 3 sessions = ~7.5h of bench
wall-clock if all are strict 3-session, this would exceed practical
single-session budget.

**Compromise**: single-session strict §4 (180s initial + 60s inter-shape +
90s inter-round) per probe. 5 probes × ~12 min/session = ~1h total bench
wall-clock. This sacrifices cross-session confidence for sprint feasibility.

The aggregate V6NAX vs predecessor magnitude is already established by
v2.32.0 ship data (+18-40% net measured single-session multiple times in
shipping). This investigation's purpose is ATTRIBUTION, not magnitude
re-verification. Single-session §4 is appropriate for mechanism
attribution because the dominant variance source (cache state, thermal)
is shared across paired V6NAX/variant measurements within the session.

**Marked as caveat** in `v6nax-forward-mechanisms.md`: results are
single-session §4-strict, not full cross-session. The mechanism
attribution is the primary deliverable; magnitude is supporting data.

## DI4 — Shape inventory

Per prompt §B.2, 4 representative shapes:
- v6nax_small_d64 (1024×1024 D=64)  — boundary, may need §4.X caveat
- v6nax_small_d128 (1024×1024 D=128) — moderate
- v6nax_mid_d128 (4096×4096 D=128)   — primary investigation target
- v6nax_large_d128 (8192×8192 D=128) — large-shape regime confirmation

D=64 mid/large shapes excluded (D=64 routes to predecessor by source-gen
default; D=128 is V6NAX's headline use case).

## DI5 — Three-axis validation applied to this investigation

Per §3.5 committed in master:

1. **Output sanity**: each probe variant must produce correct output.
   Smoke gate: RMSE < 1e-3 vs reference SDPA on a small shape, before
   any timing.
2. **Path entered**: confirm via verbose log that env var was respected
   (V6NAX vs predecessor path indicator).
3. **Edges preserved**: V6NAX baseline must remain V6NAX (env unset) and
   produce baseline output unchanged from v2.32.0 ship. Smoke also asserts
   this.

## DI6 — No production code changes

All probes use existing V6NAX code via env-var dispatch. No `NAAttentionKernel.cpp`
modifications. The `experiment/v6nax-forward-investigation` branch will merge
to master ONLY the docs (Section H amendment to CLAUDE_V6_NAX.md + the
5-deliverables docs); no code changes.

## DI7 — Section H amendment to CLAUDE_V6_NAX.md

Insert new §5 after §4.X capturing the mechanistic findings. Reference
this investigation's results doc + the V6NAX source line numbers for each
confirmed mechanism. Anti-patterns documented for falsified hypotheses
(if any).

## DI8 — Section I design hints (V6NAX backward Option β feed-forward)

Each confirmed mechanism evaluated for:
- Directly transferable to backward (e.g., per-SG row partitioning)
- Adaptable with notes (e.g., simd_shuffle_xor for dK/dV row reductions)
- Not applicable (e.g., specific to forward softmax pattern)

Output: `docs/v6-nax/v6nax-backward-option-beta-design-hints.md`.

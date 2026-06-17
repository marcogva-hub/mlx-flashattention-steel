# V3 / V4 / V5 Kernels — State-of-Truth Archaeology (no action)

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Type:** ARCHAEOLOGY — recover the true current state; NO removal, NO build, NO re-gating.
Keep-all-paths: these STAY regardless of perf. Archaeology **B of two** (Marco decides after A+B).

## TL;DR — **SPLIT verdict.** V4/V5: **CLEAN-KEEP.** V3: **GAP (queue framing stale) + STALE-VERDICT.**

The queue entry — "V3/V4/V5 — opt-in only (regress vs V2 on this hardware); kept gated" — is
**accurate for V4/V5 but doubly wrong for V3**: V3 is **auto-routed** in a winning regime (not
opt-in), and it was measured to **win** vs V2 (not regress). Worse, that promote-vs-V2 verdict is
from **M1 Max, 2026-03-20** — never re-validated on **M5 Max / 26.6**, while V3 auto-fires for real
users (windowed-causal-large-N is default-reachable on M5). All three are **correct on M5** (suite
tests pass; V5's III-9 OOB-V clamp is in). No correctness risk; the V3 issue is gating-framing +
perf-verdict currency.

---

## R.1 — Design intent + the "regress vs V2" evidence (recovered, cited)

| Kernel | Built for | vs-V2 evidence | HW / OS / measured? |
|---|---|---|---|
| **V3** | separate K_smem + V_smem → removes 2 barriers/tile (K→V, V→K[next]) vs V2's shared KV_smem | **WINS** in regime: autoresearch `mfa_attention.cpp:793-799` "Geomean V3/V2 **1.015×** (causal); wins D=64 N≥4096, D=128 N≥2048, all B·H≥4" | **M1 Max, 2026-03-20/21** — measured (24-iter sweep). NOT M5/26.6. Earlier v2.7.0 (M1, 2026-03-12) had measured **0.77–0.88× regression** → verdict already flipped once with config |
| **V4** | direct device-K reads → eliminates K_smem + 2 barriers/tile | regress: `csrc/mfa_attention.cpp:587` "disabled by default pending benchmarks"; M1 sim 0.51–0.98× V2 (MEMORY) | M1 (sim M3+). Opt-in, never promoted |
| **V5** | D-blocked: Q in registers, KV_smem reused | regress: 0.60–0.90× V2 large-N (MEMORY v2.10.0) | M1 Max. Opt-in, never promoted |

**Measured vs asserted:** all three verdicts are *measured* (sweeps), but on **M1 Max** at
2026-03-era OS. The comparison is **V{3,4,5}-vs-V2 (mlx-mfa-internal)** for the gating verdicts —
not vs SDPA — so the OS shift (26.6) moves both kernels together (relatively stable). The **M1→M5
hardware shift is uncovered.**

## R.2 — Live-code state: gating + correctness

**Gating (the key finding):**

| Kernel | Dispatch gate | Opt-in only? |
|---|---|---|
| **V4** | `csrc/mfa_attention.cpp:588` `if (is_m3_plus_steel && MFAEnvConfig::enable_v4())` | ✅ YES — never auto |
| **V5** | `mfa_attention.cpp:693` `if (MFAEnvConfig::enable_v5())` | ✅ YES — never auto |
| **V3** | `mfa_attention.cpp:808-818`: `v3_shape_ok = causal && N≥(4096\|2048) && B·H≥4`; `v3_eligible = !disable_v3 && (v3_shape_ok ‖ v3_force) && f16/bf16 && tgp_ok && !block_mask` | ❌ **NO — AUTO-ROUTED** in regime; `MFA_ENABLE_V3` only *forces* (bypasses shape guard); comment `:801` "Production routing: V3 dispatched when shape is in the winning regime" |

**V3 default-reachability on M5:** dense causal D=64/128 forward routes to **SDPA** on M5
(`dispatch_policy.py:595-598`, has_nax → return False), so the MFA primitive (where V3 lives) is
**not** reached for dense → V3 reachable on M5 only via (a) the **windowed** path (`window → MFA`
unconditionally, `dispatch_policy.py:501-507`; V3 supports window — generator `mfa_steel_fwd_v3.cpp:206-294`),
or (b) `backend="mfa"`. So V3 **does** auto-fire for real users on M5 (windowed causal large-N,
B·H≥4), on the M1-2026-03 verdict.

**Correctness (all three correct on M5):**
- **V3**: smem-staged V (`V_smem`, `mfa_steel_fwd_v3.cpp:159,171`, `load_safe`) → **NOT** a
  device-direct-V-read site (III-9 multi-gate `sprint-III-9-report.md:90` lists only V2-single-pass
  / GNA / V5) → OOB-safe by construction. `TestSteelV3::test_v3_matches_sdpa` (vs SDPA, fp32-compared)
  + GQA + bf16 tests pass in the 1820-suite. Window supported.
- **V4**: `TestSteelV4::test_v4_matches_v2` (via `MFA_FORCE_GEN=15`) passes.
- **V5**: III-9 OOB-V clamp **IN** live code (`mfa_steel_fwd_v5.cpp:389-392`, "Clamp key-row on the
  partial final tile (§AA.5.x) … v_row = kL_rem-1"); `TestSteelV5::test_v5_matches_sdpa` +
  `test_iii9_gna_v5_direct_v_clamp.py` pass; v2.55.0 post-publish smoke verified V5 at N=40 vs fp32
  = 2.7e-5. Window supported (`:281`).

**No correctness gap, no stale-guard.** The only gap is V3's **gating-framing**: it is documented
("opt-in only") and queue-described as opt-in, but is actually conditionally auto-routed.

## R.3 — Verdict-currency (Pattern #6)

- **V4, V5**: opt-in, **never auto** → their regress-vs-V2 verdict drives no production routing.
  Currency is moot (a user must explicitly set the flag). Internal-comparison anyway. **CLEAN.**
- **V3**: auto-routed on an **M1-Max-2026-03** verdict, never re-measured on **M5/26.6**.
  - V3/V2 is mlx-mfa-internal → the 26.6 **OS** shift likely moves both together (relatively
    stable). BUT the **M1→M5 hardware** shift is the uncovered axis, and V3's benefit (−2
    barriers/tile) is bought with **2× TGP → halved occupancy** — a hardware-sensitive tradeoff
    that **already flipped once on M1** (v2.7.0 0.77–0.88× regression → 2026-03 1.015× win).
  - **Flip-plausibility: real.** On M5 (gen 17: different CU count, TGP budget, register file), V3
    could be slower than V2 in its auto-fire regime → mlx-mfa would be auto-selecting a *correct but
    slower* kernel for windowed-causal-large-N users. This is the exact Pattern #6 risk (a verdict
    measured on one HW driving production routing on another).

## R.4 — Verdict (state of truth, no action)

- **V4 — CLEAN-KEEP.** Opt-in (`enable_v4`), never auto, correct on M5. Queue framing accurate.
- **V5 — CLEAN-KEEP.** Opt-in (`enable_v5`), never auto, correct on M5 (III-9 clamp in + smoke-verified).
  Queue framing accurate.
- **V3 — GAP + STALE-VERDICT.**
  - **GAP (gating-framing):** V3 is **auto-routed** in its winning regime, NOT opt-in. The queue
    entry + the docstrings ("opt-in only / regress vs V2") are **stale/wrong** (reflect the v2.7.0
    state; V3 was promoted to conditional-auto in the 2026-03-21 autoresearch and measured to *win*
    vs V2). **Not a correctness gap** (V3 is correct + OOB-safe on M5). Scoped fix (Marco-gated, not
    applied): correct the queue/doc framing to "V3 = conditionally auto-routed (causal, N≥thresh,
    B·H≥4); measured-to-win-vs-V2 on M1-2026-03."
  - **STALE-VERDICT (perf currency):** V3's auto-fire perf basis was never measured on M5/26.6, the
    M1→M5 tradeoff is uncovered, and it already flipped once. **Gating next step (Marco-gated,
    flagged NOT run — not trivial, needs the §4 multi-session sweep across D×N×{dense,windowed}):**
    a Pattern #6 re-bench of **V3-vs-V2 on M5/26.6** at the auto-fire regime (causal N≥2048/4096
    B·H≥4, windowed + `backend="mfa"`) to validate or revise V3's production auto-routing.

**Keep-all-paths honored:** nothing removed/re-gated; V3/V4/V5 all stay. V3 stays auto-routed
pending Marco's call on the re-bench — flagged, not changed.

---

## Disposition (queue entry: ACCURATE for V4/V5, STALE for V3)

- **V4/V5:** CLOSED — keep-all-paths confirmed (opt-in, correct, accurate framing).
- **V3:** **GAP + STALE-VERDICT** — Marco-gated. Two next steps for Marco's decision: (1) correct
  the stale "opt-in/regress" framing (V3 is auto-routed + won-vs-V2 on M1); (2) Pattern #6 re-bench
  V3-vs-V2 on M5/26.6 before trusting V3's production auto-routing. **No code changed this sprint.**

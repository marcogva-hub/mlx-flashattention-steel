# `MFA_FORCE_NATIVE_BWD` — State-of-Truth Archaeology (no action)

**Date:** 2026-06-17
**Executor:** Claude Opus 4.8 High
**Type:** ARCHAEOLOGY — recover the true current state; NO removal, NO deprecation action, NO
code change. Verdict is the deliverable. Archaeology **A of two** (run B before Marco decides).

## TL;DR — **REMOVE-ELIGIBLE (ghost knob on the public path); deprecation cycle ALREADY COMPLETE.**

Unlike the V34 block-sparse queue entry (which was stale), **this entry is accurate**: the flag
IS deprecated/superseded, and the supersession is **measured, not asserted** (sprint-C Track 2
matrix), and **robust to the 26.6 shift**. The flag forces a correct-but-strictly-dominated
routing (never the fastest at any cell) and is **inert on the default public API** (forward
routes to SDPA on M5 → the backward branch that reads it isn't reached without `backend="mfa"`).
Its deprecation was **announced in v2.50.0 with "target removal v2.51+"** — we are now at
**v2.55.0**, so the warn-then-remove cycle is complete and overdue. Removal is contract-clean;
the only thing keeping it is a thin research/debug value + the standing "keep deprecated" steer.
**No action taken — Marco decides after archaeology B.**

---

## R.1 — Origin + documented status (recovered, cited)

- **Introduced** v2.36-era as a debug/eval override — `CHANGELOG.md:3348, 3870` ("`MFA_FORCE_NATIVE_BWD=0|1` override precedence for debug/evaluation"); forces routing through legacy STEEL backward (`MFASteelBwdDQ`/`MFASteelBwdDKV`) — `CHANGELOG.md:1129`.
- **Deprecated** v2.50.0 (Prompt 5f Phase E) — `CHANGELOG.md:747-754`, `docs/v50/known-debt-v2.50.md:189-200`. **Original reason: BROKEN** — KD-5 bug (STEEL backward zeroed output blocks for query rows ≥1024 at D=128 N≥2048): `docs/MIGRATION_v2.39.1_to_v2.50.0.md:77-90`, **"target removal v2.51+"**.
- **Reason SHIFTED "broken" → "superseded"** after the 2026-05 whole-repo review FIXED KD-5: `docs/v50/campaign-2026-06/phase2/sprint-II-0-report.md:26` ("superseded" reword at all 3 sites: warning + ENV_VARS + ...), `sprint-C-report.md:68`, `CAMPAIGN-CLOSE.md:45`. Current live warning (`dispatch_policy.py:726-733`) says **"SUPERSEDED, not broken … correct at every cell (rmse ~4e-5) … auto dispatch picks optimal per cell … redundant. Removal remains a future Marco-gated step."**
- **Was supersession measured?** YES — see R.4. Not asserted.

## R.2 — Live-code state (strongest primary source)

**Reference map** (every non-archive reference):

| Location | Role |
|---|---|
| `mlx_mfa/dispatch_policy.py:706, 717-739` | the **only** reader. `=="1"` → emit `DeprecationWarning` + return `supported`; `=="0"` → return False; else → policy-table (`seq_len >= min_n`) |
| `mlx_mfa/attention.py:5304-5306` | sole caller of `should_use_native_backward(...)` (in the non-V34 backward branch) |
| `tests/test_v50_prompt_5f_kd5_deprecation.py` | 3 tests asserting the DeprecationWarning fires on `=1`, not on `=0`/unset |
| `tests/test_attention.py:11051-11092, 11124-11159` | override-precedence + force-on/off routing tests |
| `ENV_VARS.md:60` | public env-var doc — "DEPRECATED — superseded … Removal is Marco-gated" |
| `docs/MIGRATION_…:77`, `known-debt-v2.50.md`, `known-issues-v2.50.md`, `cache-audit/01-affecting-inputs.md:37`, `CHANGELOG.md` (historical) | docs |
| README | **not mentioned** (smaller public surface) |

**Current behavior — does it still change dispatch?**
- **On the default public API** (`backend="auto"`): on M5 the **forward** routes to Apple SDPA, so the V34/native-backward dispatch branch that consults `should_use_native_backward` **is never reached** → the flag is **inert on the default path** (`ENV_VARS.md:60`, `sprint-C-report.md` ¹).
- **On the `backend="mfa"` forced path**: `=1` DOES change routing — for a supported cell (causal, D∈{64,128}, f16/bf16) it forces native STEEL backward (`mfa_steel_backward`) where auto would pick the policy-table path (V34 or SDPA-vjp). So it is **not a pure no-op**: it forces a distinct, *correct* result.

**Classification: ghost knob on the public path / weak escape hatch on the expert path.** It only
fires under `backend="mfa"`, and the path it forces is correct-but-dominated (R.4) — never the
fastest. Its sole residual value is research/A-B/determinism (forcing STEEL backward for
comparison) + a reference for the STEEL backward kernel. Contrast the V34 block-sparse opt-in flag,
which was KEPT because it *wins* in a narrow corner — this flag wins **nowhere**.

## R.3 — Removal-impact + public-contract check

`MFA_FORCE_NATIVE_BWD` is a **documented public env var** (`ENV_VARS.md`, `MIGRATION`, `CHANGELOG`)
→ removal is a breaking change requiring a deprecation cycle. **That cycle is already COMPLETE**:
announced v2.50.0 with `DeprecationWarning` (live) + "target removal v2.51+"; we are at v2.55.0
(v2.50→v2.55 = 5 minor versions of warning). So removal in the next release is the honest,
contract-respecting action — the warn-then-remove window has run; this is NOT a silent removal.

**References removal would touch** (§AA.5.x multi-gate — find every reference, not just the obvious):
1. `dispatch_policy.py:706` docstring + `717-737` (the `=="1"`/`=="0"` branches + warning). The `should_use_native_backward` **policy-table path stays** (lines 741-745).
2. `tests/test_v50_prompt_5f_kd5_deprecation.py` (whole file) + `tests/test_attention.py:11051-11092, 11124-11159` — update/remove the force-behavior tests.
3. `ENV_VARS.md:60` (remove row / mark removed-in-vN); `cache-audit/01-affecting-inputs.md:37` (env list).
4. `CHANGELOG.md` — add a "Removed" entry; **historical** entries (747, 1129, 3348…) stay as record. `MIGRATION`/`known-debt`/`known-issues` are historical KD-5 records — leave.
5. **Keep-all-paths**: the STEEL backward kernel (`mfa_steel_backward`, `MFASteelBwdDQ/DKV`) is **retained** and stays reachable via `backend="mfa"` — removal is of the **env-var knob**, not the kernel.

## R.4 — Was "superseded" measured? YES (sprint-C Track 2, `sprint-C-report.md:47-66`)

Causal fp16 backward, ms (B=1 H=8), STEEL-bwd = what `=1` forces:

| Cell | SDPA-vjp (default) | V34 (opt-in) | STEEL-bwd (forced) | Winner |
|---|---|---|---|---|
| D64 N2048 | 2.98 | **1.37 (2.2×)** | 2.68 | V34 |
| D64 N4096 | 11.53 | **4.50 (2.6×)** | 8.36 | V34 |
| D64 N8192 | 44.54 | **17.47 (2.5×)** | 31.30 | V34 |
| D128 N2048 | **3.32** | 5.71 | 6.56 | SDPA-vjp |
| D128 N4096 | **12.44** | 24.09 | 23.07 | SDPA-vjp |
| D128 N8192 | **49.05** | 106.3 | 88.57 | SDPA-vjp |

STEEL-bwd (forced path) is **correct everywhere post-KD-5 (rmse 4e-5)** and 1.12–1.42× faster than
SDPA-bwd at D=64 — **but dominated at every cell**: by V34 at D=64, by SDPA-vjp at D=128. **It is
never the optimal choice.** sprint-C dated 2026-06-12; the III-11 26.6 re-bench found Apple SDPA
got *faster* on 26.6 → **strengthens** "SDPA-vjp wins at D=128", and V34 D=64 default-on win was
re-confirmed in the v2.55.0 perf re-statement. So the supersession is measured **and robust to
26.6** — not a STALE-VERDICT.

## R.5 — Verdict (state of truth, no action)

**REMOVE-ELIGIBLE** — auto dominates everywhere the flag would fire (measured, robust to 26.6),
the forced path is correct-but-never-optimal, and it is inert on the default public API. The
deprecation cycle is **already complete** (announced v2.50.0, target v2.51+, now v2.55.0), so
removal of the **env-var knob** in the next release (e.g. v2.56.0) is contract-clean — not silent.

**Honest nuance (the only argument for KEEP):** the flag is a thin *escape hatch* on `backend="mfa"`
— forcing STEEL backward for research/A-B/determinism — and the standing Marco dispositions
(`sprint-C:68`, `CAMPAIGN-CLOSE:45`) said **"keep deprecated"** (warn, don't remove yet). Keep-all-paths
is satisfied at the **kernel** level regardless: removing the env var does not remove the STEEL
backward kernel (still reachable via `backend="mfa"` internals).

**This is NOT a STALE-VERDICT** (supersession measured) and NOT a clean KEEP (forced path wins
nowhere). It is REMOVE-ELIGIBLE with the deprecation cycle complete. **No action this sprint.**

If Marco chooses removal after archaeology B, the path is: **Removed entry in CHANGELOG +
delete the `=="1"`/`=="0"` branches in `dispatch_policy.py` (keep the policy table) + update the 2
test files + the ENV_VARS/cache-audit docs + retain the STEEL kernel** (the R.3 reference map).

---

## Disposition (queue entry: ACCURATE, not stale)

The queue entry ("deprecated/superseded; removal a future Marco-gated step") is **corroborated** by
live code + the measured matrix + ENV_VARS + standing dispositions. Refined verdict:
**REMOVE-ELIGIBLE, deprecation cycle complete** — Marco-gated, pending archaeology B. No code changed.

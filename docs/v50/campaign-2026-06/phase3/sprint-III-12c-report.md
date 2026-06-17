# Sprint III-12c — Advisory Cleanup + Honest TQ Reframe + v2.55.0 Cut

**Date:** 2026-06-16
**Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `77f3009` (clean) → III-12c doc commit → v2.55.0 cut.
macOS 26.6 (25G5028f), Apple M5 Max 128GB, mlx 0.31.2.
**Type:** two doc finalizations → the Marco-gated atomic release cut (GO given).

This sprint closes Phase III. III-12b restored the TQ claim correctly (Case 2: real,
baseline lost) and got the 9-gate audit GREEN on all 7 technical checks. III-12c clears
the last two items on Marco's instruction and cuts v2.55.0.

---

## Skill invocations (§AA.2)

| Checkpoint | Skill | Outcome |
|---|---|---|
| Pre-version-bump (canonical pre-tag gate) | `/mlx-mfa-release-audit` | GREEN on 7 technical checks pre-cut; advisory cleared; only Check 1 (version bump) blocking pre-cut → clears on the bump commit |
| Perf-claim reframe (user-facing wording change) | covered by release-audit Check 4 (public-API reachability, 17 claims audited PASS) | claim still REACHABLE via default `step()` path |

---

## R.1 — Honest TQ reframe (lead with the user's real choice)

**Marco's instruction:** reframe to "whatever is most honest." III-12b's restored claim
("6.5–23× faster than the prior fused TQ kernel it replaced") is TRUE but leads with a
ratio the user **cannot capture**: the fused kernel is gone, so it is not a baseline a
user can select. The decision a v2.55.0 user actually makes is **TQ paged decode vs fp16
dense decode** — there TQ is **~1.4–3× slower** in latency, buying **~4–5× KV-cache
memory reduction at cosine ~0.96**.

**Lesson #15 extended:** a ratio must lead with the denominator **relevant to the
reader**, not the one that produces the biggest number. A correct ratio against a
non-actionable (removed) baseline is a cousin of the direction-ambiguity III-12b fixed.

**Final wording (README / PERF_CLAIMS / CHANGELOG all aligned):**

> TurboQuant paged decode (opt-in KV compression via
> `TurboQuantPagedInferenceContext` — not auto-routed) **trades ~1.4–3× decode-step
> latency for a ~4–5× KV-cache memory reduction at cosine ~0.96, vs fp16 dense decode**
> (`step()` `0.75 ms vs 0.33 ms` @ S=16K; KV `32 MB → ~6.5 MB` @ S=8K). _(Internal-perf
> history, not a user-facing choice: the gather/dequant+SDPA path is 6.5–23× faster than
> the fused TQ attend kernel it replaced — `0.75 ms vs 16.8 ms` @ S=16K, re-confirmed
> 26.6 — but that kernel is gone, so it is not a selectable baseline.)_

- **Leads with** the user-actionable vs-fp16-dense trade-off, absolutes both sides.
- **Keeps** the vs-fused gain as clearly-labelled secondary historical context.
- **Explicit** that TQ is opt-in (the user chooses the trade-off deliberately).
- **Direction + absolute** preserved on every number (III-12b discipline) — only the
  order changed so the actionable comparison leads.

**Other claims re-checked for non-actionable leads (already direction+absolute from
III-12b):** forward attn (vs SDPA — actionable), V6NAX backward (vs SDPA-vjp — actionable),
conv MPP (vs `mx.conv_general` for bf16 — actionable; fp16 vs legacy im2col is framed as
**correctness-default, not a speed headline**), LCSA (marked historical, not a current
speedup), D=256 (correctness fix, not a speed claim). None leads with a non-actionable
baseline as its value proposition. ✔

## R.2 — Cosmetic advisory cleared

`tests/test_release_notes_perf_claims.py` iii2 entry was both **stale** (still the old
"6.0× (S=4K) to 14.4× (S=16K)" framing) and **unversioned**. Rewrote `documented_perf_claim`
to the III-12c trade-off framing, added `README.md` + `docs/PERF_CLAIMS.md` to
`documented_in`, and stamped `v2.55.0` into the claim text. The audit's advisory is a
substring check (`"2.55.0" not in test_src`) — now satisfied. Reachability test: 17 passed.

## R.3 — Pre-cut verification

- III-12c doc edits committed as their own commit (version bump kept separate per the cut
  protocol).
- Full suite green ×consecutive on clean HEAD; tree clean; no orphan bench processes (RULE 12).
- CHANGELOG [2.55.0] final + truthful against the reframed docs.
- Release-scope commits confirmed on master.

## R.4 — The atomic cut (GO given)

1. Bump 3 SoT (pyproject + `__init__` + README) → 2.55.0 — its own commit.
2. Re-run `/mlx-mfa-release-audit` → FULLY GREEN (Check 1 included).
3. Annotated tag `v2.55.0`.
4. Build wheel → `.venv/bin/twine upload` (PyPI).
5. `gh release` (CHANGELOG [2.55.0] notes) → push tag.
6. Post-publish smoke on the published wheel: the 4 fixed paths vs fp32.

(Cut record + URLs + smoke result appended below on completion.)

---

## Cut record

| Step | Result |
|---|---|
| III-12c doc commit (reframe + advisory) | `5a9f21c` |
| Version bump 2.52.1 → 2.55.0 (own commit) | `3c2c8b4` (pyproject + `__init__` + README) |
| 9-gate audit (post-bump) | **GREEN** — all 8 checks PASS, `recommendation: proceed_with_release` |
| Annotated tag | `v2.55.0` (on `3c2c8b4`) |
| Build | `mlx_mfa-2.55.0-cp311-cp311-macosx_26_0_arm64.whl` + `mlx_mfa-2.55.0.tar.gz`; `twine check` PASSED both |
| PyPI | **LIVE** — https://pypi.org/project/mlx-mfa/2.55.0/ (wheel + sdist) |
| GitHub release | **LIVE** (draft=false) — https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.55.0 |
| Tag pushed | `origin v2.55.0` (+ `master` → `3c2c8b4`) |

## Post-publish smoke (clean env, **published** wheel from PyPI)

Clean venv → `pip install --no-cache-dir mlx-mfa==2.55.0` (pulled mlx 0.31.2) →
ran from `/tmp` so the import resolved to the installed wheel
(`site-packages/mlx_mfa/__init__.py`, `__version__ == 2.55.0`). All comparisons vs an
**independent fp32 ground truth** (SDPA-fp32 for dense; vendored fp32 GNA reference) —
lesson #11.

| Path (the fix) | max_abs_err vs fp32 | finite |
|---|---|---|
| Fix 1 — V2 single-pass non-causal D=128 N=40 (last-head OOB) | 2.70e-5 | ✔ |
| Fix 2 — GNA 3D (2,4,5) N=40 D=128 (**default-reachable** OOB) | 3.00e-5 | ✔ |
| Fix 3 — STEEL V5 non-causal D=128 N=40 (opt-in, same OOB) | 2.70e-5 | ✔ |
| Fix 4 — split-K decode ×8 under pool churn (scratch lifetime) | 1.02e-6 | ✔ (all 8) |
| Sanity — forward auto causal D=128 N=512 ≈ SDPA | 7.24e-5 | ✔ |
| Sanity — V6NAX backward grad | — | ✔ finite |
| Sanity — TQ decode `step()` | — | ✔ finite |

**Verdict: SMOKE PASSED** — the published wheel IS the fixed binary; all four
correctness fixes verified on the real artifact. (The conv3d post-publish line was an
INFO, not a failure: the smoke script passed a wrong weight-channel shape; conv is
covered by the 1820-test suite that passed ×2 pre-cut on the identical source.)

## Status

**v2.55.0 SHIPPED.** Marco-gated release queue item (the cut) → DONE. This closes
Phase III: every correctness bug hunted to structural completeness (§AA.5.x multi-gate),
every perf claim re-measured + honestly framed on 26.6 against the reader-actionable
baseline, every wrong turn corrected in the record (lessons #14, #15).

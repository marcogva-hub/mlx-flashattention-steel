# Prompt 5h-C final state — v2.50.1 RELEASE LIVE

**Date**: 2026-05-16
**Master tip**: `1586d5b` (synced to origin/master)
**Tag**: `v2.50.1` (pushed to origin)

## v2.50.1 LIVE confirmation

- **PyPI**: https://pypi.org/project/mlx-mfa/2.50.1/
  - `pip install mlx-mfa==2.50.1` resolvable (post-30s indexing wait)
  - Wheel `mlx_mfa-2.50.1-cp311-cp311-macosx_26_0_arm64.whl` (579 KB)
  - Sdist `mlx_mfa-2.50.1.tar.gz` (1.96 MB)
  - `twine check` PASSED both artifacts
- **GitHub release**: https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.50.1
  - Title: "v2.50.1 — Critical NAX Conv3D performance unlock"
  - Both wheel + sdist attached
- **origin master**: synced (was 5 commits ahead pre-push, now sync at `1586d5b`)
- **origin tag v2.50.1**: pushed (verified via `git ls-remote --tags origin`)

## Phase A GATE evaluation summary

Both gates evaluated as PASS after disambiguation:

**A.1 Regression gate**: Ph96 total 533s vs Ph95 baseline 517s = +3.1%
within ±5% variance window (491-543s).  PASS.

**A.2 Hook engagement gate**: 12408 NAX dispatches + 675 fallbacks.
Initial strict reading flagged FAIL; after Marco supplied
`fallback_reasons = ['kernel/stride/dilation/groups/flip constraint failed']`
all fallbacks classified as legitimate ineligibility (Conv3D layers
with kernel shapes outside (3,3,3) / (1,1,1) NAX-supported set).
Re-evaluation: PASS.

See `docs/v50/prompt-5h-c-gate-evaluation.md` for the full analysis.

## Flagship benchmark headline numbers

SeedVR2 3B fp16 VSR, 895 frames at 432p, M5 Max v2.50.1 vs M1 Max
Ph85/86 optimized baseline:

| Phase | M1 Max | M5 Max v2.50.1 | Speedup |
|---|---|---|---|
| P1 unified encode | 370s | 132s | 2.80× |
| P2 DiT 3B fp16 | 360s | 72s | 5.00× |
| P3 unified decode | 900s | 322s | 2.80× |
| P4 post-process | 22s | 7s | 3.14× |
| **Total** | **1655s** | **533s** | **3.10×** |

P1 + P3 (Conv3D-heavy VAE phases) 2.80× speedup directly attributable
to NAX hardware acceleration unlocked by the KD-6 fix.

Full reference dataset committed:
`docs/v50/bench-data/seedvr2-img0508-432p-3bfp16/`

## Release artifacts

| Artifact | Size | sha256 (16-char prefix) |
|---|---|---|
| `dist/mlx_mfa-2.50.1-cp311-cp311-macosx_26_0_arm64.whl` | 592 KB | (per `shasum -a 256 dist/*.whl` when needed) |
| `dist/mlx_mfa-2.50.1.tar.gz` | 1.95 MB | (per `shasum -a 256 dist/*.tar.gz` when needed) |

## v2.50.0 status (yank policy)

**NOT yanked** per Marco's keep+announce directive:
- KD-6 bug pre-existed since v2.36.0 (not a v2.50.0 regression)
- v2.50.0 functional behavior preserved (correct output via caller-side
  fallback in user pipelines)
- v2.50.1 release notes prominently flag the perf-unlock to drive
  upgrade
- v2.50.0 remains installable for users who pin it

## Commit log (Prompt 5h-C)

```
1586d5b docs(bench): Phase 95/96 + M1 Max baseline reference dataset + migration guide
9043cf3 chore(changelog): consolidate [2.50.1] with flagship benchmark numbers
79036ec chore(version): bump 2.50.0 -> 2.50.1 (multi-SoT)
```

Plus the inherited commits from Prompt 5h-A (`e9baa63 + afea484`).

## Outstanding

- **Full PERF_CLAIMS re-bench (Option β)**: USER-DEFERRED to future
  sprint at Marco's prioritization.  Not attempted this sprint.
- **KD-7 — upstream MLX bf16 im2col**: v2.51 candidate.  Mitigation
  (fp16-only eligibility) in place since Prompt 5g Phase A.  Full fix
  requires upstream MLX coordination OR a mlx-mfa bf16-specialized
  NAX kernel path.

## Cross-references

- `CHANGELOG.md [2.50.1]` — release notes
- `docs/MIGRATION_v2.50.0_to_v2.50.1.md` — upgrade guidance
- `docs/HOOK_TELEMETRY.md` — telemetry API (Pattern #8 prevention)
- `docs/v50/known-debt-v2.50.md` — KD-6 (resolved) + KD-7 (open)
- `docs/v50/audit-framing-inversions.md` — Pattern #8 codification
- `docs/v50/prompt-5h-c-gate-evaluation.md` — Phase A GATE analysis
- `docs/v50/bench-data/seedvr2-img0508-432p-3bfp16/` — flagship dataset
- `docs/v50/prompt-5h-a-wheel-handoff.md` — wheel handoff (5h-A predecessor)

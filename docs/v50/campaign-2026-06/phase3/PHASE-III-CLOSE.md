# Phase III — CLOSE (2026-06-14)

Phase III delivered the two remaining Marco-gated gain candidates, the
tagged campaign release, and a fresh exhaustive whole-repo audit run
**repeat-until-clean to its fixed point** (9 passes, zero-finding on the
final pass).

## Sprint ledger

| Sprint | Outcome | Commit |
|---|---|---|
| III-1 | **KD-7 bf16 conv lift PROMOTED** — MPP convolution2d bf16 variant probed II-2R-style (genuinely implemented), 1.4–2.7× vs the pre-lift public bf16 path; bf16 gated to the MPP branch only (legacy im2col stays fp16, loud guard) | `69977f6` |
| III-2 | **TQ paged decode PROMOTED** (§AA.5 full inversion) — per-step gather/dequant kernels + Apple SDPA; step 6.0× (S=4K)–14.4× (S=16K), attend-only 13.8–22.1×. Bonus: fixed the fused TQ kernel's **silent 2/4-bit corruption** (3-bit-only unpack, latent since it landed) | `0593f6b` |
| III-3 | **v2.51.0 tagged release** — 9-gate audit GREEN; PyPI + GitHub release; captured B→A→C + Phase II + Phase III gains | `abdaa8d` |
| III-4 | **Fresh exhaustive whole-repo audit, 9 passes to the fixed point** — ~73 fixes incl. **2 pre-existing CRITICALs** the test suite never exercised | `d2d3f8d`…`d8614ea` |

## III-4 audit ledger (9 passes, repeat-until-clean)

The empirical thesis held: every pass found new material until the
systematic classes were exhausted, then the tail thinned to ~1 isolated
peripheral pre-existing bug per pass, converging to a zero-finding pass.

| Pass | Findings (headline) | Commit |
|---|---|---|
| 1 | **66 findings.** CRITICAL: topk bisect kernel Metal-grid undercount (wrote top-K thresholds for only the first 8 query rows/head; rest read stale pool). Mixed-dtype kernel reinterpretation. M3/M4 D=128 sparse mask geometry. D7 non-divisible-N bias re-tiling (real wrong grads). D16 sparse-backward downsample contamination (dV RMSE 0.5). R1 patch_mlx_lm windowed-decode key-0 bug. +~58 contract/dtype/test/doc | `d2d3f8d` |
| 2 | Grid-spec class **clean** (topk isolated); all pass-1 fixes regression-verified. New: 6 expert-API dtype-reinterpret entries (loud-guarded); window fwd/bwd anchor inconsistency | `ff6259f` |
| 3 | C++ eval_gpu (cache-keys/overflow/is_equivalent, 12 primitives) **clean**; all backward paths **clean** at adversarial scale. New: F1/F2 empty-row → NaN (topk + lcsa dispatch) → II-6 zeros contract | `c1da2a2` |
| 4 | Empty-row §AA.5.x class sweep: 1 sibling (no-ext sparse fallback); class **closed**; rest verified | `f878e12` |
| 5 | Regression all-pass. **CRITICAL P5-1**: `mx.grad(flash_attention(return_lse=True))` returned corrupt/NaN gradients (raw 2-output C++ Primitive vjp); fixed via custom_function | `aaede0d` |
| 6 | Exhaustive gradient probe — **all 17 feature×grad combos clean**, all 11 custom_function vjps consistent. 1 LOW (attn_bias fp32 dtype-promotion crash) fixed | `484c630` |
| 7 | Least-touched surfaces. F7-1 MEDIUM: `quantize_model` silent no-op on direct-attribute models (`nn.Module` is a `dict` subclass; tree walk checked `dict` first); mlx_lm/external_cache/conv/build **clean** | `ba07f60` |
| 8 | Mask-constructor family (least-audited, 3 prior bugs) fully swept. F8-1 MEDIUM: `make_axial_temporal_mask` under-approximation on non-pow2 grids; svdquant forward numerics **correct** | `d8614ea` |
| 9 | **ZERO-FINDING.** Public helpers (8), TurboQuant 20-step decode (no drift), dispatch table (96 cells), kv-cache adapters, ShaderCache concurrency, Rule-8 — all probed clean with numbers. **Audit terminated.** | — |

### Two CRITICALs caught (both pre-existing, both invisible to the suite)
1. **topk bisect Metal-grid undercount** (pass 1) — a PROMOTED AUTO-default kernel selected top-K for only ~8/N query rows; the rest read stale pool memory. Passed tests because stale buffers usually held benign zeros.
2. **`return_lse` backward corruption** (pass 5) — `mx.grad` through `flash_attention(return_lse=True)` returned NaN/garbage gradients; no test ever gradded a `return_lse` output.

### Pattern-class lessons added in Phase III
8. **MLX `grid` is total threads, not threadgroups** — a `threadgroup_position_in_grid`-indexed kernel needs `grid.x = n_items × threadgroup_size`. An undercount silently under-writes output → stale reads.
9. **`isinstance(child, dict)` before `nn.Module`** is a silent-no-op trap — `nn.Module` IS a `dict` subclass.
10. **Test suites that only exercise one shape class hide whole bug families** — power-of-2 spatial grids (masks), `nn.Sequential` (svdquant), 0.1-scale fixtures (II-6), same-dtype inputs (dtype-reinterpret), never-gradded feature combos (return_lse). The audit's value came from deliberately breaking each assumption.

## State at the III-4 fixed point

- Version **2.51.0** unchanged in tree (the III-4 correctness fixes are
  committed but **not yet released** — see the release note below).
- Suite: **1489 passed, 2 skipped** (×3 default), **1491 passed**
  (×2 stressed `MFA_POOL_STRESS=1`). Stable, no abort, no flake.
- Zero known correctness bugs in any class swept across 9 passes.
- master pushed through `d8614ea`.

## Release note (Marco DECISION 2026-06-14: HOLD — no release yet)

**v2.51.0 on PyPI (shipped in III-3) does NOT contain the III-4 audit
fixes** — including the two CRITICALs (topk grid, `return_lse`
backward) and ~71 further correctness fixes, which all landed *after*
the tag, committed/validated on master through `8c9cbff`.

**Marco's decision: HOLD — no follow-up release yet.** master stays
as-is; the III-4 fixes remain committed but unreleased, to be bundled
into a later release (e.g. with int8 V6NAX integration as v2.52.0) rather
than shipped as a standalone v2.51.1 patch now. The 2 CRITICALs are
therefore knowingly unreleased on PyPI until that bundled release.
When the release happens, the 9-gate `/mlx-mfa-release-audit` runs
again pre-tag and the CHANGELOG must credit the III-4 correctness fixes.

## Marco-gated decision queue

| Item | Evidence | Effort |
|---|---|---|
| Bundled release (III-4 fixes + int8/follow-ups) capturing the 2 CRITICALs + ~71 fixes | per Marco's HOLD decision — defer to a later version, not a standalone v2.51.1 | S–M (release flow) |
| int8 V6NAX-generator integration (Sage-NAX revival) | 2.00× gate pass; II-2R projects 1.11–1.33× end-to-end | L/XL |
| cider tier-3: paged/TQ decode transplant | TQ floor closed by III-2; cider expert API in-tree | M |
| Non-causal windowed S-N anchoring + the small-N windowed Metal-abort | III-4 B1: forward keeps 0-anchor (documented); true position-based non-causal windows need the latent Apple-Metal small-N-windowed late-dispatch abort root-caused first | M (kernel + Metal investigation) |
| Lazy packed-V pool when only the III-2 decode path is used | III-4 R9: tq_v=True keeps both packed + fp16 V pools | S |
| make_topk_spatial_mask numpy/loop → GPU pooling (R13) | perf-debt; needs bench per Pattern #6 | S/M |
| Tagged release of int8/follow-ups | carried | — |
| V6NAX backward block-sparse NAX extension | **CLOSED — DECLINE** (premise validation 2026-06-17). Was NOT pending: greenlit (Prompt 5c Opt.1), built+shipped (Prompt 5d, 4 native sparse bwd kernels, 8 tests), win premise empirically FALSIFIED — Native/SDPA 0.09–0.77× at VSR shape (Pattern #6); only 1.13× at D=64/small-H/d=0.1 (too narrow for AUTO). Default = 5c hybrid; full native = opt-in `MFA_V6_BWD_SPARSE_NATIVE=1`. See `docs/v50/campaign-2026-06/v6nax-blocksparse-nax-premise-report.md`. | none (no build) |
| `MFA_FORCE_NATIVE_BWD` disposition (Archaeology A → Queue Closure, 2026-06-17) | **CLOSED — REMOVED v2.56.0.** Deprecation cycle complete (announced v2.50.0 "target removal v2.51+", removed at v2.55.0+1). Forced STEEL backward was dominated at every cell (sprint-C Track 2). Env-var knob removed in `dispatch_policy.py` (policy table kept); STEEL kernel retained (keep-all-paths) + tested via direct `_ext.mfa_steel_backward` binding; every reference multi-gated (2 test files, ENV_VARS, cache-audit); code orphan-free. See `queue-force-native-bwd-archaeology.md` + `queue-closure-sprint-report.md`. | done (v2.56.0) |
| V3/V4/V5 kernels disposition (Archaeology B → Queue Closure, 2026-06-17) | **CLOSED.** **V4/V5 = CLEAN-KEEP** (opt-in, never auto, correct on M5). **V3 = VALIDATED on M5/26.6** — the auto-routing concern is RESOLVED. 3-session §4-strict re-bench (V3 vs V2 the fallback): V3 faster-or-parity at every auto-fire cell — windowed D=64 N4096 **0.68× (V3 ~32% faster)**, N8192 0.92×, D=128 N4096 0.97×, N8192 ~parity; backend="mfa" D=64 0.86×, D=128 ~parity. M1 "win" verdict holds, stronger at D=64. No routing change (validated as-is, keep-all-paths); framing corrected (V3 = conditionally-auto, M5-measured). See `queue-v3v4v5-archaeology.md` + `queue-closure-sprint-report.md`. | done (validated, no routing change) |

## Reports index

`docs/v50/campaign-2026-06/phase3/`: sprint-III-1-report.md,
sprint-III-2-report.md, iii4-findings-ledger.md (full 9-pass ledger),
this file. Release record: tag `v2.51.0`,
https://pypi.org/project/mlx-mfa/2.51.0/ ,
https://github.com/marcogva-hub/mlx-flashattention-steel/releases/tag/v2.51.0

---

## Phase III COMPLETE

Both gain candidates dispositioned (III-1 + III-2, both promoted), the
campaign release shipped (III-3, v2.51.0), and the fresh exhaustive
audit run to its repeat-until-clean fixed point (III-4, 9 passes,
zero-finding).  Release follow-up decided by Marco (2026-06-14): HOLD —
the III-4 fixes stay committed/validated on master, to ship in a later
bundled release rather than a standalone v2.51.1 now.  Phase III is
closed; no open items pending an immediate action.

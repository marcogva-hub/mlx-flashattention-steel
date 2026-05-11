# Conv3D NAX — Sprint C Phase 1.0 Decisions

**Date**: 2026-05-11
**Companion to**: `conv-nax-design.md` (the primary deliverable)
**Purpose**: capture the major choices made during Phase 1.0 design writing + rationale.

---

## D1. Algorithm: Option α (materialized chunked im2col)

**Choice**: Option α — chunked materialized im2col + `mpp::tensor_ops::matmul2d` per chunk.

**Alternatives considered**: Option β (implicit on-the-fly im2col), Option γ (MPS Graph delegation).

**Rationale (§2.2 of design doc)**:
- Simplest path that delivers the perf win; leverages existing V6 NAX matmul2d wrapping expertise from `csrc/mfa_v6_nax_primitive.cpp`.
- Estimated 44% faster than MLX baseline on the largest shape (up3_resnet0: 297 ms vs MLX 530 ms; 43% over theoretical floor 207 ms).
- Validation surface is well-understood: 3 oracles (PyTorch CPU FP32, MLX `conv_general`, sentinel-fill) each catch different classes of bug. Im2col addressing and matmul correctness are isolated.
- Option β rejected for Phase 1: requires hand-rolled `NAXFrag`/`NAXTile` matmul wrapper (Sprint 3 microbench showed bare `simdgroup_matrix` rewrites are ~50× harder to make competitive than wrapping MPP). Defer until Option α ships AND a measurement shows residual headroom is worth the implementation cost.
- Option γ rejected outright: no control, no evidence Apple ships NAX-aware conv3d in MPSGraph, conflicts with Sprint C's kernel-level optimization charter.

**Reversibility**: if Phase 1.5 perf sweep shows Option α reaches < 60% of theoretical NAX peak on dominant shapes, R1 revision opens Option β reconsideration as a Phase 1.x+ extension.

## D2. Conv2D: defer entirely

**Choice**: defer Conv2D from Sprint C scope.

**Rationale (§10 of design doc)**:
- SeedVR2 VAE decoder is **0% Conv2D** (per `architecture_map.json` op_type_breakdown — Conv3d_3x3x3 91.94%, Conv3d_1x1x1 7.23%, attention 0.76%, GroupNorm 0.03%, Linear_Attention 0.04%; no Conv2D appears).
- Zero ROI on the target workload → implementing Conv2D for completeness violates scope discipline.
- ConvKey enum (§5.1) explicitly accommodates future `Conv2DDirect` kind, so adding Conv2D later is a single-line enum change + Conv2D-specific source-gen.
- Trigger conditions for future Conv2D mini-sprint documented (§10.3).

**Reversibility**: if a Conv2D workload surfaces (any VSR pipeline with 2D-encoder/2D-decoder, attention-projection conv layers, etc.), open a focused 4-6h mini-sprint to wrap `mpp::tensor_ops::convolution2d` analogously to the Conv3D Primitive.

## D3. Unified cache key (avoid Sprint A's three-maps debt)

**Choice**: single `std::unordered_map<ConvKey, void*, ConvKeyHash>` with `Kind` enum field discriminating kernel type (`Conv3DIm2colKernel`, `Conv3DMatmul`, future `Conv2DDirect`).

**Alternative considered**: separate maps per kernel kind, mirroring Sprint A's organic three-maps evolution (dQ map + dKV map + combined map).

**Rationale (§5 of design doc)**:
- Sprint A's three-maps pattern emerged organically because dQ and dKV were developed as separate primitives and a combined cache was added later for the wrapper. Cache management code touched all three maps for every operation; iterators and locking became error-prone (documented as tech debt in Sprint A review).
- Sprint C starts unified to avoid that organic accumulation.
- Single mutex, single allocation path, easy iteration / introspection / eviction.
- Future kernel kinds added via single-line enum extension.

**Reversibility**: not applicable. The unified design has zero downsides if the enum field is properly used as part of the hash key.

## D4. Weight pre-packing: Option (b) Python-side

**Choice**: weight pre-pack happens at Python module init time (e.g., on the `nn.Conv3d`-like wrapper object); pre-packed weight passed as second `inputs` array on each `eval_gpu` call.

**Alternative considered**: Option (a) — pre-pack at Primitive construction time, stored on the Primitive object.

**Rationale (§4.3 of design doc)**:
- Cleaner Primitive contract (stateless apart from `params_`).
- Explicit data flow at the call site: Python knows when the pre-pack happens.
- One-time cost (transpose) at module load is negligible.
- Mirrors typical PyTorch/MLX module pattern (weights live on module objects).

**Reversibility**: if Phase 1.5 measurement shows the per-call pre-pack overhead is non-trivial, switch to Option (a) by adding a `static thread_local` pre-packed-weight cache on the Primitive.

## D5. Sub-phase 0 microbench as Phase 1.1 precondition

**Choice**: Phase 1.1 begins with a microbench measuring sustained `matmul2d` FP16 TFLOPS on the 24-cell (M, K, N) grid representative of production Conv3D implicit-GEMM shapes.

**Alternative considered**: trust Apple's published 38 TFLOPS peak figure and proceed directly to primitive implementation.

**Rationale (§3 of design doc)**:
- Apple's 38 TFLOPS peak is from balanced-square matmul benchmarks; Conv3D implicit GEMM is heavily M-skewed (M up to 4.5M, K up to 13.8K, N down to 128).
- Without sustained-TFLOPS measurement, Phase 0's 42.6% reduction target is aspirational, not grounded.
- The microbench gates whether to proceed-as-designed (≥ 30 TF), R1-revise (20-30 TF), or pivot (< 20 TF).
- Cost: ~3h wall-clock incl. §4 cooldowns. Cheap relative to the risk of building on a wrong assumption.

**Reversibility**: not applicable — the measurement is informational.

## D6. Initial tile defaults per cluster

**Choice**: per-shape-cluster initial tile recommendations:
- Cluster 1a (N=128): M_tile=16, N_tile=128, exec_sg=8
- Cluster 1b (N=256): M_tile=16, N_tile=64, exec_sg=16
- Cluster 2 (N=512): M_tile=16, N_tile=64, exec_sg=16

Plus autoresearch knobs via env vars (`MFA_CONV3D_CHUNK_M`, `_M_TILE`, `_N_TILE`, `_EXEC_SG`, `_IM2COL_TG_SIZE`) for Phase 1.3 tuning.

**Alternative considered**: single global default, autoresearch finds the winner.

**Rationale (§6 of design doc)**:
- Per-cluster initial defaults reflect V6 NAX learnings about N-axis vs cooperative-tensor row-tile granularity.
- Single global default wastes Phase 1.3 autoresearch time on shapes where the answer is structurally clear.
- Env var override grid is the same autoresearch surface for Phase 1.3.

**Reversibility**: autoresearch may overwrite all of these. Initial defaults are starting points, not commitments.

## D7. Sub-phase ordering: 1.1 → 1.2 → 1.3 → 1.4 → 1.5

**Choice**: microbench+scaffold first (1.1), then im2col+single-chunk (1.2), then multi-chunk+working-set (1.3), then 1×1×1 fast path (1.4), then perf sweep (1.5).

**Alternative considered**: start with 1.4 (1×1×1 — simplest) for an easy first win.

**Rationale (§8 of design doc)**:
- 1.1 establishes the foundation (microbench + Primitive scaffolding + smallest shape end-to-end). All later sub-phases depend on this.
- 1.2 establishes the im2col kernel + single-chunk correctness BEFORE multi-chunk introduces chunking complexity.
- 1.3 adds multi-chunk on top of validated single-chunk.
- 1.4 (1×1×1) is the late add because it requires the Primitive to already be working — and its fast path is a `params_`-based branch off the full path.
- 1.5 perf sweep happens after all shapes pass correctness.

Doing 1.4 first would invert the dependency order: 1×1×1 fast path bypasses the im2col kernel, but the Primitive scaffolding must exist first.

**Reversibility**: not applicable. The dependency graph drives the order.

## D8. Validation: 3 oracles + sentinel

**Choice**: PyTorch CPU FP32 (Oracle 1, ground truth, RMSE bar < 1e-3) + MLX `mx.conv_general` (Oracle 2, cross-check, RMSE bar < 1e-4) + sentinel-fill (Oracle 3, coverage gate, count must equal 0).

**Alternative considered**: PyTorch CPU only.

**Rationale (§7 of design doc)**:
- Three oracles each catch a different class of bug.
- Oracle 1 = pure math reference; Oracle 2 = MLX-internal regression catch; Oracle 3 = coverage / addressing bugs that pass numerical RMSE.
- Sprint A precedent: V6 NAX V34 backward used the same three-oracle pattern (PyTorch CPU FP32 + `mx.vjp` reference + STEEL comparison + sanity asserts).
- Sentinel-fill specifically catches the "im2col missed a row" class of bug, which is subtle and easy to introduce.

**Reversibility**: Oracle 2 may be removed if MLX's `conv_general` proves to have its own precision quirks. Oracle 1 + Oracle 3 are the minimum.

## D9. Risks register: 10 risks, ranked by impact

**Choice**: explicit 10-risk register in §9 of design doc, each with likelihood + mitigation.

**Alternative considered**: high-level "we'll figure things out" handwave.

**Rationale (§9 of design doc)**:
- Sprint A precedent: explicit risk registers caught the cooldown methodology gap (Phase 1.5 v1) before it became a blocker.
- Top 2 risks are HIGH impact (sustained TFLOPS << peak; im2col memory pressure) — both have well-defined mitigations.
- MEDIUM risks (3) and LOW risks (5) round out the surface.

**Reversibility**: risks register is a living document — Phase 1.x may discover new risks or revise likelihood assessments. Updates land via R1 revisions or sub-phase commits.

## D10. R1 revision protocol

**Choice**: R1 revisions land as additional commits on the Phase 1.0 design branch, not as separate documents.

**Alternative considered**: separate `conv-nax-design-r1.md` file for each revision.

**Rationale (§12.4 of design doc)**:
- Sprint A precedent: design doc had R1-R3 revisions in-place.
- Single document is easier to maintain and reference.
- Git history preserves the revision audit trail.

**Reversibility**: not applicable.

---

## Decision summary table

| ID | Decision | Rationale source | Reversibility |
|---|---|---|---|
| D1 | Option α (materialized chunked im2col) | §2.2 design doc | If Phase 1.5 < 60% peak → reopen Option β |
| D2 | Defer Conv2D | §10 design doc, 0% workload weight | Open mini-sprint on workload trigger |
| D3 | Unified ConvKey cache | §5 design doc, Sprint A lesson | N/A (no downside) |
| D4 | Weight pre-pack Option (b) Python-side | §4.3 design doc | Switch to (a) if per-call overhead is real |
| D5 | Sub-phase 0 microbench precondition | §3 design doc | N/A (informational) |
| D6 | Initial tile defaults per cluster | §6 design doc, V6 NAX learnings | Autoresearch may overwrite |
| D7 | Sub-phase ordering 1.1→1.5 | §8 design doc, dependency graph | N/A |
| D8 | 3 oracles + sentinel | §7 design doc, Sprint A precedent | Oracle 2 droppable if MLX precision quirks |
| D9 | 10-risk register | §9 design doc, Sprint A precedent | Living document |
| D10 | R1 in-place commits | §12.4 design doc, Sprint A precedent | N/A |

## Cross-doc consistency check

This decisions.md companion is consistent with:
- `conv-nax-design.md` §1-§12 (every decision cited matches the design doc rationale)
- `survey-report.md` Phase 0 Option F (Phase 1.0 refines to Option α for Conv3D + defer Conv2D)
- `theoretical-bounds.json` + `baseline-summary.json` (§1 ROI numbers anchored to measurements)
- Sprint A design doc + Phase 1.5 R1 lessons (sub-phase pattern, ship/shelve thresholds, §4-protocol cooldowns)

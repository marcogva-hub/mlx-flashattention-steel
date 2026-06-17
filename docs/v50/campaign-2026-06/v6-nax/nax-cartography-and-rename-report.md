> **⚠ CORRECTION (2026-06-17, `compacted-kernel-increment-0-report.md`):** the **"11.4 TFLOPS sparse,
> 4× below SDPA" measurement and the puzzle-resolution** in this report are **RETRACTED** — that
> benchmark used an ASYMMETRIC mask, which on M5/26.6 routes `flash_attention_sparse` to dense Apple
> SDPA (not the matmul2d kernel the source trace described). The REAL sparse kernel (symmetric mask)
> already tracks density and beats SDPA at low density. The V34→V6 rename plan + the per-path
> source/routing map in this report are unaffected; only the sparse-throughput *number* was the
> wrong binary. See the increment-0 report.

# NAX State Cartography + V34→V6 Rename — Phase 1 (read-only) + Phase 2 plan

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `db01461` (v2.56.0 baseline), macOS 26.6, M5 Max. Phase 1 = read-only
cartography (no code change). Phase 2 = the V34→V6 rename (plan + env-var flag below).

## Headline: the V6 "port STEEL→NAX" chantier was a FALSE PREMISE born of the V34/V6 confusion.

The sparse forward is **already NAX (matmul2d cooperative_tensor), routed, swap-done.** It is 4×
below SDPA **not because it's STEEL** (it isn't) but because the **matmul2d/MPP form is slower than
SDPA's raw-`simdgroup_matrix` form** (the V6-dense finding) + the sparse kernel is less-tuned. The
measurement gate **conflated** dense-mfa(STEEL, 11.1) and sparse(NAX-matmul2d, 11.4) as one "STEEL
~11" and called the gap to SDPA "STEEL→NAX headroom" — it is actually **matmul2d→raw-simdgroup +
tuning**, a different and much harder (and uncertain) lever. There is no "port" chantier.

## Phase 1 — NAX state map (per path: MMA form × routed/orphaned, file:line)

| Path | MMA form | routed? | evidence | effective TFLOPS (M5/26.6) |
|---|---|---|---|---|
| dense `backend="mfa"` D=128/64 | **STEEL `simdgroup_matrix`** (V2/V3) | routed (expert; dense default→SDPA) | `MFAttention::eval_gpu` STEEL dispatch; `v6_nax_forward` is a *separate* binding (`bindings.cpp:406`) used only for the V34-backward O,L recompute (`attention.py:5296`) | 11.1 (measured) |
| **sparse / LCSA forward** | **NAX `matmul2d` coop-tensor** (desc 16,32,16) | **routed** | `flash_attention_sparse`→`sparse_attention_nax_with_lse` (`attention.py:2816`)→`mfa_sparse_attention.cpp` `BaseNAXFrag::mma` (**matmul2d**, `:283-312`), called by `stile`/`otile` QK^T/PV (`:548/:604`) | 11.4 (measured) |
| windowed (V3) | STEEL `simdgroup_matrix` | routed (conditional-auto) | `mfa_steel_fwd_v3.cpp` | 10.8 (measured) |
| GNA native | STEEL `simdgroup_matrix` (NAXTile load, III-9) | routed (D=128 3D) | `mfa_gna_fwd.cpp` | small-N overhead-bound (not a throughput regime) |
| paged / TQ decode | gather/dequant + **Apple SDPA** | routed (default) | `tq_decode.py` (IV-D1/D2) | sync-floor-bound (latency, not throughput) |
| conv3d NAX | **NAX `matmul2d`** (MPP) | routed (auto-hook) | `mfa_conv_nax.*` (III-1) | (conv, separate) |
| V6 NAX forward (`MFAV6Forward`/`createV34Source`) | **NAX `matmul2d`** | **semi-orphaned** | reachable only via the `v6_nax_forward` binding (backward O,L recompute) + had a `causal→STEEL` fallback gate (`attention.py:4926`); NOT the main dense forward | not the production forward |
| V34 backward family (dQ/dK/dV/fused + sparse) | **NAX `matmul2d`** | routed (D=64 causal default-on; D=128 opt-in) | `NAAttentionKernel.cpp`; `mfa_v6_nax_primitive.cpp` | (backward) |

**Key:** `metal_simdgroup_matrix` is included in `mfa_sparse_attention.cpp:51` for NAXTile *layout*
helpers, but the actual MMA (`BaseNAXFrag::mma`, `:283`) is **matmul2d** — the line-13 comment
"Phase 1.3 will swap inner GEMMs to matmul2d" is **STALE** (the swap is done).

## The 11.4-vs-SDPA puzzle — RESOLVED: **(c)+(d), NOT (a) or (b)**

- **(a) artifact?** No — 11.4 is effective (density-scaled active-FLOP, plausibility-gated: below
  the 51.8 ceiling; sparse@d=0.5 takes 6.06 ms vs SDPA dense 3.06 ms → ~4× slower per unit work).
- **(b) orphaned-NAX (the 11.4 was actually STEEL)?** No — `flash_attention_sparse` forward
  provably routes to the matmul2d `BaseNAXFrag::mma` (file:lines above). The sparse path IS NAX.
- **(c) form + overhead:** the sparse forward uses **matmul2d/MPP**, which the V6-dense sprint
  measured ~1.3–1.5× slower than SDPA's raw-`simdgroup_matrix` (MPP scheduling overhead Apple's
  path avoids). That accounts for ~1.3–1.5× of the gap.
- **(d) suboptimal tiling/occupancy + mask overhead:** the remaining ~2.5–3× — the sparse kernel's
  threadgroup/occupancy/K-loop is less tuned than Apple's hand-tuned SDPA, plus per-tile mask
  handling. (Decomposition is DEDUCED from the form+structure, not separately micro-measured —
  flagged.)

## Remaining-opportunity verdict (replacing the false premise)

- **There is NO "port STEEL→NAX" chantier for the sparse forward** — it is already NAX (matmul2d).
- The real (much harder, **uncertain**) lever for the sparse path: **switch its MMA from matmul2d/MPP
  → raw `simdgroup_matrix` (SDPA's `nax.h` form)** + tune occupancy — to recover the matmul2d-MPP
  penalty (~1.3–1.5×) and the tuning gap. This is NOT a port; it's an MMA-form rewrite + tuning of a
  kernel that already works. Its ceiling is SDPA-class on the active tiles (no SDPA competition on
  sparse), but the V6-dense finding cautions that matching SDPA's raw-simdgroup tuning is hard.
- dense `backend="mfa"` (STEEL) → routes to SDPA in production anyway (not a user-facing chantier).
- windowed-V3 (STEEL) is the production windowed path; same MMA-form-switch lever applies, same
  uncertainty.
- **The measurement gate's "SCOPED-GO, ~3–4× by porting STEEL→NAX" is RETRACTED**: the premise
  (sparse=STEEL) was wrong (sparse=NAX-matmul2d). The actual lever (matmul2d→raw-simdgroup + tune) is
  harder and its realized gain is unproven. **Re-scoped verdict: the non-dense chantier is an MMA-form
  rewrite + tuning play, NOT a port — gate it on a real raw-simdgroup-sparse prototype vs the current
  matmul2d-sparse before committing (a NEXT-session decision, not now).**

## Phase 2 — V34→V6 rename PLAN (scope + scheme + env-var flag)

**Footprint:** ~1304 C++ + 801 Python + 2942 doc occurrences across 169 files; **29 public env
vars**; emitted MSL `#define` macros (`V34_TQ`, `V34BWDF_BK`, …).

**Naming scheme (avoids collision with the existing `MFAV6Forward`/`v6_nax`):** the V34 kernel is the
**Nax** variant within the V6 generation → rename the V34 token to **`V6Nax`/`V6NAX`/`v6_nax`**:
- C++ symbols: `createV34Source`→`createV6NaxSource`, `useV34`→`useV6Nax`, `force_v34`→`force_v6nax`,
  `compile_v34_backward_pipeline`→`compile_v6nax_backward_pipeline`, `MFAV34Bwd*`→`MFAV6NaxBwd*`,
  `_v34_backward_carveout`→`_v6nax_backward_carveout`.
- MSL macros: `V34_TQ`→`V6NAX_TQ`, `V34BWDF_BK`→`V6NAXBWDF_BK`, … (rename emission AND usage together).
- Stale comment fix: `mfa_sparse_attention.cpp:13` (swap done).

**Public env vars (29) — FLAG to Marco, do NOT silently rename.** Options (Marco's call):
1. **Alias-with-deprecation (preferred, MFA_FORCE_NATIVE_BWD precedent):** add `MFA_V6_*`/`MFA_V6NAX_*`
   names, keep the 29 `*V34*` names as deprecated aliases emitting a `DeprecationWarning`, remove next
   minor. Non-breaking; ~29 alias shims.
2. **Break in the next minor (2.57.0):** rename outright, document in CHANGELOG migration. Breaking
   for any user/script setting an `MFA_*V34*` env var (these are expert/internal-tuning vars).
   *(Recommendation: option 1 for the user-facing `MFA_ENABLE_V34_BACKWARD` / `MFA_V34_BWD_SPARSE_NATIVE`
   / `MFA_V6_USE_V34`; option 2 acceptable for the deep tuning knobs `MFA_V34BWD*_BK/BQ/WM` which are
   research-only.)*

**Provenance:** a new `NAMING.md` glossary records "V34 was the internal generator name for the V6
NAX kernel; unified to V6/V6Nax in v2.57.0; the V1–V5 lineage is STEEL, V6 is NAX." Historical
sprint records (devnotes/, docs/v50) get the token rename per "toute la documentation" with NAMING.md
preserving meaning.

**Execution (revertible commits, suite green after each):** scripted token rename per file-group
(C++ → rebuild + 1827 tests; Python → tests; docs → grep-clean), `grep V34` clean except NAMING.md +
any alias definitions. This is a ~5000-occurrence change touching compiled MSL macros + the public
env surface — **the env-var disposition (option 1 vs 2) is the gating decision and is flagged to
Marco before the mechanical rename executes** (per the prompt's "flag public-API env-var renames").

## Disposition

Phase 1 cartography: **COMPLETE** — the false premise is corrected (sparse is already NAX-matmul2d;
no port chantier; the real lever is an uncertain MMA-form rewrite + tuning), the puzzle resolved
(c+d), the per-path map + rename inventory delivered. Phase 2 rename: **planned + scoped**, gated on
Marco's env-var disposition (option 1 alias-with-deprecation vs option 2 break-next-minor) — the one
public-API fork the prompt mandates flagging. No code changed this session (cartography is read-only;
the rename executes after the env-var call). No orphans.

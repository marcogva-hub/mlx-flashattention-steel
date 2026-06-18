# Audit Phase F — M5+ Orchestration Rebuild (the routing fix)

**Date:** 2026-06-18 · **Executor:** Claude Opus 4.8 (1M)
**Provenance:** master, M5 Max, macOS 26.6, mlx 0.31.2. ROUTING + mask-block-size
only — no kernel/C++ change, no tag/publish (Phase G ships). Discipline: three-axis
(fp32 correctness / which-binary / win-confirm), lesson #11 (independent fp32 oracle,
same mask the kernel sees), lesson #14 (which-binary at runtime dispatch), §AA.5.x
(enumerate every dispatch site consuming the changed input), keep-all-paths.

## Change 1 — sparse → V2 always; retire the 2^31 work-product threshold
`mlx_mfa/lcsa_nax.py::decide_auto_version` now routes by **V2-capability** (head_dim
D∈{64,128} → v2), not `qL*kL*D ≥ 2^31`. Phase E measured V1-scalar is never fastest
(V2 19–59× faster). C++ `sparse_attention_forward` falls v2→v1 internally when V2 is
ineligible (causal / block_tile≠32 / bf16), so V1 stays a correct fallback — never the
default for V2-capable shapes. Commit `874f583`.

Three-axis (default path, no env):
| shape | fp32 max_err | which-binary | win vs old V1 default |
|---|---|---|---|
| D=64 N4096 d0.25 | 5.0e-6 | decide='v2', timing≈v2-forced≠v1-forced | **8.8×** (E: ~9×) |
| D=128 N2048 d0.25 | 7.6e-6 | decide='v2', timing≈v2-forced | **19.9×** (E: 19.5×) |

## Change 2 — D=128 makers → symmetric 32×32 + NAX density gate
`masks.py::_bq_bk(128)=(32,32)` (was 32,16) re-blocks all 18 makers + the top-k
expander (same `_bq_bk`) at once; `make_causal_block_mask` + `make_sliding_window_mask`
(attention.py) aligned. Generation-time 32×32 — NOT an OR-merge (the causal/window/
strided formulas are tile-geometry-based, exact; coarser tiles can only over-include
→ safe). Density gate: symmetric masks with density ≥ `_nax_sparse_density_ceiling()`
(0.78, env `MFA_NAX_SPARSE_DENSITY_CEILING`) fall to the SDPA-bias fallback (NAX loses
near-dense); below → NAX. Commit `aa4d9a9`.

Three-axis (default auto path, B1 H8 N4096 D128):
| maker mask | fp32 max_err | route | win vs old SDPA-fallback |
|---|---|---|---|
| sink-window d0.14 | 1.4e-5 | NAX | 1.92× |
| strided stride512>BK d0.14 | 6.7e-6 | NAX | 2.06× |
| LCSA top-k d0.06 | 1.0e-5 | NAX | 2.54× |
| dense-random d0.85 | 3.1e-6 | SDPA (gate) | 0.89× (correctly declines NAX) |

Caveats (B1-deferred) verified faithful under 32×32: strided-stride>BK (over-inclusion,
6.7e-6) + LCSA-top-k (1.0e-5).

## §AA.5.x multi-gate audit — every D=128 mask-granularity consumer
1. makers → emit `_bq_bk` (32×32). 2. top-k expander → reads `_bq_bk` (consistent).
3. NAX auto-route → derives bt from shape. 4. **STEEL validator** → now accepts EITHER
mask geometry (`_bq_bk`) OR kernel geometry (`_steel_block_config`), preserving legacy/
custom asymmetric masks, then **normalizes to kernel geometry by EXACT tile-split**
(repeat, not OR-merge). 5. M5 SDPA-fallback expander → already self-derives tile from
shape. 6. non-M5 STEEL kernel → gets the normalized geometry (M1–M4 correctness kept).
(Initial validator edit hit a `NameError` — `_bq_bk` was only locally imported; fixed
with a local import mirroring the existing pattern.)

## Locks UPDATED (not bypassed)
- `test_decide_auto_version_shape_aware`: Axis-2 cells flip to v2 + new D∉{64,128}→v1.
- `test_dispatch_map_lock` / `test_fingerprint_discipline`: D=128 maker→NAX positively
  asserted (+2 cells: dense-via-gate→SDPA, large-maker→NAX); residual SDPA edges
  (asymmetric/small/dense) reframed; docstrings corrected.
- maker-shape asserts (segment/dilated/cross-stream/gna×2/causal-block/sliding) switched
  `_steel_block_config`→`_bq_bk`; gna-stride grid 8×8→16×16 (64-token grid is dense at
  32-tiling); M5+ fast-fallback correctness test bit-exact→NAX-grade (now routes NAX).

## Gotchas — status
1 (D=128 maker→silent SDPA) **FIXED**; 2 (D=64→slow V1) **FIXED**; 3 (sparse backward
dense) unchanged (declined-on-perf, Pattern #6). The `(long)p->NK` compiler bug was NOT
touched (pure routing + mask-block-size, as the cartography recommended).

## Doc-only items folded in
README: `backend="mfa"` legacy-on-M5 (SDPA 3–4× faster), sage int8 not auto-routed
(4.7× slower), V5 compiled-but-unrouted; the 4096-byte mask-size threshold documented in
the routing map + dispatch-map.md.

## Disposition
Full suite **1900 passed, 2 skipped, 0 failed**; 0 orphan processes. ROUTING + mask-
block-size only; keep-all-paths. Not tagged. **Phase G (ship-readiness gate + first
release off the audited tree) is next.**

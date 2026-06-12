# Sprint II-9 — Fused im2col conv3d (2026-06-12)

**Status**: COMPLETE — **PROMOTED, via the Apple primitive, no hand-fused
kernel needed**.
**Headline**: conv3d now routes eligible shapes through native MPP
`convolution2d` (kT-accumulated 2D convs, zero materialized im2col):
**2.37x at T8 64x64 C128 (the K=3456 headline cell), 2.50x at T16,
1.71x at C256, 2.05x at C64** — 3-session medians on the production
surface with the weight repack included.  The II-4 ceiling was 2.6x;
the headline cells land at 91–96% of it.  Suite 1391 passed x2 with
the path default-on.

## Phase R.0 — MPP convolution2d evaluation (Pattern #6 first): WINS

### Tiling semantics resolved (the II-5 open question)

Empirical 8-variant sweep + authoritative confirmation from
production code found by web search (liuliu/ccv `NAConv3DKernel.cpp`,
branch unstable — Draw Things ships conv3d on this exact primitive;
plus liuliu/example_matmul_metal4 `AGENTS.md` validated semantics):

- **descriptor destination dims = the PER-THREADGROUP tile**
  (channels full, spatial tiled); source dims = whole frame (ccv uses
  the halo'd input tile + clamp; our whole-frame-source form verified
  correct on macOS across all tiles).
- **destination tensor handle = `.slice()` view at the tile origin.**
- **`set_offsets(int2(x0, y0))` = source-window position** ((x, y)
  order); not a dest offset.
- **Cooperative destination required** (zero-init -> run x kT -> store):
  direct dest-tensor writes pass on macOS but are documented incorrect
  on M5 iPad (example_matmul_metal4).  Adopted.
- WWDC has NO public documentation of this (session 262 shows only
  matmul2d; the MPP docs page 404s).  The ccv kernel is the only
  production reference.

### conv3d decomposition validated

conv3d(k=3x3x3, same-pad) = 3 accumulated conv2d(3x3) taps via
`multiply_accumulate` into one cooperative dest, temporal zero-pad by
skipping out-of-range frames.  Bit-level prototype correctness vs CPU
fp32 reference: max_err 0.0018 (fp16 floor).

## Integration (csrc/mfa_conv_nax.cpp)

- `conv3d_mpp_source()` JIT generator + `conv3d_mpp_dispatch()` via the
  file's existing `mlx::core::fast::metal_kernel` + MPP-header pattern.
- **FLOAT cooperative destination** (fp32 accumulation across kh/kw/C
  and the kt taps): the half-dest variant failed the repo's 1e-5-rel
  parity bars (rel ~2.6e-4).  Elementwise (half) store (float coop
  dest has no direct store() into a half tensor).
- Weights repacked (C_out,3,3,3,C_in) -> [3][3][3][C_in][C_out] via a
  lazy `transpose` + the kernel's ensure_row_contiguous copy — cost
  included in all bench numbers.
- **Occupancy-aware tile pick**: 16x16 only when it yields >= 64
  threadgroups (the T8/32x32/C256 cell measured 0.85x underoccupied at
  32 TGs; with the heuristic -> 8x8 -> 1.83x), else 8x8.

### Eligibility envelope (default-ON, opt-out `MFA_DISABLE_CONV3D_MPP=1`)

fp16, B==1, k=3x3x3, stride (1,1,1), dilation (1,1,1), symmetric pad
(1,1,1) (the primitive's native centered convention), H,W % 8 == 0,
**C_in,C_out % 16 == 0 AND >= 32** (C=16 measured WRONG through the
primitive — err 0.17–0.31 vs legacy while C>=32 is exact-0; undocumented
constraint, gated).  Everything else falls through to the existing
materialized-im2col path unchanged (verified: all fallback edge cases
diff 0.0 vs legacy).

## Phase R.2 — Correctness

- Eligible cells: fp16-floor parity vs the legacy path (0.0039–0.0078
  max-abs across T4..T16, 16x16..64x64, C32..C256).
- Edge sweep: odd-H, stride-2, pad-0, 1x1x1 fast path, C%16!=0, B=2 —
  all take the legacy path (diff exactly 0.0).
- **KD-7 bf16 gate explicitly verified**: the MPP branch gates
  `dtype == float16`, so bf16 never enters it; direct-C++ bf16 calls
  crash in upstream MLX's im2col helper exactly as KD-7 documents
  (pre-existing, unchanged).  NOTE: the MPP path bypasses the buggy
  upstream helper entirely AND the impl header lists bf16 cooperative
  conv variants (`dv_bf_dv_bf_f32`) — a future S-effort probe could
  lift KD-7 for the eligible envelope (declared != implemented; needs
  the probe per the II-2R lesson).  Ledger item.
- M1: no new exposure — conv3d_nax_forward was already MPP-matmul2d-
  dependent (M5-only de facto); the Python hook gates on is_m5_plus.

## Phase R.3 — Bench (3 sessions, medians, production surface)

| cell | legacy ms | MPP ms | speedup | vs 2.6x ceiling |
|---|--:|--:|--:|--:|
| T8 64x64 C128 (K=3456) | 2.09–2.15 | 0.90 | **2.30–2.38x** | 91% |
| T16 64x64 C128 | 3.85–3.86 | 1.54 | **2.49–2.51x** | 96% |
| T8 32x32 C256 | 1.48–1.51 | 0.86–0.88 | **1.71x** | — |
| T4 64x64 C64 | 0.66–0.68 | 0.32–0.33 | **2.05x** | — |

## Phase R.4 — Decision: PROMOTED

Three-axis: (1) fp16-floor output parity across the shape grid + exact
fallback parity on every edge; (2) path-entered proof via the timing
differential + opt-out restoring legacy; (3) edges preserved (padding/
stride/dtype/batch fallbacks, KD-7 intact).  Default-on within the
envelope; suite 1391 passed x2.

Phase R.1 (hand-fused kernel) **not needed** — the Apple primitive won
outright, which is the preferred outcome (no maintenance, follows OS
improvements).  The `MFA_CONV3D_FUSED_IM2COL` hand-rolled variant is
moot; the Marco-gated fused-im2col XL ledger item is RETIRED, superseded
by this promotion.

## Skill invocations (§AA.2)

| Skill | When | Result |
|---|---|---|
| `/mlx-mfa-apple-primitives-coverage` (protocol) | R.0 | FULL_INVERSION class outcome: Apple primitive replaces the prescribed hand-built kernel |
| `/metal-kernel-dev` (protocol) | integration | tile/occupancy heuristic, fp32 accumulator, TG sizing 4 simdgroups |
| `/mlx-mfa-bench-methodology` (protocol) | R.3 | 3 sessions x 20-iter medians, warmed, repack cost included |

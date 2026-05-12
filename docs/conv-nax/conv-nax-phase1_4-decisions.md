# Phase 1.4 — Decisions Companion

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_4`

Decisions D27-D29. Continues numbering from Phase 1.3 (D23-D26). D26
was a forward-declared decision for Phase 1.4; this Phase ratifies +
extends it.

---

## D27 — 1×1×1 detection: strict `K_T=K_H=K_W=1` AND no spatial extras

**Context.** The "1×1×1 fast path" prompt §E.1 specifies skipping im2col
when the conv kernel is purely pointwise. But "1×1×1" is ambiguous:
- Is stride=2 a 1×1×1? (No: stride changes which input positions
  contribute to each output.)
- Is padding=1 a 1×1×1? (No: padding introduces zero-fill rows that
  break the M = B×T×H×W flat-reshape assumption.)

**Decision.** Fast path activates ONLY when ALL of these hold:
```
K_T == 1 AND K_H == 1 AND K_W == 1
AND pT_l == 0 AND pT_r == 0
AND pH_l == 0 AND pH_r == 0
AND pW_l == 0 AND pW_r == 0
AND sT == 1 AND sH == 1 AND sW == 1
```

**Rationale.**
- Stride > 1 breaks `M = B*T*H*W` (output M would differ from input M).
- Padding > 0 makes output T_out > input T (etc.), so a reshape doesn't
  preserve the per-position correspondence.
- Dilation is irrelevant when K_T=K_H=K_W=1 (no kernel positions to
  dilate), so we don't gate on it.

Edge case: SeedVR2 VAE's 1×1×1 layers do typically use stride=1
padding=0 (the natural pointwise pattern). Cases where any of the
constraints fail fall through to the general path and still work
correctly (just don't get the perf win).

**Validation.** All Phase 1.4 tests use the exact strict-1×1×1 config.
`test_kt1_routing` (Phase 1.2) covers K_T=1 with K_H=K_W=3 — confirms
that doesn't accidentally hit the 1×1×1 fast path (it goes through
general im2col).

---

## D28 — Reshape-only (no copy) reliance on channels-last layout

**Context.** The fast path's core operation is:
```python
x_flat = x.reshape(M, C_in)  # x is (B, T, H, W, C_in)
```

**Decision.** Rely on the channels-last layout invariant established in
Phase 1.1 D16. With `(B, T, H, W, C_in)` row-major:
- Stride is `(T*H*W*C_in, H*W*C_in, W*C_in, C_in, 1)`.
- Flattening (B, T, H, W) → M means M's stride becomes
  `B*T*H*W stride / M = C_in`.
- This matches contiguous (M, C_in) — so MLX's `.reshape` is metadata-only.

**Verification.** The 4 1×1×1 tests pass, including the bit-exact
fast-vs-general identity test. If reshape were copying, the fast path
would have additional time + memory; the 15% speedup observation is
consistent with no-copy.

**Risk.** If a caller passes a non-contiguous channels-last input
(e.g., a sliced view), the reshape may force a copy. This is
acceptable — slow but correct. The sanity assert's `dtype` + rank
checks don't enforce contiguity; relying on Python `.reshape` to
materialize if needed.

---

## D29 — Env-var escape hatch `MFA_CONV_NAX_NO_FAST_PATH`

**Context.** The 1×1×1 fast path is an optimization. Tests need to
verify that the optimization doesn't introduce numerical drift vs the
general path.

**Decision.** Implement `MFA_CONV_NAX_NO_FAST_PATH=1` env var as an
opt-out. When set, even is_pointwise shapes route through the general
path (with im2col). Default unset = fast path active.

**Rationale.**
- Tests can run both paths back-to-back without API changes.
- Production users can disable if they hit any unexpected behavior
  (rapid escape hatch beats requiring a code update).
- The env var is read at API entry per call (not cached), so
  toggling between tests doesn't require process restart.

**Test coverage.**
- `test_conv3d_nax_1x1x1_fast_equals_general` — bit-exact rmse=0
  between paths.
- `test_conv3d_nax_1x1x1_faster_than_general_path` — wall-clock
  speedup measurable.

**Rejected.**
- A `disable_fast_path: bool` kwarg in conv3d_nax_forward — too
  visible in the API surface for what's essentially a debug flag.
- A module-level mutable flag — process-global state is harder to
  test in parallel.

---

## Forward-declared in Phase 1.3 (D26) — ratified by Phase 1.4

D26 in `conv-nax-phase1_3-decisions.md` predicted:
- "Use mx.reshape to flatten the input's (B, T, H, W) dims into a
  single M-axis" — ✓ ratified by D28.
- "K = C_in = 512 (for SeedVR2 1×1×1 layers) is 27× smaller than 3×3×3.
  Most 1×1×1 cases will be single-chunk." — ✓ verified in this Phase's
  Test `test_conv3d_nax_1x1x1_faster_than_general_path` which uses a
  shape that fits single-chunk.
- "No new kernel needed." — ✓ confirmed; reused `_matmul2d_source()`
  with a separate cache key for tracking.

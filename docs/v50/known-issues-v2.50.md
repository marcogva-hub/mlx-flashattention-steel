# mlx-mfa v2.50 known issues

**Last reviewed**: 2026-05-14 (Prompt 5c Section C verification)
**Test baseline**: 1231 passed, 2 xfailed, 32 xpassed, 0 unexpected failures.

## Active xfails (acceptable scope for v2.50)

### B.5-D128 — STEEL native backward zeroed-blocks at D=128 N≥2048

**Tests**:
- `tests/test_attention.py::TestNativeBackwardRouting::test_target_shapes_native_backward_matches_sdpa_gradients[128-2048]`
- `tests/test_attention.py::TestNativeBackwardRouting::test_target_shapes_native_backward_matches_sdpa_gradients[128-4096]`

**Failure mode** (empirical, captured via `--runxfail`):
- max_diff = 0.41 vs `mx.vjp(SDPA)` baseline
- Output zeroed for query rows ≥ 1024 (16 × BQ tile boundary)
- Pattern: `out[0, 0, 1024:, :] == 0` exactly while `out[0, 0, :1024, :]` matches reference

**Affected code path**: `MFA_FORCE_NATIVE_BWD=1` env var routes through STEEL
backward kernels (`MFASteelBwdDQ` + `MFASteelBwdDKV` in
`csrc/mfa_steel_bwd.cpp`).  D=64 path works correctly; D=128 path
appears to have a tile-loop termination bug at row index 1024.

**Production status**: NOT affected.  Production AUTO path:
- D=128 backward without env: SDPA-vjp (correct, ~2× slower than V6NAX but
  numerically exact)
- D=128 backward with `MFA_ENABLE_V6_BACKWARD=1` (Prompt 5b Section D
  broadening): **V6NAX NAX-direct split kernels** (correct gradients,
  RMSE ~2e-5 vs SDPA-vjp; this IS the production path post-Section D).
- D=128 backward with `MFA_FORCE_NATIVE_BWD=1` (the test path):
  STEEL backward kernel buggy at row≥1024 — **this is the legacy
  research path being preserved for compatibility testing only**.

**Scope decision (Section C verdict (β))**: preserve xfail with this
accurate rationale.  STEEL backward kernel debugging is post-v2.50
work because:
1. V6NAX backward D=128 (the production path) is correct and shipped.
2. STEEL backward is the legacy path; deprecation is implicit (V6NAX
   carve-out broadened to cover its scope).
3. Fixing the STEEL tile-loop bug requires kernel-level investigation
   in `csrc/mfa_steel_bwd.cpp` (1295 LOC) that would not benefit any
   production user (everyone should use V6NAX via env-var opt-in or
   default SDPA-vjp).

**Future direction**: post-v2.50, STEEL backward path can either be
deprecated (recommended) or fixed via dedicated investigation
session.  No urgency given V6NAX is production-active.

## Notes

- 32 xpassed tests in the suite are **xfail decorators that no longer
  apply** (test now passes despite being marked xfail) — these are
  candidates for unmarking but are not blocking for v2.50 ship.
- 0 unexpected failures across 1231 tests confirms regression-free
  Master post-Prompt 5b accumulation.

## Cross-references

- `docs/v50/sprint-5b-section-d-dispatch-audit.md` (Section D broadening
  audit — clarified that B.5 xfails are STEEL bugs, not V6NAX)
- `tests/test_attention.py:10923-10945` (xfail decorator with full
  accurate rationale)
- `docs/HARDWARE_SUPPORT.md` Backward attention path coverage table
  (notes that STEEL backward D=128 is legacy; V6NAX is production)

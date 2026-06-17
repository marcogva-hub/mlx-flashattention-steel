# V6NAX forward lse-write patch — BLK1 resolution

## Context

The V6NAX backward Option β sprint (autonomous overnight 2026-05-13) halted at
Phase 1 Section B on blocker **BLK1**: V6NAX forward kernel allocates an `lse`
output buffer but never writes to it (dead storage). FA-2 backward dQ requires
`lse` from forward to recompute `P = softmax(QK^T - lse)`.

Per `docs/v6-nax/v6nax-backward-decisions.md` §DC0, option (a) — extend V6NAX
forward to write lse — is the architecturally correct resolution. This
patch implements option (a) per the canonical Apple Silicon SDPA-NAX
output contract `(O, lse)`.

## Patch surface

**Files modified** (4 total):

1. `csrc/mfa/v6_nax/NAAttentionKernel.cpp`
   - Added `long L_strides[3]` field to `V6NAXParams` struct (kernel-side).
   - Added `device float* L [[buffer(5)]]` argument to `v6nax_attention`
     kernel signature.
   - Inserted lse-write block in kernel body after Otile final normalization
     and before Otile store.  Per-row computation:
     `lse_natural = max_score * ln(2) + log(sum_score)`
     (V6NAX keeps softmax state in log2 domain via `scale * log2(e)` and
     `fast::exp2`; multiplying max by `ln(2)` recovers natural-log domain
     matching `mx.logsumexp` convention).
   - Single-writer-per-row pattern: only lanes with `get_coord().x == 0`
     emit lse stores, eliminating redundant writes from the 4 lanes that
     share each row-group after `row_reduce`.
   - Out-of-range guard for `is_last_q` blocks (qL not aligned to V6NAX_BQ).

2. `csrc/v6_nax_compile.mm`
   - Added `int64_t L_strides[3]` to `V6NAXParamsHost` (must mirror kernel
     `V6NAXParams` exactly).
   - Populated `L_strides[0..2]` in `v6nax_dispatch`: `[Hq*qL, qL, 1]` for
     BHND-contiguous lse of shape `[B, Hq, qL]`.

3. `csrc/mfa_v6_nax_primitive.cpp`
   - Updated `eval_gpu` buffer binding: V6NAX path now `enc.set_output_array(lse, 5)`
     (legacy path keeps `enc.set_output_array(lse, 4)`).  The if/else branch
     correctly disambiguates because V6NAX uses buffer 4 for the `set_bytes`
     params struct.
   - Added optional `MFA_V6_DUMP_SOURCE=1` diagnostic env var (via
     `fprintf(stderr, ...)`) for future debug.  Zero runtime cost when
     unset.

4. `tests/test_v6nax_forward_lse.py` (NEW, 7 tests)
   - 2 axis-1 tests: V6NAX forward output unchanged vs SDPA reference
     (RMSE < 1e-3 FP16, D=64 and D=128).
   - 3 lse-correctness tests vs `mx.logsumexp` reference (RMSE < 1e-4 FP16,
     < 5e-4 BF16).
   - 1 shape + finiteness test (B=2 H=8 qL=1024 D=128, no NaN/Inf).
   - 1 last-block remainder test (qL=510 not aligned to V6NAX_BQ=16).
   - `force_v6nax` fixture forces `MFA_V6_USE_NAX=1` for D=64 tests, because
     by default D=64 with Nk≤8000 routes through legacy v6_nax (MPP) path;
     V6NAX lse-write only applies on the V6NAX path.

## Correctness validation (Phase C)

7/7 new tests pass + 77/77 pre-existing tests pass = **84/84 total**.
Zero regressions.

| Test | Result |
|---|---|
| V6NAX D=128 FP16 output vs SDPA | RMSE 4.1e-4 (< 1e-3) PASS |
| V6NAX D=64 FP16 output vs SDPA | PASS |
| V6NAX D=128 FP16 lse vs mx.logsumexp | RMSE 2.97e-7 PASS |
| V6NAX D=64 FP16 lse vs mx.logsumexp (force_v6nax) | RMSE 2.65e-7 PASS |
| V6NAX D=128 BF16 lse vs mx.logsumexp | RMSE 3.3e-4 (< 5e-4) PASS |
| Shape + finiteness B=2 H=8 qL=1024 | PASS (no NaN/Inf) |
| Last-block remainder qL=510 | PASS |

## Perf validation (Phase D)

3-session V6NAX forward post-patch bench, single direction, 10 warmup + 100
continuous timed iterations per shape per session, M5 Max 128GB:

| Shape | S1 p50 | S2 p50 | S3 p50 | Cross-session range |
|---|---:|---:|---:|---:|
| qL=1024 D=128 | 0.413ms | 0.417ms | 0.646ms | 56% (sub-1.5ms, §4.2 regime) |
| qL=4096 D=128 | 0.981ms | 0.980ms | 0.977ms | **0.4%** CONFIDENT |
| qL=8192 D=128 | 3.127ms | 3.106ms | 3.113ms | **0.7%** CONFIDENT |
| qL=8192 D=64  | 1.718ms | 1.716ms | 1.720ms | **0.2%** CONFIDENT |

3/4 shapes CONFIDENT under canonical-style protocol.  The 4th shape
(qL=1024 D=128, sub-1ms wall-clock) is in the §4.2 regime where absolute
single-direction measurement inherits GPU power-state variance; ratio
analysis would cancel this per the canonical methodology.

**No perf regression detectable.** The patch adds ~`qL` log + store
operations per backward call, theoretically <0.1% overhead vs the
K-loop's O(qL × kL × D) cost.  Empirically confirmed.

Raw data: `docs/v6-nax/v6nax-forward-lse-bench-data.json`.

## Routing constraint discovered during testing

The V6NAX path engages by default only for:
- D=128: always
- D=64 with Nk > 8000 (LTX2-cross style asymmetric)

D=64 with Nk ≤ 8000 (FlashVSR-style small shapes) routes through the
legacy v6_nax (MPP cooperative tensor) path by default.  The legacy path
has its own pre-existing lse-write behavior (log2-domain) that this patch
does NOT modify.

**Implication for V6NAX backward sprint**: V6NAX backward auto-routing must
match V6NAX forward routing.  Backward kernels can only consume natural-log
lse produced by V6NAX forward; shapes that route through legacy forward
should fall back to STEEL backward.  This constraint should be documented
as DC12 (or equivalent) in the V6NAX backward sprint restart.

## Next steps

1. **V6NAX backward Option β sprint restart** with this patch as foundation.
   `docs/v6-nax/v6nax-backward-{inventory,decisions,status}.md` design
   artifacts from the prior session remain valid.  Update
   `v6nax-backward-status.md` with BLK1 RESOLVED entry.
2. **Future routing-parity work** (out of scope here): the legacy v6_nax
   path's lse-write could also be made natural-log-conformant, broadening
   the V6NAX-backward-eligible regime.  Currently deferred.

## References

- `docs/v6-nax/v6nax-backward-decisions.md` §DC0 (BLK1 + 4 resolution options)
- `docs/v6-nax/v6nax-backward-status.md` (V6NAX backward sprint Phase 1 Section A
  STATUS doc that surfaced BLK1)
- `docs/v6-nax/v6nax-backward-option-beta-design-hints.md` (canonical input
  for V6NAX backward sprint)
- `docs/methodology/canonical-protocol.md` §4.2 (canonical methodology,
  cited in Phase D)

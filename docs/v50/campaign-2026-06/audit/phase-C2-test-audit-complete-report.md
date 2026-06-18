# Audit Phase C2 — Complete the Test Audit (expert-binary subset, per-test fingerprinted)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `fe1e732`, M5 Max, macOS 26.6, mlx 0.31.2. **TEST-ONLY.** Closes Phase C on its
own fingerprint standard (C did the sparse zone per-test + classified the bulk by group-pattern; C2
per-test-fingerprints the expert subset — the group-reasoning the audit exists to distrust).

## (1) The enumerated expert subset + auto/SDPA spot-check
Expert-binary-claiming markers across the suite: `backend="mfa"` (13 files), STEEL-variant/split-K/
dsplit/flash_decode (13), native backward (24), conv-NAX (15), paged/decode (7), sage/topk (7) — a
~52-file union (many ARE the B1–B4/C locks, already fingerprint-asserted).
**Auto/SDPA majority spot-check:** 3/3 dense `flash_attention(backend="auto")` cells byteΔ=0.0 vs
`mx.fast.sdpa` → the group-pattern holds (the default IS SDPA, runtime-verified); the ~1700 accepted
as low-risk.

## (2) Per-test fingerprint of the expert subset — NO new wrong-binary
Key structural finding: expert tests engage the kernel in three safe ways —
- **direct `_ext.*` call** (correct-by-construction, no dispatch → no fallback): `test_v34_backward_dq`
  (`_ext.v6_nax_backward_query`), `test_v34_backward_kv`, `test_v39_fused_dkdv`,
  `test_v50_sprint_5d_sparse_backward_native` (9 `_ext` calls). The kernel named IS the kernel run.
- **forces the binary**: `backend="mfa"` (→STEEL, byteΔ=1.9e-6 vs SDPA), `MFA_ENABLE_V6_BACKWARD=1`
  (`test_v50_sprint_5b_d128_backward`: D=128 dense backward → **all native**, dQ/dK/dV byteΔ
  6e-8/6e-8/3.8e-6 — verified; completes a B3 map row).
- **dispatch-aware** (knowingly tests the fallback): `test_iii5_conv_small_channel_accuracy` (sweeps
  C_in across the MPP envelope, expects small-C → legacy fallback), `test_iii9_gna_v5_direct_v_clamp`
  (docstring: "SDPA for V5" under the trigger).

The four hunts:
| Hunt | Result |
|---|---|
| variant-blind (V3/V4/V5/split-K) | V5 ineligible→V2/SDPA is **documented in the test** (dispatch-aware); not a deceptive instance |
| backward-mix (native vs SDPA-vjp) | D=64 default-on → native (3.6e-7); D=128 opt-in → native (verified); sparse-native → direct `_ext`. All CORRECT |
| conv-fallback | small-C tests **intend** the fallback (coverage test, not a NAX claim) |
| decode-fallback | decode → SDPA-gather is what the test validates (B4); paged correct |

Plus sage→int8 (2.9e-3), topk→own (2.1e-2), backend=mfa→STEEL (1.9e-6) — all byteΔ>0, real distinct
kernels. **No NEW green-on-wrong-binary instances** beyond Phase C's 5 sparse-forward ones.

## (3) Fixes + drift-catch
Extended `tests/test_fingerprint_discipline.py` with `TestExpertPathsRunClaimedBinary` (5 cells:
backend=mfa→STEEL, D=64-bwd→native, D=128-bwd-optin→native, sage→int8, topk→own) — each asserts
byteΔ>0, so a drift to SDPA flips it to 0 and FAILS. The expert paths now carry the binary fingerprint
durably (8 cells total in the discipline module).

## (4) Real bug uncovered? NO
Every expert path, once fingerprinted, runs its claimed kernel and is correct (B1–B4 verified the math;
C2 verified the binary). No expert kernel "passed only because no test ran it." No Phase-F xfail from C2.

## (5) Phase C COMPLETE — ledger
Green-on-wrong-binary total: **5 instances**, ALL in the high-level `flash_attention_sparse` D=128
path (C), where the silent SDPA fallback is invisible AND the reference is SDPA. The **expert paths are
sound** because they bypass dispatch (`_ext`), force the binary, or are dispatch-aware. Coverage matrix
(Phase C) holds; every expert path's tests now accounted for per-test. Total which-binary locks:
`test_fingerprint_discipline.py` (8) + the 42 B1–B4 cells + `test_dispatch_map_lock.py` (11).

## Disposition
The expert paths' pre-existing tests WERE exercising their kernels (now fingerprinted, not
group-inferred) — the wrong-binary sin was confined to the high-level sparse D=128 API. Phase C is
**COMPLETE** on the audit's own standard. Suite green. No orphans. Not tagged. **Phase D
(documentation rebuild from the now-fully-verified A/B/C facts + publication cleanup) is next.**

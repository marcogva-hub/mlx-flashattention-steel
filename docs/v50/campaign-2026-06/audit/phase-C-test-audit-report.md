# Audit Phase C — Test-Correctness Audit (does each test exercise the binary it CLAIMS?)

**Date:** 2026-06-17 · **Executor:** Claude Opus 4.8 High
**Provenance:** HEAD `315dd67`, M5 Max, macOS 26.6, mlx 0.31.2. Oracle: the Phase-A dispatch map +
B1–B4 specs. **TEST-ONLY** (docstring relabels + new fingerprint tests; no kernel/routing/threshold/
bug change). Suite: 1887 → 1890 tests.

## Classification COUNTS

Audit depth (honest): the **sparse-forward zone** (where the campaign's original sin lives) was
fingerprinted per-test on M5; the ~1700 dense/backward/misc tests were classified by GROUP pattern
against the dispatch map (not per-test). The 42 B1–B4 lock cells are already fingerprint-asserted
correct-binary.

| Category | Count | Basis |
|---|---|---|
| **CORRECT-BINARY** | bulk (~1700) + 42 B1–B4 locks | dense `backend="auto"` tests exercise SDPA (what M5 runs); `backend="mfa"` tests exercise STEEL; D=64 sparse exercises the real V1-scalar kernel (byteΔ=3.8e-6) |
| **WRONG-BINARY / VACUOUS** | **5 confirmed instances** (D=128/256 sparse-forward) | fingerprinted: run SDPA, validate vs SDPA → byteΔ=0.0 (SDPA-vs-SDPA, can't fail) |
| **SKIP-MASKED** | the new lock modules skip on non-M5 | legitimate (M5-specific routes); no silent M5 coverage drop |
| **COVERAGE-GAP** | D=128-asymmetric STEEL sparse on **non-M5** | unreachable/untestable on M5 (the `(long)p->NK`-disabled path → SDPA); real D=128 sparse now covered by the symmetric B1 lock |

## THE green-on-wrong-binary instances (the original sin, enumerated with evidence)

All in `tests/test_attention.py::TestSparseAttentionKernel`, fingerprinted on M5:

| Test (cell) | Claims | Actually runs | Why it still passed |
|---|---|---|---|
| `test_all_true_mask_matches_dense[128]` | sparse kernel | **dense SDPA** (asymmetric [64,128]→fallback) | asserts sparse==dense; both are SDPA → byteΔ=0.0 |
| `test_all_true_mask_matches_dense[256]` | sparse kernel | dense SDPA / dsplit | same — vs dense=SDPA |
| `test_causal_block_mask_with_causal_matches_dense[128]` (& [256]) | "STEEL sparse path" | **dense SDPA** (N=128 small-mask → fallback on M5) | docstring is M1–M4 history; asserts vs dense-causal=SDPA |
| `test_sliding_window_matches_ref` (D=128) | sparse | **dense SDPA** | reference `_ref_sparse_sdpa` IS dense SDPA+bias → byteΔ=0.0 |

Root cause (the original sin): these assert the MATH (Δ vs an SDPA reference) without the BINARY. On M5
the D=128 sparse path IS the SDPA fallback, and the reference is also SDPA, so they compare SDPA to
SDPA — green regardless of any sparse-kernel behavior. (Exactly how the four V6-NAX sprints measured
SDPA believing it was sparse.) The D=64 cells are CORRECT-BINARY (symmetric → real kernel, byteΔ≠0).

## The fixes (test-only)

1. **Fingerprint-discipline lock** (`tests/test_fingerprint_discipline.py`, 3 cells):
   - Locks the wrong-binary instances as "runs SDPA on M5" (byteΔ==0.0) — so the vacuous state is
     EXPLICIT and a Phase-F reroute (D=128 → real sparse) flips them to byteΔ>0 and FAILS, forcing a
     deliberate update.
   - A **positive fingerprint demo**: symmetric-D128 sparse asserts byteΔ>0 vs SDPA (a real distinct
     kernel ran). **Drift-catch demonstrated**: byteΔ=7.6e-6 now; if it drifted to the SDPA fallback,
     byteΔ→0 → `assert d>0.0` RAISES → CI fails. Green-on-wrong-binary is structurally caught.
2. **Relabeled** the 3 misleading docstrings in `test_attention.py` (`test_all_true_mask_matches_dense`,
   `test_causal_block_mask_with_causal_matches_dense`, `test_sliding_window_matches_ref`) — honest
   AUDIT-C NOTE that on M5 the D=128 path is the dense-SDPA fallback, pointing to the real-kernel
   fingerprint coverage. Docstring-only; the assertions (correct as SDPA-correctness) are kept
   (keep-all-paths) — they are not deleted, just no longer mistaken for sparse-kernel coverage.
3. **Real-kernel coverage** with binary fingerprints already exists: the B1 symmetric sparse locks
   (`test_sparse_family_correctness_lock.py`) + B2/B3/B4. No re-point of 1700 tests needed.

## Real bug uncovered? NO
Re-pointing the D=128 sparse to its real (symmetric) kernel was already verified CORRECT in B1
(byteΔ=3.8e-6 vs fp32). The D=128-asymmetric STEEL sparse kernel is never run on M5 (known-disabled by
the `(long)p->NK` compiler bug → SDPA), so it is not "silently passing while broken" — it is
known-broken-and-routed-away. **No new Phase-F bug from C.** (Its non-M5 correctness is a coverage gap,
not a bug.)

## Coverage matrix (kernel × edge → fingerprint-asserted lock)
| Kernel | correctness lock (fingerprint-asserted) |
|---|---|
| sparse V2 matmul2d / V1 scalar | `test_sparse_family_correctness_lock.py` (13, byteΔ + fp32) |
| dense STEEL V1/V2/V3/V4/V5/splitK/dsplit/flash_decode | `test_dense_steel_family_lock.py` (14, fp32 + source-threshold) |
| backward dQ/dK/dV (native + SDPA-vjp, per path) | `test_backward_family_lock.py` (6, fp32 grad + byteΔ which-binary) |
| GNA / conv / topk / sage / paged | `test_b4_family_lock.py` (9, per-type oracle) |
| dispatch identity (all entries) | `test_dispatch_map_lock.py` (11) |
| **which-binary discipline** | `test_fingerprint_discipline.py` (3, drift-catch demonstrated) |

## Carry to Phase E (unchanged)
sparse V1↔V2 `2^31` PERF crossover; STEEL-vs-SDPA M5-optimality; V5 reachability; sage int8 quality-worth.

## Disposition
The suite's green-on-wrong-binary class is **named, enumerated (5 instances), and structurally locked**
— a re-pointed test now FAILS on drift to the wrong binary, and the wrong-binary instances are
explicitly documented (not silently trusted as sparse coverage). The real per-kernel coverage (B1–B4,
fingerprint-asserted) is the backbone. Suite green. No orphans. Not tagged. **Phase D (documentation
rebuild from the verified A/B/C facts + publication cleanup) is next.**

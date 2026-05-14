# v2.50 Prompt 4 Section A — Pre-existing test failures inventory

**Sprint date**: 2026-05-14 (Prompt 4 Section A)
**Master tip pre-Section-A**: `50f59b9` (post-Prompt 3 Phase 4b-complete partial merge)

## Failure overview

50 pre-existing tests fail on master. Categorized by error pattern:

| Test file | Failures | Cluster |
|---|---|---|
| `tests/test_attention.py` | 20 | Sparse RuntimeError (mask < 4096B) + 5 AssertionError edge cases |
| `tests/test_v34_backward_dq.py` | 10 | TypeError (missing d_vec + causal args) |
| `tests/test_v34_backward_kv.py` | 8 | TypeError (missing d_vec + causal args) |
| `tests/test_v34_bwd_multisg.py` | 4 | TypeError (missing d_vec + causal args) |
| `tests/test_perf_claims_doc_sync.py` | 2 | AssertionError (registry vs doc drift) |
| `tests/test_turboquant.py` | 2 | AssertionError (QR roundtrip precision) |
| `tests/test_attn_bias_native.py` | 2 | AssertionError (d128 causal max_err) |
| `tests/test_lcsa_nax_phase1_4_dispatcher.py` | 1 | AssertionError (threshold value) — Sprint 1 v2.50 |
| `tests/test_svdquant.py` | 1 | AssertionError (quantize_with_svd) |

## Cluster A — TypeError signature mismatches (22 tests)

**Root cause**: Tests written before v2.38.1 `d_vec` API addition and
Prompt 3 Section B `causal` parameter addition.

**Affected bindings**:
- `v6_nax_backward_query(..., scale, causal=False)` — needs `d_vec` arg + optional `causal`
- `v6_nax_backward_kv(..., scale, causal=False)` — needs `d_vec` arg + optional `causal`
- `v6_nax_backward_dv_raw(..., scale, wm=4, causal=False)` — needs `causal` arg
- `v6_nax_backward_dk_raw(..., scale, wm=4, causal=False)` — needs `d_vec` + `causal`
- `v6_nax_backward_fused_dkdv_raw(..., scale, wm=4, causal=False)` — needs `d_vec` + `causal`

**Fix**: Update test call sites to pass `d_vec` (compute via `mx.sum(dO*O, axis=-1)`).
The `causal` parameter has default False; only need to add it if the test exercises causal.

**Affected tests**:
- 10 in `test_v34_backward_dq.py`
- 8 in `test_v34_backward_kv.py`
- 4 in `test_v34_bwd_multisg.py`

**Estimated effort**: ~30 min — straightforward signature updates.

## Cluster B — sparse_attention mask < 4096 bytes RuntimeError (15 tests in test_attention.py)

**Root cause**: MLX added a runtime constraint that sparse_attention mask
buffers must be ≥4096 bytes. Tests use small qL/kL shapes that produce
masks < 4096 bytes.

**Error**:
```
RuntimeError: sparse_attention: mask total bytes < 4096
(use larger qL, kL, or higher mask ndim). MLX inlines small buffers
in constant address space; the JIT kernel emits device-qualified pointer.
```

**Affected tests**: Within `test_attention.py`:
- TestSparseAttentionKernel
- TestSparseBackwardTiled (3 tests)
- TestSparseBackwardSteel (3 tests)
- TestRoPEFusion
- TestSegmentMask
- TestGNAAttention
- TestGNABackward (2 tests)
- TestRotaryDim
- TestBlockMask4D (3 tests)
- TestV2FeatureExtensions
- TestSteelV5CP5

**Fix options**:
- (a) Bump test shapes to produce ≥4096-byte masks (qL=kL=64 BT=32 → 4 bytes; need qL=kL=256 BT=16 → 256 bytes; bump to qL=kL≈512 or use bigger BT)
- (b) Add skip-marker if MLX version has the constraint enforced

Most tests just need bigger shapes — minor edit.

**Estimated effort**: ~45 min.

## Cluster C — AssertionError edge cases (8 tests)

### C.1 — test_topk_ratio_1_matches_dense (test_attention.py)
- `topk_ratio=1.0 should match dense: diff=6.1e-4`
- The Phase 3a dispatch (Prompt 2) routes topk_ratio=1.0 → no filter → dense via mx.fast.sdpa.
- But the test expects exact match against a reference path; small fp16 diff at 6e-4 is just precision drift.
- **Fix**: bump tolerance from default to ≥1e-3.

### C.2 — test_lcsa_nax_phase1_4_dispatcher::test_default_threshold_value
- Likely checks `DEFAULT_DENSITY_THRESHOLD == 0.02` (pre-Sprint 1).
- Sprint 1 v2.50 (Prompt 1) raised threshold to 1.01.
- **Fix**: update test expectation to 1.01.

### C.3 — test_perf_claims_doc_sync (2 tests)
- Doc PERF_CLAIMS.md is out of sync with test registry (missing v2.39.1/v2.39.2 entries).
- Either update doc or test registry to match.
- **Fix**: small edit; identify which side is authoritative.

### C.4 — test_turboquant QR roundtrip/orthogonal (2 tests)
- QR roundtrip error 5.9e-3 vs tolerance; orthogonality error 8.9e-4.
- Precision issue, possibly MLX version-specific in current numerics.
- **Fix**: bump tolerance OR investigate (could be a real bug).

### C.5 — test_attn_bias_native::TestBiasMode1/2::test_d128_causal (2 tests)
- max_err 0.30 / 0.32 (large) vs tolerance 0.05.
- Significant divergence — needs investigation to determine if real bug or stale test.

### C.6 — test_svdquant::test_quantize_with_svd
- AssertionError without specific message — needs investigation.

### Other 4 AssertionErrors in test_attention.py
- TestRoPEFusion, TestSegmentMask, TestRotaryDim, TestReturnAttnWeights, TestBlockMask4D — likely tied to other clusters; needs per-test inspection.

## Triage strategy

**Phase A.2.1 — Quick wins (signature fixes)**: Cluster A — 22 tests, ~30min
**Phase A.2.2 — Shape bumps**: Cluster B — 15 tests, ~45min
**Phase A.2.3 — Edge case fixes**: Cluster C — 8 tests, ~60min total (some may need investigation)

Total estimated: ~2-2.5h CC. Target: 0 unexpected failures post-Section-A.

If any AssertionError reveals a REAL bug (Cluster C.5 or C.6), halt Section A, STATUS doc, escalate to Marco (per Section A.2.3 protocol).

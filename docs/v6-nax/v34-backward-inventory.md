# V6NAX backward NAX-direct (Option β) — inventory

## Goal

Implement NAX-direct backward attention kernels (dQ + dK/dV) per
`docs/v6-nax/v6nax-backward-option-beta-design-hints.md`, transferring
the V6NAX forward B+C+E mechanism bundle (cross-SG sync elim + simd_shuffle_xor
+ M5-tuned defaults) plus the anti-pattern A correction (EXEC_SG=8 default
for D=128 mid shapes).

## Foundation read (Phase 1 Section A.1-A.3)

| File | Role | Verified |
|---|---|:--:|
| `docs/v6-nax/v6nax-backward-option-beta-design-hints.md` | Canonical input (5 open Qs, B+C+E transfer plan) | ✓ |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp::createV6NAXSource()` | V6NAX forward source (658 LOC, lines 2307-2964) — clone target for backward kernel structure | ✓ |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp::loopBackwardQuery()` (line 2967) | STEEL MPP backward dQ reference (algorithm) — uses lse + D operands | ✓ |
| `csrc/mfa/v6_nax/NAAttentionKernel.cpp::loopBackwardKeyValue()` (line 3313) | STEEL MPP backward dK/dV reference | partial |
| `csrc/mfa_v6_nax_primitive.cpp::v6_nax_forward()` (line 696) | V6NAX forward Python-callable Primitive — returns `(O, lse)` pair | ✓ |
| `mlx_mfa/attention.py` custom_vjp registration | Python autograd hook point | not yet read |

## V6NAX forward kernel signature (current state, line 2759)

```
kernel void v6nax_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    constant V6NAXParams& params [[buffer(4)]],
    ...);
```

**Only 5 buffers — NO `L` (lse) buffer.** The V6NAX forward Primitive
allocates `lse_shape` as an output array (line 491:
`lse.set_data(mlx::core::allocator::malloc(lse.nbytes()))`) but the
kernel **never writes to it**. The lse output is dead storage.

The forward kernel keeps `max_score` and `sum_score` per-row state in
registers (lines 2794-2795) and uses them online for the softmax
factor + final normalization, but does not persist them to device
memory.

## Critical blocker: lse access for backward (BLK1)

FA-2 backward dQ computation requires:
- `lse` from forward (per-row log-sum-exp, for re-derive softmax P)
- `D` accumulator = `rowsum(dO ⊙ O)` (computed pre-backward per row)
- `dO`, `O`, Q, K, V (standard backward inputs)

The design hints doc assumed `lse` is available from forward. **It is
not.** V6NAX forward as currently shipped does not compute lse into
device memory. This is the foundational blocker for V6NAX backward
dQ implementation.

Resolution options (require Marco's decision):

| Option | Cost | Pros | Cons |
|---|---|---|---|
| (a) Extend V6NAX forward to write lse | ~50 LOC patch to forward kernel + Primitive | Cleanest. Forward outputs match Apple SDPA-NAX contract. Backward depends on existing path. | Modifies production forward kernel — risk of forward regression. Must re-bench V6NAX forward post-patch. Was explicitly listed "out of scope" in original prompt §1 but is necessary infrastructure for backward. |
| (b) Add `compute_lse_from_v6nax` helper kernel | ~30 LOC new kernel + binding | Forward kernel untouched. New kernel scoped tightly. | +1 dispatch per backward call. Helper kernel needs to RECOMPUTE max + sum from device output — wasteful (O(N²) recomputation). |
| (c) Use STEEL forward → V6NAX backward hybrid path | ~0 kernel change | Smallest scope. Uses existing STEEL forward (which writes lse). | V6NAX backward depends on STEEL forward state layout — fragile cross-kernel coupling. Eliminates V6NAX forward perf gain when paired with V6NAX backward. |
| (d) Recompute lse inline in backward kernels | ~20 LOC per backward kernel | No forward changes. | Backward kernel walks the full K-loop twice (once to compute lse, once for the gradient compute). +50-100% backward wall-clock. |

**Recommendation**: option (a). The "out of scope" wording in the
prompt §1 referred to *EXEC_SG shape-aware heuristic* (the
anti-pattern A bonus finding). Adding lse-write to V6NAX forward is a
different change — it is necessary infrastructure that the design
hints doc implicitly assumed but did not surface. The patch is small
(~50 LOC: add `L [[buffer(5)]]` to signature, compute final
lse = max_score + log(sum_score) at line 2949 region, store via
`L.store_rows(...)` or analog). Re-bench V6NAX forward post-patch under
canonical methodology to confirm no regression.

This is **BLK1** — top-priority resolution before Phase 1 Section B
can begin.

## Shape catalog (V6NAX backward eligibility, per design hints scope)

- D ∈ {64, 128}
- dtype ∈ {FP16, BF16}
- No causal mask (deferred per DC3)
- No block mask (deferred per DC3)
- No softcap / ALiBi / TurboQuant (deferred to STEEL fallback)

Bench shapes (post-blocker resolution, Phase 3 Section F):
- Small: qL=kL=512, D=64/128
- Mid: qL=kL=2048, D=64/128
- Large: qL=kL=8192, D=64/128
- Cross: qL=512, kL=4096, D=128
- 8-12 shapes total

## Acceptance criteria (v2.37.0 GREEN)

| Criterion | Target |
|---|---|
| dQ correctness vs STEEL backward | RMSE < 1e-3 FP16, < 1e-4 BF16 |
| dK/dV correctness vs STEEL backward | RMSE < 1e-3 FP16, < 1e-4 BF16 |
| V6NAX backward ratio vs STEEL backward | < 1.0 (V6NAX faster) across eligible regime |
| Realistic target | ratio ∈ [0.4, 0.85] (15-60% speedup) |
| Three-axis tests | All axes covered, ≥ 15 new tests |
| Auto-routing | `flash_attention()` VJP transparently uses V6NAX on M5+ eligible |
| Escape hatch | `MFA_DISABLE_V6_BACKWARD=1` falls back STEEL |
| Pre-existing tests | 77/77 still pass (no regression vs v2.36.1) |

## Sprint phase structure (per prompt §2)

| Phase | Sections | Estimated CC | Current status |
|---|---|---|---|
| 1 — Design + dQ kernel | A, B, C | 4-5h | **A in progress, B BLOCKED on BLK1** |
| 2 — dK/dV kernel + integration | D, E | 3-4h | pending Phase 1 |
| 3 — Validation + tuning | F, G | 3-4h | pending Phase 2 |
| 4 — Release v2.37.0 | H | 1-2h | pending Phase 3 |

**Current state**: Phase 1 Section A nearly complete. Phase 1 Section
B (dQ kernel implementation) is BLOCKED on BLK1 (lse-from-forward
infrastructure) until Marco decides between options (a), (b), (c), (d).

## Out of scope (per prompt §1)

- Block-sparse backward (causal mask interaction with V6NAX backward)
- Causal backward (deferred per DC3 in design hints)
- V6NAX forward EXEC_SG shape-aware heuristic patch (independent ~1h
  follow-up, not bundled)
- D ∉ {64, 128} backward (falls back STEEL)
- Softcap / ALiBi backward (falls back STEEL)
- TurboQuant backward (kept on STEEL)

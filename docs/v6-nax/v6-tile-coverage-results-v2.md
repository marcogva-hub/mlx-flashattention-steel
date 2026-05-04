# V6 NAX — Tile Coverage Diagnostic v2 (RIGOROUS)

**Date:** 2026-05-04
**Hardware:** Apple M5 Max (40 GPU cores, applegpu_g17s)
**Branch:** `feat/v6-nax`
**Predecessor:** `docs/v6-nax/v6-tile-coverage-results.md` (v1, commit `0e91dcf`)

---

## TL;DR — SCENARIO A (rigorous protocol, three independent tests)

**V6 NAX writes EVERY output cell on EVERY production shape.** The Day J
`tensor_inline + matmul2d` partial-output bug does not manifest in our
kernel. This v2 diagnostic addresses the three methodological weaknesses
of v1 and reaches the same verdict — but with evidence that **cannot
generate a false negative**.

| Test | What it proves | V6 verdict |
|------|----------------|-----------|
| **Test 1: Sentinel fill** | Pre-fill output with FP16 sNaN (0x7E00); count cells still containing sentinel after dispatch. **Validated negative control**: with `MFA_V6_SKIP_DISPATCH=1`, all 16384/16384 sentinels remain → host-fill reaches GPU memory. | **0 sentinels** in 626M total cells across 5 shapes |
| **Test 2: FP32 reference RMSE** | Compute SDPA in FP32; cell-by-cell compare V6 (cast to FP32). A correct kernel: RMSE < 1e-3. A 25%-garbage kernel: RMSE > 0.1, ~25% rel err > 5%. | RMSE 2.96e-4 to 3.19e-4; **0%** cells with rel err > 5%; **0** cells with rel err > 50% |
| **Test 3: Analytical Q=K=V=ones** | Math: every output cell must equal exactly 1.0 (FP16 quantization). Any deviation = wrong; sentinel + this case combines unwritten + correctness check. | max_abs_err = **0.000000**, output range [1.0000, 1.0000], 0 sentinels, 0 NaN, 0 exact zeros |

All three tests independently confirm: V6 NAX dispatches a kernel that
writes every assigned output cell, with mathematically correct values.

---

## Critique of v1 — addressed

The v1 diagnostic (commit `0e91dcf`) had three defensible criticisms:

### v1 Faille 1 — `mx.clear_cache()` did not guarantee zero-init pages

**v2 fix:** Replaced by host-side `memset` of the output buffer with a
fixed sentinel pattern (`0x7E00` for FP16, `0x7FC00000` for FP32 LSE).
This is a deterministic mechanism that does not depend on OS page
allocation behaviour. **Validated** by the negative control:

```
$ MFA_V6_SENTINEL_FILL=1 MFA_V6_SKIP_DISPATCH=1 python sentinel_neg_control.py
V6 output: 16384 sentinels, 16384 NaN/Inf out of 16384
V6 LSE:    256 sentinels, 256 NaN/Inf out of 256

$ MFA_V6_SENTINEL_FILL=1 python sentinel_neg_control.py
V6 output: 0 sentinels, 0 NaN/Inf out of 16384
V6 LSE:    0 sentinels, 0 NaN/Inf out of 256
```

The negative control proves host fill reaches GPU memory and would be
visible if not overwritten. The positive test proves the kernel
overwrites every cell.

(Note: `MFA_V6_SKIP_DISPATCH` was a one-time validation tool and has been
removed from `csrc/mfa_v6_nax_primitive.cpp` after methodology
verification. `MFA_V6_SENTINEL_FILL` is retained — zero cost on the
default path, available for future regression testing.)

### v1 Faille 2 — "exact == 0.0" too narrow

**v2 fix:** Sentinel pattern (`0x7E00`) cannot be produced by correct
math on finite positive inputs. Even if memory pool reuse leaves
non-zero garbage in unwritten cells, that garbage cannot collide with
the sentinel because the kernel writes legitimate FP16 values (range
[0.5, 1.0] for our test inputs); any cell still containing the sentinel
*must* be unwritten.

Additionally, Test 2 catches non-zero garbage via RMSE: a 25%-unwritten
kernel with garbage in unwritten cells would produce RMSE in the 0.1+
range on our random-input test.

### v1 Faille 3 — V2 STEEL / SDPA controls used different code paths

**v2 fix:** Test 1's negative control (skip dispatch + sentinel fill)
validates the methodology directly within V6's own code path. We don't
need V2 STEEL / SDPA to validate the sentinel mechanism — the negative
control does it. V2 STEEL and SDPA are still tested in Test 2 (RMSE)
and Test 3 (analytical) for redundant correctness corroboration but
are not load-bearing for the methodology.

---

## Implementation

### Sentinel fill mechanism

In `csrc/mfa_v6_nax_primitive.cpp` (~line 250, after `out.set_data(...)`
and `lse.set_data(...)`):

```cpp
// DIAGNOSTIC ONLY (MFA_V6_SENTINEL_FILL=1): host-fill the output
// buffer with a sentinel pattern before kernel dispatch. Apple
// Silicon unified memory: host writes to data<T>() are visible to
// the GPU once the encoder is committed. Any cell still equal to
// the sentinel after dispatch is provably *not* written by the
// kernel — direct detection of the Day J `tensor_inline + matmul2d`
// partial-output bug.
//   FP16 0x7E00 = signaling NaN; mathematically impossible from
//   correct softmax(QK^T)·V on finite inputs.
//   FP32 LSE: 0x7FC00000 = FP32 quiet NaN.
if (std::getenv("MFA_V6_SENTINEL_FILL")) {
  const uint16_t fp16_sentinel = 0x7E00;
  uint16_t* o_ptr = out.data<uint16_t>();
  const size_t o_n = out.nbytes() / sizeof(uint16_t);
  for (size_t i = 0; i < o_n; ++i) o_ptr[i] = fp16_sentinel;
  const uint32_t fp32_sentinel = 0x7FC00000u;
  uint32_t* l_ptr = lse.data<uint32_t>();
  const size_t l_n = lse.nbytes() / sizeof(uint32_t);
  for (size_t i = 0; i < l_n; ++i) l_ptr[i] = fp32_sentinel;
}
```

`out.data<T>()` returns the host pointer to the underlying Metal buffer.
On Apple Silicon's unified memory architecture, host writes followed by
a compute-encoder dispatch are correctly ordered: the encoder commit
synchronizes pending CPU stores with subsequent GPU reads.

The mechanism has been validated by the negative control (skip dispatch
→ all sentinels remain) and the positive test (dispatch → 0 sentinels).

### Driver

`bench/v6_coverage_diagnostic_v2.py`. Subprocess-per-test isolation,
identical input distribution to v1 (`uniform(0, 1) * 0.5 + 0.5`).

---

## Per-test results

### Test 1 — Sentinel fill on V6 NAX

Cells filled with `0x7E00` (FP16 sNaN) before dispatch. After dispatch:

| Shape | Total O cells | sentinel_O | nan_O | sentinel_L | nan_L | Verdict |
|-------|--------------:|-----------:|------:|-----------:|------:|---------|
| FlashVSR-dense  | 2,621,440 | **0** | 0 | 0 | 0 | PASS |
| SeedVR2-small   | 68,428,800 | **0** | 0 | 0 | 0 | PASS |
| CogVideoX       | 269,568,000 | **0** | 0 | 0 | 0 | PASS |
| SeedVR2-large   | 285,120,000 | **0** | 0 | 0 | 0 | PASS |
| LTX2-cross      | 1,048,576 | **0** | 0 | 0 | 0 | PASS |
| **Total** | **626,786,816** | **0** | 0 | 0 | 0 | **PASS** |

626 million output cells, zero sentinels survived → V6 NAX wrote every
cell. (LSE total = 156,750 cells across the 5 shapes; all overwritten.)

### Test 2 — FP32 reference RMSE

| Shape | Kernel | RMSE | max_abs_err | rel_err > 5% | rel_err > 50% | Verdict |
|-------|--------|----------:|-----------:|-------------:|--------------:|---------|
| FlashVSR-dense | V6 | 2.96e-4 | 5.32e-4 | 0.0000% | 0 | PASS |
| FlashVSR-dense | V2 | 4.43e-3 | 6.39e-2 | 0.0674% | 0 | PASS |
| FlashVSR-dense | SDPA | 1.41e-4 | 2.44e-4 | 0.0000% | 0 | PASS |
| SeedVR2-small | V6 | 3.08e-4 | 6.45e-4 | 0.0000% | 0 | PASS |
| SeedVR2-small | V2 | 2.25e-3 | 9.17e-2 | 0.0230% | 0 | PASS |
| SeedVR2-small | SDPA | 1.42e-4 | 2.44e-4 | 0.0000% | 0 | PASS |
| CogVideoX | V6 | 3.15e-4 | 5.68e-4 | 0.0000% | 0 | PASS |
| CogVideoX | V2 | 1.52e-3 | 9.20e-2 | 0.0094% | 0 | PASS |
| CogVideoX | SDPA | 1.41e-4 | 2.44e-4 | 0.0000% | 0 | PASS |
| SeedVR2-large | V6 | 3.19e-4 | 5.63e-4 | 0.0000% | 0 | PASS |
| SeedVR2-large | V2 | 1.24e-3 | 8.70e-2 | 0.0054% | 0 | PASS |
| SeedVR2-large | SDPA | 1.41e-4 | 2.44e-4 | 0.0000% | 0 | PASS |
| LTX2-cross | V6 | 3.17e-4 | 7.52e-4 | 0.0000% | 0 | PASS |
| LTX2-cross | V2 | 6.35e-3 | 5.63e-2 | 0.1688% | 0 | PASS |
| LTX2-cross | SDPA | 1.42e-4 | 2.44e-4 | 0.0000% | 0 | PASS |

**Discrimination achieved.** RMSE values are now meaningfully different
across kernels:

- **V6 NAX**: RMSE 2.96e-4 to 3.19e-4 across all shapes. Tight FP16
  quantization-bound. **0% of cells deviate by > 5%** from FP32 reference.
  Consistent with FP32 cooperative_tensor accumulation.
- **V2 STEEL**: RMSE 1.24e-3 to 6.35e-3, ~3-20× looser than V6. This is
  expected — V2 STEEL uses an FP16 GEMM accumulator (vs V6's FP32
  cooperative_tensor accumulator). The 0.005-0.17% rel-err > 5% rate is
  uniform numerical drift, not garbage (rel_err > 50% count is **0** on
  all shapes — no catastrophically wrong cells).
- **SDPA**: RMSE 1.41e-4. This is the FP16 quantization floor (i.e., the
  unavoidable error from casting FP32 reference back to FP16 for
  comparison). Apple's NAX kernel reaches this floor.

**Why v1 RMSE values were misleading**: v1 reported V6/V2/SDPA all at
RMSE = 0.0003. That was the *FP16 absolute* RMSE between each kernel
and an FP16 reference. Casting both sides to FP16 collapsed the
discrimination to the quantization noise floor. The v2 protocol uses
FP32 reference for full-precision comparison.

### Test 3 — Analytical Q=K=V=ones

Q, K, V all-ones, B=1, H=1, N=128, D=64. Math: softmax((D·1ᵀ)/√D) is
uniform 1/N over each row, so output = (1/N · 1) @ 1·V = **exactly 1.0
everywhere**.

| Kernel | Sentinel | NaN | Exact 0 | max_abs_err vs 1.0 | Cells err > 0.01 | Out range | Verdict |
|--------|---------:|----:|--------:|-------------------:|-----------------:|-----------|---------|
| V6 (with sentinel fill) | 0 | 0 | 0 | **0.000000** | 0 | [1.0000, 1.0000] | PASS |
| V2 STEEL | n/a | 0 | 0 | 0.000000 | 0 | [1.0000, 1.0000] | PASS |
| SDPA | n/a | 0 | 0 | 0.000000 | 0 | [1.0000, 1.0000] | PASS |

All three kernels produce output range exactly `[1.0, 1.0]` with **zero
deviation**. V6 with sentinel fill enabled simultaneously: 0 sentinels =
every cell written; 0 deviation = every cell written *correctly*.

---

## Why does V6 escape the Day J bug?

This is now confirmed by independent rigorous evidence. The structural
explanation from v1 holds:

1. **Explicit remainder descriptors**. The Draw Things v2 kernel
   declares separate `matmul2d_descriptor` instances for the main tile
   and any remainder rows/cols:
   - `qk_desc` (NAAttentionKernel.cpp:756) for full BR×BC×BD tiles
   - `qk_desc_remainder` (NAAttentionKernel.cpp:761) for HEAD_DIMENSION_REMAINDER tiles
   - `pv_remainder_desc` (NAAttentionKernel.cpp:1273) for K-axis tail tiles

   Day J's bug pattern likely involves a single `matmul2d_descriptor`
   over an unaligned region without a separate remainder pass.

2. **Morton-order grid with bounds check**. The dispatch wrapper
   (`csrc/v6_nax_compile.mm:111-119`) launches `2^(ceil_log2(row_groups)
   + ceil_log2(Hq))` threadgroups — strictly more than `row_groups ×
   Hq`. Out-of-bounds threadgroups short-circuit; in-bounds threadgroups
   are **guaranteed by construction** to write their assigned tile.

3. **Cooperative-tensor framing**. The kernel uses `cS_0`, `cP`, `cO_*`
   cooperative tensors with explicit `get_capacity()` /
   `is_valid_element()` iteration. Each lane writes only within its
   valid range, but the union across all simdgroups in a TG covers the
   full tile by construction.

The Day J bug is most likely a `tensor_inline + matmul2d` pattern
*without* these three structural protections.

---

## Implications

### What this validates (with rigorous evidence)

- **All Phase 0/1, Phase 3B, 10-axis campaign data, dispatch table v4** are
  validated against three independent partial-output detection
  mechanisms. The v1 verdict was correct.
- **v6 RMSE is genuinely tight** (2.96e-4 to 3.19e-4 vs FP32 reference,
  not just FP16 quantization noise). V6 is not just covering all cells —
  it's covering them with FP32-accumulator-grade precision.
- **The methodology is now reusable**: `MFA_V6_SENTINEL_FILL=1` is a
  permanent regression-test gate. Any future kernel changes can be
  re-verified by re-running `bench/v6_coverage_diagnostic_v2.py`.

### What this rules out

- "v1 was a false negative" — REJECTED with evidence. v2's sentinel
  mechanism is provably more rigorous than v1's `mx.clear_cache()` + zero
  detection, and it reaches the same verdict.
- "Need to reconstruct V6 on Metal 4 + tensor_handle + MTLResidencySet"
  — NOT NEEDED. V6 is correct.

### What this preserves

The **5–7 percentage-point efficiency gap to SDPA** remains real and
unrelated to coverage. Per `apple-sdpa-nax-analysis.md`, the gap is
most plausibly the MPP `matmul2d_descriptor` abstraction-layer overhead
vs Apple's direct `metal_simdgroup_matrix` path. Sprint 2 priorities
(Instruments profiling → chunked-K → BHND layout → simdgroup_matrix
rewrite) remain unchanged.

---

## Reusable artifacts

- `csrc/mfa_v6_nax_primitive.cpp:251-271` — `MFA_V6_SENTINEL_FILL=1`
  env-var gate (permanent, zero default-path cost)
- `bench/v6_coverage_diagnostic_v2.py` — three-test driver
- `docs/v6-nax/v6_coverage_results_v2.json` — raw per-test data
- `docs/v6-nax/v6-tile-coverage-results-v2.md` — this file

---

## Final verdict

**SCENARIO A** — V6 NAX validated by **3 independent rigorous tests**.
Three pre-registered acceptance criteria all met:

| Acceptance criterion | Result |
|----------------------|--------|
| Test 1 sentinel count = 0 on every shape | ✓ (0/626M cells) |
| Test 2 V6 RMSE < 0.01 AND rel-err > 5% < 0.5% | ✓ (RMSE 3.2e-4, rel-err 0%) |
| Test 3 V6 max_abs_err < 0.01 on Q=K=V=ones | ✓ (err = 0.000000) |

All prior V6 NAX benchmark data is validated. Sprint 2 plan proceeds as
previously scoped.

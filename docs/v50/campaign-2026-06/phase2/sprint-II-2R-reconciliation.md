# Sprint II-2R — int8 Premise Reconciliation (2026-06-12)

**Status**: Phase R.0 COMPLETE — contradiction resolved definitively.
**Verdict**: int8 MPP matmul2d **IS implemented** on macOS 26.4/M5.
Both prior reports were wrong about the *mechanism*:

- **II-2's "unimplemented" was a probe artifact**: all five variants
  declared operands as **`char`**, which in Metal C++ is a distinct
  type from `int8_t` (= `signed char`).  MPP's dispatch is keyed on
  `__is_same_v<T, int8_t>` / `uint8_t` exactly
  (MPPTensorOpsMatMul2dImpl.h combo chains), so `char` falls through
  every combo list in every operand form to the "Unsupported type"
  assert.  Verified directly: the identical device-tensor kernel at
  II-2's exact dims (64,64,128) FAILS with `char`, COMPILES with
  `int8_t` and `signed char`.
- **II-5's falsification was correct in outcome but incomplete in
  diagnosis**: the "fragment-dims constraint" it identified
  (M,N,K ∈ {16,32}, ≥ one == 32; header lines 4249-4252) applies ONLY
  when both inputs are cooperative tensors.  It explains the failure
  of II-2's full-coop variant alone; the other four failed purely on
  `char`.

## Compile matrix (verified by direct compile, 2026-06-12)

| Form | dims | int8_t | char | half |
|---|---|---|---|---|
| full-coop (both inputs coop) | 16,16,16 | dims-reject (no dim == 32) | — | dims-reject |
| full-coop | 16,32,16 | **OK** | fail (type) | OK |
| full-coop | 32,32,16 / 32,32,32 | **OK** | fail (type) | OK |
| full-coop | 64,64,128 | dims-reject | fail | dims-reject |
| device-tensor operands | 16,32,16 … 64,64,128 | **OK** | fail (type) | OK |

Header ground truth: `int8_t` appears in every operand-form's combo
chain (`int8 x int8 -> int32` at impl lines 4640/5400/6236/6832/7219/
7595); the coop-coop dims asserts are at 4249-4252 and are
dtype-independent.

## Throughput at the working forms (in-repo probe, corrected to int8_t)

| Form | int8 | fp16 | ratio |
|---|--:|--:|--:|
| full-coop (16,32,16) register fragments | **264.9 TOPS** | 132.6 TF | **2.00x** |
| device-tensor (64,64,128) | 134.0 TOPS | 134.6 TF | **0.995x** |

**Design law for R.2**: the int8 advantage exists ONLY in the
full-cooperative register-fragment form — the device-tensor path
compiles but delivers fp16-speed (no int8 MMA mode engaged).  An int8
attention kernel must compose its QK^T (and PV, if int8) from
(16,32,16)-class paired fragments — exactly the `BaseNAXFrag::mma`
shape the V34 kernels already use for fp16.

## Probe hygiene

`csrc/mpp_int8_bench.mm` updated: all legacy variants `char` →
`int8_t`; header comment block rewritten to the reconciled story.  The
corrected legacy path now compiles and measures (0.995x — itself a
finding), so the probe tracks BOTH forms across macOS updates.

## R.1 gate — attention-level accounting (next section of this sprint)

The 2.00x is a raw-MMA number.  Gate evidence: V34-fp16 forward time
vs Apple SDPA NAX at target cells + QK/PV phase shares + quant-pass
cost → net int8 ceiling.  Recorded below once measured.

## Lesson (institutional)

`char`, `signed char`, and `unsigned char` are THREE distinct types in
C++/Metal template dispatch.  A "type not supported" diagnostic against
a template library keyed on fixed-width types must be checked with the
exact `intN_t` spelling before concluding a capability gap.  This
joins the II-2R/II-5 lesson set: "unimplemented" verdicts need a
dims × forms × **type-spelling** sweep.

## R.1 gate — MEASURED attention-level accounting (2026-06-12)

V34-fp16 forward vs Apple SDPA NAX + separate-pass quant cost
(B=1 H=16 fp16, medians of 30):

| cell | V34-fp16 | SDPA | V34/SDPA | quant(q,k) | quant(q,k,v) |
|---|--:|--:|--:|--:|--:|
| D=64 N=4096 | 1.567 | 1.294 | 1.21 | 0.634 | 0.538 |
| D=128 N=4096 | 2.952 | 2.989 | **0.99** | 0.634 | 0.928 |
| D=64 N=8192 | 5.424 | 4.677 | 1.16 | 0.655 | 0.922 |
| D=128 N=8192 | 11.117 | 11.250 | **0.99** | 1.207 | 1.748 |

Ceiling model (QK ≈ PV ≈ 42% of kernel, softmax/overhead ≈ 16%):
- **int8 QK only**: net ≈ 1.01–1.13x vs SDPA after quant cost —
  does NOT survive the accounting.  Declined as a standalone variant.
- **int8 QK + int8 PV**: net ≈ 1.13x (N=4096) – 1.37x (N=8192) at
  D=128, before in-kernel quant fusion upside (Draw Things fuses the
  quant online and reports 1.24–1.41x).  D=64 must first overcome the
  1.16–1.21x V34-vs-SDPA handicap — D=128 is the target cell.

Accuracy simulation of the full recipe (int8 per-token QK + int8
row-affine PV) vs fp16 SDPA reference, N=1024 D=128 causal:

| activations | rmse | max-abs | cos |
|---|--:|--:|--:|
| unit-scale | 0.0012 | 0.023 | 0.999950 |
| unit + 8x channel outliers | 0.0076 | 0.162 | 0.999446 |
| std-4 | 0.135 | 3.07 | 0.999300 |

int8 PV adds negligible error over fp16 PV (rmse 0.0010 → 0.0012).
The std-4 degradation is QK-side and is the case `smooth_k` (already
in `mlx_mfa/quantize.py`) exists for.

**GATE VERDICT: GO for Phase R.2, scoped to the combined
QK-int8 + PV-int8 variant at D=128** (QK-only declined by accounting;
D=64 deferred until the variant proves out at D=128).

## R.2 build plan (committed scope)

1. Kernel: new JIT generator emitting a V34-forward-class kernel
   (BaseNAXFrag (16,32,16) paired fragments — the ONLY form with the
   2.00x int8 advantage) with: int8 Q/K fragments → int32 S
   accumulation → per-(q-row, k-row) scale dequant to fp32 → online
   softmax (fp32, V34 pattern) → P quantized per-row to int8 →
   int8 PV with V row-affine (zero-point folded via row-sum identity)
   → fp16 O.  `KernelType` new entry; D=128 first.
2. Quant pass: Python mx ops initially (cost measured above and
   included in every bench); in-kernel online quant as the follow-up
   optimization if the first bench lands between 1.1–1.3x.
3. Flag: `MFA_SAGE_INT8=1` opt-in only; NEVER default dispatch
   (accuracy-gated promotion is a separate later decision).
4. Acceptance: cos ≥ 0.9995 + rmse ≤ 1e-2 at unit scale per cell
   (matches the simulation); FFT forensics on any structured artifact.
5. Bench: vs Apple SDPA NAX (the real dispatch), full quant cost
   included, 3 sessions median; promote/decline on the measured
   number.  Kill: < 1.10x at D=128 N=8192.

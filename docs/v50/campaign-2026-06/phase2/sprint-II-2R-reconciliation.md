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

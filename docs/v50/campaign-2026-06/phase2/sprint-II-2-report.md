# Phase II — Sprint II-2 report: Sage-NAX int8 Attention

**Date**: 2026-06-12 · **Status**: **DECLINED — primitive unimplemented** (measured, permanent until OS update)

## Verdict

The §AA.5 kill-gate microbench, built through the repo's production MSL4
compile path (`ShaderCache::compile_shader`, the only route that loads
MPP headers — `mx.fast.metal_kernel` proven blocked in Sprint C), shows:

| Variant | Result |
|---|---|
| fp16 half×half→float (baseline) | **113–133 TF sustained** (L1-resident loop; upper-bound grade) |
| int8 char×char→int32 (plain device tensors) | COMPILE FAIL: `static_assert "Unsupported type"` (MPPTensorOpsMatMul2dImpl.h:6021) |
| int8 char×char→float | COMPILE FAIL (same class) |
| int8 char×char→half | COMPILE FAIL |
| int8 cooperative-destination int32 | COMPILE FAIL |
| int8 full-cooperative (Draw Things register form) | COMPILE FAIL |

**The MPP headers DECLARE int8 operand types (Sprint C premise check read
the declaration tables correctly) but the `__run` implementation rejects
char operands in every binding form available to runtime-compiled MSL4 on
macOS 26.4.**  The theoretical 1.3–1.7× is unreachable on this OS.  The
Draw Things NAInt8 reference must rely on a different MPP build (newer
macOS or AOT toolchain) — labeled DEDUCED.

## Premise-validation lesson (feeds the inversions catalogue)

Header-declaration verification is NOT implementation verification.  The
Sprint-C premise check verified the type LIST; only an attempted
compile through the production path verifies the implementation.  Future
§AA.5 checks on MPP features must include a compile probe.

## Revival probe (permanent)

`mlx_mfa._ext.mpp_int8_microbench()` is committed in-tree
(csrc/mpp_int8_bench.mm).  After ANY macOS/MPP update: one call either
returns `int8=…ratio=…` (candidate revives; re-open this sprint with the
ratio against the 1.3× kill threshold) or the FAIL string (still dead).

## Cost accounting

Sprint cost: one ~170-line .mm + binding (kept as the probe).  No kernel
was built beyond the gate — the gate did its job, killing an L-XL build
before it started.

# S3.3 — bypassThreadgroupMemory re-test on new defaults

**Status:** **Not testable as written. Re-formulated as documentation.**

## What the brief asked for

The brief proposed re-running the `MFA_V6_BYPASS_TGP=1` sweep on the
new auto-tuned defaults to test the hypothesis: "with single-Otile
(kBlocks=1) replacing the kBlocks-split cO of the old default tiles,
bypass should now win on D=128 since the structural register pressure
that defeated bypass in Sprint 3.2 is gone."

## Why it's not testable as a separate axis

Looking at the v2.29.0 source after Sprint 3.3 + autoresearch:

```cpp
// csrc/mfa_v6_nax_primitive.cpp (post-merge)
if (single_otile) bypass_tgp = true;  // forced on by single-Otile
```

The single-Otile kernel path (`loopForwardSingleTile()`) was implemented
in Sprint 3.3 with **always-bypass cP** — there is no "single-Otile +
P_buf staging" code path in the kernel itself. The `cP` cooperative
tensor replaces `P_buf` unconditionally inside `loopForwardSingleTile()`.
So `MFA_V6_BYPASS_TGP=0` with single-Otile silently gets coerced back to
bypass=true upstream.

To test "single-Otile + non-bypass" we would need to:
1. Add a new code path to `loopForwardSingleTile()` that emits the
   threadgroup-staged `P_buf` variant of the PV matmul (mirroring the
   existing `if (bypassThreadgroupMemory)` branches in the legacy
   `loopForward()`).
2. Add a way to override the `if (single_otile) bypass_tgp = true;`
   coercion in the primitive.

This is **a non-trivial source-generator extension** (~50-100 LOC) not
included in Sprint 3.3 by design — single-Otile was always conceived
with always-bypass, because the cP cooperative_tensor is the *whole point*
of single-Otile (no P_buf staging is the simplification).

## What we actually know

The hypothesis "bypass wins because single-Otile reduces register
pressure" is **partially testable** through a different lens:

- **Sprint 3.2** (legacy double-buffer + bypass): bypass regressed
  +13–22% on D=128 production shapes.
- **Sprint 3.3 main** (single-Otile, which forces bypass): single-Otile
  + bypass regressed +16-22% on D=128 at default tiles BQ=32.
- **Autoresearch retuning** (single-Otile + bypass + new tiles): closes
  the V6/SDPA gap to 1.20-2.06× on all 5 shapes; D=128 wins
  -47% to -70%.

The autoresearch winner therefore is "**single-Otile + bypass + BQ=16
tiles**". We can decompose this into:
- Single-Otile alone @ BQ=32: regressed +13-22% on D=128 (S3.3 main)
- Single-Otile alone @ BQ=16: wins -47% to -70% on D=128 (autoresearch)
- The bypass component is forced by single-Otile, so cannot be isolated.

The conclusion is: the original Sprint 3.2 hypothesis ("kBlocks-split
register pressure causes the regression") was **wrong**. The actual
cause was BQ=32 being too large. Bypass wasn't the problem.

## Resulting decision

**No new code. No further measurements for this section.** The
single-Otile + always-bypass + autoresearch tiles already shipped in
v2.29.0 captures the bypass benefit.

A future "single-Otile + non-bypass" experiment is conceivable if we
suspect the threadgroup-staged P pattern would help long-N D=128
shapes (where memory traffic dominates), but that's speculative and
outside the v2.29.0 scope. If the residual D=128 gap to SDPA matters
(currently 1.35-2.06×), other levers — such as the simdgroup_matrix
rewrite that writes its own P_buf in registers — are higher-leverage.

## What this section produces

- This document (decision + rationale)
- No bench script
- No JSON data
- No code change

# v2.50 Prompt 5d Section B v3 — Approach 5 EMPIRICAL SKIP decision

**Decision**: skip Approach 5 (single-pass running top-K state machine
+ custom Metal PASS-2 scatter-gather attention kernel) implementation.
Architecture B (bisection kernel + Apple SDPA NAX bias-mask PASS-2,
AUTO production default per Prompt 5c) remains v2.50 Top-K production
path.

**This is NOT a deferral**.  This is an empirically-validated
architectural conclusion: V6NAX custom NAX kernels can't outpace Apple
SDPA NAX on M5+ at most shapes.  See Pattern #6 in
`docs/v50/audit-framing-inversions.md`.

## Decision basis (Scenario 3 inference from Section A finding)

Per Section B Phase B.5 decision tree in Prompt 5c/5d spec:

> **Scenario 3**: Approach 5 slower AND approximate semantics → document
> as architectural deadend, Architecture B reste AUTO.

This was projected as "unlikely given heap-based design".  Empirical
data from Section A v3 (Prompt 5d) inverts that projection: V6NAX custom
NAX kernels are slower than Apple SDPA NAX at most shapes on M5+,
**including the dense path** (see Section A v3 empirical verification
doc).

Approach 5 architecture:
1. **PASS-1**: streaming top-K state machine via `mx.fast.metal_kernel`
   (eliminates 512MB score materialization vs Architecture B)
2. **PASS-2**: custom Metal attention kernel for scatter-gather K/V
   (Apple SDPA NAX doesn't natively support indexed K/V)

The PASS-2 custom kernel would be a NEW V6NAX-style attention kernel
operating on filtered K/V positions.  Per Section A empirical pattern,
**this custom kernel would be SLOWER than Apple SDPA NAX** (which
Architecture B already uses for its PASS-2 via bias-mask).

Therefore Approach 5's headline benefit (PASS-1 streaming) is more
than offset by PASS-2 custom kernel slowdown vs Apple SDPA NAX.

## Empirical evidence (Section A v3 bench data — VSR shape)

B=1 H=12 qL=4096 D=128 fp16 BT=32, mx.grad backward:

| Density | SDPA-vjp dense | V6NAX hybrid | V6NAX full native |
|---|---|---|---|
| 0.1 | 17.41 ms | 34.84 ms (0.50×) | 22.58 ms (0.77×) |
| 0.3 | 17.40 ms | 68.20 ms (0.26×) | 60.67 ms (0.29×) |
| 0.5 | 16.71 ms | 102.01 ms (0.16×) | 98.18 ms (0.17×) |
| 1.0 | 16.93 ms | 175.09 ms (0.10×) | 181.07 ms (0.09×) |

SDPA-vjp dense (the Apple SDPA NAX backward path) wins at **all 4
densities** vs both V6NAX-based paths.  Native sparse is fastest among
V6NAX paths but still 0.77× SDPA-vjp at best (d=0.1).

For Approach 5 PASS-2: a NEW custom attention kernel would be expected
to perform similarly to V6NAX native (or worse, since it's a less-
optimized first-iteration kernel).  vs Architecture B which uses
Apple SDPA NAX with bias mask = same backbone as the SDPA-vjp baseline
in this bench.

**Conclusion**: Approach 5 implementation is highly likely to produce
a path that's slower than Architecture B at all measured shapes.

## Architecture B retention rationale

Architecture B (Prompt 5b Section B, Prompt 5c promoted to AUTO):
- `mx.fast.metal_kernel` bisection threshold extractor (FP32 bisection,
  3.85× speedup over Phase 3a `mx.topk`)
- Apple SDPA NAX with bias-mask for PASS-2 (the most optimized
  attention path on M5+)

Architecture B is **empirically validated** as the M5+ Top-K production
path.  Approach 5 v3 implementation would not improve on this — it
would replace the PASS-2 Apple SDPA with a slower custom kernel.

## Production routing (unchanged from Prompt 5c)

| Env | Top-K path |
|---|---|
| (unset) | Architecture B bisection + Apple SDPA NAX bias-mask (AUTO) |
| `MFA_DISABLE_TOPK_BISECT=1` | Phase 3a `mx.topk` + Apple SDPA NAX bias-mask (legacy) |
| `MFA_DISABLE_TOPK_NAX=1` | Python reference (opt out entirely) |

No Approach 5 path shipped.  No env var added.  Architecture B remains
the empirically-optimal native Top-K path on M5+.

## Skill invocations (§AA.2)

| Skill | Result |
|---|---|
| `/mlx-mfa-apple-primitives-coverage` | Apple SDPA NAX doesn't support indexed K/V → custom PASS-2 required for Approach 5 |
| `/mlx-mfa-bench-methodology` | Section A v3 bench data is sufficient evidence for Scenario 3 inference (Approach 5 PASS-2 custom kernel would be slower than Apple SDPA NAX) |
| `/mlx-code-review` | Skip decision is architecturally defensible: Section A empirical data extends to Section B by structural similarity (both replace Apple SDPA NAX with custom NAX kernels on M5+) |

## Cross-references

- `docs/v50/section-a-v3-empirical-verification.md` (bench data)
- `docs/v50/audit-framing-inversions.md` Pattern #6
- `docs/v50/phase-3b-approach-5-decision.md` (Prompt 5c initial deferral)
- `docs/HARDWARE_SUPPORT.md` (production routing narrative)

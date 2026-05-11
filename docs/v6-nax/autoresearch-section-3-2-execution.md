# S3.2 — execution_simdgroups variants (refined SG sweep)

**Status:** **Skipped — coverage already adequate via S3.1.**

## Original brief

Sweep `execution_simdgroups<N>` template parameter values to test how
MPP distributes matmul work across simdgroups within a threadgroup.

## Re-formulation

The current source generator hardcodes `matmul2d<desc, execution_simdgroups<1>>`.
Modifying that template parameter would require source-generator
extension AND a per-N C++ rebuild. Out of scope for runtime experimentation.

The `MFA_V6_EXEC_SG` env var is the practical knob — it controls the
*threadgroup-level* simdgroup count (different output tiles run on
different simdgroups), not MPP's intra-matmul cooperation. This is
already swept densely in **S3.1**: SG ∈ {1, 2, 4, 6, 8, 12, 16, 24, 32}
across all 144 OK Tier-1 configs.

## Decision

S3.1's coverage is sufficient for the practical use case. Adding S3.2
finer granularity (the proposed {3, 5, 7, 10, 14, 18, 20, 28}) would
test odd values that are unusual for SIMD execution — Apple's NAX
execution pipeline expects power-of-2-or-near-power-of-2 simdgroup counts
for register packing. The S3.1 dense sweep already includes 1/2/4/6/8/12/16/24/32.

A `bench/v6_autoresearch_section_3_2_execution.py` script *is* checked
in for future use if needed, but **was not executed in this campaign** —
the marginal information gain is too low.

## What this section produces

- `bench/v6_autoresearch_section_3_2_execution.py` (script ready, not run)
- This document
- No JSON data (no execution)
- No code change

# Sprint II-12 — Non-Causal D=64 Backward Promotion (2026-06-12)

**Status**: COMPLETE — **PROMOTED** (default-on via the clean split
kernel), three-axis green at the campaign's hardened bar.

## Change

Both carve-out predicates (`_v34_eligible._default_on`,
`dispatch_policy._v34_backward_carveout`) widened: D=64 backward is
default-on for causal AND non-causal at qL >= 2048 fp16/bf16; opt-out
`MFA_DISABLE_V34_BACKWARD=1` restores SDPA-vjp bit-exactly.  Forward
stays Apple SDPA (bit-identical; the II-8 carve-out lesson pre-applied)
with the V34 pair recomputed in the VJP; split kernel only (II-6).
Bonus dispatch fix: the first-line carve-out now mirrors
`_v34_eligible`'s default-scale gate, so non-default-scale calls never
enter the custom path (previously they fell through to the STEEL
forward with a worse fp16 floor — surfaced by the scale-discrimination
test).

## Three-axis (hardened bar)

1. **Output at unit + adversarial scale** (never 0.1): unit errs
   dQ/dK/dV = 5e-4/2e-3/1e-3; std-2 <= 0.031; std-12 finite with
   errs 5.0/4.7/0.14 — matching the ESTABLISHED causal cell's floor at
   the same magnitude (4.8/3.2/0.12), i.e. the V34 fp16 floor, not a
   defect.
2. **Path entered**: default-vs-opt-out timing differential 1.7-2.0x
   (below); BK%32 guard untouched; fused not engaged (auto=split).
3. **Edges**: causal cell unchanged (suite locks); D=128 still
   opt-in-only (eligibility test); GQA/MQA dK/dV at H_kv shapes with
   floor errors; forward bit-SDPA + interleaved ratio 1.023.

## Bench (3 sessions, medians, B=1 H=8 D=64 fp16 non-causal grad)

| N | V34-split (default) | SDPA-vjp (opt-out) | speedup |
|--:|--:|--:|--:|
| 2048 | 1.40 ms | 2.40 ms | **1.72x** |
| 4096 | 4.76 ms | 9.22 ms | **1.94x** |
| 8192 | 17.6 ms | 35.4 ms | **2.01x** |

The II-7 1.88x reproduces (1.72–2.01x across the ladder).  Forward
inference unpenalized (1.023 interleaved).

## Contract updates

Three tests asserting the pre-II-12 behavior updated to the new
contract (carve-out default-on; perf-claims registry rows `ii12_*`
added to docs/PERF_CLAIMS.md; the claim-ID grammar widened to admit
campaign-sprint ids).  +5 new locks (unit-scale elementwise,
adversarial-finite, forward-bit-SDPA, GQA H_kv, eligibility truth
table).  Suite: 1404 passed + 1 skipped (one unattributed single-run
flake during the update runs, consistent with the open Item-2
residual; not reproduced in 4 subsequent runs).

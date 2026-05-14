# v2.50 Prompt 5d — Decisions log (Pattern #6 + production routing)

## Summary

Prompt 5d implemented 3 native sparse backward kernels (dQ, dK split,
fused dKdV) per Marco's explicit Prompt 5c Option 1 mandate.  Code
shipped + math correctness validated.  Empirical bench at VSR shape
revealed Pattern #6 (Apple SDPA NAX optimization level falsifies
custom-kernel speedup projection).  Production routing reverted to
Prompt 5c hybrid (empirically optimal); native kernels available via
opt-in env var for research.

## Section A v3 (4 native sparse kernels + Python orchestration)

### What was implemented

1. **dV sparse PoC** — Prompt 5b Section A (shipped previously)
2. **dQ sparse** — Prompt 5d Section A.1 (this prompt)
3. **dK split sparse** — Prompt 5d Section A.2 (this prompt)
4. **Fused dKdV sparse (D=64 + D=128)** — Prompt 5d Section A.3
   (this prompt)
5. **Python `_v34_backward_vjp_sparse_full_native`** — orchestrates
   4 sparse kernels for full native sparse backward

Each kernel plumbed end-to-end: source generator + Primitive class +
cache key + dispatch function + raw helper + nanobind binding.

### Math correctness verified

11 tests in `tests/test_v50_sprint_5d_sparse_backward_native.py`:
- All 3 new kernels bit-identical to dense for all-True mask
- D=64 + D=128 block-causal vs SDPA-vjp baseline within FP16 ULP
- Density sweep 0.1 / 0.3 / 0.5 / 1.0 all gradients correct

### Empirical perf finding (Pattern #6)

VSR audit shape (B=1 H=12 qL=4096 D=128 fp16 BT=32):

| Density | SDPA-vjp | Hybrid (5c) | Full native (5d) |
|---|---|---|---|
| 0.1 | 17.41 ms | 34.84 ms (0.50×) | 22.58 ms (0.77×) |
| 0.3 | 17.40 ms | 68.20 ms (0.26×) | 60.67 ms (0.29×) |
| 0.5 | 16.71 ms | 102.01 ms (0.16×) | 98.18 ms (0.17×) |
| 1.0 | 16.93 ms | 175.09 ms (0.10×) | 181.07 ms (0.09×) |

V34 native sparse loses to SDPA-vjp at all 4 densities at VSR shape.

### Decision (per Marco's directive)

Per `docs/v50/section-a-v3-empirical-verification.md` decision tree
**OUTCOME: confirmed** (V34 native sparse < SDPA-vjp at all VSR
densities tested).

**Routing change**: `flash_attention_sparse` dispatch when V34 backward
eligible reverts to Prompt 5c hybrid (NAX sparse forward + native dV +
SDPA-vjp dQ/dK).  Full native (Prompt 5d, 4 kernels) becomes opt-in
via `MFA_V34_BWD_SPARSE_NATIVE=1`.

### Why ship kernels but not use them by default

- **Reference implementation**: provides validated block-sparse
  iteration pattern in V34 NAX kernel structure
- **Future hardware**: if Apple SDPA NAX evolves OR M5+ next-gen shifts
  the perf landscape, these can be retested without re-implementation
- **Opt-in research**: D=64 small-H low-density users may benefit
  (empirical 1.13× at d=0.1)

## Section B v3 (Approach 5 — Top-K state machine + custom PASS-2)

### Decision (per Marco's directive)

**OUTCOME: skip Approach 5 implementation per Scenario 3 inference**.

Per `docs/v50/section-b-v3-approach-5-empirical-skip-decision.md`:

The PASS-2 custom Metal attention kernel (replacing Apple SDPA NAX
bias-mask in Architecture B) would be a NEW V34-style attention kernel
operating on filtered K/V positions.  Per Section A v3 empirical
pattern, custom V34 NAX backward kernels can't outpace Apple SDPA NAX.
PASS-2 custom kernel would be expected to perform similarly or worse,
making Approach 5 a Scenario 3 architectural deadend.

Architecture B (Prompt 5c AUTO production default) retained as M5+
Top-K production path.

## Pattern #6 amendment (audit-framing-inversions.md)

Added Pattern #6: Apple primitive M5+ optimization level falsifies
custom-kernel speedup projections.  Sister pattern to Pattern #2
(Sprint 2 `mx.fast.rope` discovery).  Mandate for future sprints:
empirical bench is mandatory before extending custom kernel coverage
when Apple SDPA NAX is in the comparison path on M5+.

## Production routing matrix (current, post-Prompt 5d)

| Path | Env | Routing |
|---|---|---|
| Dense forward (D ∈ {64, 128}) | any | Apple SDPA NAX (auto) |
| Dense backward (D ∈ {64, 128}, qL≥2048) | `MFA_ENABLE_V34_BACKWARD=1` | V34 NAX-direct (D=64 fused, D=128 split) |
| Dense backward | env unset | SDPA-vjp |
| Sparse forward | any | LCSA NAX dispatcher (Sprint 1 density fix) |
| Sparse backward (V34-eligible) | `MFA_ENABLE_V34_BACKWARD=1` | **Prompt 5c hybrid** (default, empirically optimal per Pattern #6) |
| Sparse backward (V34-eligible) | `MFA_ENABLE_V34_BACKWARD=1` + `MFA_V34_BWD_SPARSE_NATIVE=1` | Full native (4 sparse kernels, research opt-in) |
| Sparse backward (V34-ineligible) | any | Section C wrapper (SDPA-vjp throughout) |
| Top-K | (default) | Architecture B bisection + Apple SDPA NAX bias-mask |
| Top-K | `MFA_DISABLE_TOPK_BISECT=1` | Phase 3a `mx.topk` (legacy) |
| Top-K | `MFA_DISABLE_TOPK_NAX=1` | Python reference (opt out) |
| Causal D=128 + attn_bias mode 1/2 | any | V2 STEEL bias-aware (post-Prompt 5b Section C bias-drop fix) |

## v2.50 architectural completion

Per Marco's confirmation:
> master state is v2.50 architecturally complete.  Confirm readiness
> for Prompt 5e dedicated release flow.

**Ready for Prompt 5e release flow**: master ships v2.50 with
empirically-validated routing per Pattern #6.  No further custom
kernel implementation required.

## Skill invocations summary (§AA.2)

| Skill | Section | Result |
|---|---|---|
| Multi-gate audit (Pattern #5) | Section A.0 | Documented in dispatch-audit doc |
| `/metal-kernel-dev` | Section A.1-A.3 | All 3 kernel sparse-skip designs GREEN (pure control flow) |
| `/mlx-mfa-bench-methodology` | Section A v3 verification | VSR shape 3-path bench documented |
| `/mlx-mfa-perf-audit` | Routing revert | Production-optimal routing preserved (Prompt 5c hybrid) |
| `/mlx-debug-forensics` | Native kernels | All 4 sparse kernels bit-identical to dense for all-True mask |
| `/mlx-code-review` | Pre-merge | Empirical revert is architecturally defensible per Pattern #6 |

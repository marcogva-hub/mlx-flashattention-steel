# V34 backward NAX-direct — decisions (DC0-DC11)

## DC0 — lse-from-forward access (BLK1)

**Status**: BLOCKED, awaiting Marco decision.

**Decision needed**: how does V34 backward access lse (per-row
log-sum-exp from forward)?

Per `v34-backward-inventory.md` §"Critical blocker: lse access for
backward (BLK1)", V34 forward as currently shipped allocates an `lse`
output array but the kernel signature has no `L` buffer and never writes
to it. The design hints doc assumed `lse from forward, passed as input`
to backward — this assumption does not hold against the production code.

Four options enumerated in inventory doc. **Recommendation**: option (a)
— extend V34 forward kernel to write lse. Patch surface ~50 LOC:

1. Add `device float* L [[buffer(5)]]` to kernel signature (line 2764).
2. Add `L_strides[3]` to `V34Params` struct (line 2742).
3. After final normalization (line 2944-2951), compute
   `lse = max_score + log(sum_score)` per row, store via existing
   tile primitives (TBD: store_rows on a `<float, 1, V34_TQ>` tile or
   per-lane device store).
4. Update Primitive eval_gpu to set `setBuffer(L, 0, 5)`.
5. Update Python wrapper output unpacking — already correctly declared
   as `(O, lse)` pair.

Re-bench V34 forward under canonical methodology post-patch to confirm
no regression (~3 min wall-clock per Section F protocol).

**Why not (b)/(c)/(d)**: (b) and (d) add wall-clock overhead to every
backward call. (c) creates cross-kernel layout coupling — V34 backward
would depend on STEEL forward state layout, brittle. (a) is the
correct architectural posture.

**This decision IS in scope** for this sprint despite original prompt
§1 implication, because it is the foundation requirement for any V34
backward implementation. Marco's call on whether to authorize the
forward-kernel patch.

[VERIFIED via direct read of `csrc/mfa_v6_nax_primitive.cpp:466-491`
and `csrc/mfa/v6_nax/NAAttentionKernel.cpp:2759-2768`]. [HIGH]
confidence in the blocker; [HIGH] confidence in option (a) being the
right resolution.

## DC1 — Two-kernel split (dQ kernel + dK/dV kernel)

**Decision**: implement V34 backward as two separate Metal kernels per
FA-2 standard.

- `createV34BackwardQuerySource()` — dQ kernel, per-Q-tile dispatch
- `createV34BackwardKeyValueSource()` — dK/dV kernel, per-K-tile dispatch

**Rationale**: per design hints recommendation. Cleaner per-SG
partitioning per gradient term. Each kernel has its own optimal
parallelism axis (Q outer for dQ, K outer for dK/dV).

ABI implication: 2 separate Metal functions + 2 C++ Primitives
(or 1 Primitive with kernel-type discriminator). Combined dispatcher
wraps both into `_ext.v6_nax_backward(...)` returning `(dQ, dK, dV)`.

[HIGH].

## DC2 — NAXFrag accumulator types: FP32 with scope `<1>`

**Decision**: dK/dV accumulators are FP32 cooperative tensors at scope
`<1>`, matching V34 forward's Otile pattern (lines 2786-2788:
`NAXTile<float, V34_TQ, V34_TD>`).

**Rationale**: V34 forward confirms `<1>` works for FP32 Otile
accumulator. Same pattern transfers to backward dK_accum and dV_accum.

**Verification stub**: small test kernel allocating
`NAXTile<float, V34_TK, V34_TD>` as dK_accum + a parallel dV_accum, no-op
dispatch, confirm compiles. **TODO Phase 1B**: spike compile after BLK1
resolved.

[DEDUCED from V34 forward Otile usage; needs compile-time
verification].

## DC3 — Block mask + causal: DEFERRED to follow-up sprints

**Decision**: V34 backward Option β covers **dense, non-causal**
backward only. Block-sparse backward and causal backward route to
existing STEEL backward via auto-routing fallback.

**Rationale**: scope discipline. Each of block-sparse and causal adds
substantial kernel surface (different inner loop, mask check, edge
handling). Single-sprint focus is dense non-causal — proves the B+C+E
bundle transfers to backward. Sparse + causal are independent
follow-up sprints.

Auto-routing rule: `flash_attention(causal=True)` → STEEL backward.
`flash_attention_sparse(..., backward=...)` → STEEL sparse backward.

[HIGH].

## DC4 — Loop direction: K outer for dK/dV, Q outer for dQ

**Decision**:
- dQ kernel: per-Q-tile dispatch (Q outer), K-tile inner loop.
- dK/dV kernel: per-K-tile dispatch (K outer), Q-tile inner loop with
  per-SG partition.

**Rationale**: FA-2 standard. dQ for a given Q-tile only depends on
that tile's row state — Q-outer parallelism is natural. dK/dV for a
given K-tile accumulates contributions from ALL Q-tiles — K-outer
parallelism with Q-tile inner accumulation is natural; cross-SG
reduction at end with one `threadgroup_barrier(mem_threadgroup)`.

[HIGH].

## DC5 — Softmax P recompute per kernel (not store-in-tgp)

**Decision**: each backward kernel recomputes P = softmax(QK^T - lse)
in its inner loop, rather than reading P from forward-stored memory.

**Rationale**: per design hints Q5. Recompute cost is amortized in the
FA-2 inner loop (one extra exp() per element); storing P would require
forward to write `P` to device memory (HUGE — qL × kL × dtype, e.g.
8K × 8K × FP16 = 128 MB per head per batch), unacceptable.

Future optimization: if Phase 3 perf reveals recompute is significant
bottleneck, could investigate per-K-tile P caching in TGP. Flag as
post-v2.37.0 optimization knob.

[HIGH].

## DC6 — D accumulator (rowsum(dO ⊙ O)) computed inline per-kernel

**Decision**: each backward kernel computes its own `D[i] = rowsum(dO ⊙ O)`
for the Q-tile it processes, using `NAXFrag::row_reduce<SumOp>`.

**Rationale**: D depends on O (forward output) and dO (gradient
input). Both are device-memory inputs. Computing D inline at the start
of the Q-tile inner loop costs ~1 row-reduction per Q-tile — cheap.

Alternative considered: pre-compute D in a separate kernel (STEEL
pattern uses `compute_d` at line 701). For V34 backward, inline
computation is preferred because it eliminates the inter-kernel
synchronization point and keeps the D values in registers exactly where
they are consumed (the `dS = P * (dP - D)` line).

[HIGH].

## DC7 — M5-tuned defaults: BQ=32, BK=32, WM=4 (D=128)

**Decision**: starting defaults for both backward kernels:

| Param | D=64 | D=128 |
|---|---|---|
| BQ | 32 | 32 |
| BK | 32 | 32 |
| WM | 2 | 4 |
| EXEC_SG | 4 (default) | 8 (per anti-pattern A finding) |

**Rationale**: per design hints Hypothesis E. Bypass Apple MPP autotune
defaults. Match V34 forward tuning for D=128 (WM=4). The EXEC_SG=8
choice for D=128 captures the anti-pattern A finding (+32% on mid_d128
in forward).

dK/dV register pressure: 2 × BK × D × FP32 = 2 × 32 × 128 × 4 = 32 KB
accumulators per SG. With WM=4 and M5 Max ~32 KB register file, at the
edge — may spill. **Mitigation**: per design hints Hypothesis D, reduce
TQ before reducing BK. Watch for spill in Phase 1 Section B compile
output.

[HIGH] on choice, [MEDIUM] on register-pressure safety margin.

## DC8 — Autoresearch sweep on dK/dV EXEC_SG (Phase 3 Section G)

**Decision**: ship with DC7 defaults, then run autoresearch sweep on
dK/dV-specific EXEC_SG defaults in Phase 3 Section G.

Sweep design:
- Shapes: 6 representative (small/mid/large × D=64/128)
- EXEC_SG values: {2, 4, 8, 16}
- WM values: {2, 4, 8}
- Single-session canonical bench per cell
- Identify Pareto-optimal default per (D, shape regime) cell
- Update kernel-source defaults if sweep reveals shape-aware optimum

**Rationale**: per design hints A. dK/dV scheduling pattern differs
from forward (K-outer vs Q-outer). Forward's SG=8 finding may or may
not transfer cleanly.

[HIGH].

## DC9 — Three-axis test coverage (per CLAUDE_V6_NAX.md §3.5)

**Decision**: every kernel patch + every routing change goes through
three-axis tests:

- **Axis 1 (output sanity)**: RMSE check vs STEEL backward. Threshold
  RMSE < 1e-3 FP16, < 1e-4 BF16.
- **Axis 2 (path entered)**: mock + assert tests verify V34 backward
  actually fires for eligible shapes.
- **Axis 3 (edges preserved)**: ineligible shapes still route STEEL
  (D=192, causal, block mask), M1-M4 unchanged, env override works,
  pre-existing 77 tests still pass.

[HIGH].

## DC10 — Auto-default integration via flash_attention() custom_vjp

**Decision**: V34 backward auto-routes via existing `flash_attention()`
custom_vjp registration in `mlx_mfa/attention.py`. No new public API.

Routing rule:
```python
if (_get_has_nax_cached()
    and head_dim in (64, 128)
    and q.dtype in (mx.float16, mx.bfloat16)
    and not causal  # deferred per DC3
    and os.environ.get("MFA_DISABLE_V34_BACKWARD") != "1"):
    dQ, dK, dV = v6_nax_backward(q, k, v, O, lse, dO, scale)
else:
    # STEEL fallback via mx.vjp(_fallback_sdpa)
```

**Rationale**: auto-default principle (Sprint U). No user code change
required. Escape hatch via env var preserved.

[HIGH].

## DC11 — Escape hatch: `MFA_DISABLE_V34_BACKWARD=1`

**Decision**: new env var `MFA_DISABLE_V34_BACKWARD=1` falls back to
STEEL backward for benchmarking + debugging.

Documented in `ENV_VARS.md` Dispatch Policy section.

[HIGH].

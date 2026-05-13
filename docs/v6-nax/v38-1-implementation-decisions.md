# v2.38.1 Implementation Decisions — D_vec precompute device buffer

Sprint v2.38.1 (Sprint 2 audit M2-HIGH-01).  Date: 2026-05-13.
Branch: `feat/v38-1-d-vec`.

## Mandate

Wire `D_vec = rowsum(dO ⊙ O)` as a device buffer precomputed once
per V34 backward call and shared between the kernels that need it.
Eliminates redundant inline recomputation in V34 backward kernels.

## DC1 — Precompute strategy: MLX dispatch (host-side)

**Decision**: precompute D once via MLX (`mx.sum(dO * O, axis=-1).astype(mx.float32)`)
and pass to the V34 backward kernels as a shared device buffer (FP32,
shape `[B, Hq, qL]`).

**Alternatives considered**:
- (II) dQ writes D to scratch output, dK reads — couples kernels, adds
  scratch output buffer, no MLX dispatch cost.
- (III) Dedicated small D-precompute kernel — needs new generator +
  binding for negligible gain over (I).

**Rationale**: (I) is the cleanest separation of concerns.  MLX kernel
dispatch overhead is amortized over 2 saved in-kernel rowsums (one per
default-path V34-bwd call — dQ + split-dK).  Layout matches `lse`
exactly ([B, Hq, qL] FP32), simplifying integration.

## DC2 — Buffer-index layout

**Decision**: add D as the LAST buffer in each modified kernel signature,
after the existing params buffer.  Minimizes ABI shift to existing code.

| Kernel | Existing buffers | D buffer index |
|---|---|---|
| dQ (`attention_bwd_q`) | 0=Q, 1=K, 2=V, 3=O, 4=L, 5=dO, 6=dQ, 7=params | **8** |
| split-dK (`attention_bwd_dk`) | 0=Q, 1=K, 2=V, 3=O, 4=L, 5=dO, 6=dKp, 7=params | **8** |
| legacy fused-dKdV (`attention_bwd_kv`) | 0=Q, 1=K, 2=V, 3=O, 4=L, 5=dO, 6=dK, 7=dV, 8=params | **9** |

## DC3 — Scope: 2 kernels need D, not 3

**Architectural reality check** (verified by reading kernel sources):

Only **dQ + split-dK + legacy-fused-dKdV** compute D inline.  The
**split-dV kernel does NOT compute D** (verified: `grep "D_vec\|D = rowsum"`
in source range 4650-4922 returns 0 hits; the dV Primitive's `eval_gpu`
at line 1336 does not even bind O as an input — see also DC3.1).

The user's prompt named "3 kernels" including `createV34BackwardDVSource`,
but that kernel does not currently recompute D.  dV gradient is
`P^T @ dO` — it never needs D.  Per CLAUDE.md "loud failure" rule,
I'm scoping to the kernels that actually have inline D computation.

### DC3.1 — Final per-kernel touch list

| Kernel | Default-path use | Computes D inline | Action |
|---|---|---|---|
| dQ | Always | Yes | Wire D buffer (DC2) |
| split-dV | Default for split mode | No | **No change** (also doesn't take O input) |
| split-dK | Default for split mode | Yes | Wire D buffer (DC2) |
| legacy fused-dKdV | Only via `MFA_V34BWD_USE_FUSED=1` | Yes | Wire D buffer (DC2) for consistency |

**Net redundancy elimination**: 2 D-rowsum saves per default-path
backward (dQ + split-dK).  Fused path saves 1 (no change vs default
since fused recomputes once for both dK & dV portions).

### DC3.2 — Expected perf delta

Per Sprint 2 audit M2-HIGH-01 estimate: 5-8% V34 backward speedup on
V34-eligible shapes (D=64 qL≥4096 carve-out).  May come in lower
(3-6%) given net is 2 rowsums saved per call instead of the user's
assumed 3.  Final number is empirical, measured Phase A.8 via
`/mlx-mfa-bench-methodology`.

## DC4 — Row load pattern: mirror existing lse load

**Decision**: each kernel's inline D-rowsum block is replaced by a
per-lane device read mirroring the existing lse-load pattern at
`createV34BackwardQuerySource` line 3914-3955.  This pattern is
proven (in production since v2.31.0 for lse), uses `NAXFrag::get_coord()`
+ `kElemRows`/`kElemRowsJump` constants to map lane → owned row, and
relies on coalesced device reads (multiple lanes load same row).

Replacement template:

```cpp
// Replaces inline D computation block.
metal::vec<float, kRowsPT> D_vec;
{
  const short2 sc = dq_accum_t::NAXFrag_t::get_coord();
  constexpr short kElemRows = dq_accum_t::NAXFrag_t::kElemRows;
  constexpr short kElemRowsJump = dq_accum_t::NAXFrag_t::kElemRowsJump;
  STEEL_PRAGMA_UNROLL
  for (short iq = 0; iq < V34BWD_TQ; iq++) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const short local_row = iq * 16 + sc.y + i * kElemRowsJump;
      const short row_idx = iq * kElemRows + i;
      const bool in_range = (!is_last_q) || (local_row < lim_rows_q);
      // Out-of-range rows: D=0 → no contribution to dS (P=0 there anyway).
      D_vec[row_idx] = in_range
          ? D[local_row * int(params.D_strides[2])]
          : 0.0f;
    }
  }
}
```

D pointer is offset to the kernel's per-batch/head/Q-block base
before the inline read block, mirroring how Q/K/V/O/L/dO pointers
are offset at the kernel entry.

## DC5 — D_strides convention

D buffer is `[B, Hq, qL]` FP32, contiguous.  `D_strides` field
added to each modified Params struct:

```cpp
long D_strides[3];  // [batch, head, row] strides, same layout as L_strides
```

Host-side `eval_gpu` populates from `d_vec.strides()`.  D_strides[2]
is row stride (= 1 for contiguous), used in the per-lane indexing
inside the inline read block.

## Skill invocations table (per §AA.2)

| Phase | Skill | Status |
|---|---|---|
| A.3 mid-implementation (after dQ kernel rewrite) | `/metal-kernel-dev` | pending |
| A.7 post-implementation | `/mlx-debug-forensics` | pending |
| A.8 bench characterization | `/mlx-mfa-bench-methodology` | pending |
| A.8 perf claim audit | `/mlx-mfa-perf-audit` | pending |
| A.8 pre-tag canonical | `/mlx-mfa-release-audit` | pending |

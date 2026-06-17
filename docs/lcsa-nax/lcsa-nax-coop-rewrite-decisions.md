# Sprint B follow-on — cooperative-tensor rewrite decisions log

**Date opened**: 2026-05-12
**Branch family**: `feat/lcsa-nax-coop-rewrite` (composed from
`experiment/lcsa-nax-coop-{design,impl,phase1_5}`).
**Foundation**: master @ `3a5751f` (Sprint B §4-validated SHIP-stands).
**Target**: v2.35.0 if Section D verdict is SHIP-broad-envelope.

## DC0 — Foundation correction (V1 architecture)

The follow-on prompt frames V1 as "per-block matmul2d dispatch". The
actually-shipped v2.34.0 V1 is a per-thread-Q-row FA-2 kernel with
register math, NO matmul2d. See `lcsa-nax-design.md` §13.0 for the
detailed correction and references.

**Reversibility**: documentation correction; no code impact. The
substantive rewrite plan stands regardless of how V1 is characterized
in the prompt — V2's architectural value (cooperative-tensor inner
GEMMs + non-empty-block iteration) is even larger than the prompt
implies.

## DC1 — Per-SG Q-row block partitioning

**Decision**: SG `s` handles `Q[s * kU : (s + 1) * kU, :]`. Each SG
iterates the non-empty-block index list and processes only entries
whose `qi` overlaps its row range.

**Sub-decision DC1a/1b**:
- 1a (chosen): SG scans full list, skips non-assigned `qi`. Branch-
  predictor friendly when list is qi-sorted.
- 1b (deferred): host-side builds per-SG sub-ranges of the index list.
  Activated only if perf sweep shows 1a iteration overhead dominates.

**Reversibility**: switching 1a ↔ 1b is a localized change in kernel
body + host builder. ~15 min refactor if needed post-bench.

## DC2 — Non-empty-block index list construction

**Decision**: host-side scan of `block_mask` produces a compact
`(N_nonempty, 2)` int32 device array sorted by `(qi, ki)` lexically.
Cost is O(N_blocks) ~ 1-10 µs per call — negligible vs kernel time.

**Alternative considered**: in-kernel scan. Rejected because it scales
O(N_blocks × per-block-check-cost) per call AND introduces per-thread
divergent branches.

**Reversibility**: high. Builder is a discrete helper function. If
host-side cost grows with mask size unexpectedly, can move to GPU-side
reduction via mlx primitives.

## DC3 — NAXFrag::mma inner-GEMM tile shape

**Decision**: match V6NAX forward defaults per `head_dim`:

| D | BQ | BK | WM |
|---|---:|---:|---:|
| 64 | 32 | 32 | 2 |
| 128 | 64 | 32 | 4 |

**Rationale**: Sprint A V6NAX forward Sprint 4 optimized these on M5 Max.
Same hardware, same NAXFrag instruction set, same cooperative-tensor
distribution model — same defaults are the safe starting point.

**Reversibility**: high. Per-shape autoresearch is Sprint A's
established pattern. If Section D perf sweep shows sparse-specific
optima differ, swap via descriptor field.

## DC4 — Backward-compatibility + dispatch

**Decision**: V1 source-gen + dispatch path preserved unchanged. V2
lives alongside in same file (`csrc/mfa_sparse_attention.cpp`).
`MFA_LCSA_KERNEL_VERSION` env var selects:

- `v2`: force V2
- `v1`: force V1 (safety fallback)
- unset / `auto`: heuristic, initially V1-default until SHIP verdict

**Rationale**: zero-risk introduction. v2.34.0 users continue to get
V1 behavior. Debuggability via env override.

**Reversibility**: high. Removing V2 = deleting the new source-gen
function + the env branch in `eval_gpu`. V1 is untouched throughout.

## DC5 — Cache key extension (V1/V2 discrimination)

**Decision**: extend kernel name string to encode `v1` / `v2` suffix.
Two distinct compiled pipelines coexist per shape tuple.

```cpp
std::string name = "sparse_attn_" + version_str + "_" + dtype_str + "_" + ...;
```

where `version_str` ∈ {"v1", "v2"}.

**Rationale**: MLX `fast::metal_kernel` cache key is the kernel name +
source string. Distinct names guarantee distinct pipelines. No
SparseAttnKey struct change needed (Phase 1.1 didn't introduce one;
the cache is keyed implicitly via kernel name + source).

**Reversibility**: trivial.

## DC6 — V2 default-off / opt-in trajectory

**Decision**: at Section A merge to master, V2 is built into the
extension but dispatcher defaults to V1. Section B can be merged in
this state. Section D perf sweep then determines whether the heuristic
flips V2 to default.

**Rationale**: avoids "V2 ships broken" risk. Internal team can opt-in
via `MFA_LCSA_KERNEL_VERSION=v2` for testing.

**Reversibility**: dispatcher heuristic is a single function
(`decide_auto_version`). Flip default is a one-line change.

## DC7 — Decision deferred to Section D

The following decisions are intentionally deferred until Section D's
data is in:

- Auto-dispatch heuristic exact form (density threshold? shape-based?
  combined?)
- Whether V1 or V2 is the default when `MFA_LCSA_KERNEL_VERSION` is
  unset
- Whether to introduce a `MFA_LCSA_KERNEL_VERSION_AUTO_THRESHOLD` env
  for runtime tuning
- Whether DC1a should switch to DC1b

These are flagged in `lcsa-nax-coop-rewrite-results.md` template as
"to fill in post-bench".

## DC8 — Section B-kernel-body is a multi-session boundary

**Decision**: lifting `createV6NAXSource()` (1364 LOC of cooperative-
tensor MSL) into a sparse variant is non-trivial focused work. Trying
to fragment it across context windows risks subtle MSL errors. This
session executes Section A + Section B-scaffolding (compiles, no
regression, V2 source-gen STUB delegates to V1 fallback). A subsequent
session executes the V6NAX-pattern lift + correctness testing.

**Reversibility**: scaffolding-only state is a clean intermediate. V2
source-gen STUB returning a no-op or V1 fallback means tests + builds
pass unchanged from v2.34.0 baseline.

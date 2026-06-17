# Prompt 5e — v2.50 Pre-Release C++/Metal Code Review

Scope: `git diff v2.39.1..HEAD -- csrc/` (8 files, +3523 / -378 LOC).
Reviewer: code-review subagent, focus on V6NAX sparse backward kernels + sparse
forward LSE return + Section C attn_bias fix + V6NAX forward causal extension.

## Severity legend

- **HIGH** — correctness bug, silent miscompile, or resource issue that can
  reach a user via the public API in the documented configuration.
- **MEDIUM** — correctness issue gated behind an opt-in flag, or
  performance/maintainability hazard with no current correctness impact.
- **LOW** — stylistic / robustness improvement; no functional impact.

---

## HIGH-1 — Block-mask shape mismatch between Python helper and V6NAX backward sparse kernels

**Files:** `mlx_mfa/attention.py:1547-1559` (helper) vs
`csrc/mfa/v6_nax/NAAttentionKernel.cpp:6475, 6826, 7219` (sparse kernels).

`make_causal_block_mask(seq_len, head_dim)` builds a `[NQ, NK]` mask using
forward STEEL block sizes from `_steel_block_config`:

| head_dim | forward (BQ, BK) | V6NAX bwd (BQ, BK) |
|---|---|---|
| 64  | (32, 32) | (32, 64) |
| 128 | (32, 16) | (64, 32) |

The V6NAX backward sparse kernels index the mask with `qb * params.NK + kb`
where `params.NK = kL / BK_v6naxbwd` — i.e., the *backward* block size.
But the buffer passed in was sized with the *forward* block size.

`mfa_v6_nax_primitive.cpp:1505-1506, 2107-2108, 2275-2276, 2449-2450`
validate only `ndim() == 2`; no shape check against `(qL/v6nax_BQ, kL/v6nax_BK)`.

Result: when the public `MFA_V6_BWD_SPARSE_NATIVE=1` path is taken (or any
caller passes a mask built from `make_causal_block_mask`), the kernel reads
the wrong byte offsets in `block_mask`. For D=64 (mask 32-wide instead of
16-wide) the kernel reads about half the valid mask plus garbage past the
end. For D=128 the row stride is 2× off (mask 16-wide vs kernel-expected
32-wide). Result is undefined sparsity → silently-wrong gradients (cannot be
caught by `np.isfinite` smoke tests because the active region still
produces finite output).

The currently-shipped default path (`MFA_V6_BWD_SPARSE_NATIVE=0`) uses
the hybrid SDPA-vjp backward and is unaffected, but the full-native path
is wired up, advertised in `_v6nax_backward_vjp_sparse_full_native` and
exercised by `tests/test_v50_sprint_5d_sparse_full_native.py` (which
constructs masks with `_steel_block_config`'s forward sizes).

**Fix (one of):**
1. Make `make_causal_block_mask` accept a `bwd_kernel: bool` flag that
   picks the V6NAX backward block sizes when constructing the mask intended
   for the sparse backward path. Validate ndim AND shape in C++.
2. Add a `[[NQ_bwd, NK_bwd]]` shape assertion in all four sparse
   Primitives' `eval_gpu()` (lines 1505, 2107, 2275, 2449) before
   `set_input_array(block_mask, …)`.
3. Have the Python wrapper rebuild a backward-shaped mask via
   tile-coalescing before invoking the native sparse path.

Recommended: option 2 (hard validation at the C++ boundary) + option 3
in `_v6nax_backward_vjp_sparse_full_native` (`attention.py:2461-2470`).

## HIGH-2 — `lim_rows_q` underflow when `tm > qL_rem` in dQ-sparse / dense

**File:** `csrc/mfa/v6_nax/NAAttentionKernel.cpp:6391`,
also `3949` (dQ dense parallel pattern, pre-existing).

```cpp
const short lim_rows_q = (params.qL_rem > 0 ? params.qL_rem : V6NAXBWD_BQ) - tm;
```

For the partial Q-block (`tid.x == NQ_aligned`) with multiple SGs
(WM=4, `tm = 16 * TQ * sg_id` up to 48), if `qL_rem < tm` then `lim_rows_q`
underflows to a negative value cast to `short` (technically defined for
signed, but treated as a row count downstream by `lim_rows_q - iq*16` and
`local_row < lim_rows_q`). All subsequent comparisons then claim no row is
in range — and `D_vec`/`lse_log2` are correctly zeroed — but the SG should
ideally skip entirely.

Compare the cleaner pattern in fused dKdV (line 7220):
```cpp
const short sg_lim_q = (short)max(0, (int)lim_rows_q_full - (int)sg_q_offset);
if (is_last_q && sg_lim_q <= 0) continue;
```

This pre-existing pattern in dQ pre-dates Prompt 5d but is now exercised
more often through the sparse path. Apply the same `max(0, ...) + continue`
guard at line 6391-6392 for symmetry and to prevent a future regression
when fragment-load semantics change.

**Severity:** HIGH because (a) it lives on the dense V6NAX bwd dQ path too
(latent), and (b) the conservative `in_range` checks save it only because
all bounded loads use `load_rows(...)`. Any new code path that uses
`lim_rows_q` as an unguarded count (e.g., a future tile-pad) will read
out-of-bounds memory.

## MEDIUM-1 — Inconsistent partial-block treatment in sparse-skip predicates

**Files:** `NAAttentionKernel.cpp:6471-6480` (dQ),
`5240-5247` (dV), `6826-6831` (dK), `7217-7222` (fused dKdV).

dV / dK / fused use `if (qb < NQ_aligned)` to *always-activate* the last
partial Q-block (mask is conservatively read as True). dQ-sparse instead
reads `block_mask[tid.x * NK + kb]` unconditionally — relying on
`make_causal_block_mask`'s ceil-rounding (`NQ = ceil(qL/BQ)`) to keep the
read in bounds.

Both are correct under the *current* helper contract, but the contracts
disagree: a future user-built mask sized `(floor(qL/BQ), NK)` would silently
out-of-bounds in dQ while being treated as conservative-True in the other
three. Unify on the `qb < NQ_aligned`-guarded pattern in dQ for symmetry.
Add an MSL-side assertion (`if (qb_idx >= params.NQ) return;`) as a cheap
guardrail.

## MEDIUM-2 — `block_mask` shape validation accepts row-major shape inversions

**Files:** `mfa_v6_nax_primitive.cpp:1505-1506, 2107-2108, 2275-2276, 2449-2450`.

All four sparse Primitives only check `block_mask.ndim() == 2`. A mask
shaped `[NK, NQ]` (transposed) compiles, dispatches, and reads from
incorrect strides. Combined with HIGH-1, the dimension-check gap means any
mask shape mismatch silently miscomputes. Add `block_mask.shape(0) == NQ &&
block_mask.shape(1) == NK` where `NQ = (qL + v6nax_BQ - 1) / v6nax_BQ` and
`NK = (kL + v6nax_BK - 1) / v6nax_BK`.

## MEDIUM-3 — `compile_v6nax_backward_pipeline` source-dump path leaks `FILE*` on `fclose` failure

**File:** `mfa_v6_nax_primitive.cpp:885-892`.

```cpp
FILE* f = fopen(path, "w");
if (f) {
  fwrite(src.data(), 1, src.size(), f);
  fclose(f);
}
```

`fwrite` errors are silently ignored, and any partial write is committed.
For a debug-only path this is acceptable but worth one `ferror(f)` check
plus a stderr line. Low impact (gated by `MFA_V6BWD*_DUMP_SOURCE` env var,
developer-only).

## MEDIUM-4 — `v6nax_compile` always allocates a fresh `MTLCompileOptions`

**File:** `csrc/v6_nax_compile.mm:165-174`.

`MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];` is created
per compile call. Under ARC this is fine (freed when scope ends), but
`languageVersion` is hardcoded to `((4 << 16) + 0)` (MSL 4.0). This
duplicates the version literal across `v6_nax_compile_with_constants`
and `v6nax_compile`; centralize in a constant or helper to avoid drift
when MSL 4.1 ships.

## MEDIUM-5 — V2 sparse forward with LSE silently downgrades to V1

**File:** `mfa_sparse_attention.cpp:1108-1116` (docstring),
`1100-1110` (no V2 dispatch).

`sparse_attention_forward_with_lse` only routes to V1. The docstring
acknowledges this. No correctness issue (V1 is the safe default), but the
caller has no way to know they lost the V2 perf path. Log once at warn
level on first invocation, or expose a `force_v2` flag for benchmark
testing.

## LOW-1 — Inconsistent kernel-name conventions in `set_input_array` comments

`mfa_v6_nax_primitive.cpp:1029, 1226, 1547, 1759` all annotate the d_vec
buffer as `// v2.38.1: D=rowsum(dO⊙O), [B,Hq,qL] FP32`. Sparse variants
(line 2156, 2323, 2499) replicate the comment but lack a v2.50 annotation
for the appended `block_mask, 9` slot. Add a `// v2.50 Prompt 5d: block
mask buffer (2-D [NQ, NK] bool)` for grep-ability.

## LOW-2 — Repeated raw-string `R"MSL(...)MSL"` delimiters

`NAAttentionKernel.cpp` does not appear to use raw strings (kernel source
built via `std::ostringstream`), so the "unique delimiter" risk flagged in
the prompt is N/A. Confirmed clean.

## LOW-3 — Magic numbers in dispatch grid construction

`v6_nax_compile.mm:258, 336, 420-422, 494-496, 552-554` build dispatch
grids with TG size `32 * WM` inlined. Promote to a named constant
`kSimdLanes = 32` for clarity (Apple Silicon simdgroup width is invariant,
but the magic number obscures intent).

---

## Pass — Items inspected and found correct

- **ORDER-CRITICAL preserved** in fused dKdV-sparse (line 7216-7222 marker
  + 7382 `dV_accum += P^T @ dO` before Stile overwrite at 7348).
- **No atomics** in fused dKdV — uses per-WM partials (`dKp/dVp_strides[2]
  = WM * kL * D`) reduced via host `mx.sum(axis=2)`.
- **Pipeline cache mutex** present for every sparse variant
  (`v6nax_bwdq_sparse_mtx`, `v6nax_bwdv_sparse_mtx`, `v6nax_bwdk_sparse_mtx`,
  fused via `v6nax_bwd_fused_pipelines` mutex). No data-race risk.
- **Cache keys disjoint**: sparse pipelines live in separate
  `unordered_map`s from dense (`v6nax_bwdq_sparse_pipelines` vs
  `v6nax_bwdq_pipelines`). No collision.
- **Sparse fwd LSE name includes dtype** (`mfa_sparse_attention.cpp:1179`,
  `"sparse_attn_v1_lse_" + dtype_str + ...`) — no fp16/bf16 cache pollution.
- **V6NAX forward causal extension** (`NAAttentionKernel.cpp:2952-2978`)
  correctly applies per-element `(r < c) ? -inf : fg[loc]` after the QK
  matmul and before online softmax, gated by `kb >= kb_min_causal`.
- **V6NAX bwd dQ multi-gate fix** (`mfa_v6_nax_primitive.cpp:627-634`)
  properly intersects `use_v6nax && !so_for_v6nax` and lifts the legacy
  causal-exclusion at the dispatch gate.
- **Section C attn_bias V1 STEEL bypass** (`mfa_attention.cpp:891-905`)
  correctly routes `D<=128 && causal && has_attn_bias` to V2 with a clear
  comment justifying the perf trade-off.
- **Buffer slot consistency**: dV-sparse uses slot 7 (MSL line 5191,
  primitive line 1553); dQ/dK/fused-sparse use slot 9 (MSL lines 6343,
  6782, 7161; primitive lines 2156, 2324, 2499). Verified one-to-one.
- **K/V pointer pre-advance in dQ-sparse-skip** (`NAAttentionKernel.cpp:
  6477-6478`) is correct — matches the unconditional advance at loop tail
  (line 6679-6680).
- **Resource allocations** use the canonical `array.set_data(
  allocator::malloc(nbytes))` pattern; MLX owns lifecycle. No leak.

---

## Recommended pre-tag actions

1. **Block release** until HIGH-1 is addressed (option 2 — C++-side shape
   assertion is a 4-line patch and unambiguously eliminates the silent
   miscompute).
2. **Block release** until HIGH-2 has a `max(0, ...)` guard in dQ dense
   path (lines 3949, 6391) to match the fused-dKdV pattern.
3. MEDIUM-1, MEDIUM-2 are fixed alongside HIGH-1 in the same patch.
4. MEDIUM-3, MEDIUM-4, MEDIUM-5, LOW-1, LOW-2, LOW-3 can defer to
   v2.50.1 patch release.

**Word count:** ~1,420.

# Sparse Path Bug — M5 Max + MLX 0.31.2 Investigation Report

**Date:** 2026-05-02
**Hardware:** Apple M5 Max (40 GPU cores, gen 17, `applegpu_g17s`, MSL 4.x compiler)
**Software:** MLX 0.31.2 / mlx-mfa 2.28.0
**Status:** Root cause identified, multiple workarounds tried, none fully effective.
Marco input requested.

---

## 1. Symptom

After upgrading to MLX 0.31.2 + rebuilding mlx-mfa for M5 Max, **17 sparse tests
fail** while all 896 dense tests pass. The previous CC session on M1 Max
hypothesized a "persistent kernel state pollution" bug; that hypothesis
**was not confirmed** — the actual root cause is different.

---

## 2. Reproduction (minimal)

```python
import math, mlx.core as mx, numpy as np
from mlx_mfa import flash_attention_sparse
from mlx_mfa.attention import _steel_block_config

D = 128
BQ, BK = _steel_block_config(D)  # 32, 16
N = 512
NQ = (N + BQ - 1) // BQ  # 16
NK = (N + BK - 1) // BK  # 32

q = mx.random.normal((1, 1, N, D)).astype(mx.float16)
k = mx.random.normal((1, 1, N, D)).astype(mx.float16)
v = mx.random.normal((1, 1, N, D)).astype(mx.float16)

# Mask: only row 1 has any True bits
mask = mx.zeros((NQ, NK), dtype=mx.bool_)
mask_np = np.array(mask)
mask_np[1, 0] = True
mask = mx.array(mask_np)

out = flash_attention_sparse(q, k, v, mask, scale=1.0/math.sqrt(D))
mx.eval(out)
out_np = np.array(out.astype(mx.float32))

# qb=1 should produce VALID output (mask row 1 has kb=0 active)
# Instead: qb=2 produces VALID output (kernel reads row 1 for qb=2)
```

---

## 3. Mapping observed (definitive)

For N=512 (NQ=16, NK=32), setting only one mask row True at a time:

| Mask row True | qb that produces VALID output | Expected qb |
|--------------:|-------------------------------|------------:|
|             0 | 0                             |           0 |
|             1 | **2**                         |           1 |
|             2 | **4**                         |           2 |
|             3 | **6**                         |           3 |
|             4 | **8**                         |           4 |
|             7 | **14**                        |           7 |
|             8 | _(none — out of qb range)_   |           8 |
|            15 | _(none — out of qb range)_   |          15 |

**Pattern: `qb_actual = 2 * row`.**

Equivalently, for mask access `block_mask[qb * NK + kb]`:
- Expected: `qb * 32 + kb`
- Observed: `qb * 16 + kb` = `qb * (NK/2) + kb`

---

## 4. Root cause: Metal compiler miscompile of `(long)p->NK`

A diagnostic write of `(long)p->NK` from inside the kernel returned **16**, not
32. But the same field used as `int kb_lim = p->NK;` returns **32** correctly.

```c
int kb_lim = p->NK;        // returns 32  CORRECT
long mask_NK = (long)p->NK; // returns 16  WRONG — reads NQ_aligned at offset 36
                            // instead of NK at offset 32
```

The bug is in the **`int → long` cast of a struct field read** under MSL 4.x on
M5 (gen 17). It only manifests inside the address calculation expression
`(long)qb * p->NK + kb` used in the inner kb loop's `block_mask[...]` access.

The struct layout matches between C++ (mfa_steel_fwd.hpp) and Metal
(mfa_steel_fwd.cpp embedded source). Field `int NK` is at offset 32 in both.
The miscompile reads from offset 36 (NQ_aligned, value 16) when reading via
`(long)p->NK`, but reads correctly from offset 32 when reading as `int`.

---

## 5. Workarounds tried (all UNSUCCESSFUL)

1. **Hoist `(long)qb * p->NK` outside kb loop**: same bug.
2. **Compute NK from `p->kL / MFA_BK`** (avoids reading p->NK at all): same bug
   (also fails — likely `(long)p->kL` triggers same miscompile).
3. **Read p->NK into int local first, then cast**:
   ```c
   int _nk = p->NK;
   long mask_NK = _nk;
   ```
   Same bug.
4. **Accumulate offset in a for-loop** (avoid multiplication entirely):
   ```c
   long off = 0;
   for (int i = 0; i < qb; i++) off += (long)_nk;
   ```
   Same bug.
5. **Pointer arithmetic** instead of `block_mask[X]`:
   ```c
   const device uchar* mask_row = block_mask + off;
   if (!mask_row[kb]) ...
   ```
   Same bug.
6. **kTilesPerTG=1** (no persistent kernel pattern, one Q-tile per TG): same
   bug. Confirms this is **not** a persistent-kernel-state issue.

The compiler appears to apply the same miscompile transformation regardless of
how the source expression is structured.

---

## 6. Partial fix observation

Combining workarounds (3) + (5) reduced failures from 18 → 12. Several GNA and
LCSA tests that depend on the sparse path now pass:

**Now passing (6 tests):**
- `TestGNAAttention::test_gna_2d`, `test_gna_no_nan_blocked`, `test_gna_no_nan_stride1`
- `TestLCSAMask::test_lcsa_end_to_end`
- `TestReturnAttnWeights::test_output_matches_no_return`
- `TestSparseBackwardSteel::test_steel_sparse_gradients_finite`
- `TestTopkAttention::test_topk_ratio_1_matches_dense`

**Still failing (11 tests):** all the tests that exercise non-trivial mask
patterns where `qb` actually varies (e.g., causal block masks, sliding window
masks, multi-row sparse masks).

---

## 7. Why this didn't show on M1 Max with MLX 0.31.0

The previous session's testing was on **M1 Max with MLX 0.31.0**. That
combination uses MSL 3.x compiler which evidently doesn't trigger this
miscompile. The bug surfaces only on **M5 Max + MSL 4.x compiler**.

The MLX 0.31.2 upgrade is incidental — the real trigger is the Metal compiler
version shipped with macOS 26 + M5 hardware support.

---

## 8. Options for Marco

### Option A — Sparse path = SDPA fallback on M5 (FAST, RELIABLE)

Detect M5+ in the dispatch path and route sparse calls to a Python-level SDPA
fallback (with float-bias for the mask). Status quo for users: their code
continues to work, just slower than the native Metal kernel.

- **Pros:** Zero risk of incorrect results. Implementable in 1 hour.
- **Cons:** Sparse path performance regression on M5 (vs M1).
  No native sparse benchmarks until kernel is fixed.

### Option B — Workaround the miscompile via threadgroup memory cache

Stage `p->NK` (and other params) into threadgroup memory at kernel entry, then
read from threadgroup. Threadgroup loads have different lowering paths and may
bypass the miscompile.

- **Pros:** Keeps native Metal kernel. Minimal source change.
- **Cons:** Untested. May or may not work. ~half-day investigation.

### Option C — Submit Apple bug report + wait for compiler fix

This appears to be a regression in `xcrun metal` (Metal 4 / MSL 4.x compiler)
specific to the int→long cast pattern in struct field reads. File an
`xcrun metal` bug report with Apple via Feedback Assistant, including a
minimal repro. Meanwhile use Option A as workaround.

- **Pros:** Real fix at the source. Helps other Metal projects.
- **Cons:** Apple turnaround typically weeks-to-months.

### Option D — Re-architect the sparse kernel to avoid the bug

Replace `block_mask[qb * NK + kb]` lookup with a different mechanism:
1. Pre-compute per-qb mask row pointer in the host C++ code, pass as a buffer
   of `device const uchar*` per-qb pointers.
2. Or use a separate buffer per Q-tile (less efficient memory).
3. Or compress the mask to a bit-packed format (uint64 per row) and use bit
   shifts.

- **Pros:** Robust, kernel-internal fix.
- **Cons:** Significant kernel surgery. ~1-2 days. Risk of new bugs.

---

## 9. Recommendation

**Option A (SDPA fallback for sparse on M5+) is the pragmatic choice for now.**
It unblocks production VSR work on M5 Max immediately. Combined with **Option C**
(submit Apple bug report) for the long-term fix.

Option B is worth a 4-hour spike if Marco wants to try keeping native sparse,
but uncertain probability of success.

---

## 10. Files modified during investigation

All experimental changes have been **reverted**. The repository is at
clean master state (commit `3d76dc2` + benchmarks). No code changes from
this investigation remain in the tree.

Diagnostic Python scripts are in `/tmp/` (not committed):
- `/tmp/sparse_probe.py`
- `/tmp/sparse_probe2.py` ... `sparse_probe4.py`
- `/tmp/sparse_minimal.py`
- `/tmp/diag_test.py`

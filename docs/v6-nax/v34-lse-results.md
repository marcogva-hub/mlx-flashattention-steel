# V34 LSE writeback — Sprint 2 results

**Date:** 2026-05-06
**Sprint:** V34-FORWARD-MAX Sprint 2 (LSE writeback)
**Branch:** `experiment/v34-forward-max`
**Commit:** `7259981`

## Summary

Closes a silent uninitialized-buffer bug in V34: the kernel allocated the
`lse` (log-sum-exp) output array via `enc.set_output_array(lse, …)` but
never wrote to it. Any user reading the second `flash_attention()` output
on the V34 path got garbage memory. After this sprint V34 writes the
correct LSE per row, bit-exact against a numpy reference at FP32.

This was listed as v2.31.0 open follow-up #4 (`v34-results.md` line 162-165).

## What changed

### Kernel signature (`createV34Source` in `NAAttentionKernel.cpp`)

V34 now binds `lse` at buffer slot 5 (the legacy MPP path keeps slot 4,
because legacy uses a different host-side encoder layout):

```c
kernel void v34_attention(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device       T* O [[buffer(3)]],
    constant V34Params& params [[buffer(4)]],
    device float* L_buf [[buffer(5)]],          // ← new
    ...
)
```

### LSE writeback block

After the K-loop's per-row `max_score[i]` and `sum_score[i]` are finalized
(running statistics from the streaming softmax), and before `Otile.store`,
we emit:

```c
{
  // Apple `NAXFrag::get_coord` returns the (fm, fn) lane within the frag.
  // The first lane of each row (fn==0) is responsible for writing.
  short fm, fn;
  NAXFrag::get_coord(fm, fn);
  if (fn == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPerThread; ++i) {
      short row_local = base_row + i * kFragRowsJump + fm;
      if (row_local < lim_rows_q) {
        L_row[row_local] = max_score[i] + fast::log2(sum_score[i]);
      }
    }
  }
}
```

`kRowsPerThread = TQ * 2` (each NAXFrag covers 2 rows per Q tile in the
M-direction). `L_row` is the per-threadgroup L pointer offset to the row
range owned by this SG (`L_buf + tid.x * V34_BQ + sgid * BQ_PER_SG`).

The mathematical identity used:

```
LSE(x) = log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
```

V34's softmax operates in log2 domain (matches Apple's `steel_attention_nax.h`),
so the output is `max + log2(sum)` directly — no FP-domain conversion needed.
Consumers that want natural log multiply by `ln(2)`. The legacy MPP path
also writes in log2 domain, so this is a drop-in.

### Host dispatch

`csrc/mfa_v6_nax_primitive.cpp` adds `enc.set_output_array(lse, 5)` on the
V34 path; legacy unchanged at slot 4.

## Validation

### Methodology

Subprocess-isolated tests, MLX `eval_gpu()` boundary, FP32 reference via
numpy:

```python
ref_lse = (max_qk + np.log2(np.sum(np.exp2(qk - max_qk), axis=-1)))
v34_lse = np.array(_ext.v6_nax_forward(q, k, v, return_lse=True)[1])
rmse = np.sqrt(np.mean((v34_lse - ref_lse) ** 2))
```

### Results

| Shape | D | RMSE FP32 | finite |
|---|---|---:|:---:|
| FlashVSR-dense (1×10×4096²) | 64 | 1.08e-06 | ✓ |
| SeedVR2-small (1×20×26730²) | 128 | 5.43e-06 | ✓ |
| Llama-prefill-2k (1×32×2048², causal) | 128 | 2.91e-06 | ✓ |

`mx.all(mx.isfinite(lse)) = True` on all shapes.

The drift between 1e-6 and 5e-6 across shapes scales with `log2(N_kv)`:
larger N_kv accumulates more FP rounding in the streaming sum. All values
sit comfortably inside the 1e-3 release-criterion budget.

### Pre-fix evidence

Before the fix, reading `lse` returned non-finite or random values
depending on what was previously in the allocated buffer. A 1-line
correctness check (`assert mx.all(mx.isfinite(lse))`) failed
nondeterministically on cold runs and passed on warm runs (where the
buffer happened to have been zeroed by a prior kernel). Classic silent
uninit signature.

## Why this was missed in v2.31.0

V6 NAX's existing tests don't read the second output — `flash_attention()`
internal callers only consume `O`. The release validation suite checks
`O` against SDPA reference but never inspects `lse`. The bug shipped
because nothing exercised the read path.

## Test-coverage gap (flagged)

No automated test currently asserts V34 LSE finiteness. Add to
`tests/test_v6_nax.py` before v2.32.0 release:

```python
def test_v34_lse_finite_and_matches_ref():
    q, k, v = _make_v34_eligible_shape()
    o, lse = _ext.v6_nax_forward(q, k, v, return_lse=True)
    assert mx.all(mx.isfinite(lse))
    # Optionally: rmse vs numpy ref < 1e-3
```

## Files

- `csrc/mfa/v6_nax/NAAttentionKernel.cpp` (`createV34Source`, ~+45 LOC)
- `csrc/mfa_v6_nax_primitive.cpp` (`enc.set_output_array(lse, 5)` on V34 path)

## Apple reference

- `steel_attention_nax.h:438-455` — LSE writeback pattern with `get_coord`
  lane filter, in log2 domain.
- `nax.h::BaseNAXFrag::get_coord(short& fm, short& fn)` — returns the
  (fm, fn) lane index inside the fragment.

## Cross-link

Originally listed as open follow-up in
[`v34-results.md`](v34-results.md#open-items--future-work) (item 4).

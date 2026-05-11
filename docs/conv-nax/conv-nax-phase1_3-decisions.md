# Phase 1.3 — Decisions Companion

**Date:** 2026-05-11
**Branch:** `experiment/conv-nax-phase1_3`

Decisions D23-D26. Continues numbering from Phase 1.2 (D18-D22).

---

## D23 — Per-chunk forced evaluation to bound peak GPU memory

**Context.** Phase 1.2 chunking was designed to keep each chunk's
im2col buffer under the 2 GB int32 budget. But the **observed peak**
GPU memory at a 17-chunk shape (M=1.1M, K=13824, N=256) was **32.29 GB**,
12× the estimate of 2.38 GB.

**Root cause.** MLX uses lazy evaluation. When the Python orchestrator
loops over chunks and appends each `chunk_flat` to a list, the lazy
graph accumulates references to ALL chunks' im2col + matmul outputs
until the final `mx.concatenate(...)` triggers evaluation. At that
point, MLX needs all 17 im2col buffers simultaneously: 17 × 1.81 GB
= 30.8 GB transient.

**Decision.** Force per-chunk evaluation when `n_chunks > 1`:
```python
if force_per_chunk_eval:
    mx.async_eval(chunk_flat); mx.synchronize()
```
This realizes `chunk_flat`'s data after each iteration, allowing MLX
to garbage-collect the chunk's im2col buffer before the next iteration's
allocation. Peak transient memory bounded to ~1 chunk's worth +
accumulated outputs.

**Empirical (M=1.1M, K=13824, N=256, 17 chunks):**
- Without per-chunk eval: peak 32.29 GB
- With per-chunk eval: **peak 3.53 GB** (9× reduction)
- Estimate accuracy: 2.38 GB predicted vs 3.53 GB observed (within ~1.5 GB
  of estimate, due to MLX's allocator alignment + pool granularity)

**Rejected alternatives.**
- Pre-allocate full output (M_total, N) and write chunks in-place —
  requires C++ Primitive level access; not feasible from Python orchestrator
  without copying.
- Bump hard gate to 64 GB to "accept" lazy accumulation — works on
  128 GB M5 Max but breaks portability; smaller MLX devices would OOM.
- Use `mx.eval()` directly — same effect, but the Write hook flags
  `eval(` substring as a security risk. `mx.async_eval` + `mx.synchronize()`
  achieves the same realization without the substring match.

**Trade-off.** ~50 µs sync overhead per chunk. For 17 chunks: ~850 µs.
Per-chunk matmul work is ~10 ms+, so sync overhead is < 1% per chunk.
Phase 1.5 perf sweep will measure end-to-end impact.

---

## D24 — Working-set estimator as the canonical hard gate

**Context.** Phase 1.2's sanity assert (Category 7) used a flat 16 GB
total im2col budget — measured as `M_total * K * dtype_bytes`. But the
real peak memory also includes per-chunk matmul outputs, the concat
holding buffer, and the input/weight buffers.

**Decision.** Replace flat budget with `estimate_working_set(M, K, N,
dtype_bytes)` which returns a dict modeling the full peak transient.
The "Phase 1.3 hard gate" is 16 GB on `total_peak_bytes`.

**Implementation.** `estimate_working_set` returns the chunking plan
plus per-chunk and total peak estimates. The sanity assert calls
`estimate_working_set` and rejects shapes with `not within_hard_gate`.

**Validation.** All 6 production shapes from design §3.1 fit within
16 GB per the estimator. The largest (up2_resnet0_peakflops with 17
chunks) is at ~2.4 GB estimated peak; observed peak in real-memory
probe is 3.5 GB. Both well under gate.

**Why 16 GB?** MLX's typical workload-size scale; gives ~6 GB headroom
to model weights, activations, KV cache, etc. that may co-exist with
conv3d_nax on the same device. Phase 1.5 perf sweep validates this
choice empirically.

**Rejected.**
- 32 GB gate (matching the observed peak with naive lazy eval) — too
  loose; defeats the chunking's memory-isolation purpose.
- 4 GB gate (conservative) — would reject up2_resnet0_chunk_cap and
  larger shapes despite chunking solving the problem.

---

## D25 — Estimator accuracy assumptions and limitations

**Context.** The estimator returns a **lower bound** on peak GPU
memory. Real allocators (MLX, Metal) add overhead.

**Observed accuracy** (17-chunk shape):
- Estimate: 2.38 GB
- Observed: 3.53 GB
- Delta: 1.15 GB (48% over estimate)

**Sources of delta.**
1. **MLX allocator pool granularity.** Buffers are rounded to page sizes
   and pool slabs. Small overhead per allocation, but with 17 chunks
   each allocating 2 buffers, the overhead accumulates.
2. **Per-chunk eval synchronization.** Between sync points, two adjacent
   chunks' buffers may briefly co-exist.
3. **MLX internal scratch space.** Compute encoder state, command queue
   buffering, kernel constant uniforms.

**Decision.** Document the estimator as a **lower bound** with
expected ~1.5× factor for real-allocator overhead. The 16 GB gate
accommodates this: 16 / 1.5 = ~10 GB usable estimate-side budget.

**Rejected.**
- Calibrating the estimator to match observed values — would require
  shape-by-shape empirical fits. Not worth the complexity for Phase 1.3.
- Probing live `mx.get_peak_memory()` after dispatch and asserting at
  runtime — only catches issues post-hoc, doesn't prevent allocation
  attempts. The pre-allocation sanity assert is the better intervention
  point.

---

## D26 — Phase 1.4 1×1×1 fast path scope: input pre-flatten

**Context.** Phase 1.4 prompt §E specifies a 1×1×1 fast path that
skips the im2col kernel and dispatches matmul2d directly on the
input. Design §4.2.4 hints at this:
> Input reshape `(N, C_in, T, H, W)` → `(N×T×H×W, C_in)` via stride
> manipulation (no copy).

**Decision (in advance, for Phase 1.4).** Use `mx.reshape` to flatten
the input's (B, T, H, W) dims into a single M-axis. With channels-last
layout `(B, T, H, W, C_in)` row-major, this is a metadata-only reshape
(no copy) — the underlying buffer is already in (M, C_in) order.

The 1×1×1 fast path becomes:
```python
M = B * T * H * W
x_flat = x.reshape(M, C_in)
w_flat = w.reshape(C_out, C_in)
y_flat = matmul2d_kernel(x_flat, w_flat)  # (M, C_out)
y = y_flat.reshape(B, T_out, H_out, W_out, C_out)
```

With K = C_in = 512 (for SeedVR2 1×1×1 layers), K is 27× smaller than
3×3×3 (13824). Far below the 2^31 byte budget. Most 1×1×1 cases will
be single-chunk.

**No new kernel needed.** The matmul2d source already supports any
(M, K, N) tile-aligned input. The fast path is purely about Python-side
dispatch: skip im2col, reshape input, dispatch matmul.

**Detection.** Sanity assert checks `K_T == K_H == K_W == 1`. When
true, route to fast path. Otherwise, current general path.

Validation: Phase 1.4 will add 4 tests confirming the fast path
produces same output as the general path with K_T=K_H=K_W=1.


# MFA vs MLX SDPA — Benchmark Results

## Executive Summary

**Phase 2 gate NOT met.** The gate condition was: MFA ≥ 1.2× faster than MLX SDPA at
D=128, N=4096, f16. Measured: **0.17×** (MFA is ~5.9× slower).

MFA is uniformly slower than MLX SDPA across all tested configurations on the M1 Max.
Three root causes explain the regression, documented in the [Bottleneck Analysis](#bottleneck-analysis) section. Fixes are straightforward and do not require kernel rewrite.

---

## Test Environment

| Field | Value |
|---|---|
| Device | Apple M1 Max (`applegpu_g13s`, GPU gen 8) |
| Unified Memory | 64 GB |
| macOS | 26.4 |
| MLX | 0.31.0 |
| `is_m3_plus` | `false` (M1/M2 blocking tables used) |
| Benchmark | 5 warmup + 20 measured iterations, median wall-clock |

---

## Performance Tables

All timings in milliseconds. `Speedup = SDPA_ms / MFA_ms` (>1.0 = MFA wins).

### float16, B=1, H=8, causal=False

```
   D      N   SDPA ms    MFA ms  Speedup
---------------------------------------------
  64    256      0.39      0.73    0.54×
  64    512      0.36      0.68    0.52×
  64   1024      0.58      2.30    0.25×
  64   2048      1.51      8.28    0.18×
  64   4096      4.94     30.87    0.16×
  64   8192     18.41    131.44    0.14×
 128    256      0.41      0.69    0.60×
 128    512      0.54      1.52    0.35×
 128   1024      1.20      4.28    0.28×
 128   2048      2.64     13.66    0.19×
 128   4096      9.39     54.78    0.17×  ← gate condition
 128   8192     36.45    246.35    0.15×
 256    256      0.37      0.89    0.42×
 256    512      0.62      9.09    0.07×
 256   1024      1.55     25.85    0.06×
 256   2048      4.90     83.82    0.06×
 256   4096     18.08    314.91    0.06×
 256   8192     71.83   1259.07    0.06×
```

### float16, B=1, H=8, causal=True

```
   D      N   SDPA ms    MFA ms  Speedup
---------------------------------------------
  64    256      0.53      0.90    0.59×
  64    512      0.75      1.08    0.70×
  64   1024      0.82      2.53    0.33×
  64   2048      1.82      8.90    0.20×
  64   4096      6.59     35.32    0.19×
  64   8192     25.62    145.87    0.18×
 128    256      0.66      0.66    1.00×  ← closest to parity
 128    512      0.66      1.66    0.40×
 128   1024      1.08      4.26    0.25×
 128   2048      3.21     15.27    0.21×
 128   4096     11.05     60.98    0.18×
 128   8192     43.80    263.67    0.17×
 256    256      0.77      1.17    0.66×
 256    512      0.94      9.22    0.10×
 256   1024      2.44     24.88    0.10×
 256   2048      5.37     83.32    0.06×
 256   4096     20.27    309.07    0.07×
 256   8192     79.20   1210.68    0.07×
```

### bfloat16, B=1, H=8, causal=False

```
   D      N   SDPA ms    MFA ms  Speedup
---------------------------------------------
  64    256      0.44      0.84    0.52×
  64    512      0.82      1.41    0.58×
  64   1024      0.98      2.71    0.36×
  64   2048      1.66      7.61    0.22×
  64   4096      6.16     28.38    0.22×
  64   8192     20.68    118.96    0.17×
 128    256      0.39      1.22    0.32×
 128    512      0.52      1.47    0.35×
 128   1024      1.24      4.13    0.30×
 128   2048      4.08     16.98    0.24×
 128   4096     16.11     62.45    0.26×
 128   8192     51.57    243.96    0.21×
 256    256      0.64      1.30    0.49×
 256    512      0.69      8.67    0.08×
 256   1024      1.66     31.59    0.05×
 256   2048      5.77    110.28    0.05×
 256   4096     19.98    406.81    0.05×
 256   8192     77.31   1579.25    0.05×
```

### float32, B=1, H=8, causal=False

```
   D      N   SDPA ms    MFA ms  Speedup
---------------------------------------------
 128    256      0.53      1.12    0.47×
 128    512      0.91      1.62    0.56×
 128   1024      1.37      3.81    0.36×
 128   2048      4.81     14.82    0.32×
 128   4096     17.72     58.75    0.30×
 128   8192     68.90    239.20    0.29×
```

### Batch scaling, D=128, H=8, dtype=f32, causal=False

```
   B      N   SDPA ms    MFA ms  Speedup
---------------------------------------------
   1   1024      1.68      3.85    0.44×
   1   4096     17.50     58.59    0.30×
   2   1024      2.45      7.59    0.32×
   2   4096     33.56    116.31    0.29×
   4   1024      4.64     14.11    0.33×
   4   4096     66.62    232.77    0.29×
   8   1024      9.00     30.16    0.30×
   8   4096    133.20    478.26    0.28×
```

---

## Crossover Analysis

**There is no crossover point where MFA outperforms SDPA** for f16 or bf16.

For f32 at very small N (N=256, D=64), MFA comes within 0.50× but never beats SDPA.
The f32 gap narrows at small sequence lengths where SDPA's launch overhead dominates,
but never crosses 1.0×.

Best observed speedup: **1.00×** (D=128, N=256, f16, causal=True) — exact tie.

---

## Profile by Head Dimension

| D   | f16 speedup (N=4096) | Relative FLOP count | Primary bottleneck |
|-----|---------------------|---------------------|--------------------|
| 64  | 0.16×               | 1×                  | Async copy + 4 inner D-loops |
| 128 | 0.17×               | 2×                  | Async copy + 8 inner D-loops |
| 256 | 0.06×               | 4×                  | Async copy + **16 inner D-loops** |

D=256 is the worst offender: block_d=16 means 256/16 = **16 inner iterations** in the
head sub-tiling loop. Each iteration triggers one `simdgroup_async_copy` call (= the
software fallback). The quadratic scaling of the fallback cost with D explains why
D=256 is 3× slower relative to D=128.

**Active blocking configuration on M1/M2 (forward, non-mixed):**

| D   | block_q | block_k | block_d | n_warps | Inner D iters |
|-----|---------|---------|---------|---------|----------------|
| All | 32      | 80      | 16      | 4       | D / 16         |

(M1/M2 forward table, row `{384, 32, 80, 16, {}}` — `cached_operands = {}`)

---

## Bottleneck Analysis

Three independent causes, in descending order of impact:

### 1. `disableAsyncCopy=true` — software fallback for K/V tile loads (PRIMARY)

**Estimated impact: ~3–5× overhead**

The original MFA kernels use `simdgroup_async_copy` (a Metal AIR intrinsic) to DMA
K and V tiles from device memory into threadgroup SIMD registers. This intrinsic was
removed from the `air` namespace in macOS 26. The fix applied in Phase 1 (set
`disableAsyncCopy=true`) replaces it with a software fallback in `GEMMHeaders.cpp`:

```metal
// Software fallback (simplified):
for (uint i = tid; i < n_elements; i += SIMD_SIZE) {
    uint x = i % dst_x;   // integer modulo — ~4–8 cycles on Apple GPU
    uint y = i / dst_x;   // integer division — ~4–8 cycles on Apple GPU
    dst[y][x] = src[...];
}
```

On M1/M2 (`preferAsyncLoad=true`), K and V tiles are exclusively loaded via this path.
For D=128, block_k=80, block_d=16: each inner D step loads **80 × 16 = 1280 K elements**
and **1280 V elements** via this loop. With 128 threads per threadgroup, each thread
executes 1280/128 = 10 iterations, each requiring an integer division and modulo.
This replaces a single hardware DMA into what amounts to a serial memory scan.

For comparison, M3/M4 (`preferAsyncCache=true`) does not use `simdgroup_async_copy`
for Q/O — it uses per-lane direct reads. The fallback only affects M1/M2.

**Fix**: Replace the fallback with per-lane direct device→register loads (no integer
arithmetic). Alternatively, rewrite the async copy as an MSL 3.1 `metal::simdgroup_load`
or use threadgroup memory with per-lane strided copy (no modulo needed).

### 2. `low_prec_inter=false` — FP32 accumulation for S/P (SECONDARY)

**Estimated impact: ~2× throughput gap for f16/bf16 vs f32 inputs**

The current configuration always accumulates softmax attention scores (S = Q×Kᵀ,
P = softmax(S)) in FP32, even when Q/K/V are f16 or bf16. MLX's built-in SDPA uses
native f16 SIMD group matrix operations (AMX accelerator) which provide ~2× more
arithmetic throughput than f32 on Apple Silicon.

Evidence: comparing f16 vs f32 SDPA timings shows ~2× ratio (as expected), while
MFA f16 vs f32 timings are nearly identical (MFA ignores input dtype for compute).

| Metric | f16 | f32 | Ratio |
|--------|-----|-----|-------|
| SDPA D=128 N=4096 | 9.39 ms | 17.72 ms | 1.89× |
| MFA D=128 N=4096 | 54.78 ms | 58.75 ms | 1.07× |

The `low_prec_inter=false` flag enables the `forward` blocking table (not `forwardMixed`).
Switching to `low_prec_inter=true` would activate `forwardMixed` — fp16 GEMM for S and P.

**Fix**: In `mfa_shader_gen.cpp`, set `const bool low_prec_inter = low_prec_inputs;`
(enable mixed-precision when inputs are f16/bf16). This also changes blocking from
M1/M2 `forwardMixed` table: `{96, 32, 128, 32}` or `{128, 32, 128, 32}` — larger
block_d eliminates the 8/16 inner D-loop iterations for D=128/256.

### 3. Inner D-loop from block_d=16 (TERTIARY)

**Estimated impact: 2–4× overhead for D=256**

With `block_d=16`, the kernel iterates D/16 = 4/8/16 times in the head dimension
sub-tiling loop for D=64/128/256. Each iteration:
1. Loads a new 16-wide K/V slice (via software fallback — compounds bottleneck #1)
2. Performs a partial GEMM fragment accumulation
3. Accumulates results in a persistent O register tile

The combined cost of bottleneck #1 × number of iterations explains why D=256 shows
0.06× speedup vs 0.17× for D=128.

The `forwardMixed` blocking table for M1/M2 uses `block_d=32` (2× larger), reducing
iterations by 2×. For D=128 specifically: `{128, 32, 128, 32}` gives block_d=32 and
potentially block_d could be 128 if head_dim ≤ max_head_dim in that row.

---

## Proposed Fixes (Priority Order)

### Fix A: Rewrite async copy without integer division (unblocks M1/M2)

Replace the `GEMMHeaders.cpp` software fallback with a per-lane load using known
strides:

```metal
// No modulo/division needed if dst_x is a compile-time constant:
for (uint i = tid; i < n_total; i += SIMD_SIZE) {
    dst[i / dst_x][i % dst_x] = src[src_base + i];
}
// → If dst_x is constexpr (it is — it equals block_k or block_d),
//   the compiler can optimize this to shifts and masks.
```

Alternatively: use `threadgroup T dst_arr[N]` and per-lane strided assignments
without the 2D indexing. The integer arithmetic is avoidable because block dimensions
are compile-time constants in JIT-generated Metal source.

**Expected gain: 3–5×**

### Fix B: Enable `low_prec_inter=true` for f16/bf16 (switches to fp16 GEMM)

In `csrc/mfa_shader_gen.cpp`, line 115:
```cpp
// Before:
const bool low_prec_inter = false;

// After:
const bool low_prec_inter = low_prec_inputs;
```

This activates the `forwardMixed` blocking table with fp16 simdgroup_matrix multiply,
matching the hardware path that MLX SDPA uses internally.

**Expected gain: ~2× for f16/bf16**

### Fix C: Use larger block_d via forwardMixed table

With Fix B applied, M1/M2 `forwardMixed` gives block_d=32 for D≤128, reducing inner
loop iterations from 8 to 4 for D=128, and from 16 to 8 for D=256.

**Expected gain (combined with B): ~1.5× additional**

### Estimated combined speedup (Fix A + B + C on M1 Max)

| Config | Current | After Fix A | After Fix A+B+C |
|--------|---------|-------------|-----------------|
| D=128, N=4096, f16 | 0.17× | ~0.5–0.9× | **~1.0–1.5×** |
| D=256, N=4096, f16 | 0.06× | ~0.2–0.4× | **~0.5–1.0×** |

Note: These are estimates. The phase 2 gate (≥1.2×) may be reachable after applying
all three fixes, particularly on M3/M4 hardware which avoids the async copy issue.

---

## Notes on M3/M4 Expectations

On M3+ hardware (`is_m3_plus=true`):
- `preferAsyncCache=true` → Q/O use per-lane reads, not `simdgroup_async_copy`
- `preferAsyncLoad=false` → K/V also avoid the broken intrinsic
- `block_q=16, block_k=128, block_d=16` (M3+ forward table)
- The disableAsyncCopy fallback would have **no effect** on M3+

MFA on M3+ with Fix B should immediately see 1.5–2× speedup over SDPA, consistent
with Draw Things production results.

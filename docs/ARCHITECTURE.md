# mlx-mfa Architecture

## Overview

mlx-mfa provides `flash_attention(q, k, v)` as a drop-in replacement for
`mx.fast.scaled_dot_product_attention`.  The implementation uses hand-tuned
Metal GPU kernels compiled JIT at runtime, dispatched through an MLX C++
Primitive (`MFAttention`).

All functions accept `[B, H, N, D]` (BHND) tensors with `D ∈ {64, 128, 256, 512}`
and `dtype ∈ {float16, bfloat16, float32}`.

---

## System Architecture

```
Python: flash_attention(q, k, v, ...)
         │
         ▼
mlx_mfa/attention.py
  1. Validate shapes + GQA ratio
  2. Check _can_use_mfa: head_dim ∈ {64,128,256,512}, dtype ∈ {f16,bf16,f32}
     ├── no  → _fallback_sdpa: mx.fast.scaled_dot_product_attention
     └── yes → _mfa_forward
                 │
                 ▼
         mx.custom_function (_make_mfa_custom)
           Forward: mfa_forward_with_lse   ← C++ binding
           Backward: see "Backward Pass" section
                 │
                 ▼
         C++ binding: csrc/bindings.cpp
           mfa_attention_forward(q, k, v, params)
                 │
                 ▼
         MFAttention::eval_gpu (csrc/mfa_attention.cpp)
           ├── V2 eligible → SteelV2Forward      (D=64/128, f16/bf16)
           ├── Flash Decode eligible → FlashDecodePartial + Reduce
           ├── STEEL eligible → SteelForward     (all D, f16/bf16)
           ├── Varlen → SteelVarlenForward        (D≤256)
           ├── f32 → ccv path (AttentionForward)
           └── else → SDPA fallback
                 │
                 ▼
         ShaderCache::get_or_compile (csrc/shader_cache.mm)
           ├── KernelKey in cache → return cached MTLComputePipelineState
           └── key missing → newLibraryWithSource + newComputePipelineState
                 │
                 ▼
         Metal GPU kernel execution
           Output: O [B, H, N, D], L [B, H, N] (logsumexp, internal)
```

---

## STEEL Kernel Family

The STEEL (Structured Tiled Execution Engine Layer) kernel family handles all
f16/bf16 forward passes.  Kernels are JIT-generated as Metal source strings —
there are no pre-compiled `.metal` files.

### STEEL V2 (current primary, D=64/128)

STEEL V2 is the default forward path for `D ∈ {64, 128}` when sequence length
meets the activation threshold (`N ≥ 4096` for D=64, `N ≥ 8192` for D=128 on M1).

**Key innovation**: K_smem and V_smem share a single `KV_smem` buffer (sequential
K phase then V phase).  This doubles `BK` vs V1 within the same TGP budget,
halving K-tile iterations:

| D | BQ | BK | WM | TGP bytes | V1 BK | Speedup vs V1 |
|---|----|----|----|-----------:|------:|:-------------:|
| 64  | 32 | 64 | 4 | 13,824 | 32 | ~1.7× |
| 128 | 32 | 32 | 4 | 18,944 | 16 | ~1.5× |

M3+ devices (gen ≥ 15) use `BK=64` for D=128 (vs `BK=32` on M1/M2), exploiting
larger register files.

**Tiling algorithm**:

```
// Q is pre-loaded into registers (Q-hoisting)
for kb in kb_start..kb_lim:
    if causal and tile fully masked: continue   // tile-skip

    // Phase K: load K[BK × D] into KV_smem; compute S = Q_reg @ K_smem^T
    load K[kb] → KV_smem; barrier
    S = simdgroup_multiply(Q_reg, KV_smem)      // [BQ × BK]

    apply softcap / sliding-window mask
    update running_max + sum; rescale O_reg

    // Phase V: load V[BK × D] into KV_smem; accumulate O
    load V[kb] → KV_smem; barrier
    O_reg += simdgroup_multiply(P, KV_smem)     // [BQ × D]

write O_reg → device memory; write logsumexp L
```

**Causal tile-skip**: if `k_start ≥ q_start + BQ` the entire tile is skipped
with no masking overhead — halves effective iterations for causal attention.

**V2 split-K**: for under-occupied grids (`total_tgs < 0.8 × gpu_cores`), the
K dimension is split across additional threadgroups.  A reduction pass combines
partial outputs via exp2-domain LSE.  Disabled with `MFA_DISABLE_V2=1`.

**Sliding window**: `window_size=(left, right)` — `kb_start` is offset O(1)
before the K-loop; right bound clips `kb_lim`; masks applied per-element on
diagonal tiles.  Guarantees constant active tile count → 5–21× vs SDPA at
long sequences.

**Softcap**: `tanh(score × scale / softcap) × softcap` applied in log2 domain
(`log2e`/`ln2` conversion) after QK scale, before masking. Uses `precise::tanh`.

### STEEL V1 (D=256/512 and short-N fallback)

Same algorithm as V2 but with separate K_smem and V_smem buffers.
Primary use: D=256 and D=512, where V2's doubled-BK would exceed TGP budget.

| D | BQ | BK | WM | TGP bytes |
|---|----|----|----|-----------:|
| 64  | 32 | 32 | 4 | ~16 KB |
| 128 | 32 | 16 | 4 | ~20 KB |
| 256 | 32 | 16 | 4 | ~29 KB |
| 512 | D-split | 16 | 2 | <32 KB |

D=256 causes register spill on M1/M2 — performance is 0.9–1.0× SDPA.
Auto-routing never activates STEEL for D=256+.

**D=512 D-split**: head_dim=512 is processed in 4× sub-tiles of 128 per
inner loop to stay within the 32 KB TGP limit.

### Flash Decode (N_q ≤ 4, S ≥ 256)

Two-phase split-KV decode for single-token or speculative decoding steps.
Activated automatically when `N_q ≤ 4` and `S ≥ 256` (f16/bf16 only).

**Phase 1** (`FlashDecodePartial`): KV is split into `num_splits` chunks.
Grid = `(N_q × num_splits, H, B)`.  Writes partial output `O_partial [N_q, D]`
and log-sum-exp `L_partial [N_q]` to scratch buffers.

**Phase 2** (`FlashDecodeReduce`): LSE-weighted combination across splits.
Grid = `(N_q, H, B)`.  Writes final `O [B, H, N_q, D]`.

`compute_num_splits(kL, BK)` targets ≥2 K-tiles per split, capped at 32.

**Causal offset**: `qL_off = S − N_q`.  Query at position `i` sees keys
`0..(S − N_q + i)` — the K-loop must start at `qL_off`, not 0.

### Async DMA Metallib (experimental)

The Apple GPU has a hardware DMA unit that overlaps device→threadgroup copies
with shader core compute, exposed via the undocumented `simdgroup_event` API
(`__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")`).  Philip Turner's
metal-flash-attention used this to achieve 83% ALU utilisation on M1 Max.

**Apple removed access to this instruction:**

- Xcode 14.3+: public headers removed
- macOS 26: `__asm` blocked in JIT compiler; runtime silently converts
  precompiled async_copy opcodes to synchronous loads (no crash, no gain)

mlx-mfa ships `mlx_mfa/precompiled/async_v2.metallib` compiled on macOS 15
/ Xcode 16 (GitHub Actions macos-15 runner).  The metallib contains two
entry points — `mlx_mfa_v2_async_attention` (D=64) and
`mlx_mfa_v2_async_attention_d128` (D=128) — with function constants
`FC_CAUSAL` (bool, index 0) and `FC_GQA_FACTOR` (ushort, index 1).

**Async overlap schedule** (per K-tile iteration):

```
wait K[kb] DMA → threadgroup_barrier → Q@K^T GEMM
launch V[kb] DMA (overlaps softmax below)
softmax (compute while V DMA runs)
wait V[kb] DMA → threadgroup_barrier → launch K[kb+1] DMA
P@V GEMM (compute while K[kb+1] DMA runs)
```

The `threadgroup_barrier(mem_flags::mem_threadgroup)` after each
`simdgroup_event::wait()` is mandatory: `wait()` synchronises only the
calling simdgroup; without the barrier, simdgroups 1–3 may still be writing
shared K_smem/V_smem when simdgroup 0 begins reading.

**Fallback chain:** async metallib → sync AOT metallib (`~/.mlx_mfa/`) → JIT.
Disable async path: `MFA_DISABLE_ASYNC=1`.

**macOS 26 measurement (M1 Max, D=64 N=4096 causal, B=2 H=8):**

| Path | ms | vs Sync |
|------|----|---------|
| Async metallib | 5.5 | 1.14× |
| Sync V2 | 6.2 | — |

The 1.14× at D=64/N=4096 is measurement noise; at D=64/N=8192 ratio = 1.00×.
On macOS ≤15, hardware DMA is expected to provide +20–40% over sync V2 for
causal D=64/128 (ALU fully hides DMA latency at long sequences).

**Future:** Metal 4 TensorOps (M5+/A19+) provide dedicated matrix multiply
hardware that inherently overlaps with shader core compute — the official
successor to `simdgroup_async_copy`.  Reserved as `TensorOpsForward` in the
kernel type registry.

### STEEL Varlen Forward

Single Metal dispatch for packed sequences with `cu_seqlens`.
Supported for D ∈ {64, 128, 256}; D=512 falls back to per-sequence SDPA.

### Block-Sparse STEEL

Block-sparse kernel triggered by `flash_attention_sparse(block_mask=...)`.
K-tile loop checks `block_mask[qb][kb]` via uniform threadgroup branch
(zero warp divergence for fully-zero or fully-one tiles).

### GQA (Grouped Query Attention)

`gqa_factor = H_q / H_kv` is a compile-time Metal `#define`.

- Forward: Q head `h` reads KV head `h / gqa_factor`
- Backward dQ: same mapping
- Backward dKV: accumulates gradients across all `gqa_factor` Q-heads per KV-head

No K/V tensor copying or tiling — GQA is native in the kernel.

---

## Backward Pass

### STEEL native backward (f16/bf16, D ≤ 512)

Activated for f16/bf16 with `softcap==0` and `alibi==False`.
Dispatches two Metal kernels: `MFASteelBwdDQ` and `MFASteelBwdDKV`.

**Algorithm** (FlashAttention-2 backward, log2 domain):

```
P = exp2(S × scale × log2e − L_log2)    // recompute attention weights
D_i = scale × rowsum(O ⊙ dO)            // diagonal correction

dQ += scale × (P ⊙ (dO @ V^T − D_i)) @ K   [grid: (N_q/BQ, H_q, B)]
dK += scale × (P ⊙ (dO @ V^T − D_i))^T @ Q [grid: (N_k/BK, H_kv, B)]
dV += P^T @ dO
```

**D=256 D-split**: QK and `dO@V^T` are each accumulated across two D=128 sub-tiles
(three phases per K-tile) to stay within TGP.  Register tiles `Qtile_lo/hi` and
`dOtile_lo/hi` are declared outside all loops to keep them in registers.

### Varlen backward

`flash_attention_varlen` uses `mx.custom_function`.  The backward splits into
per-sequence vjp calls:

```python
for i in 0..num_seqs:
    _, (dq_i, dk_i, dv_i) = mx.vjp(flash_attention, [q_i, k_i, v_i], [dO_i])
dQ, dK, dV = concat(dq_i), concat(dk_i), concat(dv_i)
```

`cu_seqlens` lists are materialised to `list[int]` before the
`@mx.custom_function` definition so they are captured by closure, not tracked
by the MLX autograd tape.

### Fallback routes

| Condition | Backward method |
|-----------|----------------|
| f32 | `mx.vjp(mx.fast.scaled_dot_product_attention)` |
| softcap > 0 | `mx.vjp(_softcap_sdpa_ref)` via `mx.compile` |
| alibi | `mx.vjp(_alibi_sdpa_ref)` via `mx.compile` |
| sparse | `mx.vjp(sdpa)` (dense) or tiled sparse backward |

**Why not C++ Primitive vjp**: `MFAttention::vjp()` cannot access `L`
(logsumexp) because MLX prunes it from the graph before vjp runs.  The Python
`custom_function` saves `L` as a closure variable.

---

## SageAttention

SageAttention (int8 Q/K) is implemented in `csrc/mfa_sage_fwd.cpp`.

### Quantization scheme

```python
K_smooth, K_mean  = smooth_k(K)               # per-channel mean subtraction
K_int8, K_scale   = quantize_per_block(K_smooth)
Q_int8, Q_scale   = quantize_per_block(Q)
```

`smooth_k` bias cancels exactly in the softmax ratio — no output correction needed.

### Kernel (`SageForward = 11`)

JIT-generated Metal kernel:

- `Q_int8 @ K_int8^T` in int32 accumulator; dequantize via scales before softmax
- `V` kept in fp16/bf16 (memory bandwidth dominates, not compute)
- Optional `window_size=(left, right)` for sliding-window SageAttention

Block sizes by head_dim:

| D | BQ | BK |
|---|----|----|
| 64 | 16 | 32 |
| 128 | 16 | 32 |
| 256 | 8  | 32 |
| 512 | 4  | 32 |

**SageAttention is inference-only**: autograd is not supported.

### QuantizedKVCache

Pre-stores K as int8; O(1) quantization per decode step (only the new K block):

```python
cache = QuantizedKVCache(B, H_kv, D, max_seq_len)
cache.append(k_new, v_new)       # quantizes k_new once; stores k_int8, k_scale
out = sage_attention_prequantized(q_int8, cache.k_int8, cache.v,
                                   q_scale, cache.k_scale, causal=True)
```

---

## Memory Architecture

### KV cache types

| Class | Storage | Use case |
|-------|---------|----------|
| `DenseKVCache` | `[B, H, max_len, D]` f16/bf16 | Standard decode |
| `QuantizedKVCache` | K: `[B, H, max_len, D]` int8 + scale | SageAttention decode |
| `PagedKVCache` | pool: `[num_blocks, block_size, H, D]` + `block_table` | Multi-request serving |

### Paged KV gather kernel

`MFAPagedKVGather` (`csrc/mfa_paged_gather.cpp`) materialises a contiguous
`[B, H_kv, max_kv_len, D]` tensor from the page pool in a single GPU dispatch.

```
Pool:   [num_blocks, block_size, H_kv, D]   (token-major within block)
Output: [B, H_kv, max_kv_len, D]             (BHND — STEEL-ready)
```

The kernel transposes `[block_size, H_kv, D] → [H_kv, block_size, D]` during
the copy.  Grid: 1-D, one thread per output element.

After gather, sequences are sliced to actual length `[:kv_len]` before
dispatching to STEEL, preventing padded-zero positions from corrupting softmax.

### InferenceContext lifecycle

```
InferenceContext(B, H_kv, D, max_seq_len)
  ├── prefill(q, k, v)   → STEEL forward on full sequence
  ├── step(q, k, v)      → Flash Decode (N_q=1, split-KV)
  └── reset()            → zero fill_pos counter
```

`SageInferenceContext` wraps `QuantizedKVCache` and routes `step()` through
`sage_attention_prequantized`.

---

## Dispatch System

### DispatchPolicy constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `DispatchPolicy.AUTO` | `"auto"` | Empirical threshold routing (default) |
| `DispatchPolicy.MFA` | `"mfa"` | Force STEEL kernel |
| `DispatchPolicy.SDPA` | `"sdpa"` | Force MLX SDPA |
| `DispatchPolicy.SAGE` | `"sage"` | Force SageAttention (int8 Q/K) |

### Auto-routing thresholds

| D | Causal | M1 threshold N ≥ | M3+ threshold N ≥ |
|---|--------|:----------------:|:------------------:|
| 64 | yes | 4096 | 4096 |
| 128 | yes | 8192 | 2048 |
| 256+ | any | never | never |
| any | window | always | always |
| any | sparse | always | always |

### Calibration

`calibrate_dispatch()` benchmarks your device and saves thresholds to
`~/.cache/mlx_mfa/calibration.json`.

Environment overrides (set before `import mlx_mfa`):

| Variable | Effect |
|----------|--------|
| `MFA_DISABLE_V2=1` | Force V1 (benchmarking baseline) |
| `MFA_FORCE_GEN=13` | Override arch detection (13=M1, 15=M3) |
| `MFA_LOG_DISPATCH=1` | Print chosen kernel per call |

---

## Build System

```
pyproject.toml (scikit-build-core)
  ↓
CMakeLists.txt
  Languages: CXX + OBJCXX (for shader_cache.mm)
  Finds: Python.Development.Module + MLX (via python -c "import mlx")
  Frameworks: Metal, Foundation
  Output: mlx_mfa/_ext.cpython-3XX-darwin.so
```

Key decisions:

- **`NB_DOMAIN "mlx"`** — mandatory for sharing `mlx.core.array` ABI between
  MLX and the extension.  Without this, passing MLX arrays raises
  `RuntimeError: Unable to cast Python instance to C++ type`.
- **`MTLLanguageVersion3_1`** — required for `bfloat4` vectors in bf16 kernels.
- **`shader_cache.mm` is Objective-C++** — uses native Metal API with
  `void*` / `__bridge_retained` for ARC-safe pipeline management.

---

## Device Detection

```cpp
int gen = d.get_architecture_gen();   // "applegpu_g13s" → 13
bool is_m3_plus = (gen >= 15);        // 13=M1, 14=M2, 15=M3, 16=M4
bool is_m5_plus = (gen >= 17);        // M5/A19+ (Metal 4 tensor API, reserved)
```

| Gen | Chip | STEEL V2 BK(D=128) |
|:---:|------|--------------------|
| 13 | M1 | 32 |
| 14 | M2 | 32 |
| 15 | M3 | 64 |
| 16 | M4 | 64 |

`get_architecture_gen()` extracts the integer suffix from the Metal architecture
string — **not** the `MTLGPUFamilyApple` enum value.

---

## Kernel Type Registry

12 active types defined in `csrc/shader_cache.hpp`:

| Value | Name | Description |
|------:|------|-------------|
| 0 | `AttentionForward` | ccv MFA forward (f32) |
| 1 | `AttentionBackwardDQ` | ccv MFA backward dQ |
| 2 | `AttentionBackwardDKV` | ccv MFA backward dKV |
| 3 | `SteelForward` | STEEL V1/V2 forward (all D; d-split for D=512) |
| 4 | `FlashDecodePartial` | Flash Decode Phase 1: partial attn per KV split |
| 5 | `FlashDecodeReduce` | Flash Decode Phase 2: LSE reduce over splits |
| 6 | `SteelBackwardDQ` | STEEL native backward dQ (f16/bf16, D≤512) |
| 7 | `SteelBackwardDKV` | STEEL native backward dKV (f16/bf16, D≤512) |
| 8 | `SteelVarlenForward` | STEEL varlen forward (D≤256) |
| 9 | `PagedKVGather` | Paged KV gather: pool → contiguous BHND |
| 10 | `PagedSteelForward` | STEEL forward with kernel-level paged KV (D≤256) |
| 11 | `SageForward` | int8 Q/K quantized attention with window support |
| — | `TensorOpsForward` | Reserved: Metal 4 cooperative tensors (M5+/A19+ only) |

STEEL V2 shares kernel type 3 (`SteelForward`) with a `v2=true` compile-time
flag in the JIT source generator.

---

## Key Design Decisions

1. **JIT Metal compilation** — kernels are parameterized by head_dim, dtype,
   block dims, and causal flag.  Pre-compiling all combinations would require
   O(100) `.air` files shipped in the wheel; JIT compiles only what is used.

2. **`transposeState = false`** — the original ccv code set `transposeState = true`,
   coupling head-offset computation to GEMM inner loop addressing.  mlx-mfa
   unconditionally uses `transposeState = false` and forces `SEQUENCE_LENGTH`
   in the head-offset expression.  **Do not revert.**

3. **`disableAsyncCopy = true`** — `simdgroup_async_copy` (a private AIR
   intrinsic) was removed from Metal shader compilation on macOS 26.  STEEL
   was designed from the start to avoid it; ccv-path kernels use the software
   loop fallback.

4. **Why not C++ Primitive vjp** — `MFAttention::vjp()` cannot access `L`
   (logsumexp) because MLX prunes it from the graph.  Python `mx.custom_function`
   saves `L` as a closure variable.

5. **Buffer aliasing prevention** — `cotangent = ones_like(O_fwd)` carries
   lazy graph ancestry.  `_sever_lazy_graph` adds `+ zeros_like` to write a
   fresh buffer, preventing the Metal allocator from aliasing `O_backward`
   with the freed `O_fwd` buffer.

6. **bfloat16 → numpy** — `numpy` PEP 3118 does not support bfloat16.  Cast
   to float32 first: `np.array(mlx_bf16.astype(mx.float32))`.

7. **`QuantizedKVCache` contiguity** — slices from `cache.k_int8` may be
   non-contiguous after indexing.  Call `.flatten().reshape(shape)` or
   `mx.contiguous()` before C++ dispatch.

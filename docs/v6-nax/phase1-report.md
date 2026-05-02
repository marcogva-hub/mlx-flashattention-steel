# V6 NAX — Phase 0+1 Report

**Date:** 2026-05-02
**Hardware:** Apple M5 Max (40 GPU cores, gen 17 / `applegpu_g17s`, 128 GB)
**Software:** macOS 26.5 · MLX 0.31.2 · mlx-mfa 2.28.1 · Xcode toolchain 32023.884
**Branch:** `feat/v6-nax`

---

## TL;DR

| Gate | Description | Result |
|------|-------------|--------|
| G1 | V6 toolchain compiles on M5 Max | **🟢 PASS** |
| G2 | V6 runs without crash on trivial shape | 🟡 SKIP (kernel API needs work) |
| G3 | V6 FP16 D=64 correct | 🟡 SKIP |
| G4 | V6 FP16 D=128 correct | 🟡 SKIP |
| G5 | V6 BF16 D=64 correct | 🟡 SKIP |
| G6 | V6 BF16 D=128 correct | 🟡 SKIP |
| G7 | V6 beats V2 on ≥ 1 workload | **🔴 STRATEGIC RECONSIDERATION** |
| G8 | Cold-start latency acceptable | 🟡 SKIP |
| G9 | Zero V2 STEEL regression | **🟢 PASS** |

**Bottom line:** Toolchain works, but **the strategic premise of V6 NAX in mlx-mfa
needs reconsideration**. MLX 0.31.2's SDPA *already* uses Apple's NAX kernel on
M5 Max. mlx-mfa's `flash_attention()` correctly routes to SDPA on M5 for
self-attention shapes, so **users already get Apple's NAX today** without any
V6 implementation work. Building V6 NAX in mlx-mfa would mean competing
against Apple's own NAX kernel — a higher bar than competing against the older
STEEL V2.

---

## 1. Phase 0 Infrastructure Results (PASS)

### Task 0.1 — MSL 4 + MPP toolchain (PASS)

Both probes succeed via mlx-mfa's shader cache (`csrc/v6_nax_probe.cpp`):

```
$ python -c "from mlx_mfa._ext import v6_nax_probe_msl4, v6_nax_probe_mpp"
$ # ...
v6_nax_probe_msl4():  OK
v6_nax_probe_mpp():   OK
```

- MSL 4.0 stub compiles (`#include <metal_tensor>` resolves)
- MPP `matmul2d` stub compiles (`#include <MetalPerformancePrimitives/...>` resolves)
- The MPP framework header is **auto-discovered by Apple's runtime metal
  compiler** — no manual header packaging needed
- Compiler version: `32023.884` (macOS 26.5 / Xcode 26.5 toolchain)

The shader cache (`csrc/shader_cache.mm`) now supports MSL 4.0 via a
source-string marker (`// MFA_REQUIRE_MSL4`). V2 STEEL stays on MSL 3.1,
V6 NAX uses MSL 4.0. This was implemented as a non-invasive opt-in.

### Task 0.2 — Hardware detection (PASS)

```python
>>> from mlx_mfa._ext import device_has_neural_accelerators, device_has_nax_bf16
>>> device_has_neural_accelerators()  # True on M5 Max, False on M1-M4
True
>>> device_has_nax_bf16()              # True on M5 Max + macOS >= 26.1
True
```

Implementation uses `supportsFamily(MTLGPUFamilyApple10)` (raw enum 1010).
**MLX 0.31.2 declares `is_nax_available()` in `device.h` but doesn't export it
in `libmlx.dylib`** — link error if used directly. Falling back to
`supportsFamily` works reliably.

`device_has_nax_bf16()` adds an `__builtin_available(macOS 26.1)` gate for
MPP bf16 paths.

### Task 0.3 — Shader cache + V6 enum slot (DONE)

- `SteelForwardV6NAX = 22` activated in `shader_cache.hpp`
- MSL 4 language version selected per-kernel via source-string marker
- `MTLLanguageVersion4_0` encoded as `(4 << 16)` for compatibility with
  older SDK enum definitions

### Task 0.4 — V6 NAX kernel skeleton (PARTIAL)

Wrote `csrc/mfa_steel_fwd_v6_nax.{cpp,hpp}` — 267 lines, MPP-based forward
kernel skeleton patterned after MLX's bundled `steel_attention_nax.h`.

**Status: compiles via shader cache up to `cooperative_tensor` API
mismatch.** The MPP header docs reference `cT.get_mask(i)` in their example,
but the actual `cooperative_tensor` class on macOS 26.5 doesn't expose that
member function. Cooperative tensor element access patterns differ from the
Draw Things 2024 examples — likely an MPP API version skew.

**Did NOT** undertake the full ccv `NAAttentionKernel.cpp` port (2667 lines)
because the strategic finding below changes the cost-benefit equation.

### Task 0.5 — End-to-end V6 dispatch (NOT WIRED)

Skipped. Wiring V6 dispatch into `mfa_attention.cpp::eval_gpu()` only makes
sense after a working kernel exists.

---

## 2. Strategic Finding: MLX SDPA on M5 already uses NAX

This is the most important finding of this sprint. It changes the V6 NAX
roadmap.

### 2.1 — Dispatch behavior on M5 Max

The mlx-mfa dispatch policy (`mlx_mfa/dispatch_policy.py::should_use_mfa`)
on M5 Max returns:

| Shape | should_use_mfa? | Routing |
|-------|:---------------:|---------|
| SeedVR2-small (D=128, N=26730) | False | → SDPA (Apple NAX) |
| SeedVR2-large (D=128, N=111375) | False | → SDPA (Apple NAX) |
| FlashVSR-dense (D=64, N=4096) | False | → SDPA (Apple NAX) |
| CogVideoX (D=128, N=70200) | False | → SDPA (Apple NAX) |
| LTX2-cross (D=64, N_q=2048, N_kv=14000) | True | → MFA V2 STEEL |

For 4/5 production shapes, `flash_attention()` already routes to SDPA, which
on M5 Max compiles Apple's NAX attention kernel
(`mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h`).

### 2.2 — Performance comparison (M5 Max, FP16)

Measured `mx.fast.scaled_dot_product_attention` (NAX path) vs
`flash_attention(backend="mfa")` (V2 STEEL forced):

| Shape | SDPA (NAX) | V2 STEEL forced | V2 / SDPA |
|-------|-----------:|----------------:|----------:|
| SeedVR2-small  | 176.58 ms |   633.30 ms | **3.59× slower** |
| SeedVR2-large  | 4323.69 ms | 12654.14 ms | **2.93× slower** |
| FlashVSR-dense | 0.93 ms   |     3.80 ms | **4.09× slower** |
| CogVideoX      | 2462.26 ms | 7244.84 ms | **2.94× slower** |
| LTX2-cross     | 1.31 ms   |     1.03 ms | 0.79× *(MFA faster)* |

**Apple's NAX kernel is 3-4× faster than mlx-mfa's V2 STEEL** on M5 Max for
self-attention. The mlx-mfa dispatch policy correctly routes to SDPA, so
end-users see SDPA performance — they don't experience the V2 STEEL slowdown
unless they force `backend="mfa"`.

LTX2-cross (asymmetric N_q vs N_kv) is the only shape where V2 STEEL beats
SDPA — and that's because Apple's NAX kernel doesn't currently optimize for
asymmetric cross-attention.

### 2.3 — Implications for V6 NAX

**The original V6 NAX premise was**: "Beat V2 STEEL by using M5's neural
accelerators." But on M5 Max, **Apple's own NAX kernel (in MLX SDPA) is
already 3-4× faster than V2 STEEL**, and the mlx-mfa dispatch policy routes
to it automatically.

So "V6 NAX in mlx-mfa" would actually be: **build a kernel that beats
Apple's NAX implementation**. That's a much harder problem than beating
STEEL V2:
- Apple has full MPP source-level access; we use the public API
- Apple owns the runtime compiler (`32023.884`); we work around its quirks
- Apple has months of M5 hardware tuning; we have a brand-new chip

The realistic value of V6 NAX in mlx-mfa is therefore in:
1. **Asymmetric cross-attention** (LTX2-class) — SDPA underperforms here
2. **Features SDPA lacks**: paged-KV, varlen, sparse, attn_bias modes,
   custom GQA layouts
3. **Specific shapes Apple hasn't tuned for** (small N, unusual H)

Building a "general V6 NAX self-attention forward" that beats SDPA is unlikely
to pay off given how good Apple's kernel already is.

---

## 3. Gate Results

### G1 — V6 compiles on M5 Max → **PASS**
MSL 4 + MPP `matmul2d` JIT-compiles successfully via mlx-mfa's shader cache.
Toolchain is unblocked.

### G2-G6 — V6 correctness → **SKIP (no working kernel)**
The minimal V6 forward skeleton compiles to the cooperative_tensor API
mismatch. Resolving it requires either:
- Reading the actual `cooperative_tensor<>` class definition (not in SDK
  headers — embedded in the GPU compiler runtime)
- Faithfully porting `NAAttentionKernel.cpp` from ccv (2667 lines)
- Reverse-engineering the API from MLX's own NAX kernel which works

### G7 — V6 beats V2 on ≥ 1 workload → **STRATEGIC RECONSIDERATION**
Inverted finding: **SDPA (Apple's NAX) is 3-4× faster than V2 STEEL** on
M5 Max for self-attention. The mlx-mfa dispatch policy already routes to
SDPA on M5 for these shapes, so users already get NAX performance via
mlx-mfa today. V6 NAX in mlx-mfa would compete against Apple's NAX, not
against V2 STEEL.

### G8 — Cold-start latency → **SKIP**
Will be measured once V6 kernel works.

### G9 — Zero V2 STEEL regression → **PASS**
Full test suite with V6 infrastructure present: 653 passed, 2 failed (the
2 pre-existing precision tolerance failures documented in 2.28.1 — unrelated
to V6 work). V2 STEEL paths unchanged.

---

## 4. Recommendation

**Stop the "V6 NAX self-attention forward" track.** The premise (beat V2 STEEL
on M5 via NAX) is already met by Apple's NAX kernel that mlx-mfa's dispatch
already uses on M5.

**Pivot V6 to where it actually adds value:**

### Option α — V6 NAX cross-attention kernel
Build V6 only for asymmetric N_q ≠ N_kv shapes (LTX2-class). The dispatch
policy already routes these to MFA on M5; replacing V2 STEEL with an
NAX-based kernel could deliver real speedups (V2 STEEL for LTX2 is currently
1.03ms; NAX-equivalent could be 0.5ms or better).

### Option β — V6 NAX features SDPA lacks
Apply NAX/MPP to paged-KV, varlen, sparse, attn_bias modes — paths where
SDPA either doesn't apply or falls back to slow code.

### Option γ — Document and ship 2.28.1 as M5-ready
Since `flash_attention()` on M5 already routes to SDPA's NAX kernel,
mlx-mfa users get NAX performance today. Document this in the README as
"M5 Max support via Apple NAX SDPA dispatch" and don't build a competing
V6 in mlx-mfa.

**Recommended choice: γ + α.** Ship 2.28.1 as M5-ready (γ), and build V6 NAX
only for asymmetric cross-attention (α). Skip the full self-attention V6
port.

---

## 5. Files added/modified this sprint

### Added
- `csrc/v6_nax_detect.{hpp,mm}` — `device_has_neural_accelerators()` +
  `device_has_nax_bf16()` helpers
- `csrc/v6_nax_probe.cpp` — JIT compile probes for MSL 4 + MPP + V6 forward
- `csrc/mfa_steel_fwd_v6_nax.{hpp,cpp}` — V6 forward skeleton (compiles to
  cooperative_tensor API mismatch — preserved for reference, not wired)
- `docs/v6-nax/phase1-report.md` — this report

### Modified
- `csrc/shader_cache.hpp` — `SteelForwardV6NAX = 22` enum slot active
- `csrc/shader_cache.mm` — MSL 4.0 selection via `// MFA_REQUIRE_MSL4` marker
- `csrc/bindings.cpp` — Python bindings for the V6 probes + detection
- `CMakeLists.txt` — added new files to MFA_SOURCES

### Test results
- 653 attention tests pass (no V6-related regressions)
- `MFA_DISABLE_V6_NAX` env var not added (no V6 dispatch wired)

---

## 6. Decisions for Marco

1. **Approve recommendation γ + α?**
   - γ = ship 2.28.1 as M5-ready, document SDPA-NAX dispatch
   - α = build V6 NAX only for asymmetric cross-attention
2. **If continuing V6 self-attention** (rejecting the recommendation):
   - Allocate 2-3 days for full ccv `NAAttentionKernel.cpp` port
   - Accept that the goal becomes "match Apple's NAX," not "beat V2"
3. **Investment in MPP API research?**
   - MLX's bundled `steel_attention_nax.h` is the best documentation we have
   - Reverse-engineering cooperative_tensor element access from there could
     unblock the minimal kernel

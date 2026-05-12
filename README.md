# mlx-mfa

`mlx-mfa` is a Metal Flash Attention + serving-oriented runtime layer for MLX on
Apple Silicon. It provides high-performance attention kernels, runtime helpers,
and cache abstractions for dense training/inference plus modern serving flows.

Current version: **2.33.1** — Conv3D NAX SHIP-DEFAULT + sparse attention M5+ fast-fallback. Forward attention on canonical shapes (D∈{64,128}) now routes to Apple's `steel_attention_nax.h`; mlx-mfa keeps native kernels for niche / non-canonical shapes.

## Foreword

**MLX Metal Flash Attention - Why?**

I've been working on personal ports of Video Super Resolution and Video 
Reconstruction models for months, but always ended up frustrated by the 
slow inference in my M1 Max MacBook Pro. And to try to mitigate this without
having to buy a brand-new, very expensive new M4, then M5 Max, I decided to
at least try to port Flash Attention to Mac, hoping for better results. And 
having better results porting VSR/VR models to MLX than MPS, that's why I ended
up doing it.

At this point, despite the lower than hoped for results, I'm still pretty
satisfied with the results in my M1 Max MBP.

I'll be doing only reduced work on this project until June 2026, when I'll
upgrade from my M1 Max to a M5 Max MBP, with which I expect to be able to
obtain much better results, thanks to the improvements Apple has been adding
to its silicon.

v2.32.0 introduces a **strategic shift in dispatch on M5+ NAX hardware**.
Apple's MLX 0.31.2 ships an excellent NAX-based SDPA kernel
(`steel_attention_nax.h`) that matches the V34 NAX-direct path mlx-mfa
shipped in v2.31.0 — and Apple's kernel benefits from continuous upstream
tuning. Rather than compete on a surface where Apple has structural
advantages, mlx-mfa now **routes forward attention to MLX SDPA on M5+
when SDPA covers the shape and feature set optimally**, and keeps native
kernels for everything else:

- `head_dim ∉ {64, 128}` (D=80, D=96, D=192, D=256, D=512) → mlx-mfa
- Block-sparse / LCSA mask                                 → mlx-mfa
- Additive attention bias (modes 1, 2)                     → mlx-mfa native bias kernel
- Sliding window                                           → mlx-mfa STEEL window kernel
- Backward pass                                            → mlx-mfa (Apple's NAX backward NYI)
- All M1–M4 hardware (no NAX)                              → mlx-mfa V2/V3/V6 NAX legacy
- Specific empirical carve-outs from Sprint A sweep        → mlx-mfa

Override via `MFA_DISABLE_SDPA_ROUTE=1` (recovers v2.31.0 dispatch on M5+).
This preserves mlx-mfa as a unified attention toolkit across all Apple
Silicon generations while stopping unnecessary competition with Apple's
upstream optimizations on shapes Apple covers well.

The v2.31.0 performance numbers (V34 +33-40% wins on D=128) were measured
under specific environmental conditions that did not reproduce in the
v2.32.0 cross-session diagnostic. v2.32.0 ships with reproducible-conditions
methodology baked into the bench infrastructure (`bench/v32_multisession_capture.py`,
`docs/v6-nax/v32-multisession-protocol.md`, `CLAUDE_V6_NAX.md` Artifact #5).
The architectural improvements that motivated v2.31.0 (V34 NAX-direct
forward kernel, multi-SG parallelism via per-SG row partitioning) remain
in the codebase as a regression canary and as the dispatched path when
`MFA_DISABLE_SDPA_ROUTE=1` is set.

v2.31.0 shipped the **V34 NAX-direct rewrite**. V6 NAX's forward hot path
uses Apple's `NAXFrag::mma` and `NAXTile<T, TQ, TD>` primitives directly
(the pattern from `steel_attention_nax.h`), bypassing MPP cooperative_tensor
constraints that previously imposed `execution_simdgroups<1>`. Multi-SG
parallelism comes from per-SG row partitioning at the kernel level
(`tm = 16 * TQ * sgid`), not via cooperative_tensor distribution — so the
V33 cross-SG opacity issue disappears entirely.

The historic D=128 long-N gap is **closed**: production VSR/DiT shapes
that were stuck at 1.5–1.7× SDPA now run at SDPA parity. **SeedVR2-small
at 0.89× SDPA actually beats SDPA**, the first time V6 NAX has dipped
below 1.0× on a production shape. Numerics also improve 4–30× over legacy
because the manual `simd_shuffle_xor` row reductions on FP32 accumulators
inside `NAXFrag::row_reduce` are bit-exact, vs MPP's `reduce_rows` which
had tile-boundary FP rounding artifacts. Dispatch is shape-aware: V34 is
default for D=128 and D=64 N≥2048, legacy stays for D=64 small-N
(FlashVSR-dense regresses under V34 — root cause TBD).

v2.30.0 extended v2.29.0's V6 NAX work along three axes: (1) **GQA
single-Otile** — the BHND rewriter now handles `Hq % Hk == 0` so GQA
shapes use the single-Otile kernel directly, gaining 7-14% over the
v2.29.0 legacy fallback; (2) **dispatch v5** (the v6 attempt was reverted
after thermal-controlled re-bench); (3) **tgmem allocation cleanup** —
single-Otile + bypass no longer allocates the unused P_buf threadgroup
memory.

v2.29.0 shipped **V6 NAX single-Otile** for M5+ hardware: an Apple-style
single-buffer kernel (`loopForwardSingleTile`) with autoresearch-tuned
default tile config (BQ=16 universal, per-D BK/SG).

v2.27.0 added native Metal `attn_bias` kernel support (additive bias on
attention logits without SDPA fallback), a dispatch audit for 11 DiT/UNet
architectures, and varlen validation for token merging workflows.
See `CHANGELOG.md` for full details per version.

Thank you for your interest, and let me know if you've been able to improve
on my work!

## Current Repository Status

- **V2 dense** is the main production path.
- Strongest dense wins on M1 Max remain **causal D=64/128** and tile-skip
  regimes (window/sparse).
- **D=256** is narrow benchmark-backed only (not broad promotion).
- **D=512** remains SDPA-default.
- **Native dense backward** was benchmarked and not promoted.
- **Sage** is a specialized decode backend (narrow, benchmark-gated use).
- **V3/V4/V5** remain experimental/hardware-dependent.
- **TurboQuant** KV cache compression (Phase 1–4) production-ready.
- **SVDQuantLinear** W4A16 + optional SVD low-rank correction for DiT quantization.
- **GNA native kernel** inline 3D window attention (D=128, f16/bf16, forward-only).
- **Native `attn_bias`** additive bias on logits via Metal kernel (modes 1/2: per-KV and per-head per-KV broadcast).
- Serving/runtime capability surface is now substantially expanded:
  - paged KV + packed varlen query support
  - paged continuous batching/remap
  - explicit chunked prefill
  - runtime-managed prefix reuse
  - runtime speculative draft/verify flow
  - deeper splitfuse runtime integration
  - KV cache abstraction layer
  - minimal real hybrid/offload-capable cache behavior (local offload tier)
  - TurboQuant compressed KV serving (`create_decode_runtime(turboquant=True)`)

## Limitations

- Main validation hardware is **Apple M1 Max**.
- Broad parity claims against CUDA FlashAttention ecosystems are not made.
- Some advanced paths are intentionally narrow, bridge-based, or explicit-only.
- Hybrid offload is currently a **local offload milestone**, not remote/
  distributed cache infrastructure.
- Future major hardware-specific optimization work is deferred pending newer
  Apple hardware (M5+).

[See the v2.31.0 V6 NAX foreword above and the "Best M5 Max Benchmark
Highlights (v2.31.0)" table below for current numbers.]

## Best M1 Max Benchmark Highlights

Representative benchmark-backed outcomes (see `RESULTS.md` and
`docs/benchmarks/RESULTS.md` for details):

| Area | Representative result (M1 Max) | Interpretation |
|---|---|---|
| Dense causal V2 | up to ~**1.82x** vs SDPA (D=64, N=8192) | Primary production win regime |
| Dense causal V2 | up to ~**1.75x** vs SDPA (D=128, N=16384) | Strong long-sequence causal performance |
| Sliding window | up to ~**21x** vs full SDPA | Tile-skip regime remains strongest |
| D=256 | narrow causal long-N wins (for example ~**1.16x** at N=16384 f16) | Keep narrow policy only |
| D=512 | decision pass found **no broad wins** | SDPA-default remains correct |

## Best M5 Max Benchmark Highlights (v2.31.0)

V6 NAX path on production VSR/DiT shapes (cross-session multi-run, iStat performance fan profile).
The shape-aware dispatch picks V34 (NAX-direct) where it wins, legacy V6 NAX otherwise.

| Shape | D | Path | V6 NAX vs SDPA |
|---|---|---|---|
| FlashVSR-dense | 64 | legacy | 1.23× SDPA |
| LTX2-cross | 64 | **V34** | **1.07× SDPA** |
| SeedVR2-small | 128 | **V34** | **0.89× SDPA ⭐ (beats SDPA)** |
| CogVideoX | 128 | **V34** | **1.03× SDPA** (parity) |
| SeedVR2-large | 128 | **V34** | **1.01× SDPA** (parity) |

GQA shapes (Sprint B single-Otile path, legacy V6 NAX):

| Shape | V6 NAX vs SDPA |
|---|---|
| GQA-Hq32-Hk8 D=128 | 1.06× ⭐ |
| GQA-Hq16-Hk4 D=64 | 1.17× |
| GQA-Hq40-Hk8 D=128 | 1.16× |
| GQA-Hq8-Hk2 D=64 | 1.18× |

Numerical: V34 RMSE FP32 vs SDPA reference is 9e-7 to 4e-6 across all 5 shapes —
4–30× more stable than legacy V6 NAX (1.5e-5 to 6e-6). Manual simd_shuffle_xor row
reductions on FP32 accumulators are bit-exact, vs MPP's reduce_rows which had
tile-boundary FP rounding.

## Serving/Runtime Capability Summary

| Capability | Maturity | Current status |
|---|---|---|
| Paged KV decode runtime | Fully usable | Explicit runtime/API usage; no broad auto-promotion |
| Paged + packed varlen queries | Production (fused kernel) | Single-dispatch fused kernel for all query/KV length combinations |
| Paged continuous batching remap | Fully usable | Explicit `cache_batch_idx` semantics + runtime helpers |
| Chunked prefill | Fully usable (scheduler-oriented) | Operational capability; not a throughput win on current matrix |
| Runtime prefix caching | Fully usable | Register/seed/reuse path integrated with runtime metadata |
| Runtime speculative decode | Fully usable (narrow) | `speculative_step` + verify integration; scheduler engine still future work |
| Splitfuse runtime integration | Narrow/conditional | Runtime path exists; performance remains shape-sensitive |
| Hybrid KV cache + local offload tier | Narrow/conditional milestone | Real hot/cold/offloaded behavior locally; remote offload future work |
| TurboQuant KV compression (Phase 4) | Production | 5.33× K compression, WHT fused in kernel (1.1–1.4× faster) |
| SVDQuantLinear | Production | W4A16 + rank-r FP16 correction; `quantize_model()` tree walker |
| GNA native kernel | Production | Inline 3D window attention (D=128); exact per-element masking |
| Native `attn_bias` | Production | Modes 1/2 via V2 STEEL; modes 0/3 SDPA fallback |
| External cache adapter layer | Experimental groundwork | Concrete local backend provided; external backend integrations pending |

## Repository Guide

- Feature coverage: [`docs/FEATURE_COVERAGE.md`](docs/FEATURE_COVERAGE.md)
- API manual: [`docs/API_MANUAL.md`](docs/API_MANUAL.md)
- Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Inventory map: [`docs/INVENTORY.md`](docs/INVENTORY.md)
- Benchmark interpretation: [`docs/benchmarks/RESULTS.md`](docs/benchmarks/RESULTS.md)
- Root benchmark summary: [`RESULTS.md`](RESULTS.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Historical development archive: [`devnotes/`](devnotes/)
- Examples: [`examples/`](examples/)

## Production vs Narrow vs Experimental

| Status | Components |
|---|---|
| Production | V2 dense causal small-D path; window/sparse tile-skip; SDPA fallback policy; TurboQuant KV compression; SVDQuantLinear; GNA native kernel; native `attn_bias` |
| Narrow / conditional | D=256 causal long-N policy; Sage decode regimes; splitfuse/page-native runtime paths; hybrid local offload behavior |
| Experimental | V3/V4/V5 families; external/LMCache-like backend extensions beyond local adapter |

## Recommended Usage

1. Use `backend="auto"` for dense attention and let policy route between V2 and SDPA.
2. Use `create_decode_runtime(...)` for serving flows instead of stitching helper calls manually.
3. Treat paged/packed/chunked/prefix/speculative features as explicit runtime capabilities.
4. Use Sage as a specialized decode backend only when your workload matches the
   benchmark-backed regime.

## Installation

```bash
pip install -e .
```

## Minimal Usage

```python
import mlx.core as mx
from mlx_mfa import flash_attention, flash_attention_gna, create_decode_runtime
from mlx_mfa import SVDQuantLinear, quantize_model

# Dense attention
q = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
out = flash_attention(q, k, v, causal=True)

# Token merging proportional attention (native Metal, no SDPA fallback)
merge_counts = mx.ones((1, 1, 1, 1024), dtype=mx.float16)
merge_counts[..., :256] = 2.0   # first 256 tokens are merged pairs
bias = mx.log(merge_counts)     # [1, 1, 1, N_kv] — mode 1 broadcast
out_biased = flash_attention(q, k, v, attn_bias=bias)

# GNA (Generalized Neighborhood Attention) — 3D window
# Video: 8 frames of 32x32, local 3D window, sliding
q_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
k_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
v_vid = mx.random.normal((1, 8, 8192, 128)).astype(mx.float16)
out_gna = flash_attention_gna(q_vid, k_vid, v_vid,
                               seq_shape=(8, 32, 32),
                               window_size=(2, 8, 8),
                               stride=(1, 1, 1))

# SVDQuantLinear — W4A16 + SVD low-rank correction
# (quantize_model replaces nn.Linear layers in-place)
# model = quantize_model(model, group_size=64, bits=4, rank=32)

# Serving-oriented runtime
rt = create_decode_runtime(
    backend="auto",
    paged=False,
    quantized_kv=False,
    B=1,
    H_q=8,
    H_kv=8,
    D=128,
    max_seq_len=4096,
)
out_prefill = rt.prefill(q, k, v)
out_step = rt.step(
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
    mx.random.normal((1, 8, 1, 128)).astype(mx.float16),
)
```

## Conv3D NAX support (M5+ Apple Silicon)

mlx-mfa includes a NAX-accelerated 3D convolution path for shapes matching
the SeedVR2 VAE production profile. Sprint C v1.x landed a SHIP-DEFAULT
verdict (median **1.64×** speedup vs `mx.conv_general` across 6 production
shapes); Sprint D migrated the dispatch from Python orchestrator to a
C++ `_ext.conv3d_nax_forward` binding.

### Quickstart

```python
import mlx.core as mx
from mlx_mfa.conv_nax import conv3d_nax_forward

# Channels-last layout: (B, T, H, W, C_in)
x = mx.random.normal((1, 5, 64, 64, 512)).astype(mx.float16)
w = mx.random.normal((512, 3, 3, 3, 512)).astype(mx.float16)  # (C_out, K_T, K_H, K_W, C_in)
y = conv3d_nax_forward(x, w, stride=(1,1,1), padding=(1,1,1), dilation=(1,1,1))
# y.shape == (1, 5, 64, 64, 512)
```

### Supported shapes

- 3D inputs in `(N, T, H, W, C_in)` channels-last layout (matches `mx.conv_general`)
- `3×3×3` and `1×1×1` kernels (other small kernels may work but are not in the validated set)
- FP16 dtype (BF16 supported in code paths but not yet on the validated bench set)
- `stride = (1, 1, 1)`, `dilation = (1, 1, 1)`
- Symmetric padding (int or 3-tuple) **or** asymmetric padding via
  3-tuple of `(left, right)` pairs **or** flat 6-tuple
  `(T_left, T_right, H_left, H_right, W_left, W_right)`. Causal video
  conv: `causal_pad_t=True` flag or `padding=((K_T-1, 0), (pH,pH), (pW,pW))`.

### Expected speedup vs `mx.conv_general` (M5 Max, FP16)

| Shape profile (SeedVR2 VAE) | M | K | Speedup |
|---|---:|---:|---:|
| mid_resnet (small M, K=13824) |     20,480 | 13824 | **2.26×** |
| up1_resnet (med M, K=13824)   |    147,456 | 13824 | **2.00×** |
| up2_resnet0_chunk_cap         |    297,000 | 13824 | **1.64×** |
| up3_resnet_chunk_cap (K=3456) |    592,896 |  3456 | 1.02× (parity) |
| up2_resnet_full               |  1,114,112 |  6912 | **1.65×** |
| up2_resnet0_peakflops         |  1,114,112 | 13824 | **1.54×** |

Median across the SeedVR2 VAE production set: **1.64×**. See
[`docs/conv-nax/ship-shelve-decision.md`](docs/conv-nax/ship-shelve-decision.md)
for the full 3-session §4-compliant methodology.

### Caveats

- At **K ≤ 3456** (small `in_channels`), speedup approaches parity (~1.0×)
  as the workload becomes bandwidth-bound. No regression, just no gain.
- **int32 byte-offset chunking invariant.** MPP `matmul2d` uses int32 for
  internal byte addresses; single-buffer reads beyond `2^31` bytes produce
  NaN. `conv3d_nax_forward()` auto-chunks `M` to keep each chunk's
  im2col buffer below the safety limit (`2^31 × 0.875` bytes). Users
  don't need to think about this; documenting it because it's the
  Sprint C Phase 1.2 lesson learned and the institutional rule for any
  future MPP-based code in this repo.
- **C++ entry point.** Production dispatch goes through
  `mlx_mfa._ext.conv3d_nax_forward` (Sprint D migration). The Python
  orchestrator is preserved as
  `_conv3d_nax_forward_python_legacy` for diagnostics; toggle via
  `MFA_CONV_NAX_USE_PYTHON_LEGACY=1`.

### Integration with SeedVR2 VAE

For drop-in replacement in SeedVR2 VAE Python code (or any MLX model
using `mx.conv_general` for Conv3D):

```python
from mlx_mfa.integrations.seedvr2_vae import patch_seedvr2_vae
model = patch_seedvr2_vae(model)
# Walks model modules, swaps Conv3D layers matching the NAX-eligible
# profile to route through conv3d_nax_forward(). Skips ineligible layers
# (logged with reason). Restorable via patch_seedvr2_vae(model, restore=True).
```

## Sparse attention on M5+ (v2.33.1)

`mlx_mfa.flash_attention_sparse(q, k, v, block_mask, ...)` is the
block-sparse attention API for FlashVSR / SparkVSR / similar LCSA
patterns. On M5+ Apple Silicon it routes through MLX's SDPA after
expanding the block mask to a float bias, with the expanded bias
cached by `id(block_mask)` since **v2.33.1**.

- **Cache HIT** (same `block_mask` Python object reused across attention
  calls — common production pattern: build mask once per forward pass):
  full SDPA-direct performance recovered; <10% overhead vs calling
  `mx.fast.scaled_dot_product_attention` with a prebuilt bias.
- **Cache MISS** (fresh mask each call — e.g. FlashVSR's per-DiT-layer
  `generate_draft_block_mask_mlx`): falls back to the v2.33.0 expansion
  path; no faster, no slower.

The NAX-native block-skip path that exploits sparsity at the kernel level
is in development as Sprint B Phase 1.x (expected 3-15× speedup at typical
LCSA density — see `docs/lcsa-nax/survey-report.md` §10).

Pre-M5 hardware (M1-M4) is unchanged: routes through the native C++
STEEL V1 sparse kernel that already skips masked tiles.

## License

MIT. See [`LICENSE`](LICENSE).

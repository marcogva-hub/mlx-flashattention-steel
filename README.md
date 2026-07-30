# mlx-mfa

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

Since early May 2026, all development and testing has run on my M5 Max, where
the focus has been the NAX (Neural Accelerator) implementation that this
hardware makes possible. My M1 Max now serves as a secondary validation target.

v2.32.0 introduces a **strategic shift in dispatch on M5+ NAX hardware**.
Apple's MLX 0.31.2 ships an excellent NAX-based SDPA kernel
(`steel_attention_nax.h`) that matches the V6NAX NAX-direct path mlx-mfa
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

The v2.31.0 performance numbers (V6NAX +33-40% wins on D=128) were measured
under specific environmental conditions that did not reproduce in the
v2.32.0 cross-session diagnostic. v2.32.0 ships with reproducible-conditions
methodology baked into the bench infrastructure (`bench/v32_multisession_capture.py`,
`.doc-archive/docs/v6-nax/v32-multisession-protocol.md`, `CLAUDE_V6_NAX.md` Artifact #5).
The architectural improvements that motivated v2.31.0 (V6NAX NAX-direct
forward kernel, multi-SG parallelism via per-SG row partitioning) remain
in the codebase as a regression canary and as the dispatched path when
`MFA_DISABLE_SDPA_ROUTE=1` is set.

v2.31.0 shipped the **V6NAX NAX-direct rewrite**. V6 NAX's forward hot path
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
had tile-boundary FP rounding artifacts. Dispatch is shape-aware: V6NAX is
default for D=128 and D=64 N≥2048, legacy stays for D=64 small-N
(FlashVSR-dense regresses under V6NAX — root cause TBD).

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

---

The foreword above is the author's historical account. The remainder of this
page describes the current code and its executable routing contracts.

Current version: **2.62.0**

## What the package provides

`mlx-mfa` is a macOS/Apple-Silicon extension for MLX. It exposes dense,
block-sparse, grouped-neighborhood, packed-varlen, paged and quantized
attention surfaces, plus cache/runtime helpers. Public entry points choose
between mlx-mfa Metal kernels and MLX primitives according to shape, dtype,
hardware and feature gates.

The package is not a single always-on kernel. A fallback to MLX is part of the
contract whenever the native path is unsupported or outside its measured
envelope. Use dispatch tracing or hook telemetry when the selected binary
matters.

## Install

Requirements are Python 3.10 or newer, MLX 0.31.2 or newer, macOS on arm64,
CMake 3.24 or newer, and a working Apple Metal toolchain.

```bash
python -m pip install mlx-mfa
```

The distribution is source-first: installation compiles `_ext` against the
MLX present in that environment. Import performs a full-version ABI check and
warns if the extension and runtime MLX versions differ. Note: pinning an older
MLX under build isolation compiles `_ext` against the latest MLX, so it is built
against a different MLX than the runtime — this yields a loud warning and a safe
SDPA fallback (correct results, no acceleration); build with
`--no-build-isolation` against the pinned MLX to avoid it.

For a checkout:

```bash
CMAKE_ARGS="-DPython_EXECUTABLE=$PWD/.venv/bin/python" \
  .venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -c "import mlx_mfa; print(mlx_mfa.__version__)"
```

The build fails loudly on non-macOS or non-arm64 targets. The host deployment
floor is macOS 14; M5/Metal-4 functionality is detected at runtime.

## First call

Inputs use BHND layout: `[batch, heads, sequence, head_dim]`.

```python
import mlx.core as mx
from mlx_mfa import flash_attention

q = mx.random.normal((1, 8, 2048, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, 2048, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 2048, 128)).astype(mx.float16)
out = flash_attention(q, k, v, causal=True)
mx.eval(out)
```

`backend="auto"` is the normal mode. `backend="sdpa"` requests MLX SDPA;
`backend="mfa"` is an expert override and may select a path that is slower than
the automatic choice.

## Current routing model

| Public surface | Native route | Deliberate fallback |
|---|---|---|
| dense `flash_attention` | M5 D128 self-attention from the code gate; narrow decode carveouts; legacy STEEL tiers | unsupported features, D512 and cells where MLX is selected |
| `flash_attention_sparse` | measured BT32 V6 NAX cells; BT64 may expand to BT32 | all unmeasured sparse cells use masked SDPA or scalar coverage |
| `flash_attention_gna` | 3D f16/bf16: D128 at N>=2048, D64 at N>=4096 | STEEL or sparse representation outside that envelope; `MFA_DISABLE_GNA_NATIVE=1` is the escape |
| `flash_attention_varlen` | STEEL packed-varlen; narrow V6 NAX route only with `MFA_ENABLE_VARLEN_NAX=1` | fp32, D512 and other unsupported inputs use per-segment split/concat |
| dense backward | D64 V6 split backward from N>=2048 unless disabled | D128 is opt-in; unsupported cases use SDPA VJP |
| sparse backward | hybrid or full-native only under explicit controls | SDPA VJP is the default outside the opt-in contract |
| `mx.conv_general` hook | eligible Conv3D calls on M5 use NAX/MPP | original MLX function for every rejected shape |

The exact sparse cells are intentionally narrow. They are listed in
[`docs/reference/dispatch-map.md`](docs/reference/dispatch-map.md), not inferred
from a broad density rule.

## Specialized examples

```python
import mlx.core as mx
from mlx_mfa import flash_attention_sparse, make_sliding_window_mask

q = mx.random.normal((1, 4, 64, 64)).astype(mx.float16)
k = mx.random.normal((1, 4, 64, 64)).astype(mx.float16)
v = mx.random.normal((1, 4, 64, 64)).astype(mx.float16)
mask = make_sliding_window_mask(64, 16, head_dim=64)
out = flash_attention_sparse(q, k, v, mask)
mx.eval(out)
```

```python
import mlx.core as mx
from mlx_mfa import flash_attention_gna

q = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
k = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
v = mx.random.normal((1, 4, 64, 128)).astype(mx.float16)
out = flash_attention_gna(
    q, k, v,
    seq_shape=(1, 8, 8),
    window_size=(1, 3, 3),
    stride=(1, 1, 1),
)
mx.eval(out)
```

Packed-varlen input stores multiple independent sequences in the sequence axis
of a B=1 tensor. Cumulative sequence arrays delimit each segment.

```python
import mlx.core as mx
from mlx_mfa import flash_attention_varlen

q = mx.random.normal((1, 4, 24, 64)).astype(mx.float16)
k = mx.random.normal((1, 4, 24, 64)).astype(mx.float16)
v = mx.random.normal((1, 4, 24, 64)).astype(mx.float16)
cu_q = mx.array([0, 8, 24], dtype=mx.int32)
cu_k = mx.array([0, 8, 24], dtype=mx.int32)
out = flash_attention_varlen(
    q, k, v,
    cu_seqlens_q=cu_q,
    cu_seqlens_k=cu_k,
    max_seqlen_q=16,
    max_seqlen_k=16,
    causal=False,
)
mx.eval(out)
```

## Observe what ran

Dense and attention routes expose dispatch traces in the test/benchmark
infrastructure. The transparent Conv3D hook exposes counters:

```python
import mlx_mfa

mlx_mfa.reset_hook_stats()
# run the workload
print(mlx_mfa.get_hook_stats())
```

Set `MLX_MFA_HOOK_TELEMETRY=verbose` before import to warn on each hook
fallback. `MFA_DISABLE_AUTO_HOOKS=1` prevents installation; `mlx_mfa.disable()`
and `mlx_mfa.enable()` control it in process.

## Hardened measurements

The active numbers below were produced on 2026-07-13 with MLX 0.31.2 on an
Apple M5 Max running macOS 27 beta. Both arms used the same dtype, both
terminals were fingerprinted, outputs were checked against an fp32 oracle, and
sub-millisecond cells used 20 dispatches per sample.

They were revalidated on 2026-07-30 under a remediated fan curve (regulator-controlled),
on MLX 0.31.2 and 0.32.0, with an under-load frequency gate. No routing classification
flips: the sparse gate's engaging region stays 61/61 win on both MLX versions. See
[RESULTS.md](RESULTS.md) for the per-row provenance, the day null floors, and the new
longer-kL decode and N6144 evidence.

- Sparse gate map: 122 cells, classified with a 7.53% same-path noise floor;
  61 won, 55 lost and 6 were unresolved. This map is why the sparse gate is
  cell-based instead of density-only.
- Causal sparse D128, N8192, B*H=12, fp16, block density 0.30: masked SDPA took
  10.822-10.829 ms while `v6nax_sparse` took 2.787-2.814 ms, or
  **3.85-3.88x** in the direction SDPA/native.
- GNA D128, N4096, fp16, 3D 1x7x7 window: MLX SDPA/native was **2.39-2.44x**
  across the two process orders when revalidated 2026-07-30. The 2026-07-13
  value **1.18-1.24x** is retained as historical (throttle-compressed: this
  sub-millisecond window was the one ratio the beta regulator compressed).
- Decode qL=8, D64, GQA=8, non-causal, fp16, kL=4096: SDPA/MFA was
  **1.21-1.25x** across the two process orders.

Historical ratios in the foreword and published changelog predate this
measurement contract. They remain provenance, not current performance claims.

## Training

D64 dense backward is automatic for f16/bf16, sequence length at least 2048,
and supported attention features. `MFA_DISABLE_V6_BACKWARD=1` restores the
SDPA-VJP route. D128 is research opt-in through
`MFA_ENABLE_V6_BACKWARD=1`.

Sparse full-native backward uses the optional natural-log LSE emitted by the
V6 NAX sparse forward and remains opt-in through
`MFA_V6_BWD_SPARSE_NATIVE=1`. See
[`TRAINING_QUICKSTART.md`](docs/reference/TRAINING_QUICKSTART.md).

## Documentation

- [API manual](docs/reference/API_MANUAL.md)
- [Runtime dispatch map](docs/reference/dispatch-map.md)
- [Environment variables](ENV_VARS.md)
- [Feature coverage](docs/reference/FEATURE_COVERAGE.md)
- [Serving guide](docs/reference/SERVING_GUIDE.md)
- [Benchmark contract and results](RESULTS.md)
- [Naming and causal convention](NAMING.md)

## License

The project is MIT licensed. Portions derived from Draw Things/ccv and Apple
MLX carry their notices in `LICENSE-DRAWTHINGS` and `THIRD_PARTY_LICENSES`.

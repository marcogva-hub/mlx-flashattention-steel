# mlx-mfa

`mlx-mfa` is a Metal Flash Attention + serving-oriented runtime layer for MLX on
Apple Silicon. It provides high-performance attention kernels, runtime helpers,
and cache abstractions for dense training/inference plus modern serving flows.

Current version: **2.11.0** — M3/M4 optimization pass with full dual-chip benchmarks.

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

v2.11.0 includes comprehensive benchmarks on both Apple M1 Max and M4 Max.
The M3+ dispatch optimization delivers up to 2.07× over SDPA for D=64 causal
attention on M4 Max, and window masking reaches 20.8× on both chips.
See `docs/benchmarks/RESULTS.md` for complete data across all shapes, dtypes,
and both hardware generations — including where MFA loses, so you can evaluate
if it benefits your specific workload.

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
- Serving/runtime capability surface is now substantially expanded:
  - paged KV + packed varlen query support
  - paged continuous batching/remap
  - explicit chunked prefill
  - runtime-managed prefix reuse
  - runtime speculative draft/verify flow
  - deeper splitfuse runtime integration
  - KV cache abstraction layer
  - minimal real hybrid/offload-capable cache behavior (local offload tier)

## Limitations

- Main validation hardware is **Apple M1 Max**.
- Broad parity claims against CUDA FlashAttention ecosystems are not made.
- Some advanced paths are intentionally narrow, bridge-based, or explicit-only.
- Hybrid offload is currently a **local offload milestone**, not remote/
  distributed cache infrastructure.
- Future major hardware-specific optimization work is deferred pending newer
  Apple hardware (M5+).

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
| External cache adapter layer | Experimental groundwork | Concrete local backend provided; external backend integrations pending |

## Repository Guide

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
| Production | V2 dense causal small-D path; window/sparse tile-skip; SDPA fallback policy |
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
/Users/marcomarcelino/code/mlx-mfa-v2/.venv/bin/python -m pip install -e .
```

## Minimal Usage

```python
import mlx.core as mx
from mlx_mfa import flash_attention, create_decode_runtime

# Dense attention
q = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
k = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 1024, 128)).astype(mx.float16)
out = flash_attention(q, k, v, causal=True)

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

## License

MIT. See [`LICENSE`](LICENSE).

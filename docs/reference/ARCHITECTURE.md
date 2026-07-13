# Architecture

## Layers

```text
application
  -> mlx_mfa public Python API
      -> validation and route policy
          -> MLX primitive fallback
          -> mlx_mfa._ext nanobind primitive
              -> generated Metal source
              -> ShaderCache pipeline
              -> MLX command encoder
```

Python owns public validation, feature composition and measured routing.
Objective-C++ owns Metal compilation and pipeline caching. C++ primitives bind
MLX arrays, allocate outputs and encode kernels. Most Metal kernels are source
strings generated for a shape/configuration at runtime.

## Dense attention

`flash_attention` validates BHND tensors, dtype, head relationships and optional
features before calling the policy. On M5, dense D128 self-attention may use the
V6 NAX path; narrow decode cells use the STEEL primitive; other cells may use
MLX SDPA. The route is a performance decision, not a statement that one family
is universally faster.

Autograd wraps selected forwards with a custom VJP. D64 long-sequence backward
can use V6 split kernels; unsupported or disabled cases delegate to MLX VJP.

## Sparse attention

The public sparse path first validates block geometry and computes actual block
density. A single Python predicate owns the measured V6 NAX gate. BT64 support
is represented as a 2x2 expansion into BT32 before that predicate is applied.

The native source contains two paths: `v6nax_sparse` and `scalar_fallback`.
V6 NAX uses BQ32/BK32/WM2 and skips inactive key blocks before QK/PV work. Its
optional LSE store converts the online log2 state to natural log for backward.

## GNA

Grouped Neighborhood Attention maps a logical 1D/2D/3D coordinate system to a
local key neighborhood. The public path selects V6 NAX only for its measured 3D
envelope, otherwise retaining STEEL or sparse-based coverage. The V6 kernel
performs a window-range check in the key-tile loop and then applies exact
per-element masking.

## Packed and paged sequences

Packed varlen stores independent segments in one B=1 allocation. Cumulative
length arrays define segment boundaries. The public implementation can select
STEEL, a narrow opt-in V6 NAX path, or per-segment MLX calls. Metadata is
validated unless the explicit trust control is enabled.

Paged attention addresses K/V through page tables and sequence lengths. Cache
adapters expose dense, paged, quantized, hybrid and external storage behind a
capability interface used by inference contexts and decode runtimes.

## Transparent hooks

Import installs a wrapper around `mx.conv_general` unless disabled. Eligibility
is checked per call; rejected calls invoke the original MLX function. Counters
separate native executions from fallbacks and retain bounded fallback reasons.

## Source generation and cache identity

`ShaderCache` keys contain the fields that alter generated code or dispatch
semantics. Current locks cover dense/sparse tile coherence, varlen generated
constants versus grid metadata, STEEL source replacement, baked sparse scale,
and custom dispatch-table reload on file mtime.

Pipeline caches are process-local. An expert source-dump or tile override must
flow through generation, cache identity and host grid as one configuration.

## Hardware separation

STEEL simdgroup paths cover pre-M5 hardware. V6 NAX requires the runtime M5
capability. Host compilation still targets the macOS 14 floor; newer Metal
features are compiled as JIT source only after runtime checks.

## Failure model

Malformed public input raises a contextual Python exception. A valid but
unsupported configuration delegates when a correct fallback exists. Direct
expert bindings reject unsupported dimensions rather than silently invoking a
different binary.

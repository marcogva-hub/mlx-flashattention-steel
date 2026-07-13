# Serving Guide

This guide covers autoregressive inference surfaces and cache ownership.

## Basic decode

Use `flash_attention` for contiguous KV. The public dispatcher selects the narrow measured decode carveouts and otherwise delegates to MLX SDPA.

```python
import math
import mlx.core as mx
from mlx_mfa import flash_attention

q = mx.random.normal((1, 8, 8, 64)).astype(mx.float16)
k = mx.random.normal((1, 1, 4096, 64)).astype(mx.float16)
v = mx.random.normal((1, 1, 4096, 64)).astype(mx.float16)
out = flash_attention(q, k, v, scale=1 / math.sqrt(64), causal=False)
mx.eval(out)
```

GQA requires `Hq` to be divisible by `Hkv`.

## Contiguous cache

`flash_attention_kvcache` accepts cache tensors and an update position. `flash_attention_kvcache_rope_append` combines append, RoPE handling, and attention for its supported contract.

Use these functions when the cache is a dense tensor and sequence capacity is known.

## Paged cache

`PagedKVCache` owns page pools, sequence-to-page metadata, and append state. `flash_attention_paged` consumes explicit page pools, a block table, and per-sequence lengths. The `block_size` argument must equal the second dimension of the page pools.

Paged metadata is validated by default. `MFA_PAGED_TRUST_INDICES=1` skips host validation and is only appropriate when an upstream allocator already enforces the same invariants.

## Varlen and paged-varlen

`flash_attention_varlen` consumes packed contiguous sequences plus cumulative offsets. `flash_attention_paged_varlen` combines packed queries with paged KV. Cumulative arrays must start at zero, be monotone, and terminate at the packed length.

The packed V6 NAX path is a narrow β3 opt-in controlled by `MFA_ENABLE_VARLEN_NAX=1`; all other cells retain STEEL or split-concat behavior.

## Quantized caches

TurboQuant helpers compress cache entries and expose matching paged attention surfaces. Quantized layouts, scales, and bit width must remain consistent from cache creation through attention.

## Observability

Enable `MLX_MFA_VERBOSE_DISPATCH=1` to record attention terminals and use `get_hook_stats()` for transparent hook counters. Production monitoring should distinguish route ineligibility from a runtime native-kernel warning.

## Safety

Do not disable paged or varlen metadata validation until the producer has its own capacity and monotonicity locks. A wrong page index is a device-memory safety issue, not merely a quality regression.

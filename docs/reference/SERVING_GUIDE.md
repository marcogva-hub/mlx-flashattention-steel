# LLM Serving Guide

Consolidated guide for mlx-mfa's LLM inference and serving layer.

---

## 1) Quick Start

```python
from mlx_mfa import create_decode_runtime
import mlx.core as mx

# Create a dense decode runtime
rt = create_decode_runtime(
    paged=False, H_kv=8, D=128,
    max_seq_len=2048, dtype=mx.float16,
)

# Prefill (process prompt)
q = mx.random.normal((1, 32, 128, 128)).astype(mx.float16)  # [B, H, N, D]
k = mx.random.normal((1, 8, 128, 128)).astype(mx.float16)
v = mx.random.normal((1, 8, 128, 128)).astype(mx.float16)
out = rt.prefill(q, k, v)

# Decode loop (one token at a time)
for _ in range(100):
    q_step = mx.random.normal((1, 32, 1, 128)).astype(mx.float16)
    k_step = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
    v_step = mx.random.normal((1, 8, 1, 128)).astype(mx.float16)
    out = rt.step(q_step, k_step, v_step)
```

---

## 2) Cache Types

| Type | When to Use | API |
|------|------------|-----|
| **Dense** (`paged=False`) | Single sequence, simple inference | `create_decode_runtime(paged=False, ...)` |
| **Paged** (`paged=True`) | Multi-sequence batching, memory efficiency | `create_decode_runtime(paged=True, ...)` |
| **Hybrid** | Hot/cold tiering with offload | `HybridKVCache(primary=..., secondary=..., external=...)` |

Dense caches pre-allocate `[B, H, max_seq_len, D]`. Paged caches allocate
fixed-size blocks on demand, enabling efficient multi-sequence serving.

---

## 3) Hybrid Cache (Hot/Cold/Offloaded)

```python
from mlx_mfa import HybridKVCache, DenseKVCacheAdapter, LocalHostKVStoreAdapter

hybrid = HybridKVCache(
    primary=DenseKVCacheAdapter(B=1, H_kv=8, D=128, max_seq_len=2048),
    secondary=DenseKVCacheAdapter(B=1, H_kv=8, D=128, max_seq_len=2048),
    external=LocalHostKVStoreAdapter(),
    hot_seq_capacity=4,      # max sequences in hot tier
    policy="lru",
)

# Sequences auto-promote to hot on access, demote on capacity pressure
hybrid.append(k, v, seq_id=0)  # hot
hybrid.append(k, v, seq_id=1)  # hot
# ... when capacity exceeded, LRU victim demoted to cold/offloaded
```

**Tiers**:
- **Hot** (primary): fast access, limited capacity
- **Cold** (secondary): slower, larger capacity
- **Offloaded** (external): unlimited, highest latency

**Pinning**: `hybrid.pin(seq_id=0)` prevents eviction of critical sequences.

---

## 4) Multi-Sequence Paged Batching

```python
rt = create_decode_runtime(
    paged=True, H_kv=8, D=128,
    max_seq_len=4096, dtype=mx.float16,
    num_blocks=256, block_size=64,
)

# Prefill multiple sequences
for seq_id in range(4):
    rt.prefill(q, k, v, seq_id=seq_id)

# Decode with batch remapping
out = rt.step(q_batch, k_batch, v_batch, cache_batch_idx=[0, 2, 3])
```

---

## 5) Prefix Caching

```python
# Register a shared prefix (e.g., system prompt)
prefix_id = rt.register_prefix(prefix_k, prefix_v)

# Seed prefix for a new sequence
rt.seed_prefix(seq_id=5, prefix_id=prefix_id)

# Prefill with prefix (skips recomputing prefix attention)
out = rt.prefill_with_prefix(q, k, v, seq_id=5, prefix_id=prefix_id)
```

---

## 6) Speculative Decoding

```python
# Draft model generates candidate tokens
draft_ids = mx.array([[10, 20, 30]])
draft_logprobs = mx.array([[-1.0, -2.0, -3.0]])

out = rt.speculative_step(
    q_target, k_target, v_target,
    draft_ids=draft_ids,
    draft_logprobs=draft_logprobs,
)
# out["accepted_prefix_lens"] tells how many draft tokens were accepted
```

---

## 7) Chunked Prefill

For long sequences that exceed GPU memory in a single pass:

```python
out = rt.chunked_prefill(q, k, v, chunk_size=512)
```

Processes the prompt in chunks of 512 tokens, accumulating the KV cache
incrementally. Supports both `query_layout="batched"` and `query_layout="packed"`.

---

## 8) Splitfuse

Narrow path for fusing split attention patterns:

```python
out = rt.splitfuse(q, k, v, split_points=[256, 512])
```

Shape-sensitive — test with your specific workload before relying on it.

---

## 9) TurboQuant KV Compression

For memory-constrained long-context serving, TurboQuant compresses KV caches
to 2-4 bits with ~3.8× memory savings (K+V at 3-bit).

```python
from mlx_mfa import create_decode_runtime
import mlx.core as mx

rt = create_decode_runtime(
    turboquant=True,          # enable TQ compression
    tq_bits=3,                # 2, 3, or 4 bits
    tq_v=True,                # compress V as well as K (Phase 3)
    paged=True,
    H_q=32, H_kv=8, D=128,
    max_seq_len=8192,
    dtype=mx.float16,
    block_size=64,
)

# Same prefill/step API as standard runtime
out = rt.prefill(q, k, v)
out = rt.step(q_step, k_step, v_step)
```

**Trade-offs:**
- Memory: ~3.8× savings (K+V), enabling longer contexts
- Quality: cosine similarity 0.96–0.97 vs fp16 (Phase 3)
- Latency: currently higher than fp16 due to Python pack overhead

---

## 10) mlx-lm Integration

```python
from mlx_mfa import patch_mlx_lm, unpatch_mlx_lm

# Monkey-patch mlx-lm to use MFA kernels
patch_mlx_lm()

# Now mlx-lm's generate() uses flash_attention under the hood
# ...

unpatch_mlx_lm()  # restore original
```

Compatibility: mlx-lm 0.30+. Falls back to original for unsupported
configurations (quantized cache, sinks, array masks, unsupported D/dtype).

---

## 11) Environment Variables

| Variable | Effect |
|----------|--------|
| `MFA_DISABLE_V2=1` | Skip V2 kernel, use V1 |
| `MFA_FORCE_V2=1` | Force V2 even on M3+ D≤128 causal |
| `MFA_FORCE_SDPA=1` | Force SDPA fallback everywhere |
| `MFA_FORCE_GEN=15` | Simulate M3 architecture on M1 |
| `MFA_ENABLE_V3=1` | Enable V3 experimental kernel |
| `MFA_ENABLE_V4=1` | Enable V4 experimental kernel |
| `MFA_ENABLE_V5=1` | Enable V5 experimental kernel |
| `MLX_MFA_VERBOSE_DISPATCH=1` | Log kernel dispatch decisions |

---

## 12) Component Status (v2.27.0)

| Component | Status |
|-----------|--------|
| Dense decode runtime | Production |
| Paged KV (batched/packed) | Production |
| HybridKVCache (hot/cold/offloaded) | Production (local offload) |
| Prefix caching | Production |
| Speculative decode | Production (narrow) |
| Chunked prefill (batched) | Production |
| Chunked prefill (packed) | Supported (v2.14.1) |
| Splitfuse | Narrow/conditional |
| mlx-lm patch | Production |
| TurboQuant Phase 1 (non-fused) | Production (v2.21.0) |
| TurboQuant Phase 2 (K fused) | Production (v2.22.0) |
| TurboQuant Phase 3 (K+V fused) | Production (v2.23.0) |
| TurboQuant Phase 4 (optimal packing + WHT) | Production (v2.24.0) |
| SVDQuantLinear (W4A16 + SVD correction) | Production (v2.25.0) |
| GNA native kernel (3D window) | Production (v2.26.0) |
| Native `attn_bias` (modes 1/2) | Production (v2.27.0) |
| Remote/distributed offload | Deferred (M5+) |

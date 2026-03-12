# Runtime Unification Inventory (Task 1)

Date: 2026-03-12

## Current runtime/stateful entry points

### Context objects (`mlx_mfa/inference.py`)
- `InferenceContext`: dense KV decode lifecycle (`prefill`, `step`, `reset`).
- `PagedInferenceContext`: paged KV lifecycle with `seq_id`.
- `SageInferenceContext`: quantized-K decode lifecycle (`QuantizedKVCache`).
- `create_inference_context(...)`: factory with dense/paged/sage routing.

### Attention helpers (`mlx_mfa/attention.py`)
- `flash_attention_kvcache(...)`: unified dense+paged KV API, append mode, RoPE/ALiBi/window support.
- `flash_attention_paged(...)`: explicit paged-gather helper (still public and directly callable).
- `make_shared_prefix_cache(...)`: prefix precompute helper.
- `flash_attention_splitfuse(...)`: prefill+decode helper.
- `flash_attention_speculative_verify(...)`: speculative verify helper.

### Public exports (`mlx_mfa/__init__.py`)
- Exposes all of the above at top level, so users can mix styles freely.

## Observed overlap / fragmentation

1. Decode lifecycle split across 3 context classes + 1 factory + direct helper calls.
2. Shared-prefix, splitfuse, and speculative helpers are standalone and not connected to the decode runtime surface.
3. Backend selection logic appears in multiple places (factory policy + user-side branching).
4. Users can construct valid flows, but there is no small “single runtime object” that bundles common operations.

## Minimal unification targets for this pass

1. Add a lightweight `mlx_mfa/runtime.py` layer with a single decode runtime wrapper.
2. Reuse existing contexts internally (no context rewrite).
3. Expose small helper methods on the runtime surface for:
   - shared-prefix cache
   - splitfuse
   - speculative verify
4. Keep dense/paged/sage policy narrow and explicit (V2 default, Sage specialized decode).
5. Centralize backend/context selection in one place to reduce duplicated Python branching.

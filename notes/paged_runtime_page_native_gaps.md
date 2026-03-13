# Paged Runtime Page-Native Gaps (Final Serving Completion Pass)

## Current Observed Bridge/Gather Hotspots

1. **Splitfuse decode on paged runtime**
   - Current runtime-integrated splitfuse path can derive decode cache from paged
     context only by materializing attention-ready contiguous K/V for a `seq_id`.
   - This is narrow and intentionally explicit.

2. **Packed hetero chunked prefill bridge path**
   - Heterogeneous packed-query chunked prefill still uses per-active-sequence
     bridge logic inside chunk loop.
   - Correct and scheduler-friendly, but not yet fully fused/page-native.

3. **Paged speculative verify fallback (pre-pass baseline)**
   - Previously pulled contiguous K/V via adapter attention views before verify.
   - This was a direct gather-heavy runtime bridge point.

## Improvements Selected in This Pass

- **Improvement A (implemented):** paged speculative verify now routes through
  paged-native cache inputs (`k_pages/v_pages + block_table + seq_lens`) via
  `flash_attention_speculative_verify_paged(...)`, removing explicit runtime
  dense-cache reconstruction.

- **Improvement B (implemented):** runtime splitfuse integration now has
  cache-aware `splitfuse_step(...)`, reducing manual caller reconstruction for
  dense runtime and providing a narrow paged single-sequence route.

## Remaining Gaps (Future Work)

- Full page-native splitfuse without dense decode-cache materialization.
- Fully fused packed hetero paged prefill/decode scheduling path (bridge-free).
- Deeper block-level reuse between paged chunking and speculative verification.

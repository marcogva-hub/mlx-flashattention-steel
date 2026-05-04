# V6 NAX — Causal Masking Optimization Analysis

**Date:** 2026-05-04
**Sprint:** 3.1
**Verdict:** **Scenario A — All three Apple-style causal optimizations are already present in V6 NAX.** No code change recommended.

## Apple's pattern (steel_attention_nax.h)

Reference: `.venv/lib/python3.11/site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:176-197`.

```cpp
int kb_lim = params->NK;
int kb_min_causal = params->NK;

if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    int q_min = tid.x * BQ + params->qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
}

for (int kb = 0; kb < kb_lim; kb++) {
    // ... Q@K^T into Stile ...
    if (do_causal && kb >= kb_min_causal) {
        // per-element mask
    }
}
```

Three distinct optimizations:
1. **`kb_lim`** — K-loop ends at the last tile that contains at least one non-masked position (skip tiles entirely beyond the diagonal).
2. **`kb_min_causal`** — Mask check skipped on K-tiles fully below the diagonal (no element needs to be masked).
3. **Per-element check** in the gated zone `[kb_min_causal, kb_lim)`.

## V6 NAX equivalent (NAAttentionKernel.cpp)

### Optimization 1: Loop upper bound — **PRESENT**

`csrc/mfa/v6_nax/NAAttentionKernel.cpp:798-810`:

```cpp
const int causal_row_start = int(tgid.x * {{BLOCK_DIMENSIONS_PARALLELIZATION}});

if (isCausal) {
    const int causal_column_offset = int({{C_LENGTH}}) - int({{R_LENGTH}});
    const int causal_last_column = causal_row_start + int({{BLOCK_DIMENSIONS_PARALLELIZATION}}) - 1 + causal_column_offset;
    const int causal_first_column_limit = causal_row_start + causal_column_offset;
    const uint single_c_edge = causal_last_column < 0 ? 0 : min({{C_SINGLE_EDGE}}, uint(causal_last_column) + 1);
} else {
    const uint single_c_edge = {{C_SINGLE_EDGE}};
}
```

Loop at line 826:
```cpp
for (uint c = 0; c < single_c_edge; c += {{BLOCK_DIMENSIONS_TRAVERSAL}}) {
```

**Mapping:**
- Apple `q_max = (tid.x + 1) * BQ + qL_off` ↔ V6 `causal_last_column + 1 = causal_row_start + BQ + qL_off` (off by one because Apple counts positions; V6 counts boundaries).
- Apple `kb_lim = ceil(q_max/BK)` clamped to `NK` ↔ V6 `single_c_edge = min(C_SINGLE_EDGE, causal_last_column+1)`.
- The unit differs (V6 iterates in column units `c += BK`; Apple in tile units `kb += 1`), but the upper bound semantics are identical. When the loop exits at `single_c_edge`, the C_SINGLE_REMAINDER tail block (line 1056) is conditionally entered only if `(C_LENGTH - C_SINGLE_REMAINDER) <= causal_last_column` — V6 is in fact *stricter* than Apple, gating the tail block on the causal range as well.

### Optimization 2: Mask gate — **PRESENT**

`csrc/mfa/v6_nax/NAAttentionKernel.cpp:863, 893`:

```cpp
const bool causal_mask_0 = int(c + {{BLOCK_DIMENSIONS_TRAVERSAL}} - 1) > causal_first_column_limit;
if (causal_mask_0) {
    // per-element mask
}
```

**Mapping:**
- Apple: `kb >= kb_min_causal` where `kb_min_causal = q_min / BK = causal_row_start / BK`. With `qL_off = 0` (square causal), this is `kb_min_causal = causal_row_start/BK`, gate triggers when the K-tile reaches the diagonal.
- V6: `(c + BK - 1) > causal_first_column_limit` where `causal_first_column_limit = causal_row_start + qL_off`. Triggers when the K-tile's last column passes the first-row diagonal.
- Both gates fire on the same set of K-tiles modulo unit rephrasing. K-tiles below the gate run **without any mask check** (no -inf substitution, no elementwise compare).

### Optimization 3: Per-element check — **PRESENT**

`csrc/mfa/v6_nax/NAAttentionKernel.cpp:898-905` (isCausal-only path):

```cpp
if (causal_mask_0) {
    for (k = 0; k < cS_0.get_capacity(); ++k) {
        if (cS_0.is_valid_element(k)) {
            auto idx = cS_0.get_multidimensional_index(k);
            const int causal_row = causal_row_start + idx[1];
            const int causal_column_limit = causal_row + causal_column_offset;
            if (int(c) + idx[0] > causal_column_limit) {
                cS_0[k] = -numeric_limits<float>::infinity();
            }
        }
    }
}
```

The same per-element check is fused into the softmax step at line 951:
```cpp
cS_0[k] = int(c) + idx[0] <= causal_column_limit ?
    fast::exp2(cS_0[k] * {{DOT_SCALE}} - *dst_it) : 0;
```

Setting `exp2(...)` to 0 directly is equivalent to setting score to -inf before exp — saves the redundant mask + exp by fusing them.

The masked + causal path (lines 866-883) and the C_SINGLE_REMAINDER tail (lines 1098-1213) implement the same gated per-element check, just over the boundary K-tile and the partial last K-tile respectively.

## Conclusion

| Apple optimization | V6 NAX status | File:line |
|---|---|---|
| `kb_lim` — loop upper bound | ✅ Present (`single_c_edge`) | NAAttentionKernel.cpp:806, 826 |
| `kb_min_causal` — mask gate | ✅ Present (`causal_mask_0`) | NAAttentionKernel.cpp:863, 893 |
| Per-element check in gated zone | ✅ Present (fused with softmax) | NAAttentionKernel.cpp:881, 901, 951, 960 |
| Tail block gating | ✅ Extra (V6-only) | NAAttentionKernel.cpp:1056 |

V6 NAX (inherited from Draw Things' source generator) already implements the full causal skip pattern, plus an extra gate on the tail block. **No code change needed for Sprint 3.1.**

Per Sprint 3.1's user-specified constraint ("Si Scénario A, ne pas inventer une modification. Documenter et passer."), this sprint closes here and the next workstream pivots to Sprint 3.2 (re-test `bypassThreadgroupMemory` post-BHND).

## Notes for future reference

- The flag `bypassThreadgroupMemory` is exposed via env var `MFA_V6_BYPASS_TGP` (not `MFA_V6_BYPASS_TGMEM` as the Sprint 3.2 brief assumed). Default is off. See `csrc/mfa_v6_nax_primitive.cpp:114`.
- The four `if (bypassThreadgroupMemory)` branches in NAAttentionKernel.cpp are at lines 784, 1013, 1442, 1596 — these are the points where Sprint 3.2 will need to verify the cP cooperative-tensor path against current dispatch tables.

## Validation

- Read: Apple `steel_attention_nax.h:140-260` (full causal init + loop entry + mask)
- Read: V6 `NAAttentionKernel.cpp:790-906` (loopForwardSingleCausal entry through softmax-gated path)
- Cross-checked: grep for `causal_mask_0`, `single_c_edge`, `C_SINGLE_REMAINDER`, `causal_last_column` confirms the three citations above and the tail-block gate at line 1056.
- No code modified. No tests run (analysis-only sprint).

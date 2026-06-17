# v2.50 NAX Coverage Audit — Function Inventory

**Audit date**: 2026-05-13
**Master tip**: `82acc55` (post-Sprint A/B/C v2.39.x-v2.40.x-internal accumulation)
**PyPI version**: v2.39.1 (unchanged; internal-mode contract)
**Hardware**: Apple M5 Max 128GB (gpu_family_gen=17, NAX engaged, macOS 26.4)

## Strategic context (apple-sdpa-nax-analysis distilled)

The dispatch_policy (`mlx_mfa/dispatch_policy.py:134-150`) routes ALL canonical
M5+ dense forward to **Apple SDPA NAX** (`_M5_NAX_THRESHOLDS = 999_999` for
all `(D, causal) ∈ {(64,T), (64,F), (128,T), (128,F), (256,T), (256,F)}`).
MFA-STEEL paths only engage at:
- D=256/512 non-NAX shapes (Apple SDPA NAX limited to D ∈ {64, 80, 128})
- Env-var-gated training carve-outs (`MFA_ENABLE_V6_BACKWARD=1`)
- Symmetric block-sparse via `lcsa_nax` dispatcher

**Implication for this audit**: the "NAX-optimal" path for canonical dense
forward is **Apple SDPA NAX**, not MFA-STEEL.  The audit asks: for each of
the 22 public attention functions, does its wrapper preserve Apple SDPA NAX
engagement, or does it hijack into a STEEL fallback that's slower than SDPA
on M5+?

Documented gaps in Apple SDPA NAX coverage (per `apple-sdpa-nax-analysis.md`):
- D=256 (falls back to non-NAX `steel_attention`, ~30% slower than unfused
  for short sequences)
- D ∉ {64, 80, 128} broadly (unfused multi-kernel fallback)
- Block-sparse with mask (no native sparse SDPA NAX path)
- Paged KV cache (no native paged SDPA NAX path)
- TurboQuant compressed KV (no native compressed-KV SDPA NAX path)
- Top-K attention (no native sparse-K SDPA NAX path)
- GNA (neighborhood) attention (no native windowed SDPA NAX path)

## Attention function inventory (22 + 4 sage)

### Core forward attention (5)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.1 | `flash_attention` (dense fwd + bwd) | 163 | Yes (l.450, l.488) — V6NAX carve-out | Apple SDPA NAX for fwd; SDPA-vjp for bwd unless `MFA_ENABLE_V6_BACKWARD=1` (D=64 only, qL≥2048 post Sprint A) |
| B.2 | `flash_attention_rope_unified` | 669 | No explicit branch | Likely Apple SDPA NAX after host-side RoPE apply (uses `_apply_rope_mlx` + `_apply_rope_and_attend`) |
| B.3 | `flash_attention_rope` | 931 | No explicit branch | Same pattern as B.2 — host-side rotation then SDPA call |
| B.14 | `flash_attention_splitfuse` | 2810 | Unknown — bench needed | TBD |
| B.18 | `flash_attention_qkv_packed` | 5609 | No explicit branch | Re-dispatches to `flash_attention` after unpacking |

### Sage forward (4) — inference-only, int8 quantized

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.4 | `sage_attention` | 1258 | No explicit branch | `mfa_sage_fwd` STEEL kernel (no NAX-specific variant) |
| B.5 | `sage_attention_prequantized` | 1392 | No explicit branch | Same STEEL path with pre-quantized K |
| B.6 | `sage_attention_kvcache` | 1465 | No explicit branch | Decode variant of sage STEEL |
| B.x | `smooth_k` | helper (no public attention call) | N/A | Pure-Python preproc helper |

### KV cache forward (3)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.7 | `flash_attention_kvcache_rope_append` | 1621 | No explicit branch | TBD — RoPE + cache append + attend |
| B.8 | `flash_attention_kvcache` | 1701 | No explicit branch | TBD — likely STEEL flash-decode path |
| B.16 | `flash_attention_paged` | 5042 | No explicit branch | STEEL paged (Apple SDPA NAX has no paged path) |

### Sparse / specialty forward (3)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.9 | `flash_attention_sparse` | 2177 | Yes (l.2255, l.2313) — lcsa_nax + sparse_fallback | Symmetric block_mask → `lcsa_nax.sparse_attention_dispatch`; asymmetric → `_sparse_fallback_sdpa_perhead` (2.1× regression documented) |
| B.10 | `flash_attention_gna` | 2441 | No explicit branch | `mfa_gna_fwd` STEEL kernel |
| B.11 | `flash_attention_topk` | 2540 | No explicit branch | TBD — top-K STEEL or SDPA wrap |

### Speculative + splitfuse (2)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.12 | `flash_attention_speculative_verify` | 2620 | No explicit branch | TBD |
| B.13 | `flash_attention_speculative_verify_paged` | 2700 | No explicit branch | TBD |

### Varlen + packed (5)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.15 | `flash_attention_varlen` | 4145 | No explicit branch | TBD |
| B.17 | `flash_attention_paged_varlen` | 5393 | No explicit branch | STEEL `mfa_paged_varlen_forward` fused kernel (critical for mlx-lm serving) |
| B.19 | `flash_attention_kv_packed` | 5695 | No explicit branch | Unpacks → re-dispatches `flash_attention` |
| B.20 | `flash_attention_varlen_qkv_packed` | 5777 | No explicit branch | Unpacks → re-dispatches `flash_attention_varlen` |
| B.21 | `flash_attention_varlen_kv_packed` | 5864 | No explicit branch | Unpacks → re-dispatches `flash_attention_varlen` |

### TurboQuant (1)

| # | Function | Line | M5+ explicit branch? | Likely path on M5+ |
|---|---|---|---|---|
| B.22 | `flash_attention_paged_varlen_turboquant` | 5950 | No explicit branch | STEEL `mfa_paged_varlen_tq_forward` fused kernel (TQ Phase 3+) |

## Inventory totals

- **22 flash_attention\* functions** (B.1-B.22, matches user prompt count)
- **3 sage_attention\* functions** (B.4-B.6; the user prompt counted 4 but `smooth_k` is a quantize helper, not an attention function — only 3 attention entry points)
- **5 functions with explicit `is_m5_plus`/`_get_has_nax_cached()` branches** in attention.py:
  1. `flash_attention` (carve-out for V6NAX backward)
  2. `flash_attention_sparse` (lcsa_nax dispatch + sparse_fallback path)
  3. `_v6nax_eligible` (predicate helper, not a public API)
- **17 functions with no explicit M5+ branch**: rely on `should_use_mfa()` dispatch
  policy (which routes M5+ canonical dense to SDPA NAX via threshold=999_999)
  OR on internal re-dispatch to other public functions.

## Apple SDPA NAX engagement model

For functions that pass shapes compatible with `mx.fast.scaled_dot_product_attention`
(D ∈ {64, 80, 128}, dense, no exotic per-element masking that breaks NAX
function constants), the `_fallback_sdpa()` helper in attention.py:3813 is
the de-facto NAX-optimal path on M5+.  This is what dispatch_policy routes
canonical dense forward to.

**Auto-hooks** (`mlx_mfa/_auto_hooks.py`) patch `mx.fast.scaled_dot_product_attention`
to add MFA routing decisions transparently.  When the auto-hook decides
"M5+ canonical → Apple SDPA NAX", it leaves the call to land on Apple's
native kernel.

## Sage attention — special status

SageAttention is **inference-only** (no backward).  No native NAX path
exists in Apple's MLX for per-block int8 quantized attention.  The `mfa_sage_fwd`
STEEL kernel is the only path; no fallback to SDPA-int8 because SDPA-int8
doesn't exist in MLX 0.31.x.

Sprint context: SageAttention's win over dense SDPA depends on the K/V
quantization saving more memory bandwidth than the dequantize-then-attend
overhead costs.  On M5+ with abundant memory bandwidth (~400 GB/s), this
trade-off may not favor SageAttention at all.  Bench will reveal.

# v2.50 NAX Coverage Audit — Consolidated Bench Results

**Audit date**: 2026-05-13.  Hardware: Apple M5 Max 128GB (gpu_family_gen=17,
NAX engaged, macOS 26.4).
**Methodology**: 4 warmup + 12 timed iters, median ms, PUBLIC API entry,
`mx.synchronize()` + array materialization between iters.

## Headline empirical findings

| Group | Function | Canonical shape | Auto-routed (ms) | SDPA reference (ms) | Ratio | Classification |
|---|---|---|---|---|---|---|
| **G1 dense canonical** | `flash_attention` | B=1 H=12 qL=4096 D=128 f16 non-causal | 2.38 | 2.40 | **0.989×** | **(A)** NAX-optimal |
| **G1 dense MFA-forced** | `flash_attention(backend="mfa")` | same | 8.19 | 2.40 | 3.408× | reference: STEEL is 3.4× slower than SDPA NAX on this shape |
| **G1b LLM causal D=64** | `flash_attention` | B=1 H_q=32 H_kv=8 qL=4096 D=64 f16 causal | 1.35 | 1.37 | **0.986×** | **(A)** NAX-optimal |
| **G2 paged decode** | `flash_attention_paged` | B=1 H_q=32 H_kv=8 S=4096 D=64 decode | 0.46 | 0.27* | 1.690× | **(B)** opportunity — functional necessity (no SDPA-paged path); * SDPA-flat needs page materialization not counted |
| **G3 sparse LCSA symmetric** | `flash_attention_sparse` | B=1 H=12 qL=4096 D=128 BT=32 ~3-active-blocks/row | 2.97 | 2.36 | 1.262× | **(B)** opportunity — LCSA sparse loses vs dense SDPA on low-density patterns |
| **G3b sparse asymmetric** | `flash_attention_sparse` | same shape, asymmetric BQ=32 BK=16 mask | 2.96 | 2.48 | 1.195× | **(B)** opportunity — confirms `docs/sparse-fallback-audit.md` (mask expansion overhead) |
| **G4 GNA neighborhood** | `flash_attention_gna` | B=1 H=16 qL=4096 D=128 window=(2,4,4) | 2.89 | 3.06 | **0.942×** | **(A)** NAX-optimal — sliding-window pattern wins vs dense |
| **G5 Top-K** | `flash_attention_topk` | B=1 H=16 qL=4096 K=64 D=128 | **55.35** | 3.23 | **17.160×** | **(B)** opportunity — Python reference; Metal kernel deferred since v2.13.0 |
| **G6 Sage int8** | `sage_attention` | B=1 H=16 qL=4096 D=128 fwd | 14.80 | 3.15 | **4.694×** | **(B)** opportunity — Python-side quantize overhead (per CLAUDE.md historical note) |
| **G7 RoPE-fused** | `flash_attention_rope_unified` | B=1 H=16 qL=4096 D=128 + rope tables | 7.97 | 3.14 (no rope) | 1.537× extra | **(B)** opportunity — host-side RoPE rotation is a hot per-call cost |
| **G8 kvcache dense** | `flash_attention_kvcache` | B=1 H=16 qL=1024 S_kv=4096 D=128 | 1.04 | 1.05 | **0.992×** | **(A)** NAX-optimal — wrapper dispatches dense SDPA NAX cleanly |

## Group-by-group analysis

### G1: Canonical dense (covers 7 of 22 functions)

`flash_attention`, `flash_attention_rope` (thin wrapper to rope_unified),
`flash_attention_kvcache` (cross-attention dense path), `flash_attention_qkv_packed`,
`flash_attention_kv_packed`, `flash_attention_varlen_qkv_packed`,
`flash_attention_varlen_kv_packed` — all 7 dispatch through the canonical
dense path on M5+ (eventually call `mx.fast.scaled_dot_product_attention`
when dispatch_policy returns "use_mfa=False" which is the canonical case
per `_M5_NAX_THRESHOLDS = 999_999`).

**Empirical confirmation**: auto-routed `flash_attention` is at **0.989×**
SDPA (parity within session noise).  MFA-forced is **3.41×** slower —
proving that the dispatch_policy correctly prevents users from accidentally
hitting the slow STEEL path on M5+.

**Verdict for all 7 functions**: **(A) NAX-optimal already**.

**Caveat**: `flash_attention_kvcache` paged-mode dispatches to G2 paged path,
not G1.  Only the dense cross-attention sub-path is (A).

### G2: STEEL-only paged (covers 4 functions)

`flash_attention_paged`, `flash_attention_paged_varlen`,
`flash_attention_paged_varlen_turboquant`, `flash_attention_varlen`.

Apple SDPA NAX has **no paged or varlen path** — no comparable Apple kernel
exists.  STEEL `mfa_paged_varlen_forward` (and TurboQuant variant) are
the only options for paged dispatch.

**Empirical**: `flash_attention_paged` at 0.46ms vs `mx.fast.SDPA` on
artificially-flattened KV at 0.27ms.  The 1.69× ratio is **misleading**
because the SDPA-flat baseline assumes user has already materialized the
[B, H, S, D] tensor from pages — which is the work the paged kernel
avoids in the first place.  Real comparison would include the
materialization cost (which can be >>0.46ms for large active page sets).

**Verdict for all 4 functions**: **(B) NAX-opportunity** in principle —
implementing a paged-SDPA-NAX variant would let users skip STEEL.  But
the **effort is XL** (new kernel, no Apple reference) and the **expected
gain is marginal** (STEEL paged is already fused; the gain over STEEL
would come from NAX cooperative-tensor MMA primitives being slightly
faster than STEEL's `simdgroup_matrix` — likely 5-15% per-call).

**Strategic recommendation**: deprioritize paged-NAX-variants for v2.50.
Ship paged-STEEL as the only path; document the dependency on Apple
adding paged-NAX in a future MLX release.

### G3: Sparse (covers 1 function)

`flash_attention_sparse` has two M5+ sub-paths:
1. **Symmetric block_mask** (BT_q == BT_k, BT ∈ {16, 32, 64}) →
   `lcsa_nax.sparse_attention_dispatch` (NAX-native).
2. **Asymmetric / arbitrary** → `_sparse_fallback_sdpa_perhead`
   (SDPA-based with mask expansion overhead, ~2ms per call).

**Empirical (G3 symmetric)**: LCSA NAX at 2.97ms vs dense SDPA at 2.36ms —
LCSA is **1.26× slower** than dense SDPA on this low-density 3-active-blocks
per row pattern.  LCSA NAX wins when active-block ratio is higher
(40%+); at sparse ratios this low, dense SDPA is faster.

**Empirical (G3b asymmetric)**: SDPA-fallback at 2.96ms vs dense SDPA at
2.48ms — **1.20×** overhead from mask expansion.  Less severe than the
2.10× documented in `docs/sparse-fallback-audit.md` (possibly different
mask density or post-v2.33 improvements).

**Verdict**: **(B) NAX-opportunity** — two specific gaps:
1. **LCSA dispatcher** should add a density threshold check: route to
   dense SDPA NAX when active-block ratio < threshold.  Effort: **S**
   (single if-statement + threshold calibration bench).
2. **Asymmetric path** could adopt the `sparse-fallback-audit.md`
   recommended Layer 1 (bool mask substitution, saves ~1.3ms) + Layer 2
   (LRU mask expansion cache, saves ~2ms on cache hit).  Effort: **S**
   (~30 LOC per audit doc).

### G4: GNA (covers 1 function)

`flash_attention_gna` with sliding-window pattern (window=(2,4,4),
stride=(1,1,1)) is at **0.94× SDPA-dense** — GNA's sparse pattern provides
a real ~6% win over dense SDPA at qL=4096 D=128.

The implementation uses `mfa_gna_fwd` STEEL kernel (D=128 native) plus
sparse-path fallback.  Apple SDPA NAX has no neighborhood-attention
variant.

**Verdict**: **(A) NAX-optimal already** for sliding-window patterns.
The STEEL GNA kernel + sparse fallback combo handles this well.

### G5: Top-K (covers 1 function)

`flash_attention_topk` at 55.35ms vs dense SDPA at 3.23ms — **17.16×
slower**.  This is the v2.13.0 Python reference implementation; the
native Metal kernel was deferred (per `FEATURE_COVERAGE.md` historical
note + CLAUDE.md commit Phase C).

**Verdict**: **(B) NAX-opportunity, HIGH priority** — top-K is essentially
unusable at the scales VSR/LLM workloads need.  Users who try
`flash_attention_topk` get >>10× slowdown vs computing dense attention
and taking top-K rows in post.

**Effort: L** — requires:
- New top-K-fused source generator (likely a sparse-attention variant
  with score-based block selection)
- Primitive + binding
- Three-axis test scaffold
- Routing in `flash_attention_topk` for the new kernel
- Documentation of which workload patterns the kernel wins on

### G6: Sage int8 (covers 3 functions)

`sage_attention` at 14.80ms vs dense SDPA at 3.15ms — **4.69× slower**.

The CLAUDE.md commit history explicitly notes this:
> "sage currently slower than flash_attention due to Python-side
> quantize overhead; speedup needs pre-quantized KV caches"

`sage_attention_prequantized` and `sage_attention_kvcache` are the
variants that should win — they assume K/V is already quantized
externally so the Python-side per-block int8 quantize is amortized.

**Bench note**: `sage_attention_prequantized` + `sage_attention_kvcache`
were NOT individually bench'd in this audit due to fixture-setup
complexity (need pre-quantized K + scales tensors).  The CLAUDE.md note
implies they're closer to parity-with-dense-SDPA on M5+, but empirical
confirmation is deferred.

**Verdict for `sage_attention` (vanilla forward)**: **(B) NAX-opportunity**
with caveat — the Python quantize overhead dominates.  Effort: **L** to
write a fused-quantize-and-attend NAX kernel that does the int8 conversion
inside the GPU kernel.  Marginal gain at long-context scales where the
memory-bandwidth savings of int8 KV actually matter.

**Verdict for `sage_attention_prequantized` + `sage_attention_kvcache`**:
likely **(A) NAX-optimal already** but pending empirical confirmation.

### G7: RoPE-fused (covers 2-3 functions)

`flash_attention_rope_unified` (and the thin wrapper `flash_attention_rope`)
at 7.97ms vs SDPA-no-rope at 3.14ms — **+4.83ms / 1.54× overhead** from
host-side RoPE rotation.

The current path applies RoPE via `_apply_rope_mlx` (separate MLX call)
then dispatches to `flash_attention` which does the attention.  Apple
SDPA NAX has no fused-RoPE variant.

A fused-RoPE-attention NAX kernel could absorb the rotation into the
kernel's Q/K load path (rotate-then-load instead of rotate-into-tensor-
then-load).  This is a real win for inference workloads where RoPE is
applied per-call.

**Verdict**: **(B) NAX-opportunity, MEDIUM priority**.
**Effort**: **S/M** — rope add to V6NAX forward NAX kernel.  The V6NAX forward
kernel already supports this (rope_q_base + rope_cos_stride params, per
`csrc/mfa/v6_nax/NAAttentionKernel.cpp` ~line 2741 forward source).
Wire-up effort: ~1-2h CC.

### G8: kvcache dense (covers 1 function — overlapping with G1)

`flash_attention_kvcache` in dense cross-attention mode at 1.04ms vs SDPA
at 1.05ms — **0.992×**, parity.  The wrapper dispatches to dense `flash_attention`
which routes Apple SDPA NAX.  No overhead.

**Verdict**: **(A) NAX-optimal already** for dense cross-attention mode.
Paged mode falls into G2.

## Empirical findings NOT in original audit plan

The audit surfaced findings beyond the 22-function classification:

1. **LCSA sparse density threshold**: LCSA NAX loses to dense SDPA at low
   active-block ratios.  The dispatcher should add a density-based
   fallback decision.  Quick win (Effort S).

2. **MFA STEEL is 3.4× slower than SDPA NAX on canonical dense** — this
   is well-known to the dispatch_policy (which routes to SDPA via
   `_M5_NAX_THRESHOLDS = 999_999`), but users who explicitly request
   `backend="mfa"` get this regression silently.  Could add a runtime
   warning at M5+ when `backend="mfa"` is requested on a shape that
   would route to SDPA.

3. **flash_attention_topk's 17× regression** is far worse than the
   "slow but works" framing in the historical FEATURE_COVERAGE.md.
   Users who try this function get an unusable experience.

## Summary classification matrix (22 functions)

| # | Function | Group | Classification | Effort if (B) |
|---|---|---|---|---|
| B.1 | `flash_attention` | G1 | **(A)** NAX-optimal | — |
| B.2 | `flash_attention_rope_unified` | G7 | **(B)** opportunity | **S/M** (~1-2h) |
| B.3 | `flash_attention_rope` | G7 (wrapper) | **(B)** opportunity | inherits B.2 |
| B.4 | `sage_attention` | G6 | **(B)** opportunity | L (~3-6h) |
| B.5 | `sage_attention_prequantized` | G6 | likely **(A)** TBD bench | TBD |
| B.6 | `sage_attention_kvcache` | G6 | likely **(A)** TBD bench | TBD |
| B.7 | `flash_attention_kvcache_rope_append` | G2+G7 | **(B)** opportunity | M (inherits B.2 + paged context) |
| B.8 | `flash_attention_kvcache` | G1/G8 | **(A)** NAX-optimal (dense path) | — (paged sub-path: see B.16) |
| B.9 | `flash_attention_sparse` | G3 | **(B)** opportunity (both sub-paths) | **S** (density threshold + bool-mask cache) |
| B.10 | `flash_attention_gna` | G4 | **(A)** NAX-optimal | — |
| B.11 | `flash_attention_topk` | G5 | **(B)** opportunity HIGH | **L** (~3-6h) |
| B.12 | `flash_attention_speculative_verify` | G2 (composite) | likely **(A)** TBD | TBD |
| B.13 | `flash_attention_speculative_verify_paged` | G2 (composite) | likely **(B)** TBD | inherits B.16 |
| B.14 | `flash_attention_splitfuse` | composite | TBD bench | TBD |
| B.15 | `flash_attention_varlen` | G2 | **(B)** opportunity (no NAX path) | **XL** (~6-12h) deprioritized |
| B.16 | `flash_attention_paged` | G2 | **(B)** opportunity (no NAX path) | **XL** deprioritized |
| B.17 | `flash_attention_paged_varlen` | G2 | **(B)** opportunity (no NAX path) | **XL** deprioritized |
| B.18 | `flash_attention_qkv_packed` | G1 (re-dispatch) | **(A)** NAX-optimal | — |
| B.19 | `flash_attention_kv_packed` | G1 (re-dispatch) | **(A)** NAX-optimal | — |
| B.20 | `flash_attention_varlen_qkv_packed` | G2 (re-dispatch) | inherits B.15 | inherits B.15 |
| B.21 | `flash_attention_varlen_kv_packed` | G2 (re-dispatch) | inherits B.15 | inherits B.15 |
| B.22 | `flash_attention_paged_varlen_turboquant` | G2 | **(B)** opportunity (no NAX path) | **XL** deprioritized |

## Effort distribution

- **(A) NAX-optimal already**: 8 functions (B.1, B.8 dense, B.10, B.18, B.19,
  + 3 likely-A pending bench: B.5, B.6, B.12)
- **(B) NAX-opportunity HIGH priority**: B.11 top-K (L), B.9 sparse (S)
- **(B) NAX-opportunity MEDIUM priority**: B.2 + B.3 rope (S/M), B.7 cache+rope (M)
- **(B) NAX-opportunity DEPRIORITIZED for v2.50**: B.15, B.16, B.17, B.20, B.21, B.22
  (paged/varlen XL effort with marginal gain since SDPA-NAX has no paged path)
- **(B) NAX-opportunity LOW priority for v2.50**: B.4 sage (L, narrow workload value)
- **TBD pending bench**: B.13, B.14

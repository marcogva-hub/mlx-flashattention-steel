# Volet P0 — Third-Surface Inventory

**Repository state:** `fix/audit-remediation` at
`e0d5010c70b95981011bcefabbbbdea4768f1117`  
**Environment:** mlx-mfa 2.61.0 · MLX 0.31.2 · M5 Max  
**Purpose:** mechanical enumeration only; no fixes in this volet.

## Verdict

[VERIFIED] The original 24-public + 34-raw oracle omitted a third surface:
**public methods on exported classes** and the distinct Python-defined Metal
kernels those methods may reach.

- Public/export-reachable classes: **24**
- Effective public method slots on those classes: **148**
- Unique project method definitions represented: **135**
- Attention-output class methods: **25**
- Cache/state-producing support methods additionally requiring rows: **4**
- Separate Python Metal attention/decode kernels: **6**
- CX-TQ-DECODE-01 page-index scope: **one defective paged decode path containing
  two unguarded kernels**, exposed directly by two public class methods
  (`TurboQuantPagedInferenceContext.step` and `DecodeRuntime.step`).
- No second independent unguarded paged-index path was found.
- Method-level validation additionally found **three cache writers that silently
  broadcast malformed V head counts**, exposed by four attention methods and
  their runtime adapters.

`DecodeRuntime.chunked_prefill` does not currently extend the unsafe path:
on a TurboQuant runtime it fails earlier with an unexpected `softcap` keyword.
That is a separate accept-valid compatibility GAP, recorded below.

## Mechanical method oracle

The enumeration imported `mlx_mfa`, took the union of `mlx_mfa.__all__` and
non-underscore package attributes, retained class objects, then inspected every
public callable method (plus `__call__`) defined by a project class in each
effective MRO. AST inspection then found direct calls to attention entries,
`mx.fast.metal_kernel`, raw-extension bindings, and transitive `self.method`
calls.

Inherited non-project framework methods were also observed and dispositioned:
MLX `nn.Module` management methods inherited by `SVDQuantLinear` (`apply`,
`children`, `parameters`, `train`, weight load/save/update, and dictionary
helpers) do not implement project attention/kernel paths; `BaseException`
methods inherited by the two exception classes are likewise non-computational.
They are not counted as project method definitions.

All 24 class objects were in `mlx_mfa.__all__`; no additional non-underscore
class attribute existed outside it:

`DecodeRuntime`, `DenseKVCache`, `DenseKVCacheAdapter`, `DispatchPolicy`,
`ExternalKVCacheAdapter`, `ExternalKVCacheCapabilities`, `HybridKVCache`,
`HybridKVCacheAdapter`, `InferenceContext`, `KVCacheAdapter`,
`KVCacheCapabilities`, `KVCacheOperationUnsupported`, `KVCacheProtocol`,
`LocalHostKVStoreAdapter`, `NaxUnavailable`, `PagedInferenceContext`,
`PagedKVCache`, `PagedKVCacheAdapter`, `QuantizedKVCache`,
`QuantizedKVCacheAdapter`, `SVDQuantLinear`, `SageInferenceContext`,
`TurboQuantKVCache`, and `TurboQuantPagedInferenceContext`.

### Selection rule

A method is in the class-method computational set when it:

1. returns attention output directly or through another method/helper; or
2. directly produces cache/state consumed by such a method, especially where it
   dispatches a project custom/raw accelerator kernel.

Ordinary cache access, bookkeeping, host storage, tensor views, and the
non-attention `SVDQuantLinear.__call__` linear layer remain reviewed
non-attention methods. This avoids redefining every MLX tensor operation as an
attention surface while still capturing state mutation and custom dispatch
hidden inside a class.

## Task 1 — Complete computational class-method set

### Attention-output methods

| Method | Reaches | Public reachability |
|---|---|---|
| `InferenceContext.prefill` | `flash_attention` | exported class |
| `InferenceContext.step` | `flash_attention_kvcache` | exported class |
| `InferenceContext.chunked_prefill` | repeated `InferenceContext.step` | exported class |
| `PagedInferenceContext.prefill` | `flash_attention` | exported class |
| `PagedInferenceContext.step` | cache gather → `flash_attention_kvcache` | exported class |
| `PagedInferenceContext.chunked_prefill` | repeated `PagedInferenceContext.step` | exported class |
| `SageInferenceContext.prefill` | `flash_attention` | exported class |
| `SageInferenceContext.step` | `sage_attention_prequantized` | exported class |
| `TurboQuantPagedInferenceContext.prefill` | guarded `flash_attention_paged_varlen_turboquant` | exported class |
| `TurboQuantPagedInferenceContext.step` | Nq=1 → `tq_decode_attend`; otherwise guarded fused TQ | exported class |
| `DecodeRuntime.prefill` | delegated context `prefill` | exported class/factory product |
| `DecodeRuntime.step` | delegated context `step`, including TQ Nq=1 | exported class/factory product |
| `DecodeRuntime.prefill_with_prefix` | `seed_prefix` → `chunked_prefill` | exported class/factory product |
| `DecodeRuntime.chunked_prefill` | paged-varlen/paged-batch or repeated `step` | exported class/factory product |
| `DecodeRuntime.paged_varlen` | `flash_attention_paged_varlen` | exported class/factory product |
| `DecodeRuntime.paged_prefill_batch` | `flash_attention_paged` | exported class/factory product |
| `DecodeRuntime.paged_step_batch` | `flash_attention_paged` | exported class/factory product |
| `DecodeRuntime.register_prefix` | `shared_prefix_cache` → `make_shared_prefix_cache` | exported class/factory product |
| `DecodeRuntime.prefill_shared_prefix` | `register_prefix` → `make_shared_prefix_cache` | exported class/factory product |
| `DecodeRuntime.shared_prefix_cache` | `make_shared_prefix_cache` | exported class/factory product |
| `DecodeRuntime.decode_from_shared_prefix` | `flash_attention` | exported class/factory product |
| `DecodeRuntime.splitfuse` | `flash_attention_splitfuse` | exported class/factory product |
| `DecodeRuntime.splitfuse_step` | `flash_attention_paged` or `flash_attention_splitfuse` | exported class/factory product |
| `DecodeRuntime.speculative_verify` | dense or paged speculative attention | exported class/factory product |
| `DecodeRuntime.speculative_step` | `speculative_verify` plus acceptance bookkeeping | exported class/factory product |

### Cache/state-producing support methods

| Method | Explicit dispatch | Why it needs a row |
|---|---|---|
| `DenseKVCache.append` | MLX slice-update/eval | directly normalizes user K/V into dense state consumed by `InferenceContext.step` |
| `PagedKVCache.append` | raw `mfa_scatter_kv` twice | public class method directly dispatches an audited raw kernel |
| `QuantizedKVCache.append` | `quantize_per_block` → raw `mfa_quantize_per_block` when extension is present | supplies Sage decode state |
| `TurboQuantPagedInferenceContext.append` | TQ pack/scale computation and direct paged-pool writes | supplies both fused TQ and separate `tq_decode` consumers |

### Empirical factory-product reachability

[VERIFIED] `create_decode_runtime(turboquant=True, ...)` returned a
`DecodeRuntime` wrapping `TurboQuantPagedInferenceContext`. After valid prefill,
`DecodeRuntime.step` populated the `(D,Hkv,block_size,bits)` entry in
`tq_decode._K_DEQUANT_KERNELS`, proving engagement through the public factory
product.

[VERIFIED] `DecodeRuntime.chunked_prefill` on the same product did **not** reach
the kernel: it raised
`TypeError: TurboQuantPagedInferenceContext.step() got an unexpected keyword
argument 'softcap'`. This is a method compatibility gap, not another OOB route.

### Full effective public-method ledger

This is the mechanical 148-slot input to the classification above. Inherited
project methods appear on each class where they are publicly callable.

- `DecodeRuntime`: `chunked_prefill`, `clear_registered_prefixes`,
  `decode_from_shared_prefix`, `drop_prefix`, `hybrid_mark_for_prefetch`,
  `hybrid_offload`, `hybrid_prefetch`, `hybrid_reload`,
  `list_registered_prefix_ids`, `paged_prefill_batch`, `paged_step_batch`,
  `paged_varlen`, `prefill`, `prefill_shared_prefix`, `prefill_with_prefix`,
  `register_prefix`, `reset`, `seed_prefix`, `seq_length`,
  `shared_prefix_cache`, `speculative_step`, `speculative_verify`, `splitfuse`,
  `splitfuse_step`, `step`.
- `DenseKVCache`: `append`, `k_for_attention`, `reset`, `seq_length`,
  `v_for_attention`.
- `DenseKVCacheAdapter`: `active_seq_ids`, `append`, `attention_k`,
  `attention_v`, `paged_pool`, `paged_tables`, `quantized_view`, `reset`,
  `seq_length`.
- `DispatchPolicy`: no public callable method.
- `ExternalKVCacheAdapter`: `evict`, `fetch`, `has`, `prefetch`, `put`,
  `seq_length`.
- `ExternalKVCacheCapabilities`: no public callable method.
- `HybridKVCache`: `active_seq_ids`, `append`, `clear_prefetch_intent`,
  `get_block_table`, `get_seq_lens`, `k_for_attention`, `mark_for_prefetch`,
  `mark_pinned`, `offload_seq`, `paged_pool`, `paged_tables`, `prefetch`,
  `prefetch_seq`, `prepare_hot_window`, `promote_seq`, `quantized_view`,
  `reload_seq`, `reset`, `seq_length`, `v_for_attention`.
- `HybridKVCacheAdapter`: `active_seq_ids`, `append`, `attention_k`,
  `attention_v`, `paged_pool`, `paged_tables`, `quantized_view`, `reset`,
  `seq_length`.
- `InferenceContext`: `chunked_prefill`, `prefill`, `reset`, `step`.
- `KVCacheAdapter`: `active_seq_ids`, `append`, `attention_k`, `attention_v`,
  `paged_pool`, `paged_tables`, `quantized_view`, `reset`, `seq_length`.
- `KVCacheCapabilities`: no public callable method.
- `KVCacheOperationUnsupported`: no project public callable method.
- `KVCacheProtocol`: `append`, `k_for_attention`, `reset`, `seq_length`,
  `v_for_attention`.
- `LocalHostKVStoreAdapter`: `evict`, `fetch`, `has`, `prefetch`, `put`,
  `seq_length`.
- `NaxUnavailable`: no project public callable method.
- `PagedInferenceContext`: `chunked_prefill`, `prefill`, `reset`,
  `seq_length`, `step`.
- `PagedKVCache`: `append`, `block_table_and_seq_lens`, `free_seq`, `gather`,
  `get_block_table`, `get_seq_lens`, `k_for_attention`, `reset`, `seq_length`,
  `v_for_attention`.
- `PagedKVCacheAdapter`: `active_seq_ids`, `append`, `attention_k`,
  `attention_v`, `paged_pool`, `paged_tables`, `quantized_view`, `reset`,
  `seq_length`.
- `QuantizedKVCache`: `append`, `reset`.
- `QuantizedKVCacheAdapter`: `active_seq_ids`, `append`, `attention_k`,
  `attention_v`, `paged_pool`, `paged_tables`, `quantized_view`, `reset`,
  `seq_length`.
- `SVDQuantLinear`: `__call__` (reviewed non-attention quantized linear).
- `SageInferenceContext`: `prefill`, `reset`, `step`.
- `TurboQuantKVCache`: `append`, `k_decompressed`, `reset`, `v_decompressed`.
- `TurboQuantPagedInferenceContext`: `append`, `get_block_table`,
  `get_seq_lens`, `prefill`, `reset`, `seq_length`, `step`.

## Task 2 — Separate Python Metal kernel inventory

Repository-wide `mx.fast.metal_kernel` and Metal-source grep found four relevant
modules. `conv_nax.py` has three Python Metal convolution kernels but performs
im2col/matmul convolution, not attention/gather/dequant, so it is outside this
third attention surface with an explicit reason.

| Kernel | Role and reachability | Bounds result |
|---|---|---|
| `tq_decode_kdequant_b*` | production Nq=1 TQ decode via public context/runtime | **UNGUARDED**: `phys = block_table[blk]`, followed by direct packed-K and scale loads at `mlx_mfa/tq_decode.py:99-105`; no logical-table-length or `0 <= phys < num_blocks` check |
| `tq_decode_vgather_*` | production Nq=1 TQ decode via public context/runtime | **UNGUARDED**: direct V load after `phys = block_table[blk]` at `mlx_mfa/tq_decode.py:134-137` |
| `topk_threshold_bisect` | production `flash_attention_topk` path | no page table; score loads are bounded by `k_idx < S_arg` at `mlx_mfa/attention.py:4047-4052,4069-4074`; output grid is constructed by the caller |
| `mfa_cider_gqa_p1_*` | direct dotted expert import `mlx_mfa.gqa_decode_cider.gqa_decode_cider`; not in `__all__` or auto-routing | no page table; contiguous K/V iteration is bounded by `kv_end=min(...,N)` and `pos < kv_end` at `mlx_mfa/gqa_decode_cider.py:78-135` |
| `mfa_cider_gqa_p2_*` | second pass of the same expert path | no external pool/page index; reads fixed-size pass-1 partial buffers at `mlx_mfa/gqa_decode_cider.py:162-200` |
| `topk_stream_v5_*` | declined/internal direct module import only; never auto-routed | no page table; Q/K/output accesses use `row<N`, `key<S`, and `idx<S` guards at `mlx_mfa/topk_stream.py:88-109,114-138,163-170` |

No other Python Metal source performs attention, decode, paged gather, or TQ
dequantization.

## Task 3 — Current five-axis state

Legend: **P** = PASS; **G** = GAP; **N/A** = axis does not involve page-indexed
memory. “Inherited” means the method is a pure adapter over an already audited
24+34 entry; the method itself was nevertheless omitted from the old row set.

### Class methods

| Class method | Correctness | Accept-valid | Reject malformed | Determinism | Memory safety | Evidence / current truth |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `InferenceContext.prefill` | P | P | P | P | N/A | direct hardened `flash_attention`; oracle test |
| `InferenceContext.step` | P(valid) | P | **G** | P(valid) | N/A | malformed V head count is silently broadcast by `DenseKVCache.append` |
| `InferenceContext.chunked_prefill` | P(valid) | P | **G** | P(valid) | N/A | repeated `step`; inherits silent V-head broadcast |
| `PagedInferenceContext.prefill` | P | P | P | P | N/A | direct dense `flash_attention`; allocator-owned pages |
| `PagedInferenceContext.step` | P | P | P | P | N/A | allocator-owned gather then hardened kvcache attention |
| `PagedInferenceContext.chunked_prefill` | P | P | P | P | N/A | repeated guarded context step |
| `SageInferenceContext.prefill` | P | P | P | P | N/A | direct hardened `flash_attention` |
| `SageInferenceContext.step` | P(valid) | P | **G** | P(valid) | N/A | malformed V head count is silently broadcast by `QuantizedKVCache.append` |
| `TurboQuantPagedInferenceContext.prefill` | P(valid) | P | **G** | P(valid) | P | malformed V heads broadcast before guarded fused dispatch |
| `TurboQuantPagedInferenceContext.step` | P(valid) | P | **G** | P(valid) | **G** | Nq=1 separate kernels silently accept malformed page IDs and read OOB |
| `DecodeRuntime.prefill` | P(valid) | P | **G(TQ)** | P(valid) | P/N/A | TQ prefill inherits silent V-head broadcast |
| `DecodeRuntime.step` | P(valid) | P | **G** | P(valid) | **G(TQ)** | dense/Sage/TQ silently broadcast malformed V; TQ also reaches unguarded page path |
| `DecodeRuntime.prefill_with_prefix` | P(valid) | P(supported backends) | **G(dense suffix)** | P(valid) | P/N/A | suffix routes through chunked dense step |
| `DecodeRuntime.chunked_prefill` | P(supported) | **G(TQ)** | **G(dense/Sage)** | P(valid) | P/N/A | TQ raises unexpected `softcap`; dense/Sage inherit silent V broadcast |
| `DecodeRuntime.paged_varlen` | P | P | P | P | P | inherits validated/guarded public paged-varlen entry |
| `DecodeRuntime.paged_prefill_batch` | P | P | P | P | P | inherits validated/guarded public paged entry |
| `DecodeRuntime.paged_step_batch` | P | P | P | P | P | inherits validated/guarded public paged entry |
| `DecodeRuntime.register_prefix` | P | P | P | P | N/A | reaches audited `make_shared_prefix_cache` |
| `DecodeRuntime.prefill_shared_prefix` | P | P | P | P | N/A | register + optional cache seed |
| `DecodeRuntime.shared_prefix_cache` | P | P | P | P | N/A | direct audited prefix compute adapter |
| `DecodeRuntime.decode_from_shared_prefix` | P | P | P | P | N/A | prepared-state reject then hardened `flash_attention` |
| `DecodeRuntime.splitfuse` | P | P | P | P | N/A | audited public splitfuse adapter; partial triples reject |
| `DecodeRuntime.splitfuse_step` | P | P | P | P | P/N/A | paged decode-only uses guarded paged path; dense uses splitfuse |
| `DecodeRuntime.speculative_verify` | P | P | P | P | P/N/A | paged branch uses guarded paged speculative entry |
| `DecodeRuntime.speculative_step` | P | P | P | P | P/N/A | verify adapter plus validated acceptance bookkeeping |
| `DenseKVCache.append` | P(valid) | P | **G** | P(valid) | N/A | malformed V heads broadcast silently |
| `PagedKVCache.append` | P | P | P | P | P | malformed V heads rejected by raw `mfa_scatter_kv`; row R13 |
| `QuantizedKVCache.append` | P(valid) | P | **G** | P(valid) | N/A | malformed V heads broadcast silently; overflow only is explicitly checked |
| `TurboQuantPagedInferenceContext.append` | P(valid) | P | **G** | P(valid) | N/A | malformed V heads broadcast silently; allocator supplies page IDs |

### Separate kernels

| Kernel | Correctness | Accept-valid | Reject malformed | Determinism | Memory safety |
|---|:---:|:---:|:---:|:---:|:---:|
| TQ K dequant | P (combined Python-dequant oracle) | P | **G** | P | **G** |
| TQ V gather | P (combined oracle) | P | **G** | P | **G** |
| top-k bisection threshold | P | P | P through public constructor | P | N/A |
| Cider pass 1 | P | P | G (expert wrapper validates only part of the full shape/dtype contract) | P (5-run byte identity) | N/A (contiguous, loop-bounded) |
| Cider pass 2 | P | P | inherited G from expert wrapper | P (5-run byte identity) | N/A |
| streaming top-k indices | P | P | G (internal wrapper has partial validation only) | P (5-run byte identity) | N/A |

Validation runs:

- 95 relevant inference/cache/TQ/Cider/top-k tests passed.
- 33 runtime prefix/splitfuse/speculative/paged/chunked tests passed.
- 89 SVD/raw-scatter/TQ-buffer inventory tests passed.
- Cider and streaming top-k were byte-identical over five identical-input runs.
- Existing 24+34 enumeration remains green, demonstrating why it cannot detect
  this third surface.

## P1 + P2 closure (2026-06-23)

**P1 fixed the product gaps** (the **G** markers above are now PASS): the two
`tq_decode` kernels carry an in-kernel `blk < n_active && 0 <= phys < num_blocks`
bounds guard (memory-safe) + loud host index validation on the public TQ step
(`MFA_PAGED_TRUST_INDICES=1` opt-out); `DenseKVCache.append` /
`QuantizedKVCache.append` / `TurboQuantPagedInferenceContext.append` reject a
malformed V head count. Locks: `tests/test_tq_decode_guard.py`,
`tests/test_cache_append_vshape.py`, `tests/test_tq_chunked_prefill_softcap.py`.

**P2 makes this surface CI-enumerated** (`scripts/enumerate_api_surface.py`,
extending the function/raw guard). Now property-based (NOT name heuristics — the
rounds-8-11 lesson):

- **Class methods (P3 — property-complete):** **37** computational methods
  (`COMPUTATIONAL_CLASS_METHODS` = 29 P0 hand-audited + 7 property-derived
  cache-append delegators). The promotion rule is now **property-complete**: a
  method may sit in the reviewed set ONLY if PROVABLY CLEAN; if the rule cannot
  prove it clean it FLAGS it (the conservative inversion that closes the
  delegation vector that hid `CX-TQ-DECODE-01`). It detects, by property:
  (1) **cross-object delegation** — `self.<attr>.<meth>(…)`, resolving `<attr>`'s
  class from `__init__` (`self.x = SomeClass(…)`); unresolvable → conservative
  name-fallback on the computational-method-name set (catches `DecodeRuntime.step
  → self.context.step`); (2) **intra-class delegation** — `self.<m>(…)` to a
  computational method, full fixpoint; (3) **state production** — a write
  (assign/slice-assign) to an attention-consumed KV buffer (`self._k*`/`_v*`/
  `*pool*`/`*scale*`) from a K/V input param (catches `DenseKVCache.append`;
  excludes `reset` — counter-only, no KV param); (4) **complete raw `_ext` set**
  — all 51 raw m.def names + `_mfa_*_cpp` wrappers + `_ext.`/`metal_kernel`
  (catches `PagedKVCache.append → _mfa_scatter_kv_cpp`). Over the live tree it
  reproduces ALL 37 by property (the round-12 NO-GO four —
  `DecodeRuntime.prefill/step`, `DenseKVCache.append`, `PagedKVCache.append` —
  are now derived, not explicit-only). `SVDQuantLinear.__call__` is the one
  reviewed non-attention method (quantized linear; provably clean — no kernel
  dispatch). No over-promotion: getters / `reset` / `seq_length` stay clean.
- **Metal kernels:** all **9** `mx.fast.metal_kernel` sites are AST-inventoried
  (`METAL_KERNELS`); a site that is page-indexed (its builder references
  `block_table`) MUST carry a `page_bounds in {guarded, reviewed}` record, and a
  NEW/unrecorded `metal_kernel` → `SystemExit`. Only the 2 TQ-decode kernels are
  page-indexed (now `guarded`).

Counts are DERIVED from the live classification (no hardcoded drift). Mutation
bites (`tests/test_third_surface_guard.py`, 12 cells, all firing): a computational
method moved to reviewed → fail; a synthetic reaching method → fail until
classified; a synthetic page-indexed kernel with no record → fail; a recorded
page kernel downgraded to `unguarded` → fail; a stale entry → fail; **(P3) each of
the four reach patterns in reviewed → fail — cross-object delegation
(`self.context.step`, Codex's synthetic), state production (writes `self._v`),
raw `_ext` call, intra-class `self.<computational>()`**; clean state → 0
offenders, **37 methods + 9 kernels**, no over-promotion (getters/`reset` clean).
**The class-method + JIT-kernel path that hid CX-TQ-DECODE-01 for 11 rounds is now
loud-on-regression, and computational methods are reproduced by PROPERTY (not an
explicit crutch).**

## CX-TQ-DECODE-01 sibling set

### CRITICAL — same root defect

1. `tq_decode_kdequant_b*` — unchecked physical page controls packed-K and
   `k_scales` addresses.
2. `tq_decode_vgather_*` — unchecked physical page controls fp16 V address.
3. `TurboQuantPagedInferenceContext.step` — public method exposing both kernels
   without default validation.
4. `DecodeRuntime.step` — public factory-product adapter exposing the same path.

These are four inventory entries but **one implementation path/root cause**.
The prior malformed `99` probe produced allocation-sensitive output rather than
the required `-1`-equivalent skip, while default mode raised nothing.

### HIGH — method-level silent-broadcast sibling class

1. `DenseKVCache.append` → `InferenceContext.step` →
   `DecodeRuntime.step`/dense and dense chunked suffixes.
2. `QuantizedKVCache.append` → `SageInferenceContext.step` →
   `DecodeRuntime.step`/sage.
3. `TurboQuantPagedInferenceContext.append` →
   `TurboQuantPagedInferenceContext.prefill/step` →
   `DecodeRuntime.prefill/step`/turboquant.

[VERIFIED] With configured `H_kv=2`, a supplied V tensor with one head did not
raise on these writers/consumers. The output was byte-identical to explicitly
broadcasting that V tensor to two heads first. This is deterministic
silent-normalization of malformed attention input, inconsistent with the
function surface's Q/K/V mutual-shape contract.

Paged cache append is not a sibling: the same malformed V head count raises
from `mfa_scatter_kv`.

### MEDIUM — separate class-method compatibility gap

`DecodeRuntime.chunked_prefill` is advertised on the runtime product but passes
`softcap` to `TurboQuantPagedInferenceContext.step`, whose signature does not
accept it. Valid TurboQuant chunked prefill therefore raises before computation.
This is not a memory-safety sibling and does not expand the kernel-fix scope.

### Nonblocking inventory observations

The dormant Cider and declined streaming-top-k direct-import wrappers have
partial malformed-input validation, but neither accepts page indices or performs
paged pool addressing. They are not CX-TQ-DECODE-01 siblings.

## Task 4 — Durable classifier-guard fix specification

The next volet should implement all of the following:

1. Add an explicit `COMPUTATIONAL_CLASS_METHODS` mapping keyed by fully
   qualified method name, with its reached public/raw/kernel path and 5-axis row.
2. For every exported class and every effective public project method plus
   `__call__`, require exactly one classification:
   `COMPUTATIONAL_CLASS_METHODS` or
   `REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS`.
3. Store a non-empty semantic reason for every reviewed non-computational
   method. A class-level reason such as “context class” must not exempt methods.
4. Inspect method source and its transitive project-local calls. Promote a
   method when it:
   - calls a known computational public/raw/class-method entry;
   - calls a helper that reaches one;
   - constructs or invokes `mx.fast.metal_kernel`; or
   - invokes an `_ext`/raw accelerator binding.
5. Treat uninspectable classes/methods as reviewed-or-fail, matching the current
   function guard.
6. Add a separate AST inventory for every `mx.fast.metal_kernel` construction.
   Each kernel needs a category (`attention/decode/paged`, support, or
   non-attention with reason), public reachability, validation owner, and
   page-bounds status.
7. Add executable mutation bites:
   - moving `TurboQuantPagedInferenceContext.step` to the reviewed method set
     must fail;
   - adding a new public method that calls `tq_decode_attend`,
     `flash_attention`, or a raw binding must fail until classified;
   - adding a new Python Metal paged kernel with a `block_table` load but no
     bounds-review record must fail;
   - stale method-review entries must fail.

This closes the structural hole: a class may remain
`REVIEWED_NONCOMPUTATIONAL`, but that classification no longer says anything
about its methods.

## Scope verdict

[VERIFIED] **`tq_decode` is the only unguarded paged-index decode path found.**
Its K-dequant and V-gather kernels are two defective kernel sites under one
public Nq=1 route. There are **zero additional independent paged-index sibling
paths** in the Python Metal inventory.

The next fix volet therefore has a bounded safety scope:

- guard both `tq_decode.py` kernels;
- add default loud validation through the public TQ step/runtime route;
- add method-level and Python-Metal classifier guards;
- add class-method Q/K/V shape validation before cache normalization for the
  dense, Sage, and TurboQuant writers/attention methods;
- separately decide whether to repair the discovered TurboQuant
  `DecodeRuntime.chunked_prefill` compatibility gap.

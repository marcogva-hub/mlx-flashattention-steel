#!/usr/bin/env python3
"""Volet J — MECHANICAL enumeration of the mlx-mfa API surface.

Round-7 found the volet-I "complete inventory" was the familiar SUBSET labeled
complete (8 of ~31 public, 16 of 33 raw bindings). Judgment under-enumerates.
This script is the SINGLE SOURCE OF TRUTH for the inventory's row-set: it parses
`mlx_mfa/__init__.py::__all__` via AST and `csrc/bindings.cpp::m.def(...)` via
regex, classifies each entry computational vs helper, and emits the authoritative
list with an AUDITED/OMITTED column. The inventory must carry a row for every
COMPUTATIONAL entry this lists, or an explicit helper/N-A reason.

Run: .venv/bin/python scripts/enumerate_api_surface.py
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "mlx_mfa" / "__init__.py"
BINDINGS = ROOT / "csrc" / "bindings.cpp"
OUT = ROOT / "audit" / "round3_remediation" / "api_surface_enumeration.md"

# ── Already-audited entries (seed from the volet-I/H/S/I2 inventory + locks). ──
# Public entries that have a first-hand 4-axis row in surface_inventory.md.
AUDITED_PUBLIC = {
    "flash_attention", "flash_attention_sparse", "flash_attention_gna",
    "sage_attention", "sage_attention_prequantized",  # prequant: volet J
    "flash_attention_varlen", "flash_attention_paged",
    "flash_attention_paged_varlen", "flash_attention_paged_varlen_turboquant",
    # volet K3 — the final 13 public adapter entries (4-axis + fixes):
    "flash_attention_qkv_packed", "flash_attention_kv_packed",
    "flash_attention_speculative_verify", "flash_attention_splitfuse",
    "flash_attention_rope_unified", "flash_attention_rope",
    "flash_attention_kvcache_rope_append", "flash_attention_kvcache",
    "flash_attention_speculative_verify_paged",
    "flash_attention_varlen_qkv_packed", "flash_attention_varlen_kv_packed",
    "sage_attention_kvcache", "flash_attention_topk",
    "sparse_attention_dispatch",  # volet L (CX-R8-02): the 23rd, was misclassified
    "make_shared_prefix_cache",   # volet M (CX-R9-01): the 24th, hidden by make_* rule
}

# CX-R8-02 (volet L): explicit COMPUTATIONAL allowlist replaces the name-prefix
# heuristic that misclassified `sparse_attention_dispatch` (lcsa_nax.py — no
# flash_/sage_/conv3d/topk_ prefix) as "helper: other". A computational entry
# takes Q/K/V and dispatches an attention kernel. This list is the source of
# truth for classification; `classify_public` raises (via "UNCLASSIFIED") on any
# export matching neither this list nor a specific helper rule, so a future
# computational export cannot silently fall through. Must equal AUDITED_PUBLIC
# (every computational entry is audited → OMITTED computational public = 0).
COMPUTATIONAL_PUBLIC = set(AUDITED_PUBLIC)

# Explicit non-computational exports the pattern/class/module rules don't catch
# (config, diagnostics, build/warmup, cache management, quantize/mask preproc).
# Each verified first-hand to NOT take Q/K/V + dispatch an attention kernel.
# Anything matching neither this list, COMPUTATIONAL_PUBLIC, nor a pattern rule →
# UNCLASSIFIED → loud failure (a NEW export forces a human classification rather
# than silently becoming a helper — the CX-R8-02 durable fix).
HELPER_PUBLIC = {
    # classes / contexts / protocols
    "DecodeRuntime", "DenseKVCache", "DenseKVCacheAdapter", "DispatchPolicy",
    "ExternalKVCacheAdapter", "ExternalKVCacheCapabilities", "HybridKVCache",
    "HybridKVCacheAdapter", "InferenceContext", "KVCacheAdapter",
    "KVCacheCapabilities", "KVCacheOperationUnsupported", "KVCacheProtocol",
    "LocalHostKVStoreAdapter", "NaxUnavailable", "PagedInferenceContext",
    "PagedKVCache", "PagedKVCacheAdapter", "QuantizedKVCache",
    "QuantizedKVCacheAdapter", "SVDQuantLinear", "SageInferenceContext",
    "TurboQuantKVCache", "TurboQuantPagedInferenceContext",
    # config / diagnostics / introspection
    "__version__", "calibrate_dispatch", "compile_metallib", "diagnostics",
    "disable", "enable", "get_device_info", "get_hook_stats",
    "get_supported_configs", "has_nax", "hooks_status", "is_mfa_available",
    "reset_hook_stats", "warmup_kernels",
    # cache management / runtime factories
    "adapt_kv_cache", "create_decode_runtime", "create_inference_context",
    "resolve_context_cache", "resolve_context_cache_adapter",
    # mask builders (no q/k/v dispatch)
    "make_adaptive_window_mask", "make_axial_spatial_mask", "make_axial_temporal_mask",
    "make_causal_block_mask", "make_causal_segment_mask", "make_cross_stream_mask",
    "make_diagonal_mask", "make_dilated_temporal_mask", "make_gna_mask",
    "make_lcsa_mask", "make_reference_frame_mask", "make_rope_3d_tables",
    "make_segment_mask", "make_sink_window_mask", "make_sliding_window_mask",
    "make_spatial_2d_mask", "make_spatial_3d_mask", "make_strided_mask",
    "make_temporal_distance_bias", "make_temporal_group_mask", "make_topk_spatial_mask",
    "temporal_distance_bias_to_mask",
    # quantize / pack / preprocessing
    "build_tq_paged_k_pool", "build_tq_paged_v_pool", "dequantize",
    "pack_3bit_optimal", "pack_k_for_metal", "pack_v_for_metal", "quantize_model",
    "quantize_per_block", "sage_block_sizes", "sage_output_correction", "smooth_k",
    "turboquant_compress", "turboquant_decompress", "unpack_3bit_optimal",
}
# Raw bindings with a first-hand row (volet H2/I/S/I2/J).
AUDITED_RAW = {
    "mfa_paged_steel_forward", "mfa_paged_varlen_forward",
    "mfa_paged_varlen_tq_forward", "mfa_paged_kv_gather",
    "mfa_steel_backward", "mfa_steel_backward_sparse",
    "mfa_backward_query_debug", "mfa_backward_kv_debug",
    "mfa_sage_forward",  # volet J (CX-R7-01)
    "v6_nax_backward_query_raw", "v6_nax_backward_kv_raw",
    "v6_nax_backward_dk_raw", "v6_nax_backward_dv_raw",
    "v6_nax_backward_fused_dkdv_raw",
    "v6_nax_backward_query_sparse_raw", "v6_nax_backward_kv_sparse_raw",
    "v6_nax_backward_dk_sparse_raw", "v6_nax_backward_dv_sparse_raw",
    "v6_nax_backward_fused_dkdv_sparse_raw",
    # volet K1 — priority groups 1–6 (4-axis + validation fixes):
    "v6_nax_forward",                       # R15
    "mfa_attention_varlen_forward",         # R7
    "mfa_attention_rope_forward",           # R5
    "mfa_attention_alibi_forward",          # R3
    "mfa_attention_bias_forward",           # R4
    "mfa_attention_sparse_forward",         # R6
    "mfa_attention_sparse_forward_with_lse",# R6
    "sparse_attention_forward_with_lse",    # R10
    "mfa_scatter_kv",                       # R13
    "mfa_attention_forward",                # R1 (retrofit via shared validator)
    # volet K2 — remaining raw entries (4-axis + validation fixes):
    "mfa_forward_with_lse",                 # R2 (added f16/bf16-supported)
    "mfa_gna_forward",                      # R8 (added dtype + window/stride>0)
    "mfa_gna_nax_forward",                  # Op-GNA NAX expert surface
    "sparse_attention_forward",             # R9 (verify-only — comprehensive)
    "mfa_quantize_per_block",               # R11 (verify-only)
    "mfa_smooth_quantize_k",                # R12 (verify-only)
    "conv3d_nax_forward",                   # R14 (verify-only)
    "v6_nax_backward_query",                # R16 (added dtype + lse/d_vec f32)
    "v6_nax_backward_kv",                   # R16
    "v6_nax_quantized_matmul",              # Op-K qmm NAX expert surface
}

# CX-R10-01 (volet N): explicit review-set for HELPER exports that the property
# guard flags (attention-input-shaped OR uninspectable). Each is verified
# first-hand to be genuinely NON-computational (returns a mask/correction/state,
# never attention output via a dispatch). A flagged HELPER NOT in this set fails
# the enumeration — so a misclassified computational entry (even one that
# computes attention INLINE, like flash_attention_topk) cannot hide.
REVIEWED_NONCOMPUTATIONAL = {
    # classes / contexts / adapters / protocols / exceptions — stateful helpers,
    # not stateless attention ops (uninspectable as a q/k/v function):
    "DecodeRuntime": "runtime class", "DenseKVCache": "cache class",
    "DenseKVCacheAdapter": "cache adapter class", "DispatchPolicy": "config class",
    "ExternalKVCacheAdapter": "cache adapter class",
    "ExternalKVCacheCapabilities": "capabilities dataclass",
    "HybridKVCache": "cache class", "HybridKVCacheAdapter": "cache adapter class",
    "InferenceContext": "context class", "KVCacheAdapter": "adapter base class",
    "KVCacheCapabilities": "capabilities dataclass",
    "KVCacheOperationUnsupported": "exception class",
    "KVCacheProtocol": "typing protocol",
    "LocalHostKVStoreAdapter": "cache adapter class",
    "NaxUnavailable": "exception class",
    "PagedInferenceContext": "context class", "PagedKVCache": "cache class",
    "PagedKVCacheAdapter": "cache adapter class", "QuantizedKVCache": "cache class",
    "QuantizedKVCacheAdapter": "cache adapter class",
    "SVDQuantLinear": "nn.Module (linear layer, not attention)",
    "SageInferenceContext": "context class", "TurboQuantKVCache": "cache class",
    "TurboQuantPagedInferenceContext": "context class",
    "__version__": "version string constant (not callable)",
    # q/k-taking FUNCTIONS that return a mask or a correction, NOT attention:
    "make_topk_spatial_mask": "returns a top-k spatial mask (mx.array), not attention",
    "make_lcsa_mask": "returns an LCSA block mask, not attention",
    "sage_output_correction": "post-hoc correction of a precomputed O; no dispatch",
    # FUNCTIONS that CALL attention internally (warmup/calibration/diagnostics) or
    # only mention it in a docstring — non-q/k/v params, return non-attention:
    "calibrate_dispatch": "calibration harness — times flash_attention; returns a results dict",
    "diagnostics": "diagnostics probe — returns a status dict, not attention",
    "warmup_kernels": "JIT warmup — runs attention for side effects; returns None",
    "make_temporal_distance_bias": "returns a temporal-distance bias array; flash_attention is a docstring example",
}


def computational_in_helper(helper_names):
    """CX-R10-01 property-based Assertion 2. A HELPER export is an offender if it
    is (a) ATTENTION-INPUT-SHAPED — takes a Q-like AND a K-like parameter (q/
    query/queries, k/key/keys, incl. q_*/k_* / *_q/*_k) — OR (b) UNINSPECTABLE
    (a class, None, or signature/getsource fails), UNLESS it is explicitly listed
    in REVIEWED_NONCOMPUTATIONAL. This keys on WHAT THE ENTRY IS (its shape),
    not on detecting a compute-call by name — so it catches inline-compute
    attention ops (flash_attention_topk computes attention inline, takes q/k/v →
    flagged) and the previously-silently-skipped classes / getsource-failures.
    A secondary callee-name check stays as belt-and-suspenders. Returns offenders
    so main() fails loudly."""
    import inspect
    import mlx_mfa
    compute = re.compile(  # secondary (belt-and-suspenders)
        r"\b(flash_attention|sage_attention|sparse_attention_nax|"
        r"sparse_attention_dispatch)\s*\(|_ext\.\w*forward")

    def _grp(params, exact, pre, suf):
        return any(p in exact or p.startswith(pre) or p.endswith(suf) for p in params)

    bad = []
    for h in helper_names:
        if h in REVIEWED_NONCOMPUTATIONAL:
            continue  # explicitly reviewed non-computational
        obj = getattr(mlx_mfa, h, None)
        # (b) uninspectable → must have been reviewed (we're here, so it wasn't)
        if obj is None or inspect.isclass(obj):
            bad.append(h)
            continue
        try:
            params = [p.lower() for p in inspect.signature(obj).parameters]
            src = inspect.getsource(obj)
        except (OSError, TypeError, ValueError):
            bad.append(h)  # uninspectable, unreviewed
            continue
        # (a) attention-input-shaped: has Q-like AND K-like param
        qk = (_grp(params, {"q", "query", "queries"}, "q_", "_q")
              and _grp(params, {"k", "key", "keys"}, "k_", "_k"))
        if qk or compute.search(src):
            bad.append(h)
    return bad


# ════════════════════════════════════════════════════════════════════════════
# Volet P2 — THIRD SURFACE: class methods + mx.fast.metal_kernel JIT kernels.
# The function/raw guard above structurally cannot see class methods or JIT
# kernels — which is how CX-TQ-DECODE-01 (unguarded tq_decode page load reached
# only via TurboQuantPagedInferenceContext.step) survived 11 rounds. These
# guards make the third surface CI-enumerated, property-based (not name
# heuristics — the rounds-8-11 lesson applies identically).
# ════════════════════════════════════════════════════════════════════════════

# The 29 computational class-methods (P0 Task 1): 25 attention-output + 4
# cache/state-producing. Keyed "Class.method" → why it reaches compute.
COMPUTATIONAL_CLASS_METHODS = {
    "InferenceContext.prefill": "flash_attention",
    "InferenceContext.step": "flash_attention_kvcache",
    "InferenceContext.chunked_prefill": "repeated InferenceContext.step",
    "PagedInferenceContext.prefill": "flash_attention",
    "PagedInferenceContext.step": "cache gather → flash_attention_kvcache",
    "PagedInferenceContext.chunked_prefill": "repeated PagedInferenceContext.step",
    "SageInferenceContext.prefill": "flash_attention",
    "SageInferenceContext.step": "sage_attention_prequantized",
    "TurboQuantPagedInferenceContext.prefill": "flash_attention_paged_varlen_turboquant",
    "TurboQuantPagedInferenceContext.step": "Nq=1 → tq_decode_attend; else fused TQ",
    "DecodeRuntime.prefill": "delegated context.prefill",
    "DecodeRuntime.step": "delegated context.step (incl. TQ Nq=1)",
    "DecodeRuntime.prefill_with_prefix": "seed_prefix → chunked_prefill",
    "DecodeRuntime.chunked_prefill": "paged-varlen/batch or repeated step",
    "DecodeRuntime.paged_varlen": "flash_attention_paged_varlen",
    "DecodeRuntime.paged_prefill_batch": "flash_attention_paged",
    "DecodeRuntime.paged_step_batch": "flash_attention_paged",
    "DecodeRuntime.register_prefix": "shared_prefix_cache → make_shared_prefix_cache",
    "DecodeRuntime.prefill_shared_prefix": "register_prefix → make_shared_prefix_cache",
    "DecodeRuntime.shared_prefix_cache": "make_shared_prefix_cache",
    "DecodeRuntime.decode_from_shared_prefix": "flash_attention",
    "DecodeRuntime.splitfuse": "flash_attention_splitfuse",
    "DecodeRuntime.splitfuse_step": "flash_attention_paged / flash_attention_splitfuse",
    "DecodeRuntime.speculative_verify": "dense/paged speculative attention",
    "DecodeRuntime.speculative_step": "speculative_verify + bookkeeping",
    "DenseKVCache.append": "normalizes user K/V into dense state (slice-update)",
    "PagedKVCache.append": "raw mfa_scatter_kv (paged pool write)",
    "QuantizedKVCache.append": "quantize_per_block → raw mfa_quantize_per_block",
    "TurboQuantPagedInferenceContext.append": "TQ pack/scale + paged-pool writes",
    # P3: the property-complete promotion rule additionally derives these
    # cache-append DELEGATORS / state-producers (same delegation shape as
    # DecodeRuntime.step → context.step, which P0 counted). They reach a
    # computational append or produce attention state, so they are computational.
    "DenseKVCacheAdapter.append": "delegates to self.cache.append (DenseKVCache)",
    "PagedKVCacheAdapter.append": "delegates to self.cache.append (PagedKVCache)",
    "QuantizedKVCacheAdapter.append": "delegates to self.cache.append (QuantizedKVCache)",
    "HybridKVCacheAdapter.append": "delegates to self.cache.append",
    "KVCacheAdapter.append": "base adapter — delegates to self.cache.append",
    "HybridKVCache.append": "delegates to self._primary_adapter.append",
    "TurboQuantKVCache.append": "TQ-compresses K/V into cache state",
    # P4 Part C: the name-independent state-write detector surfaced this
    # previously-missed K/V state-producer (writes self._records from k/v under a
    # differently-named buffer — the false-negative the old _k/_v/pool/scale name
    # rule had). It persists K/V offload state for reload→attention.
    "LocalHostKVStoreAdapter.put": "persists K/V offload state (self._records) for reload",
}

# Reviewed non-computational class methods that warrant an explicit reason (they
# could be mistaken for computational but are verified non-attention). Each MUST
# NOT reach a computational/kernel/raw call (the promotion cross-check enforces).
REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS = {
    "SVDQuantLinear.__call__": "W4A16 quantized linear layer (mx quantized matmul); "
                               "not an attention op / no mlx_mfa kernel dispatch",
}

# Promotion signals (property, NOT name): a method reaches compute if its source
# calls a computational public entry, a raw accelerator, tq_decode_attend, a
# pack/quantize raw helper, or constructs mx.fast.metal_kernel.
_METHOD_REACH_RE = None


def _method_reach_re():
    global _METHOD_REACH_RE
    if _METHOD_REACH_RE is None:
        names = sorted(COMPUTATIONAL_PUBLIC, key=len, reverse=True)
        _METHOD_REACH_RE = re.compile(
            r"\b(" + "|".join(names) + r"|tq_decode_attend|quantize_per_block|"
            r"mfa_scatter_kv|pack_k_for_metal|pack_v_for_metal)\s*\(|"
            r"mx\.fast\.metal_kernel|\b_ext\b|\b_bwd_ext\b|\b_ext_inner\b")
    return _METHOD_REACH_RE


def _exported_classes():
    import mlx_mfa
    pub = public_exports()
    return [(n, getattr(mlx_mfa, n)) for n in pub
            if isinstance(getattr(mlx_mfa, n, None), type)]


def _project_methods(cls):
    """Public, project-defined methods of cls (+ __call__); inherited stdlib /
    builtin / BaseException methods are NOT project methods."""
    import inspect
    out = []
    for nm, obj in inspect.getmembers(cls):
        if nm.startswith("_") and nm != "__call__":
            continue
        if not callable(obj):
            continue
        mod = getattr(obj, "__module__", None) or ""
        if not mod.startswith("mlx_mfa"):
            continue
        out.append(nm)
    return out


# Volet P3 — PROPERTY-COMPLETE promotion. A method is computational if it reaches
# compute by ANY path; it may sit in the reviewed set ONLY if PROVABLY CLEAN
# (inspectable + no compute-regex + no raw _ext/wrapper + no metal_kernel + no
# cross-object/intra-class delegation to a computational method + no write to an
# attention-consumed state buffer from a K/V input). "Can't prove clean" → flag —
# the inversion that closes the delegation vector that hid CX-TQ-DECODE-01.
_RAW_CALL_RE_CACHE = None
# P4 Part C — NAME-INDEPENDENT state-write: any write to a persistent `self.<attr>`
# (assign, slice-assign, or `self.<attr>.append(...)`). The old version keyed on
# `_k/_v/pool/scale` names, so a state-producer writing a differently-named buffer
# (e.g. `self._packed_keys = ...`) was a false-negative. Combined with a K/V-typed
# param (below), this conservatively flags any method that stores K/V-derived
# state under ANY attribute name → must be classified (provably-clean-or-flag).
_STATE_WRITE_RE = re.compile(
    r"self\.\w+\s*(?:\[[^\]]*\])?\s*=(?!=)|self\.\w+\.append\s*\(")
_KV_PARAM = {"q", "k", "v", "k_new", "v_new", "query", "key", "value",
             "keys", "values", "queries"}


def _raw_call_re():
    """Complete raw-binding + wrapper detector (the full _ext set, not a subset)."""
    global _RAW_CALL_RE_CACHE
    if _RAW_CALL_RE_CACHE is None:
        raws = sorted(set(raw_bindings()), key=len, reverse=True)
        _RAW_CALL_RE_CACHE = re.compile(
            r"\b(" + "|".join(raws) + r")\s*\(|"     # any of the 51 raw m.def names
            r"_mfa_\w+_cpp\s*\(|"                     # _ext wrapper helpers (_mfa_*_cpp)
            r"\b_ext\b|\b_bwd_ext\b|\b_ext_inner\b|from\s+mlx_mfa\._ext")
    return _RAW_CALL_RE_CACHE


def _comp_method_names():
    return {k.split(".", 1)[1] for k in COMPUTATIONAL_CLASS_METHODS}


def _class_attr_classes(cls):
    """Resolve self.<attr> → exported class from __init__ assignments
    (`self.x = SomeClass(...)`); used for cross-object delegation resolution."""
    import inspect
    import textwrap
    import mlx_mfa
    out = {}
    init = getattr(cls, "__init__", None)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    except (OSError, TypeError, SyntaxError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name):
            tgt_cls = getattr(mlx_mfa, node.value.func.id, None)
            if isinstance(tgt_cls, type):
                for t in node.targets:
                    if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                            and t.value.id == "self"):
                        out[t.attr] = tgt_cls
    return out


def _method_direct_reaches(cls, nm, attr_classes):
    """All non-transitive reach signals for one method."""
    import inspect
    obj = getattr(cls, nm, None)
    if obj is None:
        return True
    try:
        src = inspect.getsource(obj)
        params = set(inspect.signature(obj).parameters)
    except (OSError, TypeError, ValueError):
        return True                                   # uninspectable → flag
    if _method_reach_re().search(src):                # compute publics / metal_kernel / _ext.
        return True
    if _raw_call_re().search(src):                    # complete raw _ext + wrappers
        return True
    # cross-object delegation: self.<attr>.<meth>(...)
    for attr, meth in re.findall(r"self\.([a-z_]\w*)\.([a-z_]\w*)\s*\(", src):
        c = attr_classes.get(attr)
        if c is not None and f"{c.__name__}.{meth}" in COMPUTATIONAL_CLASS_METHODS:
            return True
        if c is None and meth in _comp_method_names():   # unresolvable → name fallback
            return True
    # state production: writes an attention-consumed KV buffer from a K/V input
    if _STATE_WRITE_RE.search(src) and (params & _KV_PARAM):
        return True
    return False


def _class_reaches(cls):
    """Per-class {method: reaches} with intra-class transitive fixpoint."""
    import inspect
    methods = _project_methods(cls)
    attr_classes = _class_attr_classes(cls)
    reach = {m: _method_direct_reaches(cls, m, attr_classes) for m in methods}
    srcs = {}
    for m in methods:
        try:
            srcs[m] = inspect.getsource(getattr(cls, m))
        except (OSError, TypeError):
            srcs[m] = ""
    changed = True
    while changed:
        changed = False
        for m in methods:
            if reach[m]:
                continue
            for cm in re.findall(r"self\.([a-z_]\w*)\s*\(", srcs[m]):
                if reach.get(cm):
                    reach[m] = True
                    changed = True
                    break
    return reach


def _method_reaches(cls, nm):
    """Property-complete promotion: reaches compute by direct call, raw _ext/
    wrapper, metal_kernel, cross-object delegation (resolved or name-fallback),
    intra-class transitive delegation, or KV-state production. Uninspectable →
    True (conservative)."""
    return _class_reaches(cls).get(nm, True)


def class_method_offenders():
    """Returns (offenders, n_computational). An offender is:
      - a method that REACHES compute but is NOT in COMPUTATIONAL_CLASS_METHODS
        (incl. one wrongly placed in the reviewed set — Item 1.4);
      - a stale COMPUTATIONAL/REVIEWED entry whose method no longer exists.
    Non-reaching methods need no listing (the property verifies them)."""
    import mlx_mfa
    live = set()
    offenders = []
    for cn, cls in _exported_classes():
        for nm in _project_methods(cls):
            key = f"{cn}.{nm}"
            live.add(key)
            reaches = _method_reaches(cls, nm)
            if reaches and key not in COMPUTATIONAL_CLASS_METHODS:
                why = ("reviewed-but-reaches" if key in REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS
                       else "unclassified reaching method")
                offenders.append(f"{key}: {why} (reaches a computational/kernel/raw call)")
    for key in COMPUTATIONAL_CLASS_METHODS:
        if key not in live:
            offenders.append(f"{key}: stale COMPUTATIONAL entry (method no longer exists)")
    for key in REVIEWED_NONCOMPUTATIONAL_CLASS_METHODS:
        if key not in live:
            offenders.append(f"{key}: stale REVIEWED entry (method no longer exists)")
    return offenders, len([k for k in COMPUTATIONAL_CLASS_METHODS if k in live])


# ── Metal-kernel inventory (Item 2) ─────────────────────────────────────────────
# Every mx.fast.metal_kernel construction, keyed "module:enclosing_func". A kernel
# whose builder references `block_table` (a page-indexed load) MUST have
# page_bounds == "guarded" (or "reviewed"); a NEW page-indexed kernel with no
# record → fail. Seeded with the P0 six (+ conv excluded with reason).
METAL_KERNELS = {
    "mlx_mfa/tq_decode.py:_get_k_dequant_kernel": dict(
        category="decode", page_indexed=True, page_bounds="guarded",
        reason="TQ K-dequant; P1 in-kernel blk<n_active && 0<=phys<num_blocks guard"),
    "mlx_mfa/tq_decode.py:_get_v_gather_kernel": dict(
        category="decode", page_indexed=True, page_bounds="guarded",
        reason="TQ V-gather; P1 in-kernel bounds guard"),
    "mlx_mfa/attention.py:<module>": dict(
        category="attention-support", page_indexed=False, page_bounds="n/a",
        reason="topk threshold bisect; score loads bounded by k_idx<S"),
    "mlx_mfa/gqa_decode_cider.py:_p1": dict(
        category="attention", page_indexed=False, page_bounds="n/a",
        reason="cider pass-1; contiguous K/V bounded by kv_end=min(.,N)"),
    "mlx_mfa/gqa_decode_cider.py:_p2": dict(
        category="attention", page_indexed=False, page_bounds="n/a",
        reason="cider pass-2; reads fixed-size pass-1 partials"),
    "mlx_mfa/topk_stream.py:_build": dict(
        category="attention", page_indexed=False, page_bounds="n/a",
        reason="topk-stream v5; row<N/key<S/idx<S guards; internal-only"),
    # CC final-cert reconciliation: METAL_KERNELS is the per-LOGICAL-kernel registry
    # (7 records); conv has THREE mx.fast.metal_kernel CALL-SITES (im2col + general
    # matmul2d in _make_kernels, 1x1x1 pointwise in _make_pointwise_matmul_kernel),
    # all mapped onto this one record by metal_kernel_offenders' conv special-case.
    # So the curated count (7 logical) and the AST call-site count (9) are different
    # BY DESIGN, both truthful; the offender check proves all 9 sites map to a record
    # (0 offenders). conv-domain (non-attention), guarded by conv3d_nax_forward
    # (cross-checks C_in + dtype + 5-D rank, raises before dispatch — verified clean).
    "mlx_mfa/conv_nax.py:conv": dict(
        category="non-attention", page_indexed=False, page_bounds="n/a",
        reason="Conv3D im2col + matmul2d + 1x1x1 (3 call-sites); not attention, no "
               "page table — excluded; conv3d_nax_forward guards C_in/dtype/rank"),
}


def metal_kernel_sites():
    """AST-scan the package for every mx.fast.metal_kernel construction → list of
    (module, enclosing_func_or_<module>, builder_references_block_table)."""
    import mlx_mfa
    pkg = Path(mlx_mfa.__file__).parent
    sites = []
    for path in sorted(pkg.glob("*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        func_ranges = [(fn.lineno, fn.end_lineno, fn.name, fn)
                       for fn in ast.walk(tree)
                       if isinstance(fn, ast.FunctionDef)]
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "metal_kernel"):
                enc, enc_node, best = "<module>", None, -1
                for lo, hi, name, fn in func_ranges:
                    if lo <= node.lineno <= (hi or lo) and lo > best:
                        enc, enc_node, best = name, fn, lo
                # page-indexed detection: scan the ENCLOSING FUNCTION when the
                # kernel is built inside one (its `source` f-string + input_names
                # live there); for a module-level construction scan ONLY the call
                # node's own segment (its source=/input_names args) — NOT the whole
                # file (which would false-positive on unrelated block_table code).
                seg = (ast.get_source_segment(text, enc_node) if enc_node
                       else ast.get_source_segment(text, node)) or ""
                sites.append((f"mlx_mfa/{path.name}", enc, "block_table" in seg))
    return sites


def metal_kernel_offenders():
    """A metal_kernel site that is page-indexed (builder references block_table)
    MUST map to a record with page_bounds in {guarded, reviewed}; a site whose
    (module, func) has no record → fail (new unrecorded kernel)."""
    offenders = []
    for rel, enc, page in metal_kernel_sites():
        # match by a record key that is a prefix of "module:func" (functions are
        # cached by _get_*/_build/_p1.. helpers; conv shares one record).
        key = None
        for k in METAL_KERNELS:
            kmod, kfn = k.split(":", 1)
            if kmod == rel and (kfn == enc or kfn in enc or
                                (kfn == "<module>" and enc == "<module>") or
                                (kfn == "conv" and rel.endswith("conv_nax.py")) or
                                (kfn == "_build" and rel.endswith("topk_stream.py")) or
                                (kfn in ("_p1", "_p2") and rel.endswith("gqa_decode_cider.py"))):
                key = k
                break
        if key is None:
            offenders.append(f"{rel}:{enc}: UNRECORDED mx.fast.metal_kernel "
                             f"(page_indexed={page}); add a METAL_KERNELS record")
            continue
        if page and METAL_KERNELS[key].get("page_bounds") not in ("guarded", "reviewed"):
            offenders.append(f"{rel}:{enc}: page-indexed kernel without a "
                             f"bounds-review (page_bounds="
                             f"{METAL_KERNELS[key].get('page_bounds')!r})")
    return offenders


def public_exports():
    """Every name reachable as a public export, via AST (no import side effects)."""
    tree = ast.parse(INIT.read_text())
    names, lazy_modules = {}, {}
    for node in ast.walk(tree):
        # __all__ = [ ... ]
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            names[elt.value] = None
        # _LAZY_IMPORTS: dict = { "Name": "module", ... }
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == "_LAZY_IMPORTS" and node.value:
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    lazy_modules[k.value] = v.value
    for n in names:
        names[n] = lazy_modules.get(n, "mlx_mfa.attention")
    return names


def classify_public(name, module):
    """computational (kernel-bearing public attention entry) vs helper.

    CX-R8-02: classification is by an explicit COMPUTATIONAL allowlist (not a
    name prefix), and there is NO silent catch-all — an export matching neither
    the allowlist nor a SPECIFIC helper rule returns "UNCLASSIFIED", which main()
    turns into a hard failure. This is the durable fix: a new computational
    export can't be hidden as "helper: other" the way sparse_attention_dispatch
    was.
    """
    # CX-R9-02 (volet M): NO name-pattern rules — a `make_*` rule hid the 24th
    # computational entry (make_shared_prefix_cache) in round-9, after a name-
    # prefix rule hid the 23rd in round-8. Classification is now purely by two
    # explicit allowlists; anything in neither → UNCLASSIFIED → loud failure.
    if name in COMPUTATIONAL_PUBLIC:
        return "computational", "attention entry (allowlist)"
    if name in HELPER_PUBLIC:
        return "helper", "non-computational (allowlist)"
    return "UNCLASSIFIED", ("matches neither COMPUTATIONAL_PUBLIC nor HELPER_PUBLIC "
                            "— add it to the correct explicit allowlist")


def raw_bindings():
    """Every csrc/bindings.cpp m.def(...) name, via regex on the source."""
    src = BINDINGS.read_text()
    # m.def("name", ...   OR   m.def(\n  "name", ...
    return sorted(set(re.findall(r'm\.def\(\s*"([^"]+)"', src)))


def classify_raw(name):
    # Introspection / probes / env-helpers FIRST (these are not kernel entries
    # that compute attention output — probing them for the 4 axes is N-A).
    if name.startswith("_") or any(k in name for k in (
            "probe", "microbench", "_env", "device_has", "device_info",
            "_compile", "generate_source", "no_padding", "invalidate",
            "hook", "stats", "version", "available", "detect", "info",
            "clear", "reset", "dt_compile", "dt_generate",
            "cache_size", "shader_cache")):
        return "introspection", "introspection/probe/env"
    if name.endswith("_debug"):
        return "computational", "debug backward kernel"
    if any(k in name for k in ("forward", "backward", "_fwd", "_bwd",
                               "gather", "quantize", "scatter", "decode",
                               "rotation", "_raw", "conv", "topk", "wht")):
        return "computational", "kernel"
    return "computational", "kernel (default)"


def main():
    pub = public_exports()
    raw = raw_bindings()

    pub_rows, raw_rows = [], []
    for name in sorted(pub):
        cls, why = classify_public(name, pub[name])
        audited = name in AUDITED_PUBLIC
        pub_rows.append((name, pub[name], cls, why, audited))
    for name in raw:
        cls, why = classify_raw(name)
        audited = name in AUDITED_RAW
        raw_rows.append((name, cls, why, audited))

    # CX-R8-02 completeness assertion: every __all__ export must be classified as
    # computational or helper-with-a-stated-reason. An UNCLASSIFIED export (new or
    # misclassified) fails the enumeration LOUDLY rather than silently dropping a
    # computational entry's inventory row (the sparse_attention_dispatch bug).
    unclassified = [r for r in pub_rows if r[2] == "UNCLASSIFIED"]
    if unclassified:
        raise SystemExit(
            "enumerate_api_surface: UNCLASSIFIED public export(s) — the row-set is "
            "incomplete. Add to COMPUTATIONAL_PUBLIC or HELPER_PUBLIC:\n  "
            + "\n  ".join(f"{r[0]} ({r[1]}): {r[3]}" for r in unclassified))

    # CX-R9 Assertion 2 (semantic cross-check): no HELPER export may take a q/k/v
    # triple AND call a compute entry — that is a misclassified computational
    # entry (how the 23rd + 24th were hidden). Fail loudly.
    misclassified = computational_in_helper([r[0] for r in pub_rows if r[2] == "helper"])
    if misclassified:
        raise SystemExit(
            "enumerate_api_surface: HELPER export(s) that take q/k/v AND call a "
            "compute entry — these are MISCLASSIFIED computational entries; move "
            "them to COMPUTATIONAL_PUBLIC:\n  " + "\n  ".join(misclassified))

    # Volet P2 — third-surface guards (class methods + JIT kernels). These make
    # the surface that hid CX-TQ-DECODE-01 (a class-method-only / metal_kernel
    # path) CI-enumerated, property-based.
    cm_offenders, n_cm = class_method_offenders()
    if cm_offenders:
        raise SystemExit(
            "enumerate_api_surface: class-method offender(s) — a public method "
            "reaches a computational/kernel/raw call but is not in "
            "COMPUTATIONAL_CLASS_METHODS (or is stale):\n  "
            + "\n  ".join(cm_offenders))
    mk_offenders = metal_kernel_offenders()
    if mk_offenders:
        raise SystemExit(
            "enumerate_api_surface: mx.fast.metal_kernel offender(s) — an "
            "unrecorded or page-indexed-without-bounds-review kernel:\n  "
            + "\n  ".join(mk_offenders))

    pub_comp = [r for r in pub_rows if r[2] == "computational"]   # (name,mod,cls,why,aud)
    raw_comp = [r for r in raw_rows if r[1] == "computational"]   # (name,cls,why,aud)
    pub_omit = [r for r in pub_comp if not r[4]]
    raw_omit = [r for r in raw_comp if not r[3]]

    L = []
    L.append("# API-Surface Enumeration (Volet J — MECHANICAL)\n")
    L.append("Generated by `scripts/enumerate_api_surface.py` (AST of "
             "`__init__.py::__all__` + regex of `bindings.cpp::m.def`). "
             "This is the SINGLE SOURCE OF TRUTH for the inventory row-set.\n")
    L.append(f"**Counts:** public exports {len(pub_rows)} "
             f"({len(pub_comp)} computational, {len(pub_rows)-len(pub_comp)} helper); "
             f"raw bindings {len(raw_rows)} "
             f"({len(raw_comp)} computational, {len(raw_rows)-len(raw_comp)} introspection).\n")
    L.append(f"**Audited:** {len(pub_comp)-len(pub_omit)}/{len(pub_comp)} computational public, "
             f"{len(raw_comp)-len(raw_omit)}/{len(raw_comp)} computational raw.\n")
    L.append(f"**OMITTED (still need the 4-axis treatment):** "
             f"{len(pub_omit)} public + {len(raw_omit)} raw = "
             f"**{len(pub_omit)+len(raw_omit)}** — this is the true scope number.\n")

    L.append("\n## Public exports (`__all__`)\n")
    L.append("| entry | module | class | status |")
    L.append("|---|---|---|---|")
    for name, mod, cls, why, aud in pub_rows:
        st = "AUDITED" if aud else ("**OMITTED**" if cls == "computational" else "helper/N-A")
        L.append(f"| `{name}` | {mod.split('.')[-1]} | {cls}: {why} | {st} |")

    L.append("\n## Raw `_ext` bindings (`bindings.cpp`)\n")
    L.append("| binding | class | status |")
    L.append("|---|---|---|")
    for name, cls, why, aud in raw_rows:
        st = "AUDITED" if aud else ("**OMITTED**" if cls == "computational" else "introspection/N-A")
        L.append(f"| `{name}` | {cls}: {why} | {st} |")

    if pub_omit or raw_omit:
        L.append("\n## OMITTED computational entries — input to the scope decision\n")
        for name, mod, cls, why, _ in pub_omit:
            L.append(f"- public `{name}` ({mod.split('.')[-1]}) — {why}")
        for name, cls, why, _ in raw_omit:
            L.append(f"- raw `{name}` — {why}")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
    print(f"public: {len(pub_rows)} ({len(pub_comp)} computational), "
          f"raw: {len(raw_rows)} ({len(raw_comp)} computational)")
    print(f"OMITTED computational: {len(pub_omit)} public + {len(raw_omit)} raw "
          f"= {len(pub_omit)+len(raw_omit)}")
    print(f"third surface: {n_cm} computational class-methods "
          f"({len(metal_kernel_sites())} metal_kernel sites), 0 offenders")
    return 0


if __name__ == "__main__":
    sys.exit(main())

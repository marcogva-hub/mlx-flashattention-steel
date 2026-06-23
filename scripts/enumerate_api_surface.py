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
    "sparse_attention_forward",             # R9 (verify-only — comprehensive)
    "mfa_quantize_per_block",               # R11 (verify-only)
    "mfa_smooth_quantize_k",                # R12 (verify-only)
    "conv3d_nax_forward",                   # R14 (verify-only)
    "v6_nax_backward_query",                # R16 (added dtype + lse/d_vec f32)
    "v6_nax_backward_kv",                   # R16
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

# Public names that are classes/helpers/constants, not kernel-bearing entries.
# (Classification is by module + name pattern; see classify_public.)
HELPER_MODULES = ("inference", "turboquant", "svdquant", "masks", "serving")


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
    """computational (kernel-bearing public attention entry) vs helper."""
    if name[0].isupper():
        return "helper", "class/context"
    if any(m in module for m in HELPER_MODULES):
        return "helper", f"{module.split('.')[-1]} helper"
    if name.startswith(("make_", "apply_", "quantize", "patch_", "get_", "set_",
                        "is_", "has_", "install_", "compress", "decompress")):
        return "helper", "utility/builder"
    if name.startswith(("flash_attention", "sage_attention", "conv3d", "topk_")):
        return "computational", "attention entry"
    return "helper", "other"


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

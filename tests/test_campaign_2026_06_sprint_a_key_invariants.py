"""Campaign 2026-06 Sprint A Phase A.5 — static cache-key invariant enforcement.

Parses every pipeline-cache key struct in the C++ sources and asserts:
every declared data field participates in BOTH `operator==` AND the
companion `<Name>Hash` functor.

This permanently locks the omission class behind the 2026-05 C1 finding
(scale absent from 9 V6NAX backward keys) and the C6 finding (kbs/vbs/obs
in == but not the hash): a future field added to a key struct without
updating == or the hash fails this test at CI time instead of shipping
a silent wrong-kernel-reuse bug.

Scope note: hash-side omission is perf-only (bucket clustering) while
==-side omission is CORRECTNESS (wrong-kernel reuse) — both are
asserted because both violate the declared invariant.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).parent.parent / "csrc"

# Key structs and the file holding them.  Extend when adding a cache.
KEY_STRUCT_FILES = {
    "V6Key": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdQKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdKVKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdVKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdVSparseKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdKKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdFusedKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdQSparseKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdKSparseKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "V6NAXBwdFSparseKey": CSRC / "mfa_v6_nax_primitive.cpp",
    "KernelKey": CSRC / "shader_cache.hpp",
}

_FIELD_RE = re.compile(
    r"^\s*(?:unsigned\s+)?(?:int|bool|float|double|uint8_t|uint16_t|uint32_t|"
    r"uint64_t|int8_t|int16_t|int32_t|int64_t|short|char|size_t|KernelType)"
    r"(?:\s*::\s*\w+)?\s+(\w+(?:\s*,\s*\w+)*)\s*(?:=[^;]*)?;",
    re.M,
)


def _extract_struct_body(src: str, name: str) -> str:
    m = re.search(rf"struct {name}\s*\{{", src)
    assert m, f"struct {name} not found"
    depth, i = 1, m.end()
    while depth and i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    return src[m.end():i - 1]


def _fields(body: str) -> list[str]:
    out = []
    # strip comments first
    body = re.sub(r"//[^\n]*", "", body)
    # stop at operator== definition (fields precede it by convention)
    body = body.split("bool operator==")[0]
    for m in _FIELD_RE.finditer(body):
        for name in m.group(1).split(","):
            n = name.strip()
            if n:
                out.append(n)
    return out


@pytest.mark.parametrize("struct_name", sorted(KEY_STRUCT_FILES))
def test_key_struct_fields_in_eq_and_hash(struct_name):
    path = KEY_STRUCT_FILES[struct_name]
    src = path.read_text()
    body = _extract_struct_body(src, struct_name)
    fields = _fields(body)
    assert fields, f"{struct_name}: no fields parsed — parser drift, fix the test"

    # Track 6 (campaign 2026-06): tie()-migrated structs declare the
    # affecting-input set ONCE via `auto tie() const { return std::tie(...); }`
    # and derive == and hash from it mechanically — they cannot diverge.
    # For those structs the invariant becomes: every declared field appears
    # in the tie list.
    tie_m = re.search(r"std::tie\((.*?)\)\s*;", _extract_struct_body(src, struct_name), re.S)
    if tie_m:
        tie_fields = [t.strip() for t in tie_m.group(1).split(",")]
        missing_tie = [f for f in fields if f not in tie_fields]
        assert not missing_tie, (
            f"{struct_name}: fields ABSENT from tie() declaration "
            f"(== and hash derive from tie — an untied field is invisible "
            f"to both, C1-class hazard): {missing_tie}")
        return  # tie-derived == and hash cannot omit a tied field

    # Legacy pattern: operator== may be inline in the struct OR defined
    # out-of-line in a sibling .mm/.cpp (KernelKey: declared in
    # shader_cache.hpp, defined in shader_cache.mm).
    full_struct = _extract_struct_body(src, struct_name)
    eq_m = re.search(r"bool operator==\s*\([^)]*\)\s*const\s*\{(.*?)\}",
                     full_struct, re.S)
    search_srcs = [src]
    if not eq_m:
        for sibling in path.parent.glob(path.stem + ".*"):
            if sibling.suffix in (".mm", ".cpp") and sibling != path:
                sib_src = sibling.read_text()
                search_srcs.append(sib_src)
                eq_m = re.search(
                    rf"bool\s+\w*::?{struct_name}::operator==\s*\([^)]*\)\s*const\s*\{{(.*?)\n\}}",
                    sib_src, re.S)
                if eq_m:
                    break
    assert eq_m, f"{struct_name}: operator== not found (inline or out-of-line)"
    eq_body = eq_m.group(1)

    # Hash body: inline functor struct OR out-of-line operator() definition
    # (KernelKeyHash::operator() lives in shader_cache.mm).
    hash_body = None
    for s in search_srcs:
        m = re.search(rf"struct {struct_name}Hash\s*\{{(.*?)\n\}};", s, re.S)
        if m and re.search(r"\bk\.", m.group(1)):
            hash_body = m.group(1)
            break
        m = re.search(
            rf"{struct_name}Hash::operator\(\)\s*\([^)]*\)\s*const\s*\{{(.*?)\n\}}",
            s, re.S)
        if m:
            hash_body = m.group(1)
            break
    assert hash_body is not None, f"{struct_name}Hash body not found"

    missing_eq = [f for f in fields
                  if not re.search(rf"\b{f}\b\s*==", eq_body)]
    missing_hash = [f for f in fields
                    if not re.search(rf"\bk\.{f}\b", hash_body)]

    assert not missing_eq, (
        f"{struct_name}: fields ABSENT from operator== (wrong-kernel-reuse "
        f"hazard, C1 class): {missing_eq}")
    assert not missing_hash, (
        f"{struct_name}: fields ABSENT from hash (bucket-clustering, C6 "
        f"class): {missing_hash}")

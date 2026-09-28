"""2.62.3 — the V6NAX kernels built by MLX's `metal_kernel` did not compile on MLX 0.32.1 / 0.32.2.

Two copies of Apple's NAX helpers (integral_constant / BaseNAXFrag / NAXTile, lifted from an
older MLX `steel/`) are embedded in our kernel sources:

* `csrc/mfa_sparse_attention.cpp` (V6NAX sparse forward + V6NAX LSE), and
* the shared block `csrc/mfa/v6_nax/NAAttentionKernel.cpp::mlx_mfa_v6_nax_helpers_block()`
  (GNA NAX / FFN NAX / QMM NAX through `metal_kernel`; dense V6 through our ShaderCache).

MLX >= 0.32.1 compiles `metal_kernel` sources as MSL 4.1 (`__METAL_VERSION__` 410, generic
address space; 0.31.2 / 0.32.0 compile them as 4.0): `decltype(local)` carries `thread` and an
unqualified member function cannot bind a member to a `thread&` — our unqualified copies failed to
build ("Unable to build metal library from source": `NAXTile::frag_at` resolving to a
`const` / non-`thread` reference, `integral_constant<const thread int, ...>`).  Every
`metal_kernel` NAX kernel raised on MLX 0.32.1 / 0.32.2, including the public
`flash_attention_sparse` and `flash_attention_gna` default M5 routes.

These locks are static (they run on any MLX): both copies carry the MLX 0.32.1 forms.  The
runtime proof across the ABI-table MLX versions is scripts/metal_kernel_matrix_smoke.py
(a release gate).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COPIES = [
    ROOT / "csrc" / "mfa_sparse_attention.cpp",
    ROOT / "csrc" / "mfa" / "v6_nax" / "NAAttentionKernel.cpp",
]


def _naxtile_body(src: Path) -> str:
    text = src.read_text(encoding="utf-8")
    start = text.index("struct NAXTile {")
    end = text.index("\n};", start)
    return text[start:end]


def _member_functions(body: str):
    """(signature, qualifier-tail) for every non-static METAL_FUNC member function."""
    out = []
    for m in re.finditer(r"METAL_FUNC(?![^\n]*\bstatic\b)([^{;]*?)\)\s*([^{;()]*)\{", body, re.S):
        sig = " ".join(m.group(1).split())
        out.append((sig, " ".join(m.group(2).split())))
    return out


@pytest.mark.parametrize("src", COPIES, ids=lambda p: p.name)
def test_naxtile_member_functions_are_address_space_qualified(src):
    funcs = _member_functions(_naxtile_body(src))
    assert len(funcs) >= 10, funcs            # constructor, clear, frag_at x2, elems, ...
    bad = [(sig, tail) for sig, tail in funcs if tail not in ("thread", "const thread")]
    assert not bad, (f"{src.name}: NAXTile member functions without a `thread` / "
                     f"`const thread` qualifier (MLX >= 0.32.1 fails to build): {bad}")


@pytest.mark.parametrize("src", COPIES, ids=lambda p: p.name)
def test_const_members_are_const_thread(src):
    """A const member must be `const thread`: the non-const `frag_at` overload would
    otherwise be picked (or fail to bind) inside const methods."""
    for sig, tail in _member_functions(_naxtile_body(src)):
        if tail.startswith("const"):
            assert tail == "const thread", (src.name, sig, tail)


@pytest.mark.parametrize("src", COPIES, ids=lambda p: p.name)
def test_integral_constant_and_mma_types_are_address_space_free(src):
    """MSL 4.1 puts `thread` on `decltype(local)`; a type fed to a template must come from
    a prvalue decltype (never address-space qualified, valid under MSL 4.0 and 4.1)."""
    text = src.read_text(encoding="utf-8")
    assert "operator value_type() const thread noexcept" in text
    assert "operator value_type() const noexcept" not in text
    assert "return integral_constant<decltype(tv __op__ uv), res>{};" in text
    assert "integral_constant<decltype(res), res>" not in text
    assert "using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<" in text
    assert "get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>()" in text
    assert "get_destination_cooperative_tensor<decltype(ct_a)" not in text


@pytest.mark.parametrize("src", COPIES, ids=lambda p: p.name)
def test_no_remove_addrspace_dependency(src):
    """Dense V6 compiles the shared block at runtime with the user's OS Metal compiler;
    metal::remove_addrspace_t first appears in MLX 0.32.1 headers and is unverified on
    older OS compilers, so the helpers must not depend on it."""
    code = "\n".join(l.split("//")[0] for l in src.read_text(encoding="utf-8").splitlines())
    assert "remove_addrspace_t<" not in code


def test_qmm_loader_members_are_address_space_qualified():
    """The QMM NAX loader (adapted from MLX quantized_nax.h, built through metal_kernel):
    constructor and members carry `thread` / `const thread` as MLX 0.32.1 does."""
    text = (ROOT / "csrc" / "mfa_qmm_nax.cpp").read_text(encoding="utf-8")
    start = text.index("struct QMMQuantizedBlockLoader")
    body = text[start:text.index("\n};", start)]
    assert re.search(r"uint simd_lane_id\)\s*thread\s*\n\s*:", body), "constructor not `thread`"
    funcs = _member_functions(body)
    assert len(funcs) >= 5, funcs            # dequantize_one, zero_local, load_safe, load_unsafe, next
    bad = [(sig, tail) for sig, tail in funcs if tail not in ("thread", "const thread")]
    assert not bad, bad

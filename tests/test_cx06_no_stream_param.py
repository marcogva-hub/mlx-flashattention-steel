"""CX-06 — exported _ext bindings must not advertise a non-functional stream param.

The 16 forward/gather/quantize/scatter bindings previously exposed a
`stream` / `StreamOrDevice` / `device` parameter that either raised nanobind
`TypeError` on a real `mx.Stream` (no caster registered) or silently ignored it
(`nb::object`) — the op always ran on the default GPU stream. Signature surgery
removed them. These locks keep the surface honest + catch an internal caller
re-passing stream to a raw binding.
"""
from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import pytest

_ROOT = Path(__file__).resolve().parent.parent

_BINDINGS = [
    "mfa_attention_forward", "mfa_attention_alibi_forward",
    "mfa_attention_bias_forward", "mfa_attention_rope_forward",
    "mfa_attention_sparse_forward", "mfa_attention_sparse_forward_with_lse",
    "mfa_gna_forward", "mfa_attention_varlen_forward", "mfa_paged_kv_gather",
    "mfa_paged_steel_forward", "mfa_sage_forward", "mfa_quantize_per_block",
    "mfa_smooth_quantize_k", "mfa_scatter_kv", "mfa_paged_varlen_forward",
    "mfa_paged_varlen_tq_forward",
]

try:
    import mlx_mfa._ext as _ext
    _HAS_EXT = True
except Exception:  # pragma: no cover
    _HAS_EXT = False


@pytest.mark.skipif(not _HAS_EXT, reason="requires built _ext")
@pytest.mark.parametrize("name", _BINDINGS)
def test_binding_signature_has_no_stream_param(name):
    """The exposed signature must not name a stream/device parameter."""
    sig = (getattr(_ext, name).__doc__ or "").splitlines()[0]
    assert "stream" not in sig and "StreamOrDevice" not in sig and "Device" not in sig, (
        f"{name} still advertises a stream/device param: {sig}")


@pytest.mark.skipif(not _HAS_EXT, reason="requires built _ext")
@pytest.mark.parametrize("name", _BINDINGS)
def test_binding_rejects_stream_kwarg(name):
    """Passing stream= to the binding must raise TypeError (the param is gone).
    Bite target: if the param were re-added, this would NOT raise → FAIL."""
    with pytest.raises(TypeError):
        getattr(_ext, name)(stream=mx.default_stream(mx.gpu))


# ── regression lock: no internal caller passes stream to a raw binding ───────

_ALIASES = set(_BINDINGS) | {
    "_sage_fwd", "_gna_fwd", "_varlen_fwd", "_raw_paged_steel", "_pvf",
    "_mfa_scatter_kv_cpp", "_mfa_quantize_per_block_cpp", "_fused_sq",
    "_fwd", "_bias_fwd", "_rope_fwd", "_alibi_fwd",
}
_STREAMISH = re.compile(r"^(stream|s|self\.stream|eff_stream|_stream|self\._stream|st)$")


def _binding_calls_passing_stream():
    out = []
    for f in (_ROOT / "mlx_mfa").rglob("*.py"):
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        for al in _ALIASES:
            for m in re.finditer(rf"(?<![\w.]){re.escape(al)}\s*\(", text):
                ls = text[:m.start()].count("\n")
                if "import" in lines[ls] or "def " in lines[ls]:
                    continue
                depth, i = 0, m.end() - 1
                while i < len(text):
                    c = text[i]
                    if c == "(":
                        depth += 1
                    elif c == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    i += 1
                inner = text[m.start():i + 1]
                inner = inner[inner.index("(") + 1:-1]
                args, d, cur = [], 0, ""
                for ch in inner:
                    if ch in "([{":
                        d += 1
                    elif ch in ")]}":
                        d -= 1
                    if ch == "," and d == 0:
                        args.append(cur.strip()); cur = ""
                    else:
                        cur += ch
                if cur.strip():
                    args.append(cur.strip())
                if not args:
                    continue
                if any(re.match(r"stream\s*=", a) for a in args) or _STREAMISH.match(args[-1]):
                    out.append(f"{f.name}:{ls + 1}  {al}(...)")
    return out


def test_no_internal_caller_passes_stream_to_a_binding():
    """CX-06 regression: no mlx_mfa/ caller may pass stream (kw or trailing
    positional) to a raw _ext binding — the bindings have no stream param."""
    offenders = _binding_calls_passing_stream()
    assert not offenders, (
        "internal caller(s) pass stream to a raw _ext binding (which no longer "
        "accepts it): " + "; ".join(offenders))

"""2.64 B4/B5/B6 — the extended sparse envelope becomes the default (after the B1-B3 guards).

B4 (D3) non-causal law: N in [2048, 200000] (MAX_N measured to 200k — Phase 2; MIN_N
   2048 = decision D2, evidence "one shape": FlashVSR), any B*H (measured coverage
   {1, 4, 12, 16, 32, 40, 56}), fp16/bf16, density ceilings KEPT (0.30; the measured
   lower cells D128 B*H4 0.05 and D64 B*H12 0.25).  The causal policy is unchanged.
   `MFA_SPARSE_NAX_LEGACY_POLICY=1` restores the 2.63 policy for one release;
   `MFA_SPARSE_NAX_EXTENDED` is a documented, warning no-op.
B5 (D2) non-causal qL != kL routes to v6nax_sparse (csrc/mfa_sparse_attention.cpp:11;
   exact on the real FlashVSR shapes); causal qL != kL stays refused (U2).
B6 (D2) density >= D_DENSE_CUTOFF: the V6NAX kernel when it can serve the call
   (0.42x of the 2.63 path on FlashVSR), else SDPA + BOOL keep-mask — never a float bias.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import flash_attention_sparse
from mlx_mfa import lcsa_nax as ln

nax = pytest.mark.skipif(not mlx_mfa.has_nax(), reason="M5+ sparse routing")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("MFA_SPARSE_NAX_LEGACY_POLICY", "MFA_SPARSE_NAX_EXTENDED",
              "MFA_SPARSE_D_DENSE_CUTOFF", "MFA_NAX_SPARSE_DENSITY_CEILING",
              "MFA_ENABLE_V6_BACKWARD", "MFA_SPARSE_FALLBACK_MAX_BYTES"):
        monkeypatch.delenv(k, raising=False)


def _qkv(B, H, N, S=None, D=128, dtype=mx.float16, seed=0):
    S = N if S is None else S
    mx.random.seed(seed)
    q = mx.random.normal((B, H, N, D)).astype(dtype)
    k = mx.random.normal((B, H, S, D)).astype(dtype)
    v = mx.random.normal((B, H, S, D)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def _periodic(nq, nk, period):
    """Deterministic mask, density 1/period, every row non-empty."""
    return ((mx.arange(nq)[:, None] - mx.arange(nk)[None, :]) % period) == 0


def _terminal(tr):
    own = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)]
    return own[-1] if own else None


def _run(q, k, v, m, **kw):
    with dt.capture() as tr:
        o = flash_attention_sparse(q, k, v, m, **kw)
        mx.eval(o)
    return o, _terminal(tr)


def _raw(q, k, v, m, causal=False):
    from mlx_mfa import _ext
    return _ext.sparse_attention_forward(q, k, v, m.astype(mx.bool_), block_tile=32,
                                         causal=causal, scale=1.0 / math.sqrt(q.shape[-1]),
                                         kernel_version="v6nax_sparse")


def _oracle_gate(o, q, k, v, m, causal=False, rows=(0, 1, 777, -1)):
    """Per-row magnitude gates vs a CPU fp32 masked oracle (never cosine alone)."""
    from tests.sparse_gates import row_gate_report
    N, S = q.shape[2], k.shape[2]
    idx = [r % N for r in rows]
    with mx.stream(mx.cpu):
        keep = mx.repeat(mx.repeat(m.astype(mx.bool_), 32, axis=-2), 32, axis=-1)[..., :N, :S]
        if causal:
            keep = keep & (mx.arange(S)[None, :] <= mx.arange(N)[:, None] + max(0, S - N))
        qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
        s = (qf[:, :, idx] @ kf.transpose(0, 1, 3, 2)) / math.sqrt(q.shape[-1])
        s = mx.where(keep[..., idx, :], s, mx.array(-float("inf")))
        ref = mx.softmax(s, axis=-1) @ vf
        mx.eval(ref)
    rep = row_gate_report(o[:, :, idx].astype(mx.float32).reshape(-1, len(idx), q.shape[-1]),
                          ref.reshape(-1, len(idx), q.shape[-1]))
    assert rep["finite"] and rep["worst_row_rel"] < 1e-2 and rep["worst_row_norm_dev"] < 5e-3, rep
    return rep


# ── B4: the promoted non-causal law ───────────────────────────────────────────────
def test_promoted_constants():
    assert ln.SPARSE_NAX_MIN_N == 2048
    assert ln.SPARSE_NAX_MAX_N == 200_000
    assert ln.SPARSE_NAX_DENSITY_CEILING == 0.30
    assert ln.SPARSE_NAX_MEASURED_BH_COVERAGE == frozenset({1, 4, 12, 16, 32, 40, 56})


@pytest.mark.parametrize("bh", [1, 4, 12, 16, 20, 32, 40, 56])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_law_routes_any_bh(bh, dtype):
    q = mx.zeros((1, bh, 8192, 128), dtype=dtype)
    d = 0.04 if bh == 4 else 0.10          # D128 B*H4 keeps its measured 0.05 ceiling
    assert ln._nax_sparse_route_viable(q, q, 32, d, causal=False, V=q)


@pytest.mark.parametrize("N,ok", [(2047, False), (2048, True), (200_000, True), (200_032, False)])
def test_law_n_bounds(N, ok):
    q = mx.zeros((1, 12, N, 128), dtype=mx.float16)
    assert ln._nax_sparse_route_viable(q, q, 32, 0.1, causal=False, V=q) is ok


@pytest.mark.parametrize("bh,D,d,ok", [(12, 128, 0.30, True), (12, 128, 0.31, False),
                                       (4, 128, 0.05, True), (4, 128, 0.06, False),
                                       (12, 64, 0.25, True), (12, 64, 0.26, False)])
def test_ceilings_are_kept(bh, D, d, ok):
    """The lower ceilings were measured at N < 8192 (N=8192 won every cell at 0.30)."""
    q = mx.zeros((1, bh, 4096, D), dtype=mx.float16)
    assert ln._nax_sparse_route_viable(q, q, 32, d, causal=False, V=q) is ok


@nax
def test_n2048_bh12_now_engages_nax():
    q, k, v = _qkv(1, 12, 2048)
    m = _periodic(64, 64, 8)
    o, term = _run(q, k, v, m)
    assert term[0] == "v6nax_sparse", term
    assert bool(mx.array_equal(o, _raw(q, k, v, m)))
    _oracle_gate(o, q, k, v, m)


@nax
def test_unmeasured_bh_non_aligned_n_goes_nax_via_auto_pad():
    q, k, v = _qkv(1, 40, 4100)                 # B*H 40, N % 32 != 0
    m = _periodic(-(-4100 // 32), -(-4100 // 32), 10)
    o, term = _run(q, k, v, m, auto_pad=True)
    assert term[0] == "v6nax_sparse" and "auto_pad" in term[1], term
    _oracle_gate(o, q, k, v, m)


@nax
def test_legacy_policy_knob_restores_263(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_NAX_LEGACY_POLICY", "1")
    q, k, v = _qkv(1, 12, 2048)
    _, term = _run(q, k, v, _periodic(64, 64, 8))
    assert term[0] == "sdpa", term              # 2.63: N=2048 outside [4096, 8192]
    q16 = mx.zeros((1, 16, 8192, 128), dtype=mx.float16)
    assert not ln._nax_sparse_route_viable(q16, q16, 32, 0.1, causal=False, V=q16)


def test_extended_knob_is_a_warning_noop(monkeypatch):
    monkeypatch.setenv("MFA_SPARSE_NAX_EXTENDED", "1")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        on = ln._sparse_extended_enabled()
    assert on is False
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


# ── B5: rectangular non-causal ────────────────────────────────────────────────────
@nax
@pytest.mark.parametrize("Nq,Nk", [(2048, 8192), (2560, 10240)])
def test_rectangular_low_density_engages_nax(Nq, Nk):
    q, k, v = _qkv(1, 12, Nq, S=Nk)
    m = _periodic(Nq // 32, Nk // 32, 10)
    o, term = _run(q, k, v, m)
    assert term[0] == "v6nax_sparse", term
    assert bool(mx.array_equal(o, _raw(q, k, v, m)))
    _oracle_gate(o, q, k, v, m)


@nax
def test_rectangular_causal_stays_refused():
    q, k, v = _qkv(1, 12, 2048, S=8192)
    m = _periodic(64, 256, 10)
    o, term = _run(q, k, v, m, causal=True)
    assert term[0] == "sdpa", term
    _oracle_gate(o, q, k, v, m, causal=True)


# ── B6: quasi-dense ───────────────────────────────────────────────────────────────
def _window_mask(H, Wq_t, Wk_t, S):
    """FlashVSR-like: all window pairs but one per (head, query temporal window), at 32x32."""
    Wq, Wk = Wq_t * S, Wk_t * S
    flat = mx.ones((H, Wq_t, S * Wk), dtype=mx.bool_)
    flat = mx.concatenate([flat[..., :-1], mx.zeros((H, Wq_t, 1), dtype=mx.bool_)], axis=-1)
    mw = flat.reshape(H, Wq, Wk)
    return mx.repeat(mx.repeat(mw, 4, axis=-2), 4, axis=-1)          # 128-token windows


@nax
@pytest.mark.parametrize("Wq_t,Wk_t,S", [(1, 4, 16), (3, 3, 16), (1, 4, 20)])
def test_quasi_dense_goes_nax_on_real_flashvsr_shapes(Wq_t, Wk_t, S):
    q, k, v = _qkv(1, 12, Wq_t * S * 128, S=Wk_t * S * 128)
    m = _window_mask(12, Wq_t, Wk_t, S)
    assert ln.mask_density(m) >= ln._d_dense_cutoff()
    o, term = _run(q, k, v, m)
    assert term[0] == "v6nax_sparse" and "quasi-dense" in term[1], term
    assert bool(mx.array_equal(o, _raw(q, k, v, m)))
    _oracle_gate(o, q, k, v, m)


@nax
def test_quasi_dense_kernel_cannot_serve_goes_bool_sdpa(monkeypatch):
    q, k, v = _qkv(1, 12, 2048, S=8192)
    m = mx.ones((64, 512), dtype=mx.bool_)                           # 32x16: not NAX-servable
    seen = []
    real = mx.fast.scaled_dot_product_attention

    def spy(*a, mask=None, **kw):
        if mask is not None and not isinstance(mask, str):
            seen.append(mask.dtype)
        return real(*a, mask=mask, **kw)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", spy)
    _, term = _run(q, k, v, m)
    assert term[0] == "sdpa" and seen and all(d == mx.bool_ for d in seen), (term, seen)


def test_no_library_call_site_builds_a_float_bias_from_a_block_mask():
    """Structural lock (B2/B6): the float-bias builders may be DEFINED (legacy helpers)
    but no library code may CALL them — a block mask only ever becomes a bool mask."""
    names = ("_block_mask_to_float_bias", "_block_mask_to_float_bias_nd",
             "_bool_mask_to_float_bias", "_get_or_build_expanded_float_bias")
    offenders = []
    for f in (ROOT / "mlx_mfa").rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for n in names:
                if re.search(rf"(?<!def ){re.escape(n)}\(", code) and "def " + n not in code:
                    offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, offenders

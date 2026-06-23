"""Volet L — sparse_attention_dispatch hardening (CX-R8-01) + classifier
completeness (CX-R8-02).

Round-8 found the 23rd public computational entry, `sparse_attention_dispatch`
(lcsa_nax.py), was misclassified as a helper (name-prefix heuristic) and its
forced-SDPA route accepted malformed V (kv-seq / dtype mismatch → finite-wrong /
NaN; V=1 nondeterministic). Fix validates Q/K/V at the dispatcher entry, BEFORE
the native-vs-SDPA split, so BOTH routes are guarded. f32 and asymmetric D_v are
valid (SDPA route) — NOT restricted (the HARD-GUARD lesson).
"""
import math
import importlib.util
import numpy as np
import mlx.core as mx
import pytest
from mlx_mfa.lcsa_nax import sparse_attention_dispatch as sad

Hq, Hk, N, D = 4, 2, 64, 64
SC = 1.0 / math.sqrt(D)
BT = 32
_NT = (N + BT - 1) // BT


def _mk(nv=N, dtk=None, dt=mx.float16, dv=D):
    mx.random.seed(0)
    q = mx.random.normal((1, Hq, N, D)).astype(dt)
    k = mx.random.normal((1, Hk, N, D)).astype(dt)
    v = mx.random.normal((1, Hk, nv, dv)).astype(dtk or dt)
    mx.eval(q, k, v)
    return q, k, v


def _mask():
    return mx.ones((Hq, _NT, _NT), dtype=mx.bool_)


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


# ── correctness: the SDPA route vs fp64 dense oracle (all-True mask = full attn).
# NOTE: at N=64 the small all-true mask is < the native kernel's 4096-byte floor,
# so this config exercises the SDPA route regardless of threshold. Genuine
# native-route engagement (+ byteΔ-vs-SDPA proof) is locked in test_hardening_m.py
# (N=1024) — the L "native" cells at threshold=1.0 actually ran SDPA (CX-R9-02).
def test_correctness_sdpa_route():
    q, k, v = _mk()
    o = sad(q, k, v, _mask(), block_tile=BT, scale=SC, causal=False, density_threshold=0.0)
    mx.eval(o)
    qf, kf, vf = _f64(q), _f64(k), _f64(v)
    kk, vv = np.repeat(kf, Hq // Hk, 1), np.repeat(vf, Hq // Hk, 1)
    s = np.einsum("bhnd,bhmd->bhnm", qf, kk) * SC
    s -= s.max(-1, keepdims=True)
    p = np.exp(s); p /= p.sum(-1, keepdims=True)
    ref = np.einsum("bhnm,bhmd->bhnd", p, vv)
    rel = float(np.max(np.abs(_f64(o) - ref)) / (np.max(np.abs(ref)) + 1e-9))
    assert rel < 5e-3, f"sdpa relerr {rel:.3e}"


# ── accept-valid dtypes: f16/bf16 (native or SDPA per density) and f32 (SDPA
# only — the native kernel is f16/bf16-only). At this N=64 config the all-true
# mask is below the native 4096-byte floor, so every cell here runs SDPA
# regardless of threshold; the native route is exercised in test_hardening_m.py.
@pytest.mark.parametrize("thr", [0.0, 1.0])
@pytest.mark.parametrize("dt", [mx.float16, mx.bfloat16, mx.float32])
def test_accept_valid_dtypes(thr, dt):
    mx.eval(sad(*_mk(dt=dt), _mask(), block_tile=BT, scale=SC, density_threshold=thr))


def test_accept_valid_asym_dv_sdpa():
    # asymmetric D_v is valid on the SDPA route — must NOT be rejected.
    mx.eval(sad(*_mk(dv=32), _mask(), block_tile=BT, scale=SC, density_threshold=0.0))


# ── reject-malformed: both routes ───────────────────────────────────────────────
@pytest.mark.parametrize("thr", [0.0, 1.0])
@pytest.mark.parametrize("nv", [1, 2, 8, 16, 31, 33, 63])
def test_reject_v_seq_mismatch(thr, nv):
    with pytest.raises((ValueError, Exception)):
        mx.eval(sad(*_mk(nv=nv), _mask(), block_tile=BT, scale=SC, density_threshold=thr))


@pytest.mark.parametrize("thr", [0.0, 1.0])
def test_reject_dtype_mismatch(thr):
    with pytest.raises((ValueError, Exception)):
        mx.eval(sad(*_mk(dtk=mx.bfloat16), _mask(), block_tile=BT, scale=SC, density_threshold=thr))


def test_reject_batch_and_gqa():
    mx.random.seed(0)
    q = mx.random.normal((2, Hq, N, D)).astype(mx.float16)
    k = mx.random.normal((1, Hk, N, D)).astype(mx.float16)
    v = mx.random.normal((1, Hk, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    with pytest.raises((ValueError, Exception)):  # batch mismatch
        mx.eval(sad(q, k, v, _mask(), block_tile=BT, scale=SC, density_threshold=0.0))
    q3 = mx.random.normal((1, 3, N, D)).astype(mx.float16); mx.eval(q3)
    with pytest.raises((ValueError, Exception)):  # GQA 3 % 2 != 0
        mx.eval(sad(q3, k, v, _mask(), block_tile=BT, scale=SC, density_threshold=0.0))


# ── determinism: valid input, SDPA route (Apple SDPA), 20 fresh-identical runs ──
def test_determinism_sdpa_route():
    outs = []
    for _ in range(20):
        o = sad(*_mk(), _mask(), block_tile=BT, scale=SC, density_threshold=0.0)
        mx.eval(o)
        outs.append(np.array(o.astype(mx.float32)))
    md = max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, 20))
    assert md == 0.0, f"sparse_attention_dispatch SDPA route nondeterministic byteΔ={md:.3e}"


# ── CX-R8-02: classifier completeness ───────────────────────────────────────────
def _load_enum():
    spec = importlib.util.spec_from_file_location("_enum", "scripts/enumerate_api_surface.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_classifier_marks_dispatch_computational():
    m = _load_enum()
    cls, _why = m.classify_public("sparse_attention_dispatch", "mlx_mfa.lcsa_nax")
    assert cls == "computational"


def test_classifier_unknown_export_is_unclassified():
    # A genuinely-new export matching no rule must NOT silently become a helper.
    m = _load_enum()
    cls, _why = m.classify_public("some_brand_new_attention_thing", "mlx_mfa.attention")
    assert cls == "UNCLASSIFIED"


def test_enumeration_complete_public_0_omitted():
    # count is asserted exactly in test_hardening_m (24 after volet M); here just
    # lock that sparse_attention_dispatch is computational and nothing is unclassified.
    m = _load_enum()
    pub = m.public_exports()
    unclass = [n for n in pub if m.classify_public(n, pub[n])[0] == "UNCLASSIFIED"]
    assert not unclass, f"unclassified exports: {unclass}"
    assert m.classify_public("sparse_attention_dispatch", "mlx_mfa.lcsa_nax")[0] == "computational"

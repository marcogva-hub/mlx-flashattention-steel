"""Volet M — 24th entry (make_shared_prefix_cache) + durable classifier
(CX-R9-01) + sparse_attention_dispatch native-route engagement & f32 contract
(CX-R9-02).

Round-9: the classifier's `make_* → helper` name rule hid the 24th public
computational entry (make_shared_prefix_cache, which calls flash_attention and
returns attention). The sparse checks use a cell inside the hardened beta-3
gate and prove the secondary density threshold genuinely changes the route.
The native sparse kernel is f16/bf16-only; f32 routes to SDPA regardless of
density.
"""
import math
import importlib.util
import numpy as np
import mlx.core as mx
import pytest
import mlx_mfa
from mlx_mfa.lcsa_nax import sparse_attention_dispatch as sad


def _f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


# ── CX-R9-01: make_shared_prefix_cache (the 24th) ───────────────────────────────
def _qkv(B=1, Hq=8, Hk=2, N=256, D=128, dt=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((B, Hq, N, D)).astype(dt)
    k = mx.random.normal((B, Hk, N, D)).astype(dt)
    v = mx.random.normal((B, Hk, N, D)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def test_prefix_cache_correctness_byte_identical_to_flash():
    q, k, v = _qkv()
    sc = 1.0 / math.sqrt(q.shape[-1])
    out, kp, vp = mlx_mfa.make_shared_prefix_cache(q, k, v, scale=sc)
    direct = mlx_mfa.flash_attention(q, k, v, scale=sc, causal=True)
    mx.eval(out, kp, vp, direct)
    # prefix_out IS flash_attention(...) — byteΔ must be 0; k/v passed through.
    assert float(np.max(np.abs(_f64(out) - _f64(direct)))) == 0.0
    assert float(np.max(np.abs(_f64(kp) - _f64(k)))) == 0.0
    assert float(np.max(np.abs(_f64(vp) - _f64(v)))) == 0.0


def test_prefix_cache_accept_valid_gqa_and_bf16():
    for dt in (mx.float16, mx.bfloat16):
        out, _kp, _vp = mlx_mfa.make_shared_prefix_cache(*_qkv(dt=dt))
        mx.eval(out)


@pytest.mark.parametrize("mut", ["batch", "k_seq", "gqa"])
def test_prefix_cache_reject_malformed_inherited(mut):
    # validation is inherited from the hardened flash_attention core.
    q, k, v = _qkv()
    if mut == "batch":
        q = mx.broadcast_to(q, (2, 8, 256, 128))
    elif mut == "k_seq":
        v = v[:, :, :128, :]
    elif mut == "gqa":
        q, k, v = _qkv(Hq=8, Hk=3)
    with pytest.raises((ValueError, Exception)):
        out, _kp, _vp = mlx_mfa.make_shared_prefix_cache(q, k, v)
        mx.eval(out)


def test_prefix_cache_determinism():
    outs = []
    for _ in range(20):
        out, _kp, _vp = mlx_mfa.make_shared_prefix_cache(*_qkv())
        mx.eval(out)
        outs.append(np.array(out.astype(mx.float32)))
    assert max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, 20)) == 0.0


# ── CX-R9-02: native-route engagement + f32 contract ────────────────────────────
# Measured winning region: N4096, B·H4, D128, density<=0.05.
_HQ, _HK, _N, _D = 4, 2, 4096, 128
_SC = 1.0 / math.sqrt(_D)
_BT = 32
_NT = (_N + _BT - 1) // _BT


def _sq(dt=mx.float16):
    mx.random.seed(0)
    q = mx.random.normal((1, _HQ, _N, _D)).astype(dt)
    k = mx.random.normal((1, _HK, _N, _D)).astype(dt)
    v = mx.random.normal((1, _HK, _N, _D)).astype(dt)
    mx.eval(q, k, v)
    return q, k, v


def _mask():
    data = np.zeros((_HQ, _NT, _NT), dtype=np.bool_)
    data[:, :, :max(1, int(_NT * 0.04))] = True
    return mx.array(data)


def test_native_route_actually_engages():
    # The canonical gate accepts d~0.04; the legacy threshold can only narrow.
    q, k, v = _sq()
    o_sdpa = sad(q, k, v, _mask(), block_tile=_BT, scale=_SC, density_threshold=0.03)
    o_nat = sad(q, k, v, _mask(), block_tile=_BT, scale=_SC, density_threshold=0.05)
    mx.eval(o_sdpa, o_nat)
    delta = float(np.max(np.abs(_f64(o_sdpa) - _f64(o_nat))))
    assert delta > 0.0, "native route did not engage (byteΔ=0 ⇒ same path as SDPA)"
    # Native remains correct against the same-dtype masked SDPA reference.
    from mlx_mfa.lcsa_nax import _bool_mask_to_float_bias
    bias = _bool_mask_to_float_bias(_mask(), _BT, _N, _N, q.dtype)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=_SC, mask=bias)
    mx.eval(ref)
    rel = float(np.max(np.abs(_f64(o_nat) - _f64(ref))) /
                (np.max(np.abs(_f64(ref))) + 1e-9))
    assert rel < 5e-3, f"native relerr {rel:.3e}"


def test_f32_contract_consistent_not_density_dependent():
    # f32 must route to SDPA in BOTH density regimes (was: SDPA ran, native raised).
    o_a = sad(*_sq(mx.float32), _mask(), block_tile=_BT, scale=_SC, density_threshold=0.03)
    o_b = sad(*_sq(mx.float32), _mask(), block_tile=_BT, scale=_SC, density_threshold=0.05)
    mx.eval(o_a, o_b)
    assert o_a.dtype == mx.float32 and o_b.dtype == mx.float32
    # same path (SDPA) both ways → byteΔ-identical
    assert float(np.max(np.abs(_f64(o_a) - _f64(o_b)))) == 0.0


def test_native_route_determinism():
    outs = []
    for _ in range(20):
        o = sad(*_sq(), _mask(), block_tile=_BT, scale=_SC, density_threshold=0.05)
        mx.eval(o)
        outs.append(np.array(o.astype(mx.float32)))
    assert max(float(np.max(np.abs(outs[0] - outs[i]))) for i in range(1, 20)) == 0.0


# ── CX-R9-01/02: durable classifier (no name patterns + cross-check) ────────────
def _load_enum():
    spec = importlib.util.spec_from_file_location("_enum2", "scripts/enumerate_api_surface.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_classifier_24_computational_and_prefix_cache_computational():
    m = _load_enum()
    pub = m.public_exports()
    comp = [n for n in pub if m.classify_public(n, pub[n])[0] == "computational"]
    assert len(comp) == 24, f"expected 24, got {len(comp)}"
    assert "make_shared_prefix_cache" in comp
    assert "sparse_attention_dispatch" in comp


def test_classifier_no_name_patterns():
    # make_* must NOT auto-classify: make_causal_block_mask is a helper, but the
    # rule must be the explicit allowlist, not a prefix. Verify a make_* helper
    # and a make_* computational coexist correctly.
    m = _load_enum()
    assert m.classify_public("make_causal_block_mask", "mlx_mfa.attention")[0] == "helper"
    assert m.classify_public("make_shared_prefix_cache", "mlx_mfa.attention")[0] == "computational"


def test_classifier_cross_check_assertion2_clean():
    # No HELPER export currently takes q/k/v AND calls a compute entry.
    m = _load_enum()
    pub = m.public_exports()
    helpers = [n for n in pub if m.classify_public(n, pub[n])[0] == "helper"]
    assert m.computational_in_helper(helpers) == []


def test_classifier_unknown_export_unclassified():
    m = _load_enum()
    assert m.classify_public("totally_new_export_xyz", "mlx_mfa.attention")[0] == "UNCLASSIFIED"

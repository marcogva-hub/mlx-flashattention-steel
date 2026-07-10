"""BT-aware sparse_attention_dispatch routing (victory-map win window).

Locks the dispatcher-correctness fix: TILE VIABILITY is the primary gate. The
non-causal proven-viable window (BT=32, N≥2048, D∈{64,128}, density≤ceiling)
routes to native NAX. The causal CC-measured β3 window is stricter:
BT=32, 2048≤N≤8192, D∈{64,128}, density≤0.3, B·H≤128. Everything else routes
to SDPA regardless of density_threshold.

β3-measured window; the constants are documented re-validate-on-stable. NAX kernel
is Metal-4 / macOS-26 (stable track). M5-gated (the NAX kernel is M5+).
"""
import math
import numpy as np
import pytest
import mlx.core as mx

from mlx_mfa import lcsa_nax as L
from mlx_mfa.attention import _get_is_m5_plus_cached

pytestmark = pytest.mark.skipif(
    not _get_is_m5_plus_cached(), reason="NAX sparse kernel is M5+ only")


def _inputs(N, D, H=4):
    q = mx.random.normal((1, H, N, D)).astype(mx.float16)
    k = mx.random.normal((1, H, N, D)).astype(mx.float16)
    v = mx.random.normal((1, H, N, D)).astype(mx.float16)
    mx.eval(q, k, v)
    return q, k, v


def _mask(N, BT, density, seed=0):
    nq = nk = N // BT
    rng = np.random.default_rng(seed)
    m = rng.random((nq, nk)) < density
    m[np.arange(nq), np.arange(nk)] = True
    return mx.array(m)


def test_constants_documented_window():
    assert L.SPARSE_NAX_VIABLE_BLOCK_TILES == frozenset({32})
    assert L.SPARSE_NAX_MIN_N == 2048
    assert L.SPARSE_NAX_CAUSAL_MIN_N == 2048
    assert L.SPARSE_NAX_CAUSAL_MAX_N == 8192
    assert L.SPARSE_NAX_CAUSAL_MAX_BH == 128
    assert L.SPARSE_NAX_CAUSAL_DENSITY_CEILING == 0.30
    assert L.SPARSE_NAX_VIABLE_HEAD_DIMS == frozenset({64, 128})
    assert 0.0 < L.SPARSE_NAX_DENSITY_CEILING <= 1.0


@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("N", [1024, 2048, 4096])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("density", [0.05, 0.5])
def test_routing_table_matches_win_window(monkeypatch, BT, N, D, density):
    """Non-causal route to NAX iff (BT=32 AND N≥2048 AND D∈{64,128}); else SDPA."""
    called = {"nax": False}
    orig = L.sparse_attention_nax
    monkeypatch.setattr(L, "sparse_attention_nax",
                        lambda *a, **k: (called.__setitem__("nax", True), orig(*a, **k))[1])
    q, kk, v = _inputs(N, D)
    m = _mask(N, BT, density)
    L.sparse_attention_dispatch(q, kk, v, m, block_tile=BT, scale=1.0/math.sqrt(D), density=density)
    expect_nax = (BT == 32 and N >= L.SPARSE_NAX_MIN_N and D in L.SPARSE_NAX_VIABLE_HEAD_DIMS)
    assert called["nax"] is expect_nax


class _FakeArray:
    def __init__(self, shape, dtype=mx.float16):
        self.shape = shape
        self.dtype = dtype


@pytest.mark.parametrize(
    "case,B,H,N,D,BT,density,dtype,expect",
    [
        ("d128_edge", 1, 128, 2048, 128, 32, 0.30, mx.float16, True),
        ("d64_edge", 2, 64, 8192, 64, 32, 0.30, mx.bfloat16, True),
        ("too_dense", 1, 4, 2048, 128, 32, 0.3001, mx.float16, False),
        ("too_many_heads", 1, 129, 2048, 128, 32, 0.30, mx.float16, False),
        ("too_short", 1, 4, 1024, 128, 32, 0.30, mx.float16, False),
        ("too_long", 1, 4, 8193, 128, 32, 0.30, mx.float16, False),
        ("bad_tile", 1, 4, 2048, 128, 16, 0.30, mx.float16, False),
        ("bad_dim", 1, 4, 2048, 256, 32, 0.30, mx.float16, False),
        ("bad_dtype", 1, 4, 2048, 128, 32, 0.30, mx.float32, False),
    ],
)
def test_causal_route_viability_matches_cc_window(case, B, H, N, D, BT, density, dtype, expect):
    """Pure gate lock: causal default-on is bounded to CC's measured β3 window."""
    del case
    q = _FakeArray((B, H, N, D), dtype)
    k = _FakeArray((B, H, N, D), dtype)
    assert L._nax_sparse_route_viable(q, k, BT, density, causal=True) is expect


@pytest.mark.parametrize("density,expect_nax", [(0.30, True), (0.3001, False)])
def test_causal_dispatch_density_gate(monkeypatch, density, expect_nax):
    """The public dispatcher applies the causal density ceiling before calling NAX."""
    called = {"nax": False}
    orig = L.sparse_attention_nax
    monkeypatch.setattr(L, "sparse_attention_nax",
                        lambda *a, **k: (called.__setitem__("nax", True), orig(*a, **k))[1])
    q, kk, v = _inputs(2048, 64, H=1)
    m = _mask(2048, 32, 0.15)
    L.sparse_attention_dispatch(
        q, kk, v, m, block_tile=32, scale=1.0 / math.sqrt(64),
        density=density, causal=True)
    assert called["nax"] is expect_nax


def test_bt16_routes_sdpa_regardless_of_threshold():
    """The headline fix: BT=16 → SDPA even at a permissive threshold (was the
    ~5.5× mis-route), and byte-identical to an explicit SDPA-threshold call."""
    q, k, v = _inputs(4096, 128)
    m = _mask(4096, 16, 0.15)
    sc = 1.0 / math.sqrt(128)
    o_default = np.array(L.sparse_attention_dispatch(q, k, v, m, block_tile=16, scale=sc, density=0.15).astype(mx.float32))
    o_sdpa = np.array(L.sparse_attention_dispatch(q, k, v, m, block_tile=16, scale=sc, density=0.15, density_threshold=0.0).astype(mx.float32))
    assert np.array_equal(o_default, o_sdpa)  # both take the SDPA+bias path


def test_uncharacterized_tiles_route_sdpa(monkeypatch):
    """BT=64 (uncharacterized) → SDPA (conservative)."""
    called = {"nax": False}
    orig = L.sparse_attention_nax
    monkeypatch.setattr(L, "sparse_attention_nax",
                        lambda *a, **k: (called.__setitem__("nax", True), orig(*a, **k))[1])
    q, k, v = _inputs(4096, 128)
    m = _mask(4096, 64, 0.15)
    L.sparse_attention_dispatch(q, k, v, m, block_tile=64, scale=1.0/math.sqrt(128), density=0.15)
    assert called["nax"] is False


def test_viable_window_correct_vs_fp32(monkeypatch):
    """Where routed to NAX (BT=32 window), output is correct vs fp32 dense-with-mask."""
    called = {"nax": False}
    orig = L.sparse_attention_nax
    monkeypatch.setattr(L, "sparse_attention_nax",
                        lambda *a, **k: (called.__setitem__("nax", True), orig(*a, **k))[1])
    N, D, BT = 4096, 128, 32
    q, k, v = _inputs(N, D)
    m = _mask(N, BT, 0.15)
    sc = 1.0 / math.sqrt(D)
    o = np.array(L.sparse_attention_dispatch(q, k, v, m, block_tile=BT, scale=sc, density=0.15).astype(mx.float32)).reshape(-1)
    assert called["nax"] is True  # engagement: NAX ran
    el = np.kron(np.array(m), np.ones((BT, BT), bool))[:N, :N]
    ref = np.array(mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=sc,
        mask=mx.array(np.where(el, 0.0, -1e9).astype(np.float32)))).reshape(-1)
    cos = float(np.dot(o, ref) / (np.linalg.norm(o) * np.linalg.norm(ref) + 1e-12))
    assert cos >= 0.999

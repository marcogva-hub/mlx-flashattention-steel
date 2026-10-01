"""2.64 A1/A2/A4 — dense D=128 `auto` delegates to SDPA byte-identically, except the
measured static tile table; the verbose dispatch log names the terminal that runs.

Evidence (devnotes/release_2640.md §A):
  * production_shapes_2026-10 Phase 1 — 10/10 production D=128 shapes ran `nax_dense`
    under `auto` (never byte-identical to SDPA), 5-11 % slower on 5/10.
  * DAY-3 Block 2 bands re-measured against the NEW default (SDPA), solo contract
    (benchmarks/results/production_shapes_20261001/tile_vs_sdpa/): 32.32.2 D128 wins
    >= 2x floor on the table cells below; 128.32.8 loses to SDPA (-4..-5 %) -> absent.

Three axes: which-binary (dispatch trace + byte fingerprint vs SDPA / raw kernel),
correctness (fp32 oracle, per-row gates), edges (band borders, causal, custom scale,
knob precedence).
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import flash_attention

pytestmark = pytest.mark.skipif(not mlx_mfa.has_nax(), reason="M5+ NAX dense routing")

TILE = (32, 32, 2)


def _qkv(B, H, N, D=128, dtype=mx.float16, seed=0):
    mx.random.seed(seed)
    q, k, v = (mx.random.normal((B, H, N, D)).astype(dtype) for _ in range(3))
    mx.eval(q, k, v)
    return q, k, v


def _terminal(tr):
    own = [t for t in tr if not t[1].startswith(dt.REENTRANT_PREFIX)]
    return own[-1] if own else None


def _sdpa(q, k, v, causal=False, scale=None):
    sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=sc,
                                                mask="causal" if causal else None)


def _auto(q, k, v, **kw):
    with dt.capture() as tr:
        o = flash_attention(q, k, v, **kw)
        mx.eval(o)
    return o, _terminal(tr)


def _raw_nax(q, k, v, tile=None, scale=None):
    from mlx_mfa import _ext
    sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if tile is None:
        return _ext.v6_nax_forward(q, k, v, False, True, sc)[0]
    return _ext.v6_nax_forward(q, k, v, False, True, sc, *tile)[0]


def _clear_dense_env(monkeypatch):
    for k in ("MFA_ENABLE_V6_DENSE", "MFA_DISABLE_V6_DENSE", "MFA_V6_DENSE_MIN_N",
              "MLX_MFA_DISPATCH_TABLE", "MFA_V6_NAX_BQ", "MFA_V6_NAX_BK", "MFA_V6_NAX_WM"):
        monkeypatch.delenv(k, raising=False)


# ── A1: outside the table, auto == SDPA, byte for byte ────────────────────────────
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("B,H,N,causal", [
    (1, 2, 2048, False),     # B*H=2: no table row
    (1, 2, 6144, False),     # N outside every band
    (1, 4, 3072, False),     # fp16 B1H4 N3072 measured-losing (-16.7 %): must delegate
    (2, 8, 4096, True),      # table is non-causal only
    (1, 32, 2052, False),    # LTX stage-1 occupancy (B*H=32: unmeasured -> default)
])
def test_auto_d128_delegates_byte_identical(monkeypatch, dtype, B, H, N, causal):
    _clear_dense_env(monkeypatch)
    if (B, H, N, causal) == (1, 4, 3072, False) and dtype == mx.bfloat16:
        pytest.skip("bf16 B1H4 has no N3072 table row either way (band [4096, 4608])")
    q, k, v = _qkv(B, H, N, dtype=dtype)
    o, term = _auto(q, k, v, causal=causal)
    assert term is not None and term[0] == "sdpa", term
    assert bool(mx.array_equal(o, _sdpa(q, k, v, causal))), \
        "auto must be byte-identical to mx.fast.scaled_dot_product_attention"


# ── A2: the static table (32.32.2, D128, non-causal) ─────────────────────────────
TABLE_HITS = [
    (mx.float16, 2, 8, 2048), (mx.float16, 2, 8, 4096), (mx.float16, 1, 12, 3072),
    (mx.float16, 1, 4, 4096),
    (mx.bfloat16, 2, 8, 3072), (mx.bfloat16, 1, 12, 4096), (mx.bfloat16, 1, 4, 4608),
]
TABLE_MISSES = [
    (mx.float16, 2, 8, 4608),    # above the measured band
    (mx.float16, 1, 4, 2048),    # measured win, but not contiguous with the N4096 anchor
    (mx.float16, 1, 4, 4608),    # fp16 B1H4 band is {4096}
    (mx.bfloat16, 1, 4, 3584),   # below bf16 B1H4 [4096, 4608]
    (mx.float16, 1, 16, 1024),   # below every band (N < 2048)
]


@pytest.mark.parametrize("dtype,B,H,N", TABLE_HITS)
def test_table_row_routes_to_nax_with_its_tile(monkeypatch, dtype, B, H, N):
    _clear_dense_env(monkeypatch)
    q, k, v = _qkv(B, H, N, dtype=dtype)
    o, term = _auto(q, k, v)
    assert term[0] == "nax_dense" and "table" in term[1] and "32.32.2" in term[1], term
    # which-binary: the NAX kernel (not SDPA).  NB: BQ/WM only regroup query rows —
    # per-row arithmetic (BK=32 accumulation order) is unchanged, so the table tile
    # and the default tile are byte-IDENTICAL; the tile itself is proven by the
    # compiled-source channel in test_table_tile_reaches_the_compiled_kernel.
    assert bool(mx.array_equal(o, _raw_nax(q, k, v, TILE)))
    assert not bool(mx.array_equal(o, _sdpa(q, k, v)))


def test_table_tile_reaches_the_compiled_kernel(monkeypatch, capfd):
    """Which-binary for the tile: MFA_V6_DUMP_SOURCE prints the compiled NAX tile on a
    pipeline-cache MISS (a unique scale forces a fresh pipeline)."""
    _clear_dense_env(monkeypatch)
    monkeypatch.setenv("MFA_V6_DUMP_SOURCE", "1")
    q, k, v = _qkv(2, 8, 4096)
    capfd.readouterr()
    o, term = _auto(q, k, v, scale=0.0871)
    err = capfd.readouterr().err
    assert term[0] == "nax_dense", term
    assert "BQ=32 BK=32 BD=128 WM=2" in err, err[-400:]


@pytest.mark.parametrize("dtype,B,H,N", TABLE_MISSES)
def test_table_band_edges_delegate(monkeypatch, dtype, B, H, N):
    _clear_dense_env(monkeypatch)
    q, k, v = _qkv(B, H, N, dtype=dtype)
    o, term = _auto(q, k, v)
    assert term[0] == "sdpa", term
    assert bool(mx.array_equal(o, _sdpa(q, k, v)))


def test_table_row_is_oracle_correct(monkeypatch):
    """Correctness axis: the table route vs a CPU fp32 oracle, per-row gates."""
    _clear_dense_env(monkeypatch)
    from tests.sparse_gates import row_gate_report
    q, k, v = _qkv(2, 8, 4096)
    o, term = _auto(q, k, v)
    assert term[0] == "nax_dense"
    rows = mx.array([0, 1, 1000, 2047, 4095])
    with mx.stream(mx.cpu):
        qf, kf, vf = (x[:, :, :, :].astype(mx.float32) for x in (q, k, v))
        s = (qf[:, :, rows] @ kf.transpose(0, 1, 3, 2)) / math.sqrt(128)
        ref = mx.softmax(s, axis=-1) @ vf
        mx.eval(ref)
    rep = row_gate_report(o[:, :, rows].astype(mx.float32).reshape(-1, 5, 128),
                          ref.reshape(-1, 5, 128))
    assert rep["finite"] and rep["worst_row_rel"] < 1e-2 and rep["worst_row_norm_dev"] < 5e-3, rep


def test_table_row_honours_custom_scale(monkeypatch):
    _clear_dense_env(monkeypatch)
    q, k, v = _qkv(2, 8, 4096)
    o, term = _auto(q, k, v, scale=0.05)
    assert term[0] == "nax_dense"
    assert bool(mx.array_equal(o, _raw_nax(q, k, v, TILE, scale=0.05)))


# ── knobs: explicit NAX dense (2.63 behaviour) and the kill switch ────────────────
def test_enable_knob_restores_nax_dense_default_tile(monkeypatch):
    _clear_dense_env(monkeypatch)
    monkeypatch.setenv("MFA_ENABLE_V6_DENSE", "1")
    q, k, v = _qkv(1, 2, 6144)
    o, term = _auto(q, k, v)
    assert term[0] == "nax_dense", term
    assert bool(mx.array_equal(o, _raw_nax(q, k, v)))


def test_enable_knob_respects_min_n(monkeypatch):
    _clear_dense_env(monkeypatch)
    monkeypatch.setenv("MFA_ENABLE_V6_DENSE", "1")
    monkeypatch.setenv("MFA_V6_DENSE_MIN_N", "8192")
    q, k, v = _qkv(1, 2, 6144)
    _, term = _auto(q, k, v)
    assert term[0] == "sdpa", term


def test_disable_knob_beats_the_table(monkeypatch):
    _clear_dense_env(monkeypatch)
    monkeypatch.setenv("MFA_DISABLE_V6_DENSE", "1")
    q, k, v = _qkv(2, 8, 4096)
    o, term = _auto(q, k, v)
    assert term[0] == "sdpa", term
    assert bool(mx.array_equal(o, _sdpa(q, k, v)))


# ── A4: the verbose log names the terminal that actually ran ──────────────────────
@pytest.mark.parametrize("case", ["delegate", "table", "d64", "sparse_nax"])
def test_verbose_log_names_the_terminal(monkeypatch, capsys, case):
    _clear_dense_env(monkeypatch)
    from mlx_mfa import dispatch_policy
    shape = {"delegate": (1, 2, 6144, 128), "table": (2, 8, 4096, 128),
             "d64": (1, 2, 4096, 64), "sparse_nax": (1, 12, 4096, 128)}[case]
    q, k, v = _qkv(*shape[:3], D=shape[3])
    dispatch_policy._set_verbose(True)
    try:
        capsys.readouterr()
        if case == "sparse_nax":
            from mlx_mfa import flash_attention_sparse
            nb = shape[2] // 32
            m = (mx.arange(nb)[:, None] - mx.arange(nb)[None, :]) % 8 == 0   # density 1/8
            with dt.capture() as tr:
                o = flash_attention_sparse(q, k, v, m)
                mx.eval(o)
            term = _terminal(tr)
        else:
            _, term = _auto(q, k, v)
        lines = [l for l in capsys.readouterr().out.splitlines() if "[MFA dispatch]" in l]
    finally:
        dispatch_policy._set_verbose(False)
    terminal_lines = [l for l in lines if "terminal=" in l]
    assert terminal_lines, f"no terminal line logged: {lines}"
    assert f"terminal={term[0]} " in terminal_lines[-1], (terminal_lines[-1], term)
    # every other dispatch line is a POLICY line, never a route claim (the H3 lie)
    for l in lines:
        assert "terminal=" in l or "] policy:" in l, f"unlabelled route claim: {l}"


def test_gqa_never_borrows_a_table_row(monkeypatch):
    """Sibling audit (A1): every table row was measured with Hq == Hk.  A GQA call in
    a row's (dtype, B*H, N) key must NOT borrow that evidence — it delegates."""
    _clear_dense_env(monkeypatch)
    mx.random.seed(0)
    q = mx.random.normal((2, 8, 4096, 128)).astype(mx.float16)
    k = mx.random.normal((2, 2, 4096, 128)).astype(mx.float16)
    v = mx.random.normal((2, 2, 4096, 128)).astype(mx.float16)
    o, term = _auto(q, k, v)
    assert term[0] == "sdpa", term
    assert bool(mx.array_equal(o, _sdpa(q, k, v)))


@pytest.mark.parametrize("N", [3000, 2049])
def test_table_row_at_non_aligned_n_is_correct(monkeypatch, N):
    """Review gap: a table hit at an N that is not a multiple of 32 (ragged last Q tile)."""
    _clear_dense_env(monkeypatch)
    from tests.sparse_gates import row_gate_report
    q, k, v = _qkv(2, 8, N)
    o, term = _auto(q, k, v)
    assert term[0] == "nax_dense" and "32.32.2" in term[1], term
    rows = mx.array([0, N // 2, N - 1])
    with mx.stream(mx.cpu):
        qf, kf, vf = (x.astype(mx.float32) for x in (q, k, v))
        ref = mx.softmax((qf[:, :, rows] @ kf.transpose(0, 1, 3, 2)) / math.sqrt(128), axis=-1) @ vf
        mx.eval(ref)
    rep = row_gate_report(o[:, :, rows].astype(mx.float32).reshape(-1, 3, 128), ref.reshape(-1, 3, 128))
    assert rep["finite"] and rep["worst_row_rel"] < 1e-2 and rep["worst_row_norm_dev"] < 5e-3, rep

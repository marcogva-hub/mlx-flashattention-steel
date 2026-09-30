"""2.64 A3 — on-device calibration of dense NAX tile PRIORS (never defaults).

The priors (dispatch_policy.DENSE_TILE_PRIORS: the second 32.32.2 zone N6144-8192,
128.32.8 B*H12, 64.32.4 D64) are measured against SDPA on the user's machine by
`calibrate_dispatch` and promoted into ``dense_nax_tiles`` of the calibrated table only
under the SAME 2x-floor guard as the static table; `MLX_MFA_DISPATCH_TABLE` activates
them.  A hand-edited row outside every prior, or one failing the guard on its own
evidence, is refused loudly (Rule 8).
"""
from __future__ import annotations

import json

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa import dispatch_policy as dp
from mlx_mfa import flash_attention

nax = pytest.mark.skipif(not mlx_mfa.has_nax(), reason="M5+ NAX dense routing")

PRIOR = dp.DenseTileRow(128, "float16", 12, 6144, 8192, (32, 32, 2), 0.0, (), "test prior")


def _flat(v, n=10):
    return [v] * n


def _noisy(v, spread, n=10):
    return [v * (1 + spread * ((i % 5) - 2) / 2) for i in range(n)]


# ── the pure rule ─────────────────────────────────────────────────────────────────
def test_perm_floor_needs_ten_samples():
    with pytest.raises(ValueError):
        dp.perm_floor([1.0] * 9)
    assert dp.perm_floor([1.0] * 10) == 0.0


def test_all_points_pass_gives_the_full_band():
    pts = {6144: (_noisy(10.0, 0.002), _flat(9.0)), 7168: (_noisy(12.0, 0.002), _flat(11.0)),
           8192: (_noisy(14.0, 0.002), _flat(13.0))}
    row = dp.decide_prior_row(PRIOR, pts)
    assert row is not None and (row.n_lo, row.n_hi) == (6144, 8192)
    assert dp.dense_tile_row_passes_guard(row)


def test_a_losing_interior_point_splits_the_band():
    pts = {6144: (_noisy(10.0, 0.002), _flat(9.0)), 7168: (_noisy(12.0, 0.002), _flat(13.0)),
           8192: (_noisy(14.0, 0.002), _flat(13.0))}
    row = dp.decide_prior_row(PRIOR, pts)
    assert (row.n_lo, row.n_hi) == (6144, 6144)          # tie -> lower N
    assert all(m >= 2 * row.floor for n, m in row.evidence if row.n_lo <= n <= row.n_hi)


def test_margin_under_twice_the_floor_is_not_promoted():
    pts = {6144: (_noisy(10.0, 0.05), _flat(9.95)), 8192: (_noisy(10.0, 0.05), _flat(9.95))}
    assert dp.decide_prior_row(PRIOR, pts) is None


# ── the calibrated table: validation + activation ────────────────────────────────
def _write_table(tmp_path, rows):
    p = tmp_path / "dispatch_table.json"
    p.write_text(json.dumps({"dense_nax_tiles": rows}))
    return str(p)


def _row(**kw):
    base = {"D": 128, "dtype": "float16", "bh": 12, "n_lo": 6144, "n_hi": 8192,
            "tile": [32, 32, 2], "floor": 0.004,
            "evidence": [[6144, 0.03], [7168, 0.02], [8192, 0.025]], "source": "test"}
    base.update(kw)
    return base


def test_row_outside_every_prior_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", _write_table(tmp_path, [_row(bh=20)]))
    with pytest.raises(ValueError, match="outside every"):
        dp._load_calibrated_tiles()


def test_row_failing_its_own_guard_is_refused(tmp_path, monkeypatch):
    bad = _row(evidence=[[6144, 0.03], [7168, 0.001], [8192, 0.025]])
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", _write_table(tmp_path, [bad]))
    with pytest.raises(ValueError, match="guard"):
        dp._load_calibrated_tiles()


@nax
def test_calibrated_row_routes_only_when_activated(tmp_path, monkeypatch):
    for k in ("MFA_ENABLE_V6_DENSE", "MFA_DISABLE_V6_DENSE", "MFA_V6_DENSE_MIN_N"):
        monkeypatch.delenv(k, raising=False)
    mx.random.seed(0)
    q, k, v = (mx.random.normal((1, 12, 7168, 128)).astype(mx.float16) for _ in range(3))
    monkeypatch.delenv("MLX_MFA_DISPATCH_TABLE", raising=False)
    with dt.capture() as tr:
        mx.eval(flash_attention(q, k, v))
    assert tr[-1][0] == "sdpa", tr[-1]                    # a prior is never a default
    monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", _write_table(tmp_path, [_row()]))
    with dt.capture() as tr:
        mx.eval(flash_attention(q, k, v))
    assert tr[-1][0] == "nax_dense" and "table" in tr[-1][1], tr[-1]


@nax
def test_calibration_smoke_emits_only_guarded_rows(tmp_path, monkeypatch):
    """Real on-device measurement on one tiny prior: whatever the verdict, an emitted
    row round-trips through the validating loader."""
    tiny = dp.DenseTileRow(128, "float16", 4, 2048, 2048, (32, 32, 2), 0.0, (), "smoke")
    monkeypatch.setattr(dp, "DENSE_TILE_PRIORS", (tiny,))
    rows = dp.calibrate_dense_tile_priors((tiny,), target_ms=5.0, verbose=False)
    assert isinstance(rows, list) and len(rows) <= 1
    if rows:
        monkeypatch.setenv("MLX_MFA_DISPATCH_TABLE", _write_table(tmp_path, rows))
        assert len(dp._load_calibrated_tiles()) == 1

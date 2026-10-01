"""2.64 C1 — doc-code lock: the README / dispatch-map statements of WHEN a Metal kernel
engages on M5 are rebuilt from the code constants.  Changing the policy in code
without the docs (or the docs without the code) fails here.
"""
from __future__ import annotations

from pathlib import Path

from mlx_mfa import attention as att
from mlx_mfa import dispatch_policy as dp
from mlx_mfa import lcsa_nax as ln

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DMAP = (ROOT / "docs" / "reference" / "dispatch-map.md").read_text(encoding="utf-8")
SHORT = {"float16": "fp16", "bfloat16": "bf16"}


def _band(r):
    return f"{r.n_lo}" if r.n_lo == r.n_hi else f"{r.n_lo}–{r.n_hi}"


def test_readme_dense_table_matches_code():
    assert "32·32·2" in README
    for r in dp.DENSE_TILE_TABLE:
        assert r.tile == (32, 32, 2), "README documents 32.32.2 rows only — update both"
        row = f"| {SHORT[r.dtype]} | {r.bh} | {_band(r)} |"
        assert row in README, f"README is missing the table row {row!r} ({r.row_id})"
    rows = [l for l in README.splitlines()
            if l.startswith("| fp16 |") or l.startswith("| bf16 |")]
    assert len(rows) == len(dp.DENSE_TILE_TABLE), rows


def test_readme_sparse_law_matches_code():
    expect = [
        f"[{ln.SPARSE_NAX_MIN_N:,}, {ln.SPARSE_NAX_MAX_N:,}]",
        ", ".join(str(b) for b in sorted(ln.SPARSE_NAX_MEASURED_BH_COVERAGE)),
        f"≤ {ln.SPARSE_NAX_DENSITY_CEILING:.2f}",
        f"≤ {ln.SPARSE_NAX_D128_BH4_DENSITY_CEILING:.2f} for D128",
        f"≤ {ln.SPARSE_NAX_D64_BH12_DENSITY_CEILING:.2f} for D64",
        f"N < {ln.SPARSE_NAX_LOWER_CEILING_BELOW_N:,}",
        f"density ≥ {ln.SPARSE_NAX_D_DENSE_CUTOFF:.2f}",
        f"{ln.SPARSE_NAX_MIN_MASK_BYTES:,} bytes",
        "MFA_SPARSE_FALLBACK_MAX_BYTES",
        f"({att._SPARSE_FALLBACK_MAX_BYTES_DEFAULT // 2**30} GiB)",
        "MFA_SPARSE_NAX_LEGACY_POLICY=1",
        "MFA_ENABLE_V6_DENSE=1",
        "MFA_DISABLE_V6_DENSE=1",
    ]
    missing = [e for e in expect if e not in README]
    assert not missing, missing


def test_dispatch_map_matches_code():
    expect = [
        f"{ln.SPARSE_NAX_MIN_N}..{ln.SPARSE_NAX_MAX_N}",
        ", ".join(str(b) for b in sorted(ln.SPARSE_NAX_MEASURED_BH_COVERAGE)),
        f"< {ln.SPARSE_NAX_LOWER_CEILING_BELOW_N}",
        f"({ln.SPARSE_NAX_D_DENSE_CUTOFF:.2f})",
        "`sdpa` (2.64: byte-identical delegation)",
        "MFA_SPARSE_NAX_LEGACY_POLICY=1",
    ]
    missing = [e for e in expect if e not in DMAP]
    assert not missing, missing

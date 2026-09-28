"""Locks for the release metal_kernel x MLX-version matrix (2.62.3).

`scripts/metal_kernel_matrix_smoke.py` is a release gate: it must probe EVERY shipped
`metal_kernel` call site on EVERY MLX version of the nanobind ABI table.  These static
locks keep it honest as the code moves (no GPU, no MLX version needed):

* every shipped source file that calls `metal_kernel(` is mapped in `SITES`, with its call
  count pinned here — a new call site fails until someone decides which probe covers it;
* the versions the tool iterates are exactly the ABI table's;
* no C++ kernel site catches exceptions (a kernel that fails to build must RAISE — that is
  what lets "the probe ran" mean "the kernel compiled");
* the receipt validator rejects incomplete / failing / tampered / dirty receipts.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load("metal_kernel_matrix_smoke")
checker = _load("check_metal_kernel_matrix")

# Pinned `metal_kernel(` call counts per shipped file.  Update together with SITES in
# scripts/metal_kernel_matrix_smoke.py (and add a probe) when a site is added/removed.
_SITE_COUNTS = {
    "csrc/mfa_sparse_attention.cpp": 3,   # fwd (V6NAX|scalar source), V6NAX LSE, scalar LSE
    "csrc/mfa_gna_nax.cpp": 1,
    "csrc/mfa_ffn_nax.cpp": 1,
    "csrc/mfa_qmm_nax.cpp": 1,
    "csrc/mfa_conv_nax.cpp": 4,           # pointwise, MPP, im2col, mm
    "mlx_mfa/conv_nax.py": 3,             # legacy im2col, mm, pointwise
    "mlx_mfa/attention.py": 1,            # top-k bisection threshold
    "mlx_mfa/tq_decode.py": 2,            # K dequant, V gather
}


def _sdist_excludes() -> set[str]:
    # Regex, not tomllib: requires-python >= 3.10 and tomllib is 3.11+.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = text[text.index("[tool.scikit-build.sdist]"):]
    block = re.search(r"^exclude\s*=\s*\[(.*?)\]", section, re.M | re.S).group(1)
    return set(re.findall(r'"([^"]+)"', block))


def _shipped_metal_kernel_files() -> dict[str, int]:
    excluded = _sdist_excludes()
    out = {}
    for pattern in ("csrc/**/*.cpp", "csrc/**/*.mm", "mlx_mfa/**/*.py"):
        for f in ROOT.glob(pattern):
            rel = f.relative_to(ROOT).as_posix()
            if rel in excluded:
                continue
            n = f.read_text(encoding="utf-8", errors="replace").count("metal_kernel(")
            if n:
                out[rel] = n
    return out


def test_every_shipped_metal_kernel_site_is_mapped_and_pinned():
    shipped = _shipped_metal_kernel_files()
    assert shipped == _SITE_COUNTS, (
        "metal_kernel call sites changed: map the new site to a probe in "
        "scripts/metal_kernel_matrix_smoke.py::SITES and update _SITE_COUNTS "
        f"(found {shipped})")
    mapped_files = {k.split(":")[0] for k in tool.SITES}
    assert mapped_files == set(_SITE_COUNTS), mapped_files


def test_site_function_keys_exist_in_their_files():
    for key in tool.SITES:
        if ":" in key:
            f, fn = key.split(":", 1)
            assert re.search(rf"\b{re.escape(fn)}\s*\(",
                             (ROOT / f).read_text(encoding="utf-8")), key


def test_every_probe_is_implemented_in_the_probe_body():
    body = (ROOT / "scripts" / "metal_kernel_matrix_smoke.py").read_text(encoding="utf-8")
    for p in tool.PROBES:
        assert f'run("{p}"' in body, p


def test_versions_are_exactly_the_abi_table():
    table = (ROOT / "csrc" / "cmake" / "MlxNanobindAbi.cmake").read_text(encoding="utf-8")
    rows = re.findall(r'"(\d+\.\d+\.\d+)=v', table)
    assert rows and tool.abi_table_versions() == rows
    assert "0.32.1" in rows   # the version that broke 2.62.2 stays covered


@pytest.mark.parametrize("f", [k for k in _SITE_COUNTS if k.startswith("csrc/")])
def test_cpp_kernel_sites_do_not_catch(f):
    src = (ROOT / f).read_text(encoding="utf-8")
    assert not re.search(r"\bcatch\s*\(", src), (
        f"{f}: a catch around a metal_kernel build would let the matrix pass on a "
        "silently-substituted path")


# ───────────────────────────────────────────────────────── receipt validator
def _good_receipt(version="9.9.9"):
    versions = tool.abi_table_versions()
    row = lambda v: {"mlx": v, "build_mlx": v, "has_nax": True, "mlx_mfa": version, "mfa_env": [],
                     "results": {p: {"ok": True, "detail": "rel=1e-4"} for p in tool.PROBES}}
    matrix = {v: row(v) for v in versions}
    return {"release_version": version, "git_dirty": False, "sdist_matches_head": True,
            "matrix": matrix,
            "matrix_sha256": hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest(),
            "all_pass": True}


def _rehash(r):
    r["matrix_sha256"] = hashlib.sha256(json.dumps(r["matrix"], sort_keys=True).encode()).hexdigest()
    return r


def _validate(r, version="9.9.9"):
    return checker.validate_receipt(r, version, tool.abi_table_versions(), tool.PROBES)


def test_validator_accepts_a_complete_passing_receipt():
    assert _validate(_good_receipt()) is None


def test_validator_rejects_wrong_version():
    assert "release_version" in _validate(_good_receipt("1.0.0"))


def test_validator_rejects_missing_abi_version():
    r = _good_receipt(); r["matrix"].pop(tool.abi_table_versions()[-1]); _rehash(r)
    assert "not covered" in _validate(r)


def test_validator_rejects_a_failing_kernel():
    r = _good_receipt()
    v = tool.abi_table_versions()[-1]
    r["matrix"][v]["results"]["sparse_v6nax"] = {"ok": False, "detail": "Unable to build"}
    _rehash(r)
    assert "sparse_v6nax" in _validate(r)


def test_validator_rejects_a_missing_probe():
    r = _good_receipt()
    r["matrix"][tool.abi_table_versions()[0]]["results"].pop("gna_nax"); _rehash(r)
    assert "gna_nax" in _validate(r)


def test_validator_rejects_non_isolated_install():
    r = _good_receipt()
    v = tool.abi_table_versions()[-1]
    r["matrix"][v]["build_mlx"] = "0.31.2" if v != "0.31.2" else "0.32.0"; _rehash(r)
    assert "isolated" in _validate(r)


def test_validator_rejects_tampering_and_dirty_tree():
    r = _good_receipt(); r["matrix"][tool.abi_table_versions()[0]]["has_nax"] = False
    assert "tampered" in _validate(r)                       # edited without rehash
    r2 = copy.deepcopy(_good_receipt()); r2["git_dirty"] = True
    assert "uncommitted" in _validate(r2)


def test_validator_rejects_unbound_sdist_wrong_package_and_leaked_env():
    r = _good_receipt(); r["sdist_matches_head"] = False; r["sdist_mismatch"] = ["csrc/x.cpp: differs"]
    assert "HEAD" in _validate(r)
    v = tool.abi_table_versions()[0]
    r = _good_receipt(); r["matrix"][v]["mlx_mfa"] = "0.0.1"; _rehash(r)
    assert "installed mlx-mfa" in _validate(r)
    r = _good_receipt(); r["matrix"][v]["mfa_env"] = ["MFA_DISABLE_CONV3D_MPP"]; _rehash(r)
    assert "leaked" in _validate(r)


def test_staleness_scope_includes_build_definition():
    assert {"csrc", "mlx_mfa", "CMakeLists.txt", "pyproject.toml"} <= set(checker._SOURCE_DIRS)


def test_probe_env_is_stripped_of_mfa_knobs(monkeypatch):
    monkeypatch.setenv("MFA_DISABLE_CONV3D_MPP", "1")
    monkeypatch.setenv("PYTHONPATH", "/x")
    env = tool._clean_env(FOO="1")
    assert "MFA_DISABLE_CONV3D_MPP" not in env and "PYTHONPATH" not in env and env["FOO"] == "1"


def test_known_failures_name_real_probe_cells():
    body = (ROOT / "scripts" / "metal_kernel_matrix_smoke.py").read_text(encoding="utf-8")
    for key, why in tool.KNOWN_FAILURES.items():
        probe, label = key.split("/", 1)
        assert probe in tool.PROBES, key
        assert f'("{label}"' in body, key           # the cell label exists in the sweep
        assert len(why) > 40, key                   # a real reason, not a placeholder

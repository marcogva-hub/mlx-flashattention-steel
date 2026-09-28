"""BLD-01 / BLD-10 (code review 2026-09) — MLX -> nanobind ABI mapping + install specifier.

BLD-01: ``mlx>=0.31.2`` was uncapped in BOTH ``[build-system].requires`` and
``[project].dependencies``.  Under pip's default build isolation the build env
resolves the LATEST MLX (0.32.2 at review time), which the CMake table did not map
-> configure FATAL for every documented ``pip install mlx-mfa``.

BLD-10: the table matched with ``VERSION_EQUAL``, so MLX dev/rc/local builds
(``0.32.0.dev2026…``, ``0.31.2rc1``) silently mapped onto a release nanobind tag —
the "never guess a tag" rule was bypassed.

Invariants locked here (offline; no network, no build):
  * the mapping lives in ``csrc/cmake/MlxNanobindAbi.cmake`` (csrc/ so the sdist guard ships it) and resolves each VERIFIED
    MLX release to its nanobind tag; anything else (unmapped, dev, rc, malformed)
    is a loud FATAL_ERROR;
  * the pyproject specifier is identical in both requirement lists, admits every
    mapped version and NOTHING above the top of the table (so a future MLX release
    can never be picked by build isolation before it is mapped).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:  # stdlib from Python 3.11; requires-python is >= 3.10 (tests/test_py310_compat.py)
    import tomllib
except ModuleNotFoundError:
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "csrc" / "cmake" / "MlxNanobindAbi.cmake"

# VERIFIED 2026-09-27 at source (raw GitHub, per tag): MLX vX CMakeLists.txt
# FetchContent_Declare(nanobind ... GIT_TAG) and nanobind src/nb_abi.h NB_INTERNALS_VERSION.
EXPECTED = {
    "0.31.2": "v2.12.0",  # NB_INTERNALS 19
    "0.32.0": "v2.13.0",  # NB_INTERNALS 20
    "0.32.1": "v2.13.0",  # NB_INTERNALS 20
    "0.32.2": "v2.15.0",  # NB_INTERNALS 21
}

_CMAKE = shutil.which("cmake")
needs_cmake = pytest.mark.skipif(_CMAKE is None, reason="cmake not on PATH (required to build _ext)")


def _resolve(version: str, tmp_path: Path) -> subprocess.CompletedProcess:
    script = tmp_path / "resolve.cmake"
    script.write_text(
        f'include("{MODULE.as_posix()}")\n'
        f'mfa_resolve_nanobind_tag("{version}" _tag)\n'
        'message(STATUS "TAG=${_tag}")\n'
    )
    return subprocess.run([_CMAKE, "-P", str(script)], capture_output=True, text=True, timeout=30)


def _mlx_requirement(reqs: list[str]) -> Requirement:
    found = [Requirement(r) for r in reqs if Requirement(r).name == "mlx"]
    assert len(found) == 1, f"expected exactly one mlx requirement, got {reqs!r}"
    return found[0]


def _toml_string_array(text: str, table: str, key: str) -> list[str]:
    """Python 3.10 fallback: the string array ``key = [...]`` of ``[table]`` (enough for the
    two requirement lists; cross-checked against tomllib where available)."""
    body = re.search(rf"^\[{re.escape(table)}\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S).group(1)
    arr = re.search(rf"^{re.escape(key)}\s*=\s*\[(.*?)\]", body, re.M | re.S).group(1)
    return re.findall(r'"([^"]+)"', "\n".join(l.split("#")[0] for l in arr.splitlines()))


def _requirement_lists(use_tomllib: bool = True):
    text = (ROOT / "pyproject.toml").read_text()
    if use_tomllib and tomllib is not None:
        data = tomllib.loads(text)
        return data["build-system"]["requires"], data["project"]["dependencies"]
    return (_toml_string_array(text, "build-system", "requires"),
            _toml_string_array(text, "project", "dependencies"))


def _pyproject_specifiers():
    build_reqs, runtime_reqs = _requirement_lists()
    build = _mlx_requirement(build_reqs).specifier
    runtime = _mlx_requirement(runtime_reqs).specifier
    return build, runtime


@pytest.mark.skipif(tomllib is None, reason="cross-check needs tomllib (Python >= 3.11)")
def test_py310_fallback_parser_agrees_with_tomllib():
    assert _requirement_lists(use_tomllib=False) == _requirement_lists(use_tomllib=True)


def test_mapping_module_exists_and_is_used_by_cmakelists():
    assert MODULE.is_file(), "csrc/cmake/MlxNanobindAbi.cmake must hold the MLX->nanobind table"
    text = (ROOT / "CMakeLists.txt").read_text()
    assert "MlxNanobindAbi.cmake" in text and "mfa_resolve_nanobind_tag(" in text
    assert "VERSION_EQUAL" not in text, "BLD-10: VERSION_EQUAL maps dev/rc builds onto release tags"


@needs_cmake
@pytest.mark.parametrize("version,tag", sorted(EXPECTED.items()))
def test_mapped_release_resolves_to_verified_tag(version, tag, tmp_path):
    r = _resolve(version, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"TAG={tag}" in r.stdout + r.stderr


@needs_cmake
@pytest.mark.parametrize(
    "version",
    [
        "0.32.0.dev20260601",       # dev build
        "0.31.2.dev20260101+g1a2b",  # dev + local
        "0.31.2rc1",                 # release candidate
        "0.31.2.1",                  # four components
        "0.32.00",                   # malformed patch
        "0.32.3",                    # future release, not yet verified
        "0.31.1",                    # below the ABI floor (nanobind v18)
        "",                          # empty
    ],
)
def test_unmapped_or_non_release_version_fails_loud(version, tmp_path):
    r = _resolve(version, tmp_path)
    assert r.returncode != 0, f"{version!r} must FATAL, got: {r.stdout}{r.stderr}"
    assert "TAG=" not in r.stdout + r.stderr


def test_build_and_runtime_mlx_specifiers_are_identical():
    build, runtime = _pyproject_specifiers()
    assert str(build) == str(runtime), (build, runtime)


def test_specifier_admits_every_mapped_version():
    build, _ = _pyproject_specifiers()
    for v in EXPECTED:
        assert Version(v) in build, f"mapped MLX {v} rejected by {build}"


def test_specifier_is_capped_at_top_of_table():
    """BLD-01 class: nothing newer than the verified table may be resolvable."""
    build, _ = _pyproject_specifiers()
    top = max(Version(v) for v in EXPECTED)
    for candidate in ("0.32.3", "0.33.0", "0.40.0", "1.0.0"):
        v = Version(candidate)
        assert v > top
        assert v not in build, f"{candidate} would be picked by build isolation but is unmapped"
    assert Version("0.31.1") not in build, "0.31.1 ships nanobind v18 (below the ABI floor)"

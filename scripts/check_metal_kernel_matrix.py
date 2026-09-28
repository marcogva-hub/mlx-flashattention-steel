#!/usr/bin/env python3
"""Release precondition: a fresh, complete metal_kernel x MLX-version matrix receipt.

`scripts/metal_kernel_matrix_smoke.py` installs the release sdist in isolation against every
MLX version of the nanobind ABI table (`csrc/cmake/MlxNanobindAbi.cmake`) and runs every
`metal_kernel`-built kernel family (plus the dense V6 consumers of the shared NAX helpers)
against an independent reference.  It writes `release-gate/metal-kernel-matrix-<version>.json`.

This check FAILS (blocks the release audit / publish) if that receipt is absent, not
all-pass, does not cover every ABI-table MLX version or every probe the tool defines, was
tampered with, or is STALE (any `csrc/` or `mlx_mfa/` source changed since it ran).

Why (2.62.3): 2.62.2 opened MLX 0.32.1 / 0.32.2 in the ABI table while its install smoke only
exercised dense kernels; every `metal_kernel` NAX kernel (sparse, GNA, FFN, QMM) failed to
compile on those versions and shipped that way.

Usage:
    python scripts/check_metal_kernel_matrix.py            # version from pyproject
    python scripts/check_metal_kernel_matrix.py --version 2.62.3
    python scripts/check_metal_kernel_matrix.py --receipt path/to.json
Exit 0 = verified; non-zero = BLOCK.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_DIRS = ["csrc", "mlx_mfa"]  # a change in either => the receipt is stale


def _tool():
    """The smoke tool is the single source of the probe list and the ABI-table parser."""
    path = os.path.join(_REPO, "scripts", "metal_kernel_matrix_smoke.py")
    spec = importlib.util.spec_from_file_location("metal_kernel_matrix_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fail(msg: str) -> int:
    print(f"❌ metal_kernel matrix precondition FAILED — release BLOCKED:\n   {msg}",
          file=sys.stderr)
    return 1


def _pyproject_version() -> str:
    txt = open(os.path.join(_REPO, "pyproject.toml"), encoding="utf-8").read()
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("could not parse version from pyproject.toml")
    return m.group(1)


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _REPO, *args], capture_output=True, text=True)


def validate_receipt(r: dict, version: str, abi_versions: list[str], probes) -> str | None:
    """Content checks (no git).  Returns an error message, or None when valid."""
    if r.get("release_version") != version:
        return f"receipt release_version {r.get('release_version')!r} != {version!r}."
    if r.get("git_dirty"):
        return "receipt was produced from a working tree with uncommitted csrc/mlx_mfa changes."
    matrix = r.get("matrix") or {}
    expect = hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest()
    if r.get("matrix_sha256") != expect:
        return "receipt matrix_sha256 does not match its matrix (tampered or malformed)."
    missing_v = [v for v in abi_versions if v not in matrix]
    if missing_v:
        return f"MLX versions of the ABI table not covered by the matrix: {missing_v}."
    for ver in abi_versions:
        row = matrix[ver]
        if "error" in row:
            return f"MLX {ver}: {row['error'][:200]}"
        if row.get("build_mlx") != ver or row.get("mlx") != ver:
            return (f"MLX {ver}: built against {row.get('build_mlx')!r}, ran on "
                    f"{row.get('mlx')!r} — not an isolated {ver} install.")
        if not row.get("has_nax"):
            return f"MLX {ver}: NAX not live — the matrix must run on an M5+ host."
        results = row.get("results") or {}
        missing_p = [p for p in probes if p not in results]
        if missing_p:
            return f"MLX {ver}: probes missing from the receipt: {missing_p}."
        bad = [p for p in probes if not results[p].get("ok")]
        if bad:
            return f"MLX {ver}: kernels FAILED: " + "; ".join(
                f"{p}: {results[p].get('detail', '')[:120]}" for p in bad)
    if not r.get("all_pass"):
        return "receipt all_pass is false."
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--receipt", default=None)
    args = ap.parse_args()

    tool = _tool()
    version = args.version or _pyproject_version()
    path = args.receipt or os.path.join(_REPO, "release-gate",
                                        f"metal-kernel-matrix-{version}.json")
    if not os.path.exists(path):
        return _fail(
            f"no metal_kernel matrix receipt for v{version} at {os.path.relpath(path, _REPO)}.\n"
            f"   Build the sdist, run `python scripts/metal_kernel_matrix_smoke.py --sdist "
            f"dist/mlx_mfa-{version}.tar.gz` on an M5+ host and commit the receipt.")
    try:
        r = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return _fail(f"receipt {path} is not valid JSON: {e}")

    err = validate_receipt(r, version, tool.abi_table_versions(), tool.PROBES)
    if err:
        return _fail(err)

    sha = r.get("git_sha")
    if not sha or sha == "UNKNOWN":
        return _fail("receipt has no git_sha — cannot bind it to the released source.")
    if not os.path.exists(os.path.join(_REPO, ".git")):
        return _fail("not a git checkout — cannot verify receipt freshness.")
    if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return _fail(f"receipt git_sha {sha[:12]} is not a commit in this repo.")
    if _git("merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        return _fail(f"receipt git_sha {sha[:12]} is not an ancestor of HEAD.")
    diff = _git("diff", "--name-only", sha, "HEAD", "--", *_SOURCE_DIRS).stdout.strip()
    if diff:
        changed = diff.splitlines()
        return _fail(
            f"STALE matrix: {len(changed)} source file(s) under {_SOURCE_DIRS} changed since "
            f"it ran at {sha[:12]} (e.g. {changed[0]}). Re-run the matrix on the current "
            f"source and re-commit the receipt.")

    vers = ", ".join(tool.abi_table_versions())
    print(f"✓ metal_kernel matrix verified for v{version}: {len(tool.PROBES)} kernels x MLX "
          f"{{{vers}}} all pass (isolated installs, M5 {r.get('device', '?')}), sha "
          f"{sha[:12]} == released source, dated {r.get('date_utc')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

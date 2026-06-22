#!/usr/bin/env python3
"""Publish precondition: verify a fresh, authentic M5/NAX gate receipt (volet D).

The M5/NAX release gate (`scripts/release_m5_nax_gate.py`) can only run on a real
M5+ host (GitHub-hosted runners are M1 → NAX never engages; a self-hosted M5
runner on a PUBLIC repo is a security risk).  So the gate is run by the
maintainer on M5, which writes a TRACKED receipt to
`release-gate/m5-gate-<version>.json`.  `publish.yml` runs THIS script before
upload; it FAILS the publish if the receipt is absent, not PASS / NAX-not-live,
or **stale** (any `csrc/` or `mlx_mfa/` source changed since the gate ran).

This makes the manual M5 gate a HARD precondition for publishing — you cannot
ship without a gate run that matches the exact source being released.

FLAG-FOR-SIGNOFF: the durable alternative is a self-hosted M5 CI runner that runs
the M5 locks + gate directly (also closes CC-14 / CC-23). Cryptographic signing
of the receipt (vs the git-SHA freshness binding used here) is a further harden.

Usage:
    python scripts/check_m5_gate_fingerprint.py            # version from pyproject
    python scripts/check_m5_gate_fingerprint.py --version 2.61.0
    python scripts/check_m5_gate_fingerprint.py --receipt path/to.json
Exit 0 = gate verified; non-zero = BLOCK the publish.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_DIRS = ["csrc", "mlx_mfa"]  # a change in either ⇒ the gate is stale


def _fail(msg: str) -> int:
    print(f"❌ M5/NAX gate precondition FAILED — publish BLOCKED:\n   {msg}",
          file=sys.stderr)
    return 1


def _pyproject_version() -> str:
    txt = open(os.path.join(_REPO, "pyproject.toml")).read()
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("could not parse version from pyproject.toml")
    return m.group(1)


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _REPO, *args],
                          capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=None)
    ap.add_argument("--receipt", default=None)
    args = ap.parse_args()

    version = args.version or _pyproject_version()
    receipt_path = args.receipt or os.path.join(
        _REPO, "release-gate", f"m5-gate-{version}.json")

    # 1) present
    if not os.path.exists(receipt_path):
        return _fail(
            f"no M5 gate receipt for v{version} at {os.path.relpath(receipt_path)}.\n"
            f"   Run `python scripts/release_m5_nax_gate.py` on an M5+ host and commit "
            f"release-gate/m5-gate-{version}.json before publishing.")
    try:
        r = json.load(open(receipt_path))
    except Exception as e:
        return _fail(f"receipt {receipt_path} is not valid JSON: {e}")

    # 2) gate verdict + NAX live + M5 hardware
    if r.get("gate") != "PASS":
        return _fail(f"receipt gate verdict is {r.get('gate')!r}, not PASS.")
    if not r.get("has_nax"):
        return _fail("receipt has_nax is false — NAX was not live when the gate ran.")
    if not r.get("is_m5_plus"):
        return _fail("receipt is_m5_plus is false — the gate did not run on an M5+ host.")
    if r.get("release_version") != version:
        return _fail(
            f"receipt release_version {r.get('release_version')!r} != pyproject {version!r}.")

    # 3) fingerprint integrity (the receipt's hash must match its own fingerprints)
    fps = r.get("fingerprints", {})
    expect = hashlib.sha256(json.dumps(fps, sort_keys=True).encode()).hexdigest()
    if r.get("fingerprints_sha256") != expect:
        return _fail("receipt fingerprints_sha256 does not match its fingerprints "
                     "(tampered or malformed).")

    # 4) authenticity + freshness: the gate's git_sha must be a real ancestor of
    #    HEAD with NO source change since (else the published source != gated source).
    sha = r.get("git_sha")
    if not sha or sha == "UNKNOWN":
        return _fail("receipt has no git_sha — cannot bind it to the released source.")
    if not (_REPO and os.path.exists(os.path.join(_REPO, ".git"))):
        # Source archive / non-git context: cannot verify freshness → block (the
        # precondition must run in the git checkout that publish.yml checks out).
        return _fail("not a git checkout — cannot verify gate freshness; run the "
                     "precondition in the release checkout.")
    if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        return _fail(f"receipt git_sha {sha[:12]} is not a commit in this repo.")
    head = _git("rev-parse", "HEAD").stdout.strip()
    if _git("merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        return _fail(
            f"receipt git_sha {sha[:12]} is not an ancestor of HEAD {head[:12]} — the "
            f"gate ran on a different/diverged source than the one being published.")
    diff = _git("diff", "--name-only", sha, "HEAD", "--", *_SOURCE_DIRS).stdout.strip()
    if diff:
        changed = diff.splitlines()
        return _fail(
            f"STALE gate: {len(changed)} source file(s) under {_SOURCE_DIRS} changed since "
            f"the gate ran at {sha[:12]} (e.g. {changed[0]}). Re-run the M5 gate on the "
            f"current source and re-commit the receipt.")

    print(f"✓ M5/NAX gate verified for v{version}: PASS, NAX live ({r.get('nax_reason')}), "
          f"M5+ ({r.get('chip','?')}), gate sha {sha[:12]} == released source "
          f"(no csrc/mlx_mfa drift), dated {r.get('date_utc')}, MLX {r.get('mlx_version')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

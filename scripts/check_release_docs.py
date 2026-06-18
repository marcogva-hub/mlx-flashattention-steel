#!/usr/bin/env python3
"""Release-doc sanity check (v2.58.1 P4) — the doc-side analogue of the built-sdist
publish-surface guard.

Catches the doc-drift class the two cold reviews found in 2.58.0:
  (a) the current version string is present in the core PyPI-facing docs;
  (b) no stale forward-pointer "next: vX.Y" at/below the current version;
  (c) no CLICKABLE `.doc-archive/...` markdown link in any PyPI-facing doc — `.doc-archive/`
      is gitignored (not shipped), so such links 404 on GitHub/PyPI. (Non-clickable
      code-span mentions are intentional "internal archive" provenance pointers — allowed.)
  (d) no broken *local relative* markdown link in README.md.

Run: `python scripts/check_release_docs.py` (exit 0 = clean, 1 = findings).
Also imported by tests/test_release_docs.py so it runs in the suite.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPI_FACING = ["README.md", "CHANGELOG.md", "RESULTS.md", "ENV_VARS.md"]
CORE_VERSION_DOCS = ["README.md", "docs/reference/API_MANUAL.md"]

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_CLICKABLE_ARCHIVE = re.compile(r"\]\(\.?/?\.doc-archive/[^)]*\)")
_NEXT_VER = re.compile(r"next:?\s*v?(\d+)\.(\d+)", re.IGNORECASE)


def _version() -> str:
    txt = (ROOT / "mlx_mfa" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', txt)
    if not m:
        raise RuntimeError("could not read __version__ from mlx_mfa/__init__.py")
    return m.group(1)


def check() -> list[str]:
    problems: list[str] = []
    ver = _version()
    vmaj, vmin = (int(x) for x in ver.split(".")[:2])

    # (a) current version present in the core docs
    for rel in CORE_VERSION_DOCS:
        p = ROOT / rel
        if p.exists() and ver not in p.read_text(encoding="utf-8"):
            problems.append(f"(a) current version {ver} not found in {rel}")

    for rel in PYPI_FACING:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # (b) stale forward-pointer "next: vX.Y" at/below current
        for mj, mn in _NEXT_VER.findall(text):
            if (int(mj), int(mn)) <= (vmaj, vmin):
                problems.append(f"(b) {rel}: stale 'next: v{mj}.{mn}' <= current v{vmaj}.{vmin}")
        # (c) clickable .doc-archive link (404 on the published surface)
        for m in _CLICKABLE_ARCHIVE.findall(text):
            problems.append(f"(c) {rel}: clickable .doc-archive link 404s on PyPI/GitHub: {m}")

    # (d) broken local relative links in README
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in _MD_LINK.findall(readme):
        t = target.split("#", 1)[0].strip()
        if not t or t.startswith(("http://", "https://", "mailto:")):
            continue
        if t.startswith(".doc-archive/"):
            continue  # handled by (c); intentional archive pointers otherwise
        if not (ROOT / t).exists():
            problems.append(f"(d) README.md: broken local link → {t}")
    return problems


def main() -> int:
    problems = check()
    if problems:
        print(f"release-doc check: {len(problems)} finding(s):")
        for x in problems:
            print(f"  - {x}")
        return 1
    print(f"release-doc check: clean (version {_version()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

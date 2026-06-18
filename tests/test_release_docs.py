"""Suite hook for the release-doc sanity check (v2.58.1 P4).

Runs scripts/check_release_docs.py inside the suite so doc drift (stale version,
broken/.doc-archive links, stale 'next: vX.Y') fails CI — the doc-side analogue of
the built-sdist publish-surface guard.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "check_release_docs.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_release_docs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_release_docs_clean():
    problems = _load().check()
    assert not problems, "release-doc check found drift:\n" + "\n".join(f"  - {p}" for p in problems)


def test_release_doc_checker_catches_synthetic(tmp_path, monkeypatch):
    """Self-test: the checker flags a clickable .doc-archive link + a stale 'next:' marker."""
    mod = _load()
    bad = tmp_path / "README.md"
    bad.write_text("see [x](.doc-archive/docs/foo.md)\n| next: v0.1 | TBD |\n", encoding="utf-8")
    # point the checker's PyPI-facing list at the synthetic file
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "PYPI_FACING", ["README.md"])
    monkeypatch.setattr(mod, "CORE_VERSION_DOCS", [])
    monkeypatch.setattr(mod, "_version", lambda: "2.58.1")
    problems = mod.check()
    assert any("clickable .doc-archive" in p for p in problems), problems
    assert any("stale 'next:" in p for p in problems), problems

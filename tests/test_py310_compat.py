"""Python 3.10 compatibility of everything that ships (pyproject: requires-python >= 3.10).

2.63.0: `tests/test_mlx_nanobind_abi_mapping.py` imported `tomllib` (3.11+) at module
level, so the whole module was a collection ERROR on Python 3.10.  Static locks (they run
on any interpreter):

* every shipped .py (mlx_mfa/, scripts/, tests/, examples/) parses as Python 3.10 syntax;
* no shipped module imports a 3.11+-only stdlib module at module level unguarded (an
  import inside ``try: ... except ModuleNotFoundError/ImportError`` is fine).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_PY311_ONLY_STDLIB = {"tomllib", "wsgiref.types"}
_TREES = ("mlx_mfa", "scripts", "tests", "examples")


def _shipped_py():
    for tree in _TREES:
        for f in sorted((ROOT / tree).rglob("*.py")):
            if "__pycache__" not in f.parts:
                yield f


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    names = {"ImportError", "ModuleNotFoundError"}
    tp = handler.type
    if isinstance(tp, ast.Name):
        return tp.id in names
    if isinstance(tp, ast.Tuple):
        return any(isinstance(e, ast.Name) and e.id in names for e in tp.elts)
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _unguarded_311_imports(tree: ast.Module):
    """Module-level imports of 3.11-only stdlib modules that would run on 3.10.  Guarded =
    inside the ``try`` BODY of a try whose handlers catch ImportError/ModuleNotFoundError
    (not its except/else/finally); ``if TYPE_CHECKING:`` blocks never run; function bodies
    only run when called."""
    bad = []

    def visit(stmts, guarded):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(st, ast.ClassDef):
                    visit(st.body, guarded)          # class bodies run at import
                continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in st.names] if isinstance(st, ast.Import) else [st.module or ""]
                bad.extend((st.lineno, n) for n in names if n in _PY311_ONLY_STDLIB and not guarded)
            elif isinstance(st, ast.Try):
                catches = any(_catches_import_error(h) for h in st.handlers)
                visit(st.body, guarded or catches)
                for h in st.handlers:
                    visit(h.body, guarded)
                visit(st.orelse, guarded)
                visit(st.finalbody, guarded)
            elif isinstance(st, ast.If):
                if not _is_type_checking(st.test):
                    visit(st.body, guarded)
                visit(st.orelse, guarded)
            elif isinstance(st, (ast.With, ast.For, ast.While)):
                visit(st.body, guarded)
                visit(getattr(st, "orelse", []), guarded)

    visit(tree.body, False)
    return bad


def test_lock_semantics_on_synthetic_sources():
    f = lambda src: _unguarded_311_imports(ast.parse(src))
    assert f("import tomllib\n") == [(1, "tomllib")]
    assert f("from tomllib import loads\n") == [(1, "tomllib")]
    assert f("try:\n    import tomllib\nexcept ModuleNotFoundError:\n    tomllib = None\n") == []
    assert f("try:\n    import x\nexcept ImportError:\n    import tomllib\n") == [(4, "tomllib")]
    assert f("try:\n    import tomllib\nexcept ValueError:\n    pass\n") == [(2, "tomllib")]
    assert f("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import tomllib\n") == []
    assert f("def g():\n    import tomllib\n") == []


@pytest.mark.parametrize("f", list(_shipped_py()), ids=lambda f: str(f.relative_to(ROOT)))
def test_parses_as_python_310_and_no_unguarded_311_imports(f):
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(f), feature_version=(3, 10))
    assert not _unguarded_311_imports(tree), (
        f"{f.relative_to(ROOT)} imports a 3.11+-only module unguarded at module level "
        "(requires-python >= 3.10): guard it with try/except ModuleNotFoundError + a fallback")

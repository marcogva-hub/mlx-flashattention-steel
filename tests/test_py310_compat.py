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


def _guarded(node, parents) -> bool:
    for p in parents:
        if isinstance(p, ast.Try) and any(
                isinstance(h.type, ast.Name) and h.type.id in ("ImportError", "ModuleNotFoundError")
                or isinstance(h.type, ast.Tuple) and any(
                    isinstance(e, ast.Name) and e.id in ("ImportError", "ModuleNotFoundError")
                    for e in h.type.elts)
                for h in p.handlers):
            return True
    return False


def _unguarded_311_imports(tree: ast.Module):
    bad = []

    def visit(node, parents):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue                     # function-local imports only run when called
            if isinstance(child, ast.Import):
                names = [a.name for a in child.names]
            elif isinstance(child, ast.ImportFrom):
                names = [child.module or ""]
            else:
                names = []
            for n in names:
                if n in _PY311_ONLY_STDLIB and not _guarded(child, parents + [node]):
                    bad.append((child.lineno, n))
            visit(child, parents + [node])

    visit(tree, [])
    return bad


@pytest.mark.parametrize("f", list(_shipped_py()), ids=lambda f: str(f.relative_to(ROOT)))
def test_parses_as_python_310_and_no_unguarded_311_imports(f):
    src = f.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(f), feature_version=(3, 10))
    assert not _unguarded_311_imports(tree), (
        f"{f.relative_to(ROOT)} imports a 3.11+-only module unguarded at module level "
        "(requires-python >= 3.10): guard it with try/except ModuleNotFoundError + a fallback")

"""Doc-example execution harness — examples were the WORST staleness failure
(2.61.0 audit found two SERVING_GUIDE examples that *failed as written*).

This extracts ```python fenced blocks from the example-bearing docs and executes
each in a fresh namespace.  Classification (chosen to catch the real breakage
WITHOUT flaking on illustrative fragments):

  - **FAIL** — the block raises a STRUCTURAL API error: TypeError / AttributeError
    / ValueError / SyntaxError / ImportError-of-mlx_mfa.  This is exactly the
    SERVING-class breakage (wrong signature, missing arg, renamed API) — a user
    copying it would hit the same error.  CI fails.
  - **skip (fragment)** — NameError / IndentationError at top level: the block is
    an illustrative fragment that references symbols defined in prose, not a
    standalone runnable example.  Counted, not failed.
  - **skip (heavy)** — references a model load / external dep (from_pretrained,
    safetensors, .load(, mlx_lm, patch_seedvr2/flashvsr, download): would make CI
    slow/flaky.  Counted, not failed.

A block must be opted IN to execution by being self-contained: it must contain an
`import` of mlx or mlx_mfa.  Blocks with no import are treated as fragments.
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _isolated_env():
    """Snapshot/restore os.environ so a doc example that sets an MFA_* env var
    (legitimate usage shown in docs, e.g. TRAINING_QUICKSTART) cannot LEAK into
    the rest of the suite and flip routing in later tests."""
    snap = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snap)

_ROOT = Path(__file__).parent.parent

# Example-bearing, current-state docs (NOT historical/changelog/migration).
_DOCS = [
    "README.md",
    "docs/reference/SERVING_GUIDE.md",
    "docs/reference/TRAINING_QUICKSTART.md",
    "docs/reference/API_MANUAL.md",
]

_HEAVY = re.compile(
    r"from_pretrained|safetensors|\.load\(|load_model|mlx_lm|patch_seedvr2|"
    r"patch_flashvsr|patch_mlx_lm|\.npz|\.gguf|download|hf_hub|snapshot_download",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _blocks(rel: str):
    txt = (_ROOT / rel).read_text()
    return _FENCE.findall(txt)


def _classify_and_run(code: str):
    """Return ('pass'|'skip-fragment'|'skip-heavy', detail). Raises on structural fail."""
    if "import mlx" not in code and "from mlx_mfa" not in code and "import mlx_mfa" not in code:
        return ("skip-fragment", "no import — illustrative fragment")
    if _HEAVY.search(code):
        return ("skip-heavy", "model-load / external dep")
    ns: dict = {}
    try:
        with _isolated_env():
            exec(compile(code, "<doc-example>", "exec"), ns)
        return ("pass", "executed")
    except (NameError, IndentationError) as e:
        # references prose-defined symbols → fragment, not a runnable example
        return ("skip-fragment", f"{type(e).__name__}: {e}")
    except ModuleNotFoundError as e:
        if "mlx_mfa" in str(e) or "mlx" == str(e).split("'")[1].split(".")[0]:
            raise  # our own package must import
        return ("skip-heavy", f"optional dep missing: {e}")


# Build the (doc, index, code) param list at collection time.
_PARAMS = []
for _rel in _DOCS:
    for _i, _code in enumerate(_blocks(_rel)):
        _PARAMS.append((_rel, _i, _code))


@pytest.mark.parametrize("rel,idx,code", _PARAMS, ids=[f"{p[0]}#{p[1]}" for p in _PARAMS])
def test_doc_example_executes_or_is_fragment(rel, idx, code):
    """A runnable doc example must execute; a structural API error fails CI."""
    try:
        verdict, detail = _classify_and_run(code)
    except (TypeError, AttributeError, ValueError, SyntaxError, ImportError,
            ModuleNotFoundError) as e:
        pytest.fail(
            f"{rel} example #{idx} raises a STRUCTURAL API error — a user copying "
            f"it hits this: {type(e).__name__}: {e}\n--- block ---\n{code[:400]}")
    if verdict.startswith("skip"):
        pytest.skip(f"{rel}#{idx}: {verdict} ({detail})")


def test_doc_examples_coverage_report(capsys):
    """Emit how many examples ran vs were skipped (visibility, never fails)."""
    from collections import Counter
    c = Counter()
    for rel, idx, code in _PARAMS:
        try:
            v, _ = _classify_and_run(code)
        except Exception:
            v = "FAIL"
        c[v] += 1
    # At least one example must actually execute (else the harness is vacuous).
    assert c["pass"] >= 1, f"no doc example executed — harness vacuous? {dict(c)}"
    print(f"[doc-examples] {dict(c)} across {len(_DOCS)} docs")

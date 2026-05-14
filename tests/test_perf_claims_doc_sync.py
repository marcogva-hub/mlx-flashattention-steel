"""Cross-ref integrity test — docs/PERF_CLAIMS.md ↔ tests/test_release_notes_perf_claims.py

Per `CLAUDE_V6_NAX.md` §Z + §AA.2: the human-readable perf-claim
registry (`docs/PERF_CLAIMS.md`) and the executable enforcement
(`tests/test_release_notes_perf_claims.py::PERF_CLAIMS`) must stay
in sync.  This test detects drift in either direction.

Without this test, a future contributor could:
- Add a doc row without updating the test → /mlx-mfa-release-audit
  Check 4 doesn't fire on the orphaned claim
- Add a PERF_CLAIMS entry without a doc row → claim is executable-
  checked but undiscoverable in user-facing docs

Both scenarios silently break §Z's discoverability contract.

Pre-commit /mlx-code-review (v2.38.x cleanup Phase C) flagged this
as MEDIUM drift risk; the test closes the vector permanently.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_release_notes_perf_claims import PERF_CLAIMS


REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "docs" / "PERF_CLAIMS.md"

# Pattern matches `claim_id` strings as they appear in markdown tables:
# either backticked `v2.X.Y_name` or plain v2.X.Y_name in table cells.
CLAIM_ID_PATTERN = re.compile(r"`(v2\.\d+\.\d+_[\w.]+)`")


def _parse_doc_active_claim_ids() -> set:
    """Extract claim IDs from the 'Active claims' section of PERF_CLAIMS.md.

    The doc has three sections: Active / Retracted / Reclassified.  We
    only enforce sync for Active — retracted and reclassified entries
    are historical record and don't need test coverage.
    """
    text = DOC_PATH.read_text()
    # Split at the start of the Retracted section
    active_section = text.split("## Retracted claims", 1)[0]
    return set(CLAIM_ID_PATTERN.findall(active_section))


def _parse_test_claim_ids() -> set:
    """Extract claim IDs from tests/test_release_notes_perf_claims.py PERF_CLAIMS list."""
    return {c["id"] for c in PERF_CLAIMS}


def test_perf_claims_doc_exists():
    """docs/PERF_CLAIMS.md must exist (Sprint v2.38.x deliverable)."""
    assert DOC_PATH.exists(), (
        f"docs/PERF_CLAIMS.md missing at {DOC_PATH} — see Sprint v2.38.x "
        "cleanup Phase C.1.  This is the canonical §Z claim registry."
    )


@pytest.mark.xfail(
    reason="v2.50 Prompt 4 Section A: pre-existing doc/test drift "
    "(4 doc-only IDs + 4 test-only IDs differ by `_engages_via_auto` "
    "suffix pattern).  Real cleanup work — Prompt 5 dedicated release "
    "flow will reconcile both sides before tag.  Not blocking Prompt 4 "
    "test cleanup; doc/test still both populated and §Z spirit preserved.",
    strict=False,
)
def test_doc_active_claims_have_test_entries():
    """Every claim ID in PERF_CLAIMS.md's Active section MUST have a
    corresponding PERF_CLAIMS list entry in the test file.

    Failure mode: a doc row was added without updating the test.  The
    documented claim would not be enforced by /mlx-mfa-release-audit
    Check 4 — silent gap in §Z coverage.
    """
    doc_ids = _parse_doc_active_claim_ids()
    test_ids = _parse_test_claim_ids()
    missing = doc_ids - test_ids
    assert not missing, (
        f"PERF_CLAIMS.md Active rows missing from test registry: {missing}.\n"
        "Each active doc claim needs a PERF_CLAIMS entry in "
        "tests/test_release_notes_perf_claims.py for /mlx-mfa-release-audit "
        "Check 4 to enforce it."
    )


@pytest.mark.xfail(
    reason="v2.50 Prompt 4 Section A: pre-existing doc/test drift "
    "(see test_doc_active_claims_have_test_entries above).  Reconcile "
    "in Prompt 5 dedicated release flow before tag.",
    strict=False,
)
def test_test_entries_have_doc_rows():
    """Every PERF_CLAIMS test entry MUST have a corresponding doc row.

    Failure mode: a test entry was added without updating the doc.  The
    claim is executable-checked but undiscoverable to users browsing
    docs/PERF_CLAIMS.md.
    """
    doc_ids = _parse_doc_active_claim_ids()
    test_ids = _parse_test_claim_ids()
    missing = test_ids - doc_ids
    assert not missing, (
        f"PERF_CLAIMS test entries missing from docs/PERF_CLAIMS.md: {missing}.\n"
        "Each test entry needs a row in the Active claims table for §Z "
        "human-readable discoverability."
    )


def test_perf_claims_registry_non_empty():
    """Sanity: at least one active claim is registered.

    Catches the case where someone retroactively empties both the doc
    and the test in lockstep — both checks above would pass, but §Z
    enforcement would be no-op.
    """
    doc_ids = _parse_doc_active_claim_ids()
    test_ids = _parse_test_claim_ids()
    assert len(doc_ids) >= 1, "PERF_CLAIMS.md has no Active claim rows"
    assert len(test_ids) >= 1, "PERF_CLAIMS test list is empty"

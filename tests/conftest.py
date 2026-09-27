"""Shared pytest fixtures for the mlx-mfa suite.

Metal buffer-pool fence (Sprint III-6)
--------------------------------------
MLX caches GPU buffers in a process-global pool and recycles them across
kernel dispatches.  Within a single pytest process the suite runs 1500+
kernels back-to-back, so a test can observe buffer-pool state left by an
earlier test — a documented cross-test contamination class (v1.3.0 Phase
3: clear_cache fences fixed downstream stale-data NaN; see MEMORY).  It
surfaces as an order-dependent failure in a numerically-sensitive test
(e.g. sage non-causal, which has no widened tolerance) that passes in
isolation.

`mx.clear_cache()` frees the *unused* cached buffers between tests so each
test starts from a clean pool.  This is purely a test-isolation fence: it
CANNOT mask an intra-dispatch correctness bug (e.g. a kernel that
under-writes its own output and reads stale memory within one call —
the top-K CRITICAL class), because that corruption happens inside a single
dispatch, not across the pool boundary.  It only removes the cross-test
ordering artifact.

Surfaced in III-6 when the conv small-channel regression file shifted the
global collection order; the underlying contamination is pre-existing and
latent.
"""
import pytest
import mlx.core as mx


@pytest.fixture(autouse=True)
def _mlx_pool_fence():
    yield
    # Teardown: free unused cached GPU buffers so the next test starts
    # from a clean pool (Rule 13: mx.clear_cache, not mx.metal.clear_cache).
    mx.clear_cache()


# ── No-extension environments (CI "Fallback tests" job, BLD-06 / A12) ──────────
# Test modules that import `mlx_mfa._ext` unconditionally at MODULE level cannot
# even be collected when the extension is absent (ImportError at collection made
# the whole fallback job red since 2026-06).  Such modules exercise the native
# kernels by construction, so they are not collected there — LOUDLY: the terminal
# summary lists every module left out and why.  Inert whenever _ext imports.
import importlib.util as _ilu
import re as _re

_EXT_IMPORT_RE = _re.compile(
    r"^(from mlx_mfa import [^\n]*\b_ext\b|from mlx_mfa\._ext import|import mlx_mfa\._ext)",
    _re.M)
_EXT_AVAILABLE = _ilu.find_spec("mlx_mfa._ext") is not None
_NOT_COLLECTED_NO_EXT: list = []


def pytest_ignore_collect(collection_path, config):
    if _EXT_AVAILABLE or collection_path.suffix != ".py" \
            or not collection_path.name.startswith("test_"):
        return None
    try:
        src = collection_path.read_text()
    except OSError:
        return None
    if _EXT_IMPORT_RE.search(src):
        _NOT_COLLECTED_NO_EXT.append(collection_path.name)
        return True
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _NOT_COLLECTED_NO_EXT:
        terminalreporter.write_sep(
            "=", f"{len(_NOT_COLLECTED_NO_EXT)} module(s) NOT collected: they import "
                 "mlx_mfa._ext at module level and the extension is absent")
        for name in sorted(_NOT_COLLECTED_NO_EXT):
            terminalreporter.write_line(f"  {name}")

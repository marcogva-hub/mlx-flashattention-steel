"""Phase II-6 — sparse all-False-row contract on the SDPA+bias fallback.

The native STEEL/V34 sparse kernels write ZEROS for a query row whose
block-mask row has no active blocks (Track-B "all-false row" contract).
The SDPA+float-bias fallback produced NaN for such rows (softmax over an
all--inf bias row), and the v2.50 Sprint-1 dispatch migration silently
moved most M5 sparse shapes onto that path — changing public semantics
from zeros to NaN.  `_get_sparse_row_active()` + the bias-sanitize +
where-fixup in `_sparse_fallback_sdpa_perhead` restore the kernel
contract; these tests lock it (including the causal x mask interaction
and per-head masks).

SUBPROCESS ISOLATION: each test runs in a fresh interpreter.  In-process,
these workloads (all-False rows -> -inf bias tensors + zero-row outputs)
left the shared Metal buffer pool in a state that flaked 2 unrelated
finite-value assertions later in the suite (topk-bisect thresholds,
sparse-native engagement) in ~3/5 full-suite runs — even after dropping
the module bias caches and mx.clear_cache().  The contamination chain
(which buffer, which consumer reads it uninitialized) is queued as a
Phase II-7/II-8 investigation item — see
docs/v50/campaign-2026-06/phase2/sprint-II-6-report.md "Open items".
Isolation keeps the contract locked without destabilizing the suite.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_PY = sys.executable
_REPO = Path(__file__).resolve().parent.parent

_COMMON = """
import mlx.core as mx
import numpy as np
from mlx_mfa import flash_attention_sparse, make_causal_block_mask, get_device_info

if not bool(get_device_info().get("is_m5_plus", False)):
    print("SKIP-NO-NAX")
    raise SystemExit(0)

B, H, N, D = 1, 8, 512, 64
BQ = 32  # block rows for D=64 (mask is 16x16 at N=512)
mx.random.seed(7)
q = mx.random.normal((B, H, N, D), dtype=mx.float16)
k = mx.random.normal((B, H, N, D), dtype=mx.float16)
v = mx.random.normal((B, H, N, D), dtype=mx.float16)
mx.eval(q, k, v)
"""


def _run(snippet: str):
    proc = subprocess.run(
        [_PY, "-c", _COMMON + snippet], capture_output=True, text=True,
        cwd=str(_REPO), timeout=300,
    )
    if "SKIP-NO-NAX" in proc.stdout:
        pytest.skip("V34/M5 NAX not available")
    assert proc.returncode == 0, (
        f"subprocess failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


class TestAllFalseRowContract:
    def test_allfalse_block_row_yields_zeros(self):
        _run("""
bm = np.array(make_causal_block_mask(N, D)); bm[2, :] = False  # rows 64..96
o = flash_attention_sparse(q, k, v, mx.array(bm))
mx.eval(o)
on = np.array(o.astype(mx.float32))
assert np.isfinite(on).all(), "NaN leaked from all-False row (contract: zeros)"
assert (on[:, :, 2*BQ:3*BQ, :] == 0).all(), "all-False block-row must be zeros"
assert np.abs(on[:, :, :2*BQ, :]).max() > 0, "active rows must be untouched"
""")

    def test_causal_future_only_row_yields_zeros(self):
        """Row-block whose only active block is causally unreachable."""
        _run("""
bm = np.zeros((16, 16), dtype=bool)
bm[np.arange(16), np.arange(16)] = True
bm[1, 1] = False
bm[1, 10] = True  # strictly future for row-block 1
o = flash_attention_sparse(q, k, v, mx.array(bm), causal=True)
mx.eval(o)
on = np.array(o.astype(mx.float32))
assert np.isfinite(on).all()
assert (on[:, :, BQ:2*BQ, :] == 0).all(), "causally-unreachable-only row-block must be zeros"
""")

    def test_perhead_3d_mask_only_affected_head_zeroed(self):
        _run("""
bm3 = np.broadcast_to(np.array(make_causal_block_mask(N, D)), (H, 16, 16)).copy()
bm3[3, 4, :] = False  # head 3, element rows 128..160
o = flash_attention_sparse(q, k, v, mx.array(bm3))
mx.eval(o)
on = np.array(o.astype(mx.float32))
assert np.isfinite(on).all()
assert (on[:, 3, 4*BQ:5*BQ, :] == 0).all()
others = np.delete(on, 3, axis=1)
assert np.abs(others[:, :, 4*BQ:5*BQ, :]).max() > 0, "must not zero unaffected heads"
""")

    def test_fully_active_mask_unchanged(self):
        """Common case: no all-False rows — fixup must be a no-op."""
        _run("""
bm = np.array(make_causal_block_mask(N, D))
o = flash_attention_sparse(q, k, v, mx.array(bm))
mx.eval(o)
on = np.array(o.astype(mx.float32))
assert np.isfinite(on).all()
assert np.abs(on).max() > 0
""")

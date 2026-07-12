"""Public packed-varlen V6 NAX routing and tile-coherence locks."""

import math
import re

import mlx.core as mx
import pytest

from mlx_mfa import _dispatch_trace as dtrace
from mlx_mfa import _ext, flash_attention_varlen


def _prefix(lengths):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return values


def _tiles(lengths, bq=32):
    values = [0]
    for length in lengths:
        values.append(values[-1] + math.ceil(length / bq))
    return mx.array(values, dtype=mx.int32)


@pytest.mark.parametrize("causal", [False, True])
def test_public_varlen_nax_engages_and_matches_expert(monkeypatch, tmp_path, causal):
    """The opt-in public route must select NAX and use one coherent tile triple."""
    lengths = [1750] * 19 + [1769]  # 35019: unique cache key for this lock
    cu = mx.array(_prefix(lengths), dtype=mx.int32)
    mx.random.seed(20260713 + int(causal))
    q = mx.random.normal((1, 2, sum(lengths), 128)).astype(mx.float16)
    k = mx.random.normal((1, 1, sum(lengths), 128)).astype(mx.float16)
    v = mx.random.normal((1, 1, sum(lengths), 128)).astype(mx.float16)
    scale = 1.0 / math.sqrt(128)
    dump_path = tmp_path / ("varlen_causal.metal" if causal else "varlen_nc.metal")

    monkeypatch.setenv("MFA_ENABLE_VARLEN_NAX", "1")
    monkeypatch.delenv("MFA_V6_NAX_BQ", raising=False)
    monkeypatch.delenv("MFA_V6_NAX_BK", raising=False)
    monkeypatch.delenv("MFA_V6_NAX_WM", raising=False)
    monkeypatch.setenv("MFA_V6_VARLEN_DUMP_PATH", str(dump_path))

    with dtrace.capture() as events:
        public = flash_attention_varlen(
            q, k, v, cu, cu, max(lengths), max(lengths),
            scale=scale, causal=causal,
        )
        mx.eval(public)

    expert, _ = _ext.v6_nax_varlen_forward(
        q, k, v, cu, cu, _tiles(lengths), scale, causal, 32, 32, 2
    )
    mx.eval(expert)

    assert events == [
        ("varlen_v6nax", "opt-in beta-3 packed V6 NAX (BQ32/BK32/WM2 explicit)")
    ]
    assert float(mx.max(mx.abs(public.astype(mx.float32) - expert.astype(mx.float32)))) == 0.0

    source = dump_path.read_text()
    assert re.search(r"^#define V6NAX_BQ 32$", source, re.MULTILINE)
    assert re.search(r"^#define V6NAX_BK 32$", source, re.MULTILINE)
    assert re.search(r"^#define V6NAX_WM 2$", source, re.MULTILINE)


def test_varlen_nax_opt_in_off_preserves_native_path(monkeypatch):
    lengths = [32, 40]
    cu = mx.array(_prefix(lengths), dtype=mx.int32)
    q = mx.zeros((1, 2, sum(lengths), 64), dtype=mx.float16)
    k = mx.zeros((1, 1, sum(lengths), 64), dtype=mx.float16)
    v = mx.zeros((1, 1, sum(lengths), 64), dtype=mx.float16)
    monkeypatch.delenv("MFA_ENABLE_VARLEN_NAX", raising=False)

    with dtrace.capture() as events:
        out = flash_attention_varlen(
            q, k, v, cu, cu, max(lengths), max(lengths),
            scale=1.0 / math.sqrt(64), causal=False,
        )
        mx.eval(out)

    assert events == [("varlen_native", "STEEL varlen _ext (D<=256 f16/bf16)")]

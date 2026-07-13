"""Structural guards for the routed-path benchmark's two-arm gate."""

from __future__ import annotations

import mlx.core as mx
import pytest

from benchmarks.bench_final_routed_spots import _require_fingerprints


def _arrays(dtype=mx.float16):
    public = mx.array([1.0, 2.0], dtype=dtype)
    baseline = mx.array([1.0, 2.001], dtype=dtype)
    oracle = mx.array([1.0, 2.0], dtype=mx.float32)
    mx.eval(public, baseline, oracle)
    return public, baseline, oracle


def test_benchmark_gate_requires_baseline_terminal():
    public, baseline, oracle = _arrays()
    with pytest.raises(RuntimeError, match="baseline terminal"):
        _require_fingerprints(
            label="guard", input_dtype=str(mx.float16),
            public_trace=[("v6nax_sparse", "test")], baseline_trace=[],
            expected_public="v6nax_sparse", public_output=public,
            baseline_output=baseline, oracle_output=oracle,
        )


def test_benchmark_gate_rejects_cross_dtype_baseline():
    public, _, oracle = _arrays()
    baseline = mx.array([1.0, 2.001], dtype=mx.float32)
    mx.eval(baseline)
    with pytest.raises(RuntimeError, match="baseline output dtype"):
        _require_fingerprints(
            label="guard", input_dtype=str(mx.float16),
            public_trace=[("v6nax_sparse", "test")],
            baseline_trace=[("sdpa", "test")],
            expected_public="v6nax_sparse", public_output=public,
            baseline_output=baseline, oracle_output=oracle,
        )


def test_benchmark_gate_accepts_distinct_same_dtype_paths():
    public, baseline, oracle = _arrays()
    result = _require_fingerprints(
        label="guard", input_dtype=str(mx.float16),
        public_trace=[("v6nax_sparse", "test")],
        baseline_trace=[("sdpa", "test")],
        expected_public="v6nax_sparse", public_output=public,
        baseline_output=baseline, oracle_output=oracle,
    )
    assert result["input_dtype"] == str(mx.float16)
    assert result["public"]["terminal"][0] == "v6nax_sparse"
    assert result["baseline"]["terminal"][0] == "sdpa"

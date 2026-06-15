"""III-7 Class C findings — quantize_model Rule-8 guards.

The conv3d sweep (III-7) hunted single-shape-class coverage gaps (lesson
#10) hiding silent-failure bugs (lesson #9 family). Two were found in
`quantize_model`, both invisible because every prior svdquant test used
Linears wrapped in a container with all dims a multiple of group_size:

  Finding 1 — a BARE top-level `nn.Linear` was a SILENT NO-OP. The walker
  (`_replace_layers`) only tests/replaces a module's *children*, and
  cannot setattr-replace the top-level module itself, so it returned
  layers=[] and reported success while quantizing nothing (Rule 8). Now
  raises an actionable error.

  Finding 2 — a layer whose in_features is not divisible by group_size
  passed the dims>=256 default predicate, then raised deep inside
  `mx.quantize` AFTER earlier layers were already mutated (partial state).
  Now: the default predicate skips it cleanly, and a custom predicate that
  forces it fails up front (atomic — nothing mutated).

These are NOT validated against another kernel path (lesson #11) — they
assert structural behavior (raise / skip / atomicity) directly.
"""
from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_mfa.svdquant import quantize_model
from mlx_mfa.svdquant.linear import SVDQuantLinear


class _Container(nn.Module):
    def __init__(self, *linears):
        super().__init__()
        for i, lin in enumerate(linears):
            setattr(self, f"fc{i}", lin)


class TestQuantizeModelTopLevelGuard:
    def test_bare_top_level_linear_raises_not_silent_noop(self):
        """Finding 1: a top-level nn.Linear cannot be replaced in place, so
        rather than a silent no-op it must raise (Rule 8)."""
        with pytest.raises(ValueError, match="top-level nn.Linear"):
            quantize_model(nn.Linear(512, 256))

    def test_container_with_no_quantizable_layers_is_clean_noop(self):
        """The guard must NOT over-fire: a container whose layers are all
        below the size threshold is a legitimate no-op (layers=[]), not an
        error."""
        m = _Container(nn.Linear(16, 16), nn.Linear(32, 32))
        stats = quantize_model(m)
        assert stats["layers"] == []
        assert stats["overall_compression"] == 1.0


class TestQuantizeModelGroupSizeGuard:
    def test_default_predicate_skips_group_misaligned_cleanly(self):
        """Finding 2a: in_features not divisible by group_size is SKIPPED by
        the default predicate (no raise, no partial state); aligned layers
        in the same model still quantize."""
        m = _Container(nn.Linear(512, 512), nn.Linear(300, 256))  # 300 % 64 != 0
        stats = quantize_model(m, group_size=64)
        assert len(stats["layers"]) == 1, "only the aligned layer quantizes"
        assert isinstance(m.fc0, SVDQuantLinear)      # aligned → quantized
        assert isinstance(m.fc1, nn.Linear)           # misaligned → skipped
        assert not isinstance(m.fc1, SVDQuantLinear)

    def test_custom_predicate_forcing_misaligned_raises_atomically(self):
        """Finding 2b: a custom predicate that forces a group-misaligned
        layer fails UP FRONT — before any layer is mutated (atomicity)."""
        m = _Container(nn.Linear(512, 512), nn.Linear(300, 256))
        force_all = lambda path, mod: isinstance(mod, nn.Linear)  # noqa: E731
        with pytest.raises(ValueError, match="not divisible by group_size"):
            quantize_model(m, group_size=64, class_predicate=force_all)
        # atomic: the aligned layer must NOT have been mutated before the raise
        assert isinstance(m.fc0, nn.Linear)
        assert not isinstance(m.fc0, SVDQuantLinear)

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_group_size_divisibility_respected(self, group_size):
        """A layer aligned to the chosen group_size quantizes; the same
        in_features under a larger non-dividing group_size is skipped.
        in_features=288 is >= the 256 dim threshold AND divisible by 32 but
        not 64/128 (288 % 64 == 32), so it isolates the group-size gate."""
        m = _Container(nn.Linear(288, 256))  # weight [256, 288], K=288
        stats = quantize_model(m, group_size=group_size)
        if 288 % group_size == 0:
            assert len(stats["layers"]) == 1
            assert isinstance(m.fc0, SVDQuantLinear)
        else:
            assert len(stats["layers"]) == 0
            assert isinstance(m.fc0, nn.Linear)

"""II-8 addendum item 2 — buffer-pool poison CANARY (permanent).

Reconstructs the pre-II-6 NaN-temporary churn (SDPA over all--inf bias
rows -> NaN intermediates released to the Metal pool) at the START of
the suite (test_aa_* sorts first).  If pool-recycling stale-value
sensitivity ever regresses, the downstream finite-value kernel tests
(topk-bisect thresholds, mixed-dtype STEEL, sparse-native) flake —
this file is the stressor that makes such a regression visible.
Evidence base: sprint-II-8 addendum item 2 (27 directed poisoning
rounds + 7 stressed full-suite runs, all clean post-II-6)."""
import os

import mlx.core as mx
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("MFA_POOL_STRESS") != "1",
                    reason="pool-stress canary is opt-in (MFA_POOL_STRESS=1); "
                           "run via the release-audit stress step")
def test_churn_nan_temporaries():
    # the pre-II-6 mechanism: SDPA over an all--inf bias row produces NaN
    # rows; the result is an intermediate whose buffer returns to the pool.
    for _ in range(6):
        q = mx.random.normal((1, 8, 512, 64), dtype=mx.float16)
        bias = mx.full((1, 8, 512, 512), float("-inf"), dtype=mx.float16)
        out = mx.fast.scaled_dot_product_attention(q, q, q, scale=0.125, mask=bias)
        s = mx.sum(out * 0)  # consume; NaN*0 = NaN, then buffer freed
        mx.eval(s)
    assert True

"""Volet N — property-based classifier guard (CX-R10-01) + stale-f32-doc (CX-R10-02).

Round-10: Assertion 2 was a hardcoded callee-name regex, so moving
`flash_attention_topk` (which computes attention INLINE — no flash_attention
call to detect) into HELPER passed enumeration silently. The guard is now
PROPERTY-based: any HELPER export that is attention-input-shaped (takes a Q-like
AND a K-like param) OR is uninspectable (class / signature / getsource failure)
MUST be in REVIEWED_NONCOMPUTATIONAL, else the enumeration fails. These are the
executable mutation tests round-10 said were missing.
"""
import importlib.util
import functools
import mlx_mfa


def _enum():
    spec = importlib.util.spec_from_file_location("_enumN", "scripts/enumerate_api_surface.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── decisive bite: inline-compute attention op moved to HELPER must FAIL ─────────
def test_topk_in_helper_is_flagged():
    m = _enum()
    # flash_attention_topk takes q/k/v and computes attention inline; if it were
    # (mis)classified HELPER, the property guard must flag it (it is not in
    # REVIEWED_NONCOMPUTATIONAL). The old callee-name regex MISSED this.
    assert m.computational_in_helper(["flash_attention_topk"]) == ["flash_attention_topk"]


def test_make_shared_prefix_cache_in_helper_is_flagged():
    m = _enum()
    # the 24th entry: takes prefix_q/k/v → Q+K-shaped → flagged if mislabeled.
    assert "make_shared_prefix_cache" in m.computational_in_helper(["make_shared_prefix_cache"])


# ── bite: uninspectable exports (class / getsource-fail) must FAIL, not skip ─────
def test_synthetic_class_in_helper_is_flagged():
    m = _enum()
    mlx_mfa._n_test_class = type("Synthetic", (), {})
    try:
        assert m.computational_in_helper(["_n_test_class"]) == ["_n_test_class"]
    finally:
        del mlx_mfa._n_test_class


def test_synthetic_getsource_failure_in_helper_is_flagged():
    m = _enum()
    # a functools.partial has an inspectable signature but getsource() fails →
    # must be flagged (uninspectable), not silently skipped.
    mlx_mfa._n_test_partial = functools.partial(lambda q, k, v: None)
    try:
        assert m.computational_in_helper(["_n_test_partial"]) == ["_n_test_partial"]
    finally:
        del mlx_mfa._n_test_partial


# ── clean state: every real HELPER is accounted for (no offenders) ──────────────
def test_clean_state_no_offenders():
    m = _enum()
    pub = m.public_exports()
    helpers = [n for n in pub if m.classify_public(n, pub[n])[0] == "helper"]
    assert m.computational_in_helper(helpers) == []


def test_reviewed_set_only_contains_real_helpers():
    # every name in REVIEWED_NONCOMPUTATIONAL must actually be a HELPER export
    # (no stale entries) — keeps the review-set honest.
    m = _enum()
    pub = m.public_exports()
    helpers = {n for n in pub if m.classify_public(n, pub[n])[0] == "helper"}
    stale = [n for n in m.REVIEWED_NONCOMPUTATIONAL if n not in helpers]
    assert not stale, f"REVIEWED_NONCOMPUTATIONAL has non-helper entries: {stale}"


def test_enumeration_24_public_clean():
    m = _enum()
    pub = m.public_exports()
    comp = [n for n in pub if m.classify_public(n, pub[n])[0] == "computational"]
    assert len(comp) == 24


# ── CX-R10-02: f32 routes to SDPA only (not native), consistent across density ──
def test_f32_routes_sdpa_only():
    import math
    import numpy as np
    import mlx.core as mx
    from mlx_mfa.lcsa_nax import sparse_attention_dispatch as sad
    Hq, Hk, N, D = 8, 2, 1024, 128
    bt = 32
    nt = (N + bt - 1) // bt
    bm = mx.ones((Hq, nt, nt), dtype=mx.bool_)

    def mk():
        mx.random.seed(0)
        q = mx.random.normal((1, Hq, N, D)).astype(mx.float32)
        k = mx.random.normal((1, Hk, N, D)).astype(mx.float32)
        v = mx.random.normal((1, Hk, N, D)).astype(mx.float32)
        mx.eval(q, k, v)
        return q, k, v
    sc = 1.0 / math.sqrt(D)
    # density-1.0 mask: thr=1.0 → SDPA regime; thr=1.01 → would be native, but
    # f32 forces SDPA → both byteΔ-identical (not density-dependent).
    a = sad(*mk(), bm, block_tile=bt, scale=sc, density_threshold=1.0)
    b = sad(*mk(), bm, block_tile=bt, scale=sc, density_threshold=1.01)
    mx.eval(a, b)
    assert a.dtype == mx.float32 and b.dtype == mx.float32
    assert float(np.max(np.abs(np.array(a) - np.array(b)))) == 0.0

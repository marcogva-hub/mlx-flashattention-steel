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

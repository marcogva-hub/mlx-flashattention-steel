"""R6 / NEPB-04 + DOC-06 (code review 2026-09, P0 / P2 PUBLISHED) — sparse bias caches.

_SPARSE_BIAS_CACHE / _SPARSE_ROWFIX_CACHE / _SPARSE_SANITIZED_BIAS_CACHE were keyed by
id(block_mask) + shape + dtype.  A strong reference prevents id reuse (ABA), but MLX
arrays mutate IN PLACE with the same id (m[...] = x) -> the old mask's bias was
reused, silently (NEPB-04).  The key also omitted head_dim, on which the expansion
tile depends (DOC-06).

Fix (review decision, measured 2026-09-27, M5 Max, MLX 0.31.2): a CRC32 of the mask
bytes joins the key — 0.43 ms per lookup at NQ=NK=4509 (20.3 MB, zero-copy host view;
blake2b 13.7 ms, GPU weighted-sum 0.85 ms rejected; MLX exposes no version counter),
under the 1 ms budget; cost is linear in mask bytes.  head_dim joins all three keys.
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

import mlx_mfa.attention as att
from mlx_mfa import is_mfa_available
from mlx_mfa.attention import flash_attention_sparse

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")


def _f32(x):
    return x.astype(mx.float32)


def _clear_caches():
    # 2.64 B2: the per-head fallback caches a BOOL keep-mask (was a float bias); the
    # block-level row-fix / sanitized-bias caches are no longer on its path.
    att._SPARSE_MASK_CACHE.clear()
    att._SPARSE_ROWFIX_CACHE.clear()
    att._SPARSE_SANITIZED_BIAS_CACHE.clear()


def _inputs(N=2048, D=128, seed=8):
    mx.random.seed(seed)
    q, k, v = (mx.random.normal((1, 2, N, D)).astype(mx.float16) for _ in range(3))
    nq = N // 32
    mx.random.seed(11)
    m = mx.random.uniform(shape=(nq, nq)) < 0.3
    m = m | m.T | mx.eye(nq).astype(mx.bool_)
    mx.eval(q, k, v, m)
    return q, k, v, m


def test_inplace_mutated_mask_uses_the_new_pattern():
    """NEPB-04: default gate (N=2048, D=128) -> per-head SDPA fallback (cached bias)."""
    _clear_caches()
    q, k, v, m = _inputs()
    sc = 1.0 / math.sqrt(q.shape[-1])
    o_old = flash_attention_sparse(q, k, v, m, scale=sc)
    mx.eval(o_old)
    same_id = id(m)
    m[:, :] = mx.eye(m.shape[0]).astype(mx.bool_)     # in place: block-diagonal now
    mx.eval(m)
    assert id(m) == same_id
    o_mut = flash_attention_sparse(q, k, v, m, scale=sc)
    o_fresh = flash_attention_sparse(q, k, v, mx.array(m), scale=sc)
    mx.eval(o_mut, o_fresh)
    assert float(mx.max(mx.abs(_f32(o_mut) - _f32(o_old)))) > 1e-2   # the mask did change
    assert float(mx.max(mx.abs(_f32(o_mut) - _f32(o_fresh)))) == 0.0


def test_unmutated_mask_still_hits_the_cache():
    """The content key must not defeat caching: a repeat call returns the SAME entry."""
    _clear_caches()
    _, _, _, m = _inputs()
    b1 = att._get_or_build_expanded_bool_mask(m, 2048, 2048, head_dim_d7=128)
    b2 = att._get_or_build_expanded_bool_mask(m, 2048, 2048, head_dim_d7=128)
    assert b1 is b2 and len(att._SPARSE_MASK_CACHE) == 1


def test_same_mask_different_head_dim_gets_distinct_entries():
    """DOC-06: the expansion tile depends on head_dim -> it must be part of every key."""
    _clear_caches()
    N = 100
    m = mx.eye(4).astype(mx.bool_) | mx.array([[False, True, False, False]] * 4)
    mx.eval(m)
    b64 = att._get_or_build_expanded_bool_mask(m, N, N, head_dim_d7=64)
    b128 = att._get_or_build_expanded_bool_mask(m, N, N, head_dim_d7=128)
    assert len(att._SPARSE_MASK_CACHE) == 2
    # each cached entry equals its own uncached computation
    for hd, b in ((64, b64), (128, b128)):
        _clear_caches()
        fresh = att._get_or_build_expanded_bool_mask(m, N, N, head_dim_d7=hd)
        assert mx.array_equal(fresh, b).item()


@pytest.mark.parametrize("mdt", [mx.bfloat16, mx.float16, mx.uint8],
                         ids=["bf16", "f16", "u8"])
def test_non_bool_block_masks_still_accepted(mdt):
    """Pre-merge review 2026-09-28 (N1): the content key hashed the mask in its OWN
    dtype, so a bf16 mask crashed in numpy ("PEP 3118 item size") — before R6 it was
    converted to bool first.  Non-bool masks must keep working and match bool."""
    N, D = 512, 64
    mx.random.seed(51)
    q, k, v = (mx.random.normal((1, 1, N, D)).astype(mx.float16) for _ in range(3))
    nb = N // 32
    mb = mx.eye(nb, dtype=mx.bool_) | (mx.random.uniform(shape=(nb, nb)) < 0.3)
    ref = flash_attention_sparse(q, k, v, mb)
    out = flash_attention_sparse(q, k, v, mb.astype(mdt))
    mx.eval(ref, out)
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)))) == 0.0

"""V6 hybrid sparse backward and per-head masks (added to 2.62.2 by decision 2026-09).

The hybrid's dQ/dK leg collapsed 3-D/4-D block masks with `.any()` (a cross-head
UNION) — the class the 2026-05 review removed from the default wrapper.  Verified at
source during remediation: that branch is UNREACHABLE — the public gate routes only
`block_mask.ndim == 2` to the hybrid (attention.py `_v6nax_hybrid_eligible`, "PoC
scope") and the native dV kernel refuses any other mask
(mfa_v6_nax_primitive.cpp "block_mask must be 2-D").  So no silent-wrong gradient was
reachable; the union is replaced by an explicit contract so a future relaxation of the
gate cannot silently use it.

Locks: (1) per-head masks under MFA_ENABLE_V6_BACKWARD=1 at a hybrid-eligible shape
get correct PER-HEAD gradients (they take the default wrapper); (2) the hybrid entry
refuses a non-2-D mask with a clear ValueError.
"""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_mfa import is_mfa_available
from mlx_mfa.attention import _v6nax_sparse_hybrid_vjp, flash_attention_sparse

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

B, H, N, D, BT = 1, 2, 4096, 64, 64   # per-head [2,64,64] = 8 KB -> NAX-route zone


def _inputs():
    mx.random.seed(21)
    q, k, v = (mx.random.normal((B, H, N, D)).astype(mx.float16) for _ in range(3))
    nb = N // BT
    eye = mx.eye(nb, dtype=mx.bool_)
    m0 = eye | (mx.random.uniform(shape=(nb, nb)) < 0.15)
    m1 = eye | (mx.random.uniform(shape=(nb, nb)) < 0.15)
    mask3 = mx.stack([m0, m1])                              # [H, NQ, NK] per-head
    mx.random.seed(22)
    g = mx.random.normal((B, H, N, D)).astype(mx.float32)
    mx.eval(q, k, v, mask3, g)
    return q, k, v, mask3, g


def _per_head_oracle_grads(q, k, v, mask3, g):
    tok = mx.repeat(mx.repeat(mask3, BT, axis=-2), BT, axis=-1)   # [H, N, S]
    with mx.stream(mx.cpu):
        bias = mx.where(tok, 0.0, float("-inf"))[None]

        def f(a, b, c):
            s = (a @ mx.swapaxes(b, -1, -2)) * (D ** -0.5) + bias
            return ((mx.softmax(s, axis=-1) @ c) * g).sum()
        grads = mx.grad(f, argnums=(0, 1, 2))(*(x.astype(mx.float32) for x in (q, k, v)))
        mx.eval(*grads)
    return grads


def test_per_head_mask_with_v6_backward_gets_per_head_gradients(monkeypatch):
    monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
    q, k, v, mask3, g = _inputs()
    grads = mx.grad(lambda a, b, c: (flash_attention_sparse(a, b, c, mask3, scale=D ** -0.5)
                                     .astype(mx.float32) * g).sum(), argnums=(0, 1, 2))(q, k, v)
    mx.eval(*grads)
    for name, x, y in zip(("dQ", "dK", "dV"), grads, _per_head_oracle_grads(q, k, v, mask3, g)):
        err = float(mx.max(mx.abs(x.astype(mx.float32) - y)))
        assert err < 3e-2, (name, err)


def test_hybrid_entry_refuses_non_2d_mask():
    q, k, v, mask3, g = _inputs()
    with pytest.raises(ValueError, match="2-D"):
        _, gr = mx.vjp(lambda a, b, c: _v6nax_sparse_hybrid_vjp(a, b, c, mask3, BT, D ** -0.5, False),
                       [q, k, v], [g.astype(mx.float16)])
        mx.eval(*gr)

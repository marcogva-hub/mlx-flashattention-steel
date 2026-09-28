"""R5 / NEPB-03 (code review 2026-09, P0 PUBLISHED) — sparse NAX backward on an empty row.

The M5 NAX sparse forward writes ZEROS for a query row whose block-mask row is all
False (II-6 contract).  Its custom vjp rebuilt an UNSANITIZED float bias (-inf across
the whole row) and ran mx.vjp(SDPA): softmax of an all -inf row is NaN, which poisoned
dQ for that row and, through P^T, EVERY column of dK/dV (nan_frac 0.4% / 100% / 100%).
Same leg in the V6 hybrid (MFA_ENABLE_V6_BACKWARD=1).  The per-head SDPA fallback
already sanitized empty rows, so gradient finiteness depended on the route.

Fix: graph-safe sanitization in both SDPA-vjp legs — empty rows get a finite bias and
their output (hence cotangent) is forced to 0, i.e. exactly the gradient of the
forward's "empty row -> 0" contract.
"""
from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_mfa import is_mfa_available
from mlx_mfa import _dispatch_trace as dt
from mlx_mfa.attention import _sparse_fallback_sdpa_perhead, flash_attention_sparse

pytestmark = pytest.mark.skipif(not is_mfa_available(), reason="MFA extension required")

B, H, N, D = 1, 1, 8192, 64       # default beta-3 gate cell (N=8192, B*H=1, d<=0.30)
SCALE = 1.0 / math.sqrt(D)
EMPTY_BLOCK_ROW = 5               # query rows 160..191 have no active key block
GRAD_TOL = 2e-2                   # f16 NAX-vs-fallback gradient agreement


def _inputs():
    mx.random.seed(6)
    q, k, v = (mx.random.normal((B, H, N, D)).astype(mx.float16) for _ in range(3))
    nq = N // 32
    mx.random.seed(1)
    m = mx.random.uniform(shape=(nq, nq)) < 0.1
    m = m | m.T | mx.eye(nq).astype(mx.bool_)
    m = mx.where((mx.arange(nq) == EMPTY_BLOCK_ROW)[:, None], False, m)
    mx.random.seed(7)
    dO = mx.random.normal((B, H, N, D)).astype(mx.float16)
    mx.eval(q, k, v, m, dO)
    return q, k, v, m, dO


def _finite(x):
    return bool(mx.all(mx.isfinite(x.astype(mx.float32))).item())


def _err(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


@pytest.fixture(scope="module")
def reference():
    """Independent route with the correct semantics: the per-head SDPA fallback."""
    q, k, v, m, dO = _inputs()
    o, g = mx.vjp(lambda a, b, c: _sparse_fallback_sdpa_perhead(a, b, c, m, SCALE, False),
                  [q, k, v], [dO])
    mx.eval(o, *g)
    assert _finite(o[0]) and all(_finite(x) for x in g)
    return o[0], g


@pytest.mark.parametrize("hybrid", [False, True], ids=["default-wrapper", "v6-hybrid"])
def test_empty_row_backward_is_finite_and_matches_fallback(hybrid, reference, monkeypatch):
    if hybrid:
        monkeypatch.setenv("MFA_ENABLE_V6_BACKWARD", "1")
    else:
        monkeypatch.delenv("MFA_ENABLE_V6_BACKWARD", raising=False)
    q, k, v, m, dO = _inputs()
    with dt.capture() as tr:
        o, g = mx.vjp(lambda a, b, c: flash_attention_sparse(a, b, c, m, scale=SCALE),
                      [q, k, v], [dO])
        mx.eval(o, *g)
    o = o[0]
    assert tr and tr[-1][0].startswith("v6nax_sparse"), [t[0] for t in tr]   # which-binary
    lo, hi = EMPTY_BLOCK_ROW * 32, (EMPTY_BLOCK_ROW + 1) * 32
    assert _finite(o) and float(mx.max(mx.abs(o[:, :, lo:hi])).item()) == 0.0
    names = ("dQ", "dK", "dV")
    assert all(_finite(x) for x in g), [n for n, x in zip(names, g) if not _finite(x)]
    assert float(mx.max(mx.abs(g[0][:, :, lo:hi])).item()) == 0.0   # empty row: dQ == 0
    ref_o, ref_g = reference
    assert _err(o, ref_o) < 5e-3
    for n, x, y in zip(names, g, ref_g):
        assert _err(x, y) < GRAD_TOL, (n, _err(x, y))


# ── Sibling legs (same class: unsanitized all -inf bias row in an SDPA-vjp) ─────
def _small_inputs(N=512, Dh=64):
    mx.random.seed(11)
    q, k, v = (mx.random.normal((1, 2, N, Dh)).astype(mx.float16) for _ in range(3))
    nq = N // 32
    m = mx.eye(nq).astype(mx.bool_) | (mx.random.uniform(shape=(nq, nq)) < 0.2)
    m = mx.where((mx.arange(nq) == 3)[:, None], False, m)          # empty query-block row
    dO = mx.random.normal(q.shape).astype(mx.float16)
    mx.eval(q, k, v, m, dO)
    return q, k, v, m, dO


def _small_reference(q, k, v, m, dO, sc):
    _, g = mx.vjp(lambda a, b, c: _sparse_fallback_sdpa_perhead(a, b, c, m, sc, False),
                  [q, k, v], [dO])
    mx.eval(*g)
    return g


def test_no_ext_fallback_backward_is_finite():
    """_sparse_fallback_sdpa (no-extension path) zeroed only the OUTPUT row; SDPA's
    backward recomputed P = NaN on that row (0 * NaN = NaN) and poisoned dK/dV."""
    from mlx_mfa.attention import _sparse_fallback_sdpa
    q, k, v, m, dO = _small_inputs()
    sc = 1.0 / math.sqrt(64)
    _, g = mx.vjp(lambda a, b, c: _sparse_fallback_sdpa(a, b, c, m, 32, 32, sc, False),
                  [q, k, v], [dO])
    mx.eval(*g)
    assert all(_finite(x) for x in g)
    for x, y in zip(g, _small_reference(q, k, v, m, dO, sc)):
        assert _err(x, y) < GRAD_TOL


def test_steel_sparse_custom_sdpa_backward_leg_is_finite():
    """_make_mfa_sparse_custom (M1-M4 STEEL sparse route) backward='sdpa' leg.  Its
    gradients depend only on (q, k, v, mask, dO), so the leg is testable on M5 even
    though the STEEL sparse FORWARD is not used on M5 [route itself UNTESTABLE-HERE]."""
    from mlx_mfa.attention import _make_mfa_sparse_custom
    q, k, v, m, dO = _small_inputs()
    sc = 1.0 / math.sqrt(64)
    impl = _make_mfa_sparse_custom(sc, False, head_dim=64, backward="sdpa")
    _, g = mx.vjp(lambda a, b, c: impl(a, b, c, m.astype(mx.uint8))[0], [q, k, v], [dO])
    mx.eval(*g)
    assert all(_finite(x) for x in g)
    for x, y in zip(g, _small_reference(q, k, v, m, dO, sc)):
        assert _err(x, y) < GRAD_TOL


# ── Pre-merge review 2026-09-28 (B1): 397144e inserted _sanitize_empty_rows_graph
# BETWEEN `@functools.lru_cache` and _make_sparse_nax_with_sdpa_vjp — the maker lost
# its cache (a new custom_function per call) and the sanitizer cached arrays by
# identity: every backward pinned ~268 MB (N=8192) for the last 64 calls.
import ast as _ast  # noqa: E402
import gc  # noqa: E402
import pathlib as _pathlib  # noqa: E402

import mlx_mfa.attention as _att  # noqa: E402


def test_lru_cache_is_on_the_maker_not_the_sanitizer():
    assert hasattr(_att._make_sparse_nax_with_sdpa_vjp, "cache_info")
    assert not hasattr(_att._sanitize_empty_rows_graph, "cache_info")


def test_no_lru_cache_over_array_arguments():
    """Class lock: a functools cache keyed by an mx.array pins device memory (and
    goes stale on in-place mutation).  Every cached function in the package must
    take hashable config only — its parameters must not be annotated mx.array."""
    pkg = _pathlib.Path(_att.__file__).parent
    offenders = []
    for path in sorted(pkg.glob("*.py")):
        for fn in _ast.walk(_ast.parse(path.read_text())):
            if not isinstance(fn, _ast.FunctionDef):
                continue
            cached = any("lru_cache" in _ast.unparse(d) or _ast.unparse(d).endswith("cache")
                         for d in fn.decorator_list)
            if cached and any(a.annotation is not None and "array" in _ast.unparse(a.annotation)
                              for a in fn.args.args + fn.args.kwonlyargs):
                offenders.append(f"{path.name}:{fn.lineno}:{fn.name}")
    assert not offenders, offenders


def test_sparse_nax_backward_memory_is_flat():
    """Measured: +268 MB kept per backward call at this shape before the fix."""
    if not _att._get_has_nax_cached():
        pytest.skip("NAX sparse route is M5+ only")
    N, D = 8192, 64
    mx.random.seed(41)
    q, k, v = (mx.random.normal((1, 1, N, D)).astype(mx.float16) for _ in range(3))
    nb = N // 32
    bm = mx.eye(nb, dtype=mx.bool_) | (mx.random.uniform(shape=(nb, nb)) < 0.1)
    mx.eval(q, k, v, bm)

    def step():
        g = mx.grad(lambda a, b, c: _att.flash_attention_sparse(a, b, c, bm)
                    .astype(mx.float32).sum(), argnums=(0, 1, 2))(q, k, v)
        mx.eval(*g)
        del g
        # Deterministic reading: without gc + a GPU sync, buffers of the last
        # command buffer are freed asynchronously and the reading swings by
        # +-1 bias (131/256 MB, both signs; flaked ~25%).  Synced: 0 MB 15/15,
        # and the simulated B1 bug still reads +768 MB (3 x 256 MB).
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        return mx.get_active_memory()

    step()
    base = step()
    for _ in range(3):
        last = step()
    assert last - base < 64 * 2 ** 20, f"active memory grew {(last - base) / 2**20:.0f} MB over 3 calls"

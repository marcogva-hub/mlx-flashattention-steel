"""Audit remediation Tier-0 locks — input validation at the public boundary.

Two findings from the 2026-06-21 CC+Codex audit:

* **C-01 (CRITICAL, OOB read).** `flash_attention` validated rank, Q/K head_dim
  and GQA head-divisibility, but never that K and V agree with each other / with
  Q on *batch*, *kv-seq*, or *kv-heads*.  The MFA primitive derives B from Q and
  ALL K/V strides from K (`V_batch_stride = Hk*Sk*D`), so a mismatched K/V reads
  out of bounds → SILENT-WRONG finite output.  Reproduced on M5/MLX-0.31.2:
  `backend="mfa"`, Bq=2, Bk=Bv=1, nonzero V → `out[1]` diverged ~1.3 from the
  correct broadcast while `out[0]` matched SDPA to fp16 noise.

* **H-01 (dropout range).** Contract is `dropout_p ∈ [0, 1)` but nothing enforced
  it: `p=1.0` → NaN (÷ by 1-p), `p>1` → garbage, `p<0` → silently disabled.

Bite-proof: on the PRE-fix code every `*_raises` case below returned a finite
(silently-wrong) array instead of raising, so the asserts fail without the fix.
The validation happens at the Python boundary BEFORE any GPU dispatch — these
tests never call `mx.eval`, so a raise proves pre-dispatch rejection.

The C++ binding carries the same guard as defense-in-depth
(`csrc/mfa_attention.cpp::mfa_attention_forward`); exercised here via the direct
`_ext` entry so the guard is not reachable only through the Python layer.
"""

import math

import mlx.core as mx
import pytest

import mlx_mfa
from mlx_mfa import flash_attention

H, D = 4, 128


def _rn(shape):
    return mx.random.normal(shape).astype(mx.float16)


# --------------------------------------------------------------------------- #
# C-01 — K/V shape contract (batch / kv-seq / kv-heads)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", ["auto", "mfa"])
def test_mismatched_batch_raises_before_dispatch(backend):
    q, k, v = _rn((2, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 64, D))
    with pytest.raises(ValueError, match="batch"):
        # no mx.eval — must raise at the boundary, before any GPU work
        flash_attention(q, k, v, causal=False, backend=backend)


@pytest.mark.parametrize("backend", ["auto", "mfa"])
def test_mismatched_kv_seq_raises(backend):
    q, k, v = _rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 32, D))
    with pytest.raises(ValueError, match="sequence length"):
        flash_attention(q, k, v, causal=False, backend=backend)


@pytest.mark.parametrize("backend", ["auto", "mfa"])
def test_mismatched_kv_heads_raises(backend):
    q, k, v = _rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, 2, 64, D))
    with pytest.raises(ValueError, match="number of heads"):
        flash_attention(q, k, v, causal=False, backend=backend)


# --------------------------------------------------------------------------- #
# H-01 — dropout_p range
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("p", [1.0, 1.1, -0.2, float("nan"), float("inf")])
def test_dropout_out_of_range_raises(p):
    q, k, v = _rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 64, D))
    with pytest.raises(ValueError, match="dropout_p"):
        flash_attention(q, k, v, causal=False, dropout_p=p)


# --------------------------------------------------------------------------- #
# Valid shapes must STILL pass (no false-positive rejection)
# --------------------------------------------------------------------------- #

def test_valid_matched_still_passes():
    q, k, v = _rn((2, H, 64, D)), _rn((2, H, 64, D)), _rn((2, H, 64, D))
    out = flash_attention(q, k, v, causal=False)
    mx.eval(out)
    assert out.shape == (2, H, 64, D)
    assert bool(mx.all(mx.isfinite(out)))


def test_valid_gqa_still_passes():
    # GQA: kv-heads < q-heads is legal as long as K and V agree.
    q = _rn((1, H, 64, D))
    k = _rn((1, 2, 64, D))
    v = _rn((1, 2, 64, D))
    out = flash_attention(q, k, v, causal=False)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))


def test_valid_dropout_in_range_passes():
    q, k, v = _rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 64, D))
    out = flash_attention(q, k, v, causal=False, dropout_p=0.1)
    mx.eval(out)
    assert bool(mx.all(mx.isfinite(out)))


# --------------------------------------------------------------------------- #
# C++ defense-in-depth — direct _ext binding must also reject
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not mlx_mfa.has_nax(), reason="needs the compiled _ext")
class TestCppBindingGuard:
    def _fwd(self, q, k, v):
        from mlx_mfa import _ext
        out = _ext.mfa_attention_forward(q, k, v, 1.0 / math.sqrt(D), False, 0.0, -1, -1)
        mx.eval(out)
        return out

    def test_direct_binding_rejects_mismatched_batch(self):
        with pytest.raises(Exception, match="batch"):
            self._fwd(_rn((2, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 64, D)))

    def test_direct_binding_rejects_mismatched_kv_seq(self):
        with pytest.raises(Exception, match="sequence length"):
            self._fwd(_rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 32, D)))

    def test_direct_binding_rejects_mismatched_kv_heads(self):
        with pytest.raises(Exception, match="number of heads"):
            self._fwd(_rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, 2, 64, D)))

    def test_direct_binding_valid_passes(self):
        out = self._fwd(_rn((1, H, 64, D)), _rn((1, H, 64, D)), _rn((1, H, 64, D)))
        assert bool(mx.all(mx.isfinite(out)))

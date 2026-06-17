"""v2.56.0 — MFA_FORCE_NATIVE_BWD removal regression test.

The MFA_FORCE_NATIVE_BWD override knob (introduced v2.36-era, deprecated
v2.50.0 Prompt 5f Phase E with "target removal v2.51+") was REMOVED in
v2.56.0 after its deprecation cycle completed (5 minor versions of
DeprecationWarning) and forced STEEL backward was measured to be
dominated at every cell (V34 at D=64, SDPA-vjp at D=128; sprint-C Track 2).

The STEEL backward kernel itself remains reachable via backend="mfa"
(keep-all-paths); only the env-var knob was removed.

This test guards the removal: the env var must now be INERT — setting it
must NOT change routing and must NOT emit a DeprecationWarning. Routing
follows the benchmark-backed policy table regardless of the (now-ignored)
env var.
"""
from __future__ import annotations

import warnings

import mlx.core as mx

from mlx_mfa.dispatch_policy import should_use_native_backward


def _route(**kw):
    return should_use_native_backward(
        head_dim=kw.get("head_dim", 64),
        seq_len=kw.get("seq_len", 4096),
        causal=kw.get("causal", True),
        dtype=kw.get("dtype", mx.float16),
    )


def test_force_native_bwd_is_inert_and_silent(monkeypatch):
    """MFA_FORCE_NATIVE_BWD=1 must no longer change routing or warn (removed v2.56.0)."""
    baseline = _route()  # policy-table result, no env var
    monkeypatch.setenv("MFA_FORCE_NATIVE_BWD", "1")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always", DeprecationWarning)
        forced = _route()
    # Routing unchanged by the (removed) knob:
    assert forced == baseline, (
        f"MFA_FORCE_NATIVE_BWD=1 changed routing ({forced} vs baseline "
        f"{baseline}); the knob should be removed/inert in v2.56.0")
    # No deprecation warning (the knob and its warning are gone):
    dep = [w for w in w_list if issubclass(w.category, DeprecationWarning)
           and "MFA_FORCE_NATIVE_BWD" in str(w.message)]
    assert len(dep) == 0, f"Removed knob must not warn; got {[str(w.message) for w in dep]}"


def test_force_native_bwd_zero_is_inert(monkeypatch):
    """MFA_FORCE_NATIVE_BWD=0 must also be inert (no special opt-out path remains)."""
    baseline = _route()
    monkeypatch.setenv("MFA_FORCE_NATIVE_BWD", "0")
    assert _route() == baseline


def test_routing_follows_policy_table_only(monkeypatch):
    """With or without the env var, routing is the policy-table decision."""
    monkeypatch.delenv("MFA_FORCE_NATIVE_BWD", raising=False)
    unset = _route()
    monkeypatch.setenv("MFA_FORCE_NATIVE_BWD", "1")
    assert _route() == unset

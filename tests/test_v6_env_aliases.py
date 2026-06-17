"""V34->V6 env-var alias layer contract (v2.57.0 rename).

Locks the deprecation contract: new name canonical, old name a deprecated
alias honored with a one-shot DeprecationWarning, collisions -> NAX, and the
C++ alias table (csrc/mfa_env_aliases.hpp) stays in lockstep with the Python
table (mlx_mfa/_env_aliases.py). See NAMING.md.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from mlx_mfa._env_aliases import V6_ENV_ALIAS_MAP, getenv_aliased

_HPP = Path(__file__).resolve().parent.parent / "csrc" / "mfa_env_aliases.hpp"


def test_map_size_and_collision_rule():
    assert len(V6_ENV_ALIAS_MAP) == 30
    # collisions (old name already had V6) map V34 -> NAX
    assert V6_ENV_ALIAS_MAP["MFA_V6_USE_NAX"] == "MFA_V6_USE_V34"
    assert V6_ENV_ALIAS_MAP["MFA_V6_NAX_BK"] == "MFA_V6_V34_BK"
    # non-collision maps V34 -> V6
    assert V6_ENV_ALIAS_MAP["MFA_ENABLE_V6_BACKWARD"] == "MFA_ENABLE_V34_BACKWARD"
    # every new name is V34-free; every old name contains V34
    for new, old in V6_ENV_ALIAS_MAP.items():
        assert "V34" not in new and "V34" in old


def test_new_name_takes_precedence_no_warning(monkeypatch):
    monkeypatch.setenv("MFA_V6BWD_WM", "8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert getenv_aliased("MFA_V6BWD_WM", "4") == "8"
        assert not w


def test_deprecated_alias_honored_with_one_shot_warning(monkeypatch):
    # fresh process-warned set isolation: use a knob not warned elsewhere
    import mlx_mfa._env_aliases as A
    A._WARNED.discard("MFA_V34BWDK_BK")
    monkeypatch.setenv("MFA_V34BWDK_BK", "32")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert getenv_aliased("MFA_V6BWDK_BK") == "32"
        assert getenv_aliased("MFA_V6BWDK_BK") == "32"  # 2nd read
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep) == 1, f"expected one-shot warning, got {len(dep)}"


def test_unset_returns_default(monkeypatch):
    monkeypatch.delenv("MFA_V6BWDV_WM", raising=False)
    monkeypatch.delenv("MFA_V34BWDV_WM", raising=False)
    assert getenv_aliased("MFA_V6BWDV_WM", "4") == "4"
    assert getenv_aliased("MFA_V6BWDV_WM") is None


def test_cpp_and_python_alias_maps_are_in_lockstep():
    """csrc/mfa_env_aliases.hpp must encode exactly the same new->old pairs."""
    src = _HPP.read_text(encoding="utf-8")
    pairs = re.findall(r'\{"(MFA_[A-Z0-9_]+)",\s*"(MFA_[A-Z0-9_]+)"\}', src)
    cpp_map = dict(pairs)
    assert cpp_map == V6_ENV_ALIAS_MAP, (
        "C++ (mfa_env_aliases.hpp) and Python (_env_aliases.py) alias tables "
        "diverged — they MUST stay identical (see NAMING.md)."
    )

"""Centralized V34->V6 env-var alias table (v2.57.0).

The ``V34`` token was the internal generator name for the V6 NAX kernel
(cooperative-tensor ``matmul2d``). v2.57.0 unified the nomenclature to
``V6``; see ``NAMING.md`` for the full provenance.

The NEW ``MFA_V6*`` name is canonical. The OLD ``MFA_*V34*`` name is a
DEPRECATED alias: still honored (existing scripts keep working) but it
emits a one-shot ``DeprecationWarning`` per process. Aliases are removed
in v3.0.0.

Rename rule: ``V34 -> V6``, EXCEPT where the name already contained
``V6`` (a collision), in which case ``V34 -> NAX`` (e.g.
``MFA_V6_USE_V34 -> MFA_V6_USE_NAX``). This mirrors the C++ table in
``csrc/mfa_env_aliases.hpp`` exactly.
"""

from __future__ import annotations

import os
import warnings

from ._knobs import parse_bool_value

# new (canonical) -> old (deprecated alias). Single source of truth for the
# Python side; mirrors csrc/mfa_env_aliases.hpp::v6_env_alias_map().
V6_ENV_ALIAS_MAP: dict[str, str] = {
    # V34 -> V6 (no collision)
    "MFA_ENABLE_V6_BACKWARD": "MFA_ENABLE_V34_BACKWARD",
    "MFA_DISABLE_V6_BACKWARD": "MFA_DISABLE_V34_BACKWARD",
    "MFA_ENABLE_V6_D128": "MFA_ENABLE_V34_D128",
    "MFA_V6_BWD_KERNEL": "MFA_V34_BWD_KERNEL",
    "MFA_V6_BWD_SPARSE_NATIVE": "MFA_V34_BWD_SPARSE_NATIVE",
    "MFA_V6_DUMP_SOURCE": "MFA_V34_DUMP_SOURCE",
    "MFA_V6BWD": "MFA_V34BWD",
    "MFA_V6BWD_BK": "MFA_V34BWD_BK",
    "MFA_V6BWD_BQ": "MFA_V34BWD_BQ",
    "MFA_V6BWD_WM": "MFA_V34BWD_WM",
    "MFA_V6BWD_USE_FUSED": "MFA_V34BWD_USE_FUSED",
    "MFA_V6BWD_DUMP_SOURCE": "MFA_V34BWD_DUMP_SOURCE",
    "MFA_V6BWDF_BK": "MFA_V34BWDF_BK",
    "MFA_V6BWDF_BQ": "MFA_V34BWDF_BQ",
    "MFA_V6BWDF_WM": "MFA_V34BWDF_WM",
    "MFA_V6BWDF_DUMP_PATH": "MFA_V34BWDF_DUMP_PATH",
    "MFA_V6BWDF_DUMP_SOURCE": "MFA_V34BWDF_DUMP_SOURCE",
    "MFA_V6BWDK_BK": "MFA_V34BWDK_BK",
    "MFA_V6BWDK_BQ": "MFA_V34BWDK_BQ",
    "MFA_V6BWDK_WM": "MFA_V34BWDK_WM",
    "MFA_V6BWDV_BK": "MFA_V34BWDV_BK",
    "MFA_V6BWDV_BQ": "MFA_V34BWDV_BQ",
    "MFA_V6BWDV_WM": "MFA_V34BWDV_WM",
    "MFA_V6BWDKV_BK": "MFA_V34BWDKV_BK",
    "MFA_V6BWDKV_BQ": "MFA_V34BWDKV_BQ",
    "MFA_V6BWDKV_WM": "MFA_V34BWDKV_WM",
    # collisions (name already had V6) -> NAX
    "MFA_V6_USE_NAX": "MFA_V6_USE_V34",
    "MFA_V6_NAX_BK": "MFA_V6_V34_BK",
    "MFA_V6_NAX_BQ": "MFA_V6_V34_BQ",
    "MFA_V6_NAX_WM": "MFA_V6_V34_WM",
}

# old names already warned about this process (one-shot per alias).
_WARNED: set[str] = set()


def _warn_once(old_name: str, new_name: str) -> None:
    if old_name not in _WARNED:
        _WARNED.add(old_name)
        warnings.warn(
            f"env var {old_name} is deprecated (renamed to {new_name} in "
            f"v2.57.0; alias removed in v3.0.0). See NAMING.md.",
            DeprecationWarning,
            stacklevel=3,
        )


def getenv_aliased(new_name: str, default: str | None = None) -> str | None:
    """``os.environ.get`` with deprecated-alias fallback.

    Pass the NEW canonical name. Returns the new var if set; else the
    deprecated ``MFA_*V34*`` alias (warning once per process); else
    ``default``. New-takes-precedence keeps a both-set state quiet.
    """
    val = os.environ.get(new_name)
    if val is not None:
        return val
    old_name = V6_ENV_ALIAS_MAP.get(new_name)
    if old_name is not None:
        old_val = os.environ.get(old_name)
        if old_val is not None:
            _warn_once(old_name, new_name)
            return old_val
    return default


def get_bool_env_aliased(
    new_name: str, *, default: bool | None = False
) -> bool | None:
    """Resolve a canonical/deprecated alias and apply strict bool parsing."""
    return parse_bool_value(
        new_name, getenv_aliased(new_name), default=default)

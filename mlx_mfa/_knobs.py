"""Central env-knob registry (Lot-1 B1, additive — 2026, off master 7831544).

The canonical enumeration of every ``MFA_*`` / ``MLX_MFA_*`` env var the library
reads. This module is **additive and non-invasive**: it does NOT change how any
knob is read or its default — it only provides a typo/ghost-knob check.

``validate_env(strict=...)`` warns (never raises) on an ``MFA_*`` / ``MLX_MFA_*``
var present in the environment but absent from ``KNOWN_KNOBS`` — a likely typo
(e.g. ``MFA_DISABLE_V6BWD`` vs ``MFA_DISABLE_V6_BACKWARD``) that would otherwise
silently no-op. Gated behind ``MFA_KNOB_STRICT=1`` (OFF by default) so a knob
missing from this list can never disrupt an existing setup.

Full migration of every read through typed ``KnobDef`` (type/default validation)
is a deliberate follow-up; the enumeration below is the foundation. The
C++-side knobs are listed for completeness (read in csrc/, not via this module).
"""
from __future__ import annotations
import os
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class KnobDef:
    name: str
    help: str = ""


# Complete enumeration (grep of mlx_mfa/ + csrc/, master 7831544). Names only —
# reproducing today's reads exactly; this list is the typo/ghost-knob oracle.
KNOWN_KNOBS: frozenset[str] = frozenset({
    # CX-09/CC-21 (audit): generated Metal `#define` tile params (MFA_BD/BK/BQ/
    # TD/TK/TQ/TGP/WM/WN/DTYPE/GQA/PAD/ROWS_PT/D_SPLITS/DIRECT_READS/NO_PADDING),
    # C++ header guards (*_HPP_/_HPP), and C macros (MFA_PRECONDITION/MFA_CHECK_ERROR)
    # were PURGED — they are NOT env vars (zero getenv reads), so a bogus tile macro
    # like MFA_BD=999 must NOT pass validate_env. Only real env knobs remain.
    # CC-12 (volet E): six MORE non-env entries removed (verified 0 getenv reads —
    # they were comments / a shader-source marker / module constants, never env
    # vars): MFA_CONV3D_MPP (comment-only force-enable; the real knob is the
    # inverse MFA_DISABLE_CONV3D_MPP), MFA_V6_BHND + MFA_V6_MATMUL_EXEC_SG
    # (comment-only), MFA_REQUIRE_MSL4 (a `// MFA_REQUIRE_MSL4` MSL source marker,
    # not an env var), MFA_SUPPORTED_DTYPES + MFA_SUPPORTED_HDIMS (module
    # constants _MFA_SUPPORTED_*, never env-overridable). They no longer advertise
    # non-existent tuning DOF via validate_env(strict=True).
    "MFA_CONV_NAX_NO_FAST_PATH",
    "MFA_CONV_NAX_USE_PYTHON_LEGACY",
    "MFA_DEBUG_SHADERS",
    "MFA_DISABLE_ASYNC",
    "MFA_DISABLE_AUTO_HOOKS",
    "MFA_DISABLE_CONV3D_MPP",
    "MFA_DISABLE_GNA_NATIVE",
    "MFA_DISABLE_ROPE_NAX",
    "MFA_DISABLE_SDPA_ROUTE",
    "MFA_DISABLE_TOPK_BISECT",
    "MFA_DISABLE_TOPK_NAX",
    "MFA_DISABLE_TQ_DECODE_SDPA",
    "MFA_DISABLE_V2",
    "MFA_DISABLE_V3",
    "MFA_DISABLE_V34_BACKWARD",
    "MFA_DISABLE_V6_BACKWARD",
    "MFA_DISABLE_V6_DENSE",
    "MFA_ENABLE_MACOS27_ROUTING",  # opt-in experimental macOS-27 M5+ routing (default off)
    "MFA_ENABLE_V3",
    "MFA_ENABLE_V34_BACKWARD",
    "MFA_ENABLE_V6_BACKWARD",
    "MFA_FORCE_D256_PATH",
    "MFA_FORCE_D512_PATH",
    "MFA_FORCE_GEN",
    "MFA_FORCE_SAGE_DECODE",
    "MFA_FORCE_SDPA_ROUTE",
    "MFA_FORCE_SPLITK",
    "MFA_FORCE_V2",
    "MFA_HOOK_VERBOSE",
    "MFA_IR_INVESTIGATE",
    "MFA_LCSA_KERNEL_VERSION",
    "MFA_NAX_SPARSE_DENSITY_CEILING",
    "MFA_NO_PADDING",  # real C++ shader-generator knob (env_bool, csrc/mfa_env.hpp)
    "MFA_PAGED_TRUST_INDICES",  # perf opt-out: skip the host block_table/seq_lens
    #   value-range sync on the paged decode hot path (kernel still bounds-guards)
    "MFA_VARLEN_TRUST_METADATA",  # perf opt-out: skip the host cu_seqlens/tile_offsets
    #   VALUE-validation sync on varlen forwards (mirrors MFA_PAGED_TRUST_INDICES)
    "MFA_SPLITK_MAX_N_D",
    "MFA_V2_BD_HALF_D512",
    "MFA_V2_BQ64",
    "MFA_V2_FORCE_BK",
    "MFA_V2_FORCE_BK_D256",
    "MFA_V2_FORCE_BK_D512",
    "MFA_V2_FORCE_BQ_D512",
    "MFA_V34BWDF_BK",
    "MFA_V34BWDF_BQ",
    "MFA_V34BWDF_DUMP_PATH",
    "MFA_V34BWDF_DUMP_SOURCE",
    "MFA_V34BWDF_WM",
    "MFA_V34BWDKV_BK",
    "MFA_V34BWDKV_BQ",
    "MFA_V34BWDKV_WM",
    "MFA_V34BWDK_BK",
    "MFA_V34BWDK_BQ",
    "MFA_V34BWDK_WM",
    "MFA_V34BWDV_BK",
    "MFA_V34BWDV_BQ",
    "MFA_V34BWDV_WM",
    "MFA_V34BWD_BK",
    "MFA_V34BWD_BQ",
    "MFA_V34BWD_DUMP_SOURCE",
    "MFA_V34BWD_USE_FUSED",
    "MFA_V34BWD_WM",
    "MFA_V34_BWD_KERNEL",
    "MFA_V34_BWD_SPARSE_NATIVE",
    "MFA_V34_DUMP_SOURCE",
    "MFA_V3_FORCE_BK_",
    "MFA_V3_FORCE_BK_D128",
    "MFA_V3_FORCE_BK_D64",
    "MFA_V6BWDF_BK",
    "MFA_V6BWDF_BQ",
    "MFA_V6BWDF_DUMP_PATH",
    "MFA_V6BWDF_DUMP_SOURCE",
    "MFA_V6BWDF_WM",
    "MFA_V6BWDKV_BK",
    "MFA_V6BWDKV_BQ",
    "MFA_V6BWDKV_WM",
    "MFA_V6BWDK_BK",
    "MFA_V6BWDK_BQ",
    "MFA_V6BWDK_WM",
    "MFA_V6BWDV_BK",
    "MFA_V6BWDV_BQ",
    "MFA_V6BWDV_WM",
    "MFA_V6BWD_BK",
    "MFA_V6BWD_BQ",
    "MFA_V6BWD_DUMP_SOURCE",
    "MFA_V6BWD_USE_FUSED",
    "MFA_V6BWD_WM",
    "MFA_V6_BLOCK_C",
    "MFA_V6_BLOCK_D",
    "MFA_V6_BLOCK_R",
    "MFA_V6_BNHD_LEGACY",
    "MFA_V6_BWD_KERNEL",
    "MFA_V6_BWD_SPARSE_NATIVE",
    "MFA_V6_BYPASS_TGP",
    "MFA_V6_DENSE_MIN_N",
    "MFA_V6_DUMP_SOURCE",
    "MFA_V6_EXEC_SG",
    "MFA_V6_FORCE_DYNAMIC_K",
    "MFA_V6_MAX_THREADS",
    "MFA_V6_NAX_BK",
    "MFA_V6_NAX_BQ",
    "MFA_V6_NAX_SINGLE_OTILE",
    "MFA_V6_NAX_WM",
    "MFA_V6_RELAXED_PRECISION",
    "MFA_V6_SENTINEL_FILL",
    "MFA_V6_UNROLL_MODE",
    "MFA_V6_USE_NAX",
    "MFA_V6_USE_V34",
    "MFA_V6_V34_BK",
    "MFA_V6_V34_BQ",
    "MFA_V6_V34_WM",
    # H4 FIX (audit, 2026-06-21): these are LIVE (read in attention.py:266/269 +
    # _knobs.py:229) but were absent — strict-validate false-flagged them, incl.
    # the knob that enables validation.
    "MFA_KNOB_STRICT",
    "MFA_REQUIRE_NAX",
    "MFA_SILENCE_NAX_WARNING",
    "MFA_UNSAFE_D128_SPARSE",  # DIAGNOSTIC-ONLY C++ knob (csrc/mfa_env.hpp): opens the
                              # D=128 sparse OOB guard for OS re-characterization. Default
                              # off; NEVER enable in production (known-incorrect kernel).
    "MLX_MFA_DISPATCH_TABLE",
    "MLX_MFA_HOOK_TELEMETRY",
    "MLX_MFA_VERBOSE_DISPATCH",
})

# C++-side knobs (read in csrc/, documented here for completeness; not validated
# by this module since they are read in the extension, not Python).
CPP_KNOBS: frozenset[str] = frozenset({
    # (none)
})

# M7/L-01 FIX (audit, 2026-06-21): knobs once registered/documented but now with
# ZERO read site anywhere in mlx_mfa/ or csrc/ (verified by grep).  Kept in a
# SEPARATE registry so a user who still sets one gets a LOUD "removed — no
# effect" signal under strict validation, instead of it being silently accepted
# as valid (it advertised tuning DOF that no longer exists).  Distinct from a
# typo (which lands in the "unrecognized" bucket).
REMOVED_KNOBS: frozenset[str] = frozenset({
    "MFA_BD_FRAGS",          # never read — advertised tile-frag DOF never existed
    "MFA_BD_TILE",           # never read
    "MFA_D_CHUNKS",          # never read
    "MFA_FORCE_NATIVE_BWD",  # removed (kernel retained) — CLAUDE.md status
    "MFA_TOPK_BISECT",       # never read
    # CC-22 (audit): ghost knobs — advertised (some with docstrings implying they
    # work) but with ZERO read site anywhere. Moved here so they warn "removed".
    "MFA_V6",  # ghost — registered/advertised but never read (CC-22)
    "MFA_GQA_DECODE_CIDER",  # ghost — registered/advertised but never read (CC-22)
    "MFA_TOPK_STREAM_V5",  # ghost — registered/advertised but never read (CC-22)
    "MFA_V6BWD",  # ghost — registered/advertised but never read (CC-22)
    # CC-13 (volet E): the V4/V5 STEEL forward knobs are documented as removed in
    # ENV_VARS.md (V4/V5 dropped from the build, Lot-2) but were absent here, so
    # strict-validate miswarned them as a "typo". They WERE real env vars → here,
    # for the distinct "removed — no effect" diagnostic. Verified 0 read sites.
    "MFA_ENABLE_V4",
    "MFA_ENABLE_V5",
    "MFA_V5_FORCE_BK",
    "MFA_V5_FORCE_BD_TILE",
    "MFA_V5_FORCE_BQ",
    "MFA_V5_FORCE_WM",
    # CC-03/CC-04 (volet E2): ghost knobs — registered + aliased but never read.
    # Verified 0 read sites (py + cpp getenv). D=128 V6 routing is governed by
    # MFA_DISABLE_V6_DENSE / MFA_V6_DENSE_MIN_N, not these. MFA_V34BWD is the
    # deprecated alias of MFA_V6BWD (already removed above) — moved for parity.
    "MFA_ENABLE_V6_D128",
    "MFA_ENABLE_V34_D128",
    "MFA_V34BWD",
})


# Runtime-templated knob families: the live name is built from params
# (e.g. MFA_SPLITK_MAX_N_D128_C1_A0_W0). A bare-name match can't catch these,
# so strict validation allows any var starting with one of these prefixes.
PREFIX_KNOBS: tuple[str, ...] = (
    "MFA_SPLITK_MAX_N_D",   # templated by D / causal / alibi / window
    "MFA_V3_FORCE_BK_",     # templated per-config force
)


def validate_env(strict: bool | None = None) -> list[str]:
    """Warn on environment ``MFA_*``/``MLX_MFA_*`` vars not in KNOWN_KNOBS.

    Returns the list of unrecognized names. No-op (returns []) unless
    ``strict`` is True or ``MFA_KNOB_STRICT=1`` — so a knob missing from the
    registry never disrupts an existing setup. Never raises (RULE 8: loud but
    non-fatal for an observability helper). Runtime-templated families
    (``PREFIX_KNOBS``) are matched by prefix to avoid false positives.
    """
    if strict is None:
        strict = os.environ.get("MFA_KNOB_STRICT") == "1"
    if not strict:
        return []
    known = KNOWN_KNOBS | CPP_KNOBS
    env_mfa = [
        k for k in os.environ
        if (k.startswith("MFA_") or k.startswith("MLX_MFA_"))
        and k not in known
        and not any(k.startswith(p) for p in PREFIX_KNOBS)
    ]
    # Removed/ghost knobs get a DISTINCT message ("removed — no effect") so a
    # user setting one knows it was real once but no longer does anything,
    # rather than reading it as a typo.
    removed = sorted(k for k in env_mfa if k in REMOVED_KNOBS)
    unknown = sorted(k for k in env_mfa if k not in REMOVED_KNOBS)
    for k in removed:
        warnings.warn(
            f"[mlx-mfa] knob {k!r} was REMOVED — it has no effect "
            f"(see _knobs.REMOVED_KNOBS).", RuntimeWarning, stacklevel=2)
    for k in unknown:
        warnings.warn(
            f"[mlx-mfa] unrecognized knob {k!r} (not in the registry) — "
            f"possible typo; it will have no effect.", RuntimeWarning, stacklevel=2)
    return removed + unknown

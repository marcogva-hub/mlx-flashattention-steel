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
    "MFA_BD",
    "MFA_BD_128",
    "MFA_BD_64",
    "MFA_BD_FRAGS",
    "MFA_BD_HALF",
    "MFA_BD_TILE",
    "MFA_BK",
    "MFA_BK_128",
    "MFA_BK_64",
    "MFA_BQ",
    "MFA_BQ_128",
    "MFA_BQ_64",
    "MFA_CHECK_ERROR",
    "MFA_CODE_WRITER_HPP_",
    "MFA_CONV3D_MPP",
    "MFA_CONV_NAX_NO_FAST_PATH",
    "MFA_CONV_NAX_USE_PYTHON_LEGACY",
    "MFA_DEBUG_SHADERS",
    "MFA_DEVICE_PROPERTIES_HPP_",
    "MFA_DIRECT_READS",
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
    "MFA_DTYPE",
    "MFA_D_CHUNKS",
    "MFA_D_SPLITS",
    "MFA_ENABLE_V3",
    "MFA_ENABLE_V34_BACKWARD",
    "MFA_ENABLE_V34_D128",
    "MFA_ENABLE_V4",
    "MFA_ENABLE_V5",
    "MFA_ENABLE_V6_BACKWARD",
    "MFA_ENABLE_V6_D128",
    "MFA_FORCE_D256_PATH",
    "MFA_FORCE_D512_PATH",
    "MFA_FORCE_GEN",
    "MFA_FORCE_NATIVE_BWD",
    "MFA_FORCE_SAGE_DECODE",
    "MFA_FORCE_SDPA_ROUTE",
    "MFA_FORCE_SPLITK",
    "MFA_FORCE_V2",
    "MFA_GQA",
    "MFA_GQA_DECODE_CIDER",
    "MFA_GQA_FACTOR",
    "MFA_HOOK_VERBOSE",
    "MFA_IR_INVESTIGATE",
    "MFA_LCSA_KERNEL_VERSION",
    "MFA_NAX_SPARSE_DENSITY_CEILING",
    "MFA_NO_PADDING",
    "MFA_PAD",
    "MFA_PRECONDITION",
    "MFA_REQUIRE_MSL4",
    "MFA_ROWS_PT",
    "MFA_SPLITK_MAX_N_D",
    "MFA_SUPPORTED_DTYPES",
    "MFA_SUPPORTED_HDIMS",
    "MFA_TD",
    "MFA_TD_128",
    "MFA_TD_64",
    "MFA_TD_HALF",
    "MFA_TGP",
    "MFA_TGP_128",
    "MFA_TGP_64",
    "MFA_TGP_SIZE",
    "MFA_TK",
    "MFA_TK_128",
    "MFA_TK_64",
    "MFA_TOPK_BISECT",
    "MFA_TOPK_STREAM_V5",
    "MFA_TQ",
    "MFA_TQ_128",
    "MFA_TQ_64",
    "MFA_V2_BD_HALF_D512",
    "MFA_V2_BQ64",
    "MFA_V2_FORCE_BK",
    "MFA_V2_FORCE_BK_D256",
    "MFA_V2_FORCE_BK_D512",
    "MFA_V2_FORCE_BQ_D512",
    "MFA_V34BWD",
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
    "MFA_V5_FORCE_",
    "MFA_V5_FORCE_BD_TILE",
    "MFA_V5_FORCE_BK",
    "MFA_V5_FORCE_BQ",
    "MFA_V5_FORCE_WM",
    "MFA_V6",
    "MFA_V6BWD",
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
    "MFA_V6_BHND",
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
    "MFA_V6_MATMUL_EXEC_SG",
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
    "MFA_WM",
    "MFA_WM_128",
    "MFA_WM_64",
    "MFA_WN",
    "MLX_MFA_DISPATCH_TABLE",
    "MLX_MFA_HOOK_TELEMETRY",
    "MLX_MFA_V6_NAX_NAATTENTIONKERNELDESCRIPTOR_HPP",
    "MLX_MFA_V6_NAX_NAATTENTIONKERNEL_HPP",
    "MLX_MFA_VERBOSE_DISPATCH",
})

# C++-side knobs (read in csrc/, documented here for completeness; not validated
# by this module since they are read in the extension, not Python).
CPP_KNOBS: frozenset[str] = frozenset({
    # (none)
})


# Runtime-templated knob families: the live name is built from params
# (e.g. MFA_SPLITK_MAX_N_D128_C1_A0_W0). A bare-name match can't catch these,
# so strict validation allows any var starting with one of these prefixes.
PREFIX_KNOBS: tuple[str, ...] = (
    "MFA_SPLITK_MAX_N_D",   # templated by D / causal / alibi / window
    "MFA_V3_FORCE_BK_",     # templated per-config force
    "MFA_V5_FORCE_",
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
    unknown = sorted(
        k for k in os.environ
        if (k.startswith("MFA_") or k.startswith("MLX_MFA_"))
        and k not in known
        and not any(k.startswith(p) for p in PREFIX_KNOBS)
    )
    for k in unknown:
        warnings.warn(
            f"[mlx-mfa] unrecognized knob {k!r} (not in the registry) — "
            f"possible typo; it will have no effect.", RuntimeWarning, stacklevel=2)
    return unknown

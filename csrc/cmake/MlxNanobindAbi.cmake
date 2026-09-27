# MLX version -> nanobind GIT_TAG (ABI-matched), for mlx_mfa._ext.
#
# _ext shares MLX's mlx::core::array type via nanobind NB_DOMAIN "mlx". The capsule
# key embeds NB_INTERNALS_VERSION (nanobind src/nb_abi.h), so _ext MUST FetchContent
# the SAME nanobind tag MLX itself was built with. A mismatch silently breaks
# cross-extension type sharing: every array-taking _ext function raises TypeError at
# runtime (the 2.62.0 / MLX-0.32.0 consumer-install incident).
#
# Each entry is "<mlx release>=<nanobind tag>=<NB_INTERNALS>", VERIFIED at source from
# that MLX tag's own CMakeLists.txt FetchContent_Declare(nanobind ... GIT_TAG) and the
# nanobind tag's src/nb_abi.h (last verified 2026-09-27):
#   MLX 0.31.2 -> nanobind v2.12.0 (NB_INTERNALS 19)
#   MLX 0.32.0 -> nanobind v2.13.0 (NB_INTERNALS 20)
#   MLX 0.32.1 -> nanobind v2.13.0 (NB_INTERNALS 20)
#   MLX 0.32.2 -> nanobind v2.15.0 (NB_INTERNALS 21)   # NOT 0.32.0's tag — never extrapolate
#
# Keep pyproject.toml's mlx specifier (BOTH [build-system].requires and
# [project].dependencies) capped at the top of this table: pip's build isolation
# resolves the LATEST allowed MLX, so an uncapped specifier breaks `pip install
# mlx-mfa` for everyone the day MLX ships an unmapped release (BLD-01, 2.62.1).
# Locked by tests/test_mlx_nanobind_abi_mapping.py.
#
# To add MLX X.Y.Z: read the nanobind GIT_TAG in MLX's CMakeLists.txt at its vX.Y.Z
# tag, confirm NB_INTERNALS_VERSION in nanobind src/nb_abi.h at that tag, add the
# entry, raise the pyproject cap to X.Y.Z, update the test's EXPECTED table, then
# re-run the consumer compile-install + M5/NAX gate on that MLX.

set(MFA_MLX_NANOBIND_ABI_TABLE
  "0.31.2=v2.12.0=19"
  "0.32.0=v2.13.0=20"
  "0.32.1=v2.13.0=20"
  "0.32.2=v2.15.0=21"
)

# mfa_resolve_nanobind_tag(<mlx_version> <out_var>)
# Sets <out_var> to the verified nanobind tag, or FATAL_ERRORs. Matching is an EXACT
# string compare on a plain X.Y.Z release: dev/rc/local builds (0.32.0.dev…,
# 0.31.2rc1, +local) are refused instead of being mapped onto a release tag (BLD-10).
function(mfa_resolve_nanobind_tag mlx_version out_var)
  set(_known "")
  foreach(_entry IN LISTS MFA_MLX_NANOBIND_ABI_TABLE)
    string(REPLACE "=" ";" _parts "${_entry}")
    list(GET _parts 0 _v)
    list(GET _parts 1 _t)
    list(GET _parts 2 _nb)
    string(APPEND _known "\n    ${_v} -> nanobind ${_t} [NB_INTERNALS ${_nb}]")
  endforeach()
  set(_how
    "  Refusing to guess a nanobind tag: a mismatched NB_INTERNALS silently breaks "
    "mlx::core::array type-sharing, so every native _ext call raises TypeError at runtime.\n"
    "  To support a new MLX: read the nanobind GIT_TAG in MLX's CMakeLists.txt at that tag, "
    "confirm NB_INTERNALS in nanobind src/nb_abi.h, add it to csrc/cmake/MlxNanobindAbi.cmake, "
    "raise the pyproject mlx cap, and re-run the consumer compile-install + M5/NAX gate.")
  string(CONCAT _how ${_how})
  if(NOT mlx_version MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR
      "mlx-mfa: MLX '${mlx_version}' is not a plain X.Y.Z release (dev/rc/local/malformed "
      "builds are not ABI-verifiable). Verified releases:${_known}\n${_how}")
  endif()
  foreach(_entry IN LISTS MFA_MLX_NANOBIND_ABI_TABLE)
    string(REPLACE "=" ";" _parts "${_entry}")
    list(GET _parts 0 _v)
    list(GET _parts 1 _t)
    if(mlx_version STREQUAL _v)
      set(${out_var} "${_t}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR
    "mlx-mfa: MLX ${mlx_version} is not in the nanobind-ABI mapping. Verified "
    "releases:${_known}\n${_how}")
endfunction()

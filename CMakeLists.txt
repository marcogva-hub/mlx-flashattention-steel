cmake_minimum_required(VERSION 3.24)
project(mlx_mfa LANGUAGES CXX OBJCXX)

# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
option(MLX_MFA_BUILD_TESTS "Build C++ tests" OFF)

# --------------------------------------------------------------------------- #
# Global settings
# --------------------------------------------------------------------------- #
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------------- #
# macOS deployment floor (audit Phase G-pre, 2026-06-18)
# --------------------------------------------------------------------------- #
# The portability floor is macOS 14.0 — MLX's minimum.  The V6/NAX Metal 4 /
# macOS-26 features are RUNTIME capabilities (gated by the M5/macOS check, e.g.
# `@available(macOS 26.1, *)` in csrc/v6_nax_detect.mm) and by runtime-compiled
# JIT shader source — NOT a build-time dependency.  Verified: a clean control
# build at target 14.0 against the macOS-26 SDK compiles with
# `-Werror=unguarded-availability-new` (no unguarded 26-only host API).  Set the
# floor explicitly so the sdist's compile-at-install always targets 14.0
# regardless of the build machine's macOS; override via -DCMAKE_OSX_DEPLOYMENT_TARGET
# or the MACOSX_DEPLOYMENT_TARGET env var.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
  set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target" FORCE)
endif()

# --------------------------------------------------------------------------- #
# Platform check
# --------------------------------------------------------------------------- #
if(NOT APPLE)
  message(FATAL_ERROR "mlx-mfa requires macOS with Apple Silicon")
endif()
# M-09 FIX (audit, 2026-06-21): assert ARM64, not just APPLE — the Metal/NAX
# kernels are arm64-only; an x86_64 macOS build would compile then fail at
# runtime.  CMAKE_OSX_ARCHITECTURES (if set) takes precedence over the host
# processor.  Allow universal2 (contains arm64).
set(_MFA_ARCH "${CMAKE_OSX_ARCHITECTURES}")
if(NOT _MFA_ARCH)
  set(_MFA_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()
if(NOT _MFA_ARCH MATCHES "arm64|aarch64")
  message(FATAL_ERROR
    "mlx-mfa requires Apple Silicon (arm64); detected architecture "
    "'${_MFA_ARCH}'. x86_64 is not supported.")
endif()

# M6 FIX (audit, 2026-06-21): the V6-NAX bring-up probes + the int8 microbench
# are dev-only diagnostics with ZERO callers — gated OFF by default so they do
# NOT ship in every _ext.  Enable for kernel bring-up with
# -DMFA_BUILD_PROBES=ON; the matching bindings in bindings.cpp are #ifdef'd on
# the MFA_BUILD_PROBES compile definition.
option(MFA_BUILD_PROBES "Compile the dev-only V6-NAX probe / microbench sources + bindings" OFF)

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #

# -- Python --
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# -- MLX (detect via Python — pip-installed MLX has no CMake config) --
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import mlx; print(mlx.__path__[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_PYTHON_DIR
  RESULT_VARIABLE MLX_PYTHON_RC
)
if(NOT MLX_PYTHON_RC EQUAL 0)
  message(FATAL_ERROR
    "Could not find MLX via Python. Install with: pip install mlx")
endif()

# MLX include path
set(MLX_INCLUDE_DIR "${MLX_PYTHON_DIR}/include")
if(NOT EXISTS "${MLX_INCLUDE_DIR}/mlx/mlx.h")
  message(FATAL_ERROR
    "MLX headers not found at ${MLX_INCLUDE_DIR}. "
    # M-09 FIX (audit, 2026-06-21): the real floor is 0.31.2 (nanobind v2.12.0 /
    # NB_INTERNALS v19 ABI — see pyproject [build-system]), not 0.18.0.
    "Ensure mlx >= 0.31.2 is installed.")
endif()
message(STATUS "MLX include: ${MLX_INCLUDE_DIR}")

# MLX shared library
find_library(MLX_LIB mlx PATHS "${MLX_PYTHON_DIR}/lib" NO_DEFAULT_PATH)
if(NOT MLX_LIB)
  find_library(MLX_LIB mlx PATHS "${MLX_PYTHON_DIR}" NO_DEFAULT_PATH)
endif()
if(NOT MLX_LIB)
  message(FATAL_ERROR "libmlx not found in ${MLX_PYTHON_DIR}")
endif()
message(STATUS "MLX library: ${MLX_LIB}")

# -- MLX runtime version (drives the nanobind ABI mapping below + MLX_BUILD_VERSION) --
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mlx.core; print(mlx.core.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_RUNTIME_VERSION
  RESULT_VARIABLE MLX_VERSION_RC
)
if(NOT MLX_VERSION_RC EQUAL 0)
  message(FATAL_ERROR
    "mlx-mfa: could not query mlx.core.__version__ — required to select the "
    "nanobind ABI that matches the installed MLX. Ensure mlx >= 0.31.2 is importable "
    "with '${Python_EXECUTABLE}'.")
endif()
message(STATUS "MLX build version: ${MLX_RUNTIME_VERSION}")

# -- nanobind via FetchContent, ABI-matched to the DETECTED MLX (D: auto-derive) --
# _ext shares MLX's mlx::core::array type via nanobind NB_DOMAIN "mlx". The capsule
# key embeds NB_INTERNALS_VERSION (nanobind src/nb_abi.h), so _ext MUST FetchContent
# the SAME nanobind tag MLX itself was built with. A mismatch silently breaks
# cross-extension type sharing: mlx::core::array passed from Python becomes
# "incompatible" to nanobind's type matcher and EVERY array-taking _ext function
# raises TypeError at runtime (the 2.62.0 / MLX-0.32.0 consumer-install incident —
# see devnotes/release_2_62_0_STOP.md).
#
# VERIFIED mapping (MLX version -> nanobind GIT_TAG -> NB_INTERNALS), each derived
# from that MLX tag's own CMakeLists FetchContent(nanobind) + nanobind src/nb_abi.h:
#   MLX 0.31.2 -> nanobind v2.12.0 (NB_INTERNALS v19)
#   MLX 0.32.0 -> nanobind v2.13.0 (NB_INTERNALS v20)   # v19->v20 = the 2.12->2.13 bump
# Declared support is mlx>=0.31.2; 0.31.2 and 0.32.0 are the ONLY releases that
# currently exist at/above that floor, so exactly those two are mapped. Any other
# MLX version FAILS LOUD below — we never guess a tag (a wrong guess reintroduces
# the silent break). To add a new MLX X: read the nanobind GIT_TAG in MLX's
# CMakeLists.txt at its vX tag, confirm its NB_INTERNALS in nanobind src/nb_abi.h,
# add an entry here, then re-run the consumer compile-install + M5/NAX gate on X.
if(MLX_RUNTIME_VERSION VERSION_EQUAL "0.31.2")
  set(NANOBIND_GIT_TAG "v2.12.0")   # NB_INTERNALS v19
elseif(MLX_RUNTIME_VERSION VERSION_EQUAL "0.32.0")
  set(NANOBIND_GIT_TAG "v2.13.0")   # NB_INTERNALS v20
else()
  message(FATAL_ERROR
    "mlx-mfa: MLX ${MLX_RUNTIME_VERSION} is not in the nanobind-ABI mapping "
    "(known: 0.31.2 -> nanobind v2.12.0 [NB_INTERNALS v19]; "
    "0.32.0 -> nanobind v2.13.0 [NB_INTERNALS v20]).\n"
    "  Refusing to guess a nanobind tag: a mismatched NB_INTERNALS silently breaks "
    "mlx::core::array type-sharing so every native _ext call raises TypeError at "
    "runtime.\n"
    "  To support MLX ${MLX_RUNTIME_VERSION}: (1) read the nanobind GIT_TAG in MLX's "
    "CMakeLists.txt at its v${MLX_RUNTIME_VERSION} tag, (2) confirm its NB_INTERNALS "
    "in nanobind src/nb_abi.h, (3) add an entry to this mapping in CMakeLists.txt, "
    "(4) re-run the consumer compile-install + M5/NAX gate on that MLX. "
    "See devnotes/release_2_62_0_STOP.md.")
endif()
message(STATUS "nanobind ABI: MLX ${MLX_RUNTIME_VERSION} -> nanobind ${NANOBIND_GIT_TAG}")

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        ${NANOBIND_GIT_TAG}
  GIT_SHALLOW    ON
)
FetchContent_MakeAvailable(nanobind)

# -- Metal & Foundation frameworks --
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# --------------------------------------------------------------------------- #
# C++ extension module: mlx_mfa._ext
# --------------------------------------------------------------------------- #
set(MFA_SOURCES
  csrc/bindings.cpp
  csrc/mfa_attention.cpp
  csrc/mfa_shader_gen.cpp
  csrc/mfa_steel_fwd.cpp
  csrc/mfa_steel_bwd.cpp
  csrc/mfa_paged_gather.cpp
  csrc/mfa_sage_fwd.cpp
  csrc/mfa_quantize.cpp
  csrc/mfa_scatter.cpp
  csrc/mfa_smooth_quant.cpp
  csrc/mfa_gna_fwd.cpp
  csrc/mfa_steel_fwd_v2.cpp
  csrc/mfa_steel_fwd_v3.cpp
  csrc/mfa_steel_paged_varlen_fwd.cpp
  csrc/mfa_steel_paged_varlen_tq_fwd.cpp
  csrc/v6_nax_detect.mm
  csrc/v6_nax_compile.mm
  csrc/metal41_probe.mm             # functional Metal-4.1 capability probe (default build)
  csrc/mfa_steel_fwd_v6_nax.cpp
  csrc/mfa_v6_nax_primitive.cpp
  csrc/mfa_conv_nax.cpp
  csrc/mfa_sparse_attention.cpp
  csrc/mfa_qmm_nax.cpp
  csrc/mfa_ffn_nax.cpp
  csrc/mfa_gna_nax.cpp
  csrc/mfa/v6_nax/NAAttentionKernelDescriptor.cpp
  csrc/mfa/v6_nax/NAAttentionKernel.cpp
  csrc/shader_cache.mm
  # ccv-derived Metal shader generator
  csrc/mfa/AttentionKernel.cpp
  csrc/mfa/AttentionKernelDescriptor.cpp
  csrc/mfa/GEMMHeaders.cpp
  csrc/mfa/CodeWriter.cpp
)

# M6 (audit 2026-06-21): dev-only probe/microbench sources — appended ONLY when
# -DMFA_BUILD_PROBES=ON (default OFF), with the MFA_BUILD_PROBES macro defined so
# the matching bindings in bindings.cpp compile in lockstep.
if(MFA_BUILD_PROBES)
  list(APPEND MFA_SOURCES
    csrc/v6_nax_toolchain_probe.cpp    # was v6_nax_probe.cpp (L4 rename): MSL4+MPP toolchain bring-up + DT-port compile
    csrc/v6_nax_primitives_probe.cpp   # was v6nax_probe.cpp  (L4 rename): NAX-direct (nax.h-inline) source-gen
    csrc/mpp_int8_bench.mm
  )
endif()

# NB_STATIC: statically links nanobind runtime into this extension.
# NB_DOMAIN "mlx": MUST match MLX's domain (mlx.core uses NB_DOMAIN "mlx").
#   Both extensions use the same capsule key in PyInterpreterState_GetDict(),
#   sharing a single type registry so mlx::core::array passes between them.
#   Requires nanobind pip version with NB_INTERNALS_VERSION matching MLX's
#   (e.g., nanobind==2.10.2 for MLX 0.31.0).
nanobind_add_module(_ext NB_STATIC NB_DOMAIN "mlx" ${MFA_SOURCES})

target_include_directories(_ext PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/mfa
  ${MLX_INCLUDE_DIR}
  ${MLX_INCLUDE_DIR}/metal_cpp   # metal-cpp: Metal/Metal.hpp
)

target_link_libraries(_ext PRIVATE
  ${MLX_LIB}
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
)

# MLX_RUNTIME_VERSION is queried once, above (it drives the nanobind ABI mapping),
# and reused here for the _check_abi runtime version compare.
target_compile_definitions(_ext PRIVATE
  # CC-20 (volet E): MLX_MFA_METAL_PATH removed — it baked the build machine's
  # absolute csrc/kernels path into _ext with ZERO consumers (real kernels are
  # JIT-generated). MLX_BUILD_VERSION is live (the _check_abi version compare).
  MLX_BUILD_VERSION="${MLX_RUNTIME_VERSION}"
)
# M6: define MFA_BUILD_PROBES so bindings.cpp registers the probe defs only when
# their sources are actually compiled in.
if(MFA_BUILD_PROBES)
  target_compile_definitions(_ext PRIVATE MFA_BUILD_PROBES)
endif()

# Install into the Python package directory
install(TARGETS _ext LIBRARY DESTINATION mlx_mfa)

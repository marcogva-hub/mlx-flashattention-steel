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
# Platform check
# --------------------------------------------------------------------------- #
if(NOT APPLE)
  message(FATAL_ERROR "mlx-mfa requires macOS with Apple Silicon")
endif()

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
    "Ensure mlx >= 0.18.0 is installed.")
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

# -- nanobind (pip-installed) --
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_DIR_RC
)
if(NB_DIR_RC EQUAL 0 AND EXISTS "${NB_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()
find_package(nanobind CONFIG REQUIRED)

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
  csrc/shader_cache.mm
  # ccv-derived Metal shader generator
  csrc/mfa/AttentionKernel.cpp
  csrc/mfa/AttentionKernelDescriptor.cpp
  csrc/mfa/GEMMHeaders.cpp
  csrc/mfa/CodeWriter.cpp
)

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

target_compile_definitions(_ext PRIVATE
  MLX_MFA_METAL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/csrc/kernels"
)

# Install into the Python package directory
install(TARGETS _ext LIBRARY DESTINATION mlx_mfa)

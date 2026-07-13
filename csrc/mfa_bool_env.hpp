/// Strict boolean environment parsing shared by C++ dispatch/generator code.

#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mlx_mfa {

inline bool parse_bool_env_value(const char* name, const char* value) {
  if (std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  throw std::invalid_argument(
      std::string("[mlx-mfa] boolean knob ") + name +
      " must be '0' or '1'; got '" + value + "'");
}

inline bool get_bool_env(const char* name, bool default_value = false) {
  const char* value = std::getenv(name);
  return value ? parse_bool_env_value(name, value) : default_value;
}

}  // namespace mlx_mfa

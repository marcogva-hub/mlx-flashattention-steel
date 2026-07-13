/// mfa_env_aliases.hpp — Centralized V34→V6 env-var alias table (v2.57.0).
///
/// The V34 token was the internal generator name for the V6 NAX kernel
/// (cooperative-tensor matmul2d). v2.57.0 unified the nomenclature to V6;
/// see NAMING.md for the full provenance.
///
/// The NEW MFA_V6* name is canonical. The OLD MFA_*V34* name is a
/// DEPRECATED alias: still honored (so existing scripts keep working) but
/// it emits a one-shot DeprecationWarning per process. The aliases are
/// scheduled for removal in v3.0.0.
///
/// Rename rule: V34 -> V6, EXCEPT where the name already contained V6
/// (a collision), in which case V34 -> NAX to keep the new name
/// unambiguous (e.g. MFA_V6_USE_V34 -> MFA_V6_USE_NAX, not _V6).
///
/// Resolution order in getenv_aliased(new_name):
///   1. If MFA_V6* (new) is set -> use it, no warning.
///   2. Else if MFA_*V34* (old) is set -> use it, warn once, point to new.
///   3. Else -> nullptr.
/// New-takes-precedence makes a both-set state deterministic and quiet.

#pragma once
#include "mfa_bool_env.hpp"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace mlx_mfa {

/// new (canonical) -> old (deprecated alias). Single source of truth for
/// the C++ side; the Python side mirrors this in mlx_mfa/_env_aliases.py.
inline const std::unordered_map<std::string, std::string>& v6_env_alias_map() {
  static const std::unordered_map<std::string, std::string> m = {
      // V34 -> V6 (no collision)
      {"MFA_ENABLE_V6_BACKWARD", "MFA_ENABLE_V34_BACKWARD"},
      {"MFA_DISABLE_V6_BACKWARD", "MFA_DISABLE_V34_BACKWARD"},
      {"MFA_ENABLE_V6_D128", "MFA_ENABLE_V34_D128"},
      {"MFA_V6_BWD_KERNEL", "MFA_V34_BWD_KERNEL"},
      {"MFA_V6_BWD_SPARSE_NATIVE", "MFA_V34_BWD_SPARSE_NATIVE"},
      {"MFA_V6_DUMP_SOURCE", "MFA_V34_DUMP_SOURCE"},
      {"MFA_V6BWD", "MFA_V34BWD"},
      {"MFA_V6BWD_BK", "MFA_V34BWD_BK"},
      {"MFA_V6BWD_BQ", "MFA_V34BWD_BQ"},
      {"MFA_V6BWD_WM", "MFA_V34BWD_WM"},
      {"MFA_V6BWD_USE_FUSED", "MFA_V34BWD_USE_FUSED"},
      {"MFA_V6BWD_DUMP_SOURCE", "MFA_V34BWD_DUMP_SOURCE"},
      {"MFA_V6BWDF_BK", "MFA_V34BWDF_BK"},
      {"MFA_V6BWDF_BQ", "MFA_V34BWDF_BQ"},
      {"MFA_V6BWDF_WM", "MFA_V34BWDF_WM"},
      {"MFA_V6BWDF_DUMP_PATH", "MFA_V34BWDF_DUMP_PATH"},
      {"MFA_V6BWDF_DUMP_SOURCE", "MFA_V34BWDF_DUMP_SOURCE"},
      {"MFA_V6BWDK_BK", "MFA_V34BWDK_BK"},
      {"MFA_V6BWDK_BQ", "MFA_V34BWDK_BQ"},
      {"MFA_V6BWDK_WM", "MFA_V34BWDK_WM"},
      {"MFA_V6BWDV_BK", "MFA_V34BWDV_BK"},
      {"MFA_V6BWDV_BQ", "MFA_V34BWDV_BQ"},
      {"MFA_V6BWDV_WM", "MFA_V34BWDV_WM"},
      {"MFA_V6BWDKV_BK", "MFA_V34BWDKV_BK"},
      {"MFA_V6BWDKV_BQ", "MFA_V34BWDKV_BQ"},
      {"MFA_V6BWDKV_WM", "MFA_V34BWDKV_WM"},
      // collisions (name already had V6) -> NAX
      {"MFA_V6_USE_NAX", "MFA_V6_USE_V34"},
      {"MFA_V6_NAX_BK", "MFA_V6_V34_BK"},
      {"MFA_V6_NAX_BQ", "MFA_V6_V34_BQ"},
      {"MFA_V6_NAX_WM", "MFA_V6_V34_WM"},
  };
  return m;
}

/// Emit a DeprecationWarning to stderr exactly once per (old_name, process).
inline void warn_deprecated_env_once(const char* old_name, const char* new_name) {
  static std::mutex mu;
  static std::unordered_set<std::string> warned;
  std::lock_guard<std::mutex> lk(mu);
  if (warned.insert(old_name).second) {
    std::fprintf(stderr,
                 "[mlx-mfa] DeprecationWarning: env var %s is deprecated "
                 "(renamed to %s in v2.57.0; alias removed in v3.0.0). "
                 "See NAMING.md.\n",
                 old_name, new_name);
  }
}

/// std::getenv with deprecated-alias fallback. Pass the NEW canonical name.
/// Returns the new var if set; else the deprecated alias (warning once);
/// else nullptr.
inline const char* getenv_aliased(const char* new_name) {
  if (const char* v = std::getenv(new_name)) return v;
  const auto& m = v6_env_alias_map();
  auto it = m.find(new_name);
  if (it != m.end()) {
    if (const char* v = std::getenv(it->second.c_str())) {
      warn_deprecated_env_once(it->second.c_str(), new_name);
      return v;
    }
  }
  return nullptr;
}

inline bool get_bool_env_aliased(const char* new_name,
                                 bool default_value = false) {
  const char* value = getenv_aliased(new_name);
  return value ? parse_bool_env_value(new_name, value) : default_value;
}

}  // namespace mlx_mfa

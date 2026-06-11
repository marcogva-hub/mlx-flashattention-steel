/// mfa_env.hpp — Cached environment variable configuration.
///
/// Reads all MFA_* env vars once at first access (thread-safe via
/// Meyers' static singleton). Replaces per-dispatch std::getenv()
/// calls in the hot path (eval_gpu and block config selection).
///
/// Shader-generator env vars (MFA_NO_PADDING, MFA_IR_INVESTIGATE,
/// MFA_DISABLE_ASYNC) are NOT cached here — they run only during
/// shader compilation (cold path) and are acceptable as-is.

#pragma once
#include <cstdlib>
#include <mutex>

namespace mlx_mfa {

struct MFAEnvConfig {
  // ── Dispatch gates ─────────────────────────────────────────────
  // NOT cached — tests use os.environ patching (setenv) which must
  // take effect immediately. These are cheap single-reads per eval_gpu().
  static bool enable_v3()    { return env_bool("MFA_ENABLE_V3"); }
  static bool enable_v4()    { return env_bool("MFA_ENABLE_V4"); }
  static bool enable_v5()    { return env_bool("MFA_ENABLE_V5"); }
  static bool disable_v2()   { return env_bool("MFA_DISABLE_V2"); }
  static bool disable_v3()   { return env_bool("MFA_DISABLE_V3"); }
  static bool force_v2()     { return env_bool("MFA_FORCE_V2"); }
  static int  force_splitk() { return env_tristate("MFA_FORCE_SPLITK"); }

  // Repo review 2026-05: MFA_NO_PADDING changes the GENERATED Metal source
  // (smem padding expression) but is absent from the shader-cache KernelKey.
  // A mid-process toggle therefore returned a stale kernel compiled with the
  // other padding mode.  Freezing the value at first read makes the env var
  // load-time-only: consistent kernels and keys for the whole process.
  static bool no_padding() {
    static const bool v = env_bool("MFA_NO_PADDING");
    return v;
  }

  // ── Architecture override (int, 0 = use hardware detection) ────
  int force_gen;        // MFA_FORCE_GEN — override GPU architecture gen

  // ── V2 config overrides ────────────────────────────────────────
  int v2_force_bk;      // MFA_V2_FORCE_BK      (0 = auto)
  bool v2_bq64;         // MFA_V2_BQ64           (false = BQ=32)
  int v2_force_bk_d256; // MFA_V2_FORCE_BK_D256  (0 = auto)
  int v2_force_bk_d512; // MFA_V2_FORCE_BK_D512  (0 = auto)
  int v2_force_bq_d512; // MFA_V2_FORCE_BQ_D512  (0 = auto)
  int v2_bd_half_d512;  // MFA_V2_BD_HALF_D512   (0 = auto)

  // ── V3 config overrides ────────────────────────────────────────
  int v3_force_bk_d64;  // MFA_V3_FORCE_BK_D64   (0 = auto)
  int v3_force_bk_d128; // MFA_V3_FORCE_BK_D128  (0 = auto)

  // ── V5 config overrides ────────────────────────────────────────
  int v5_force_bk;      // MFA_V5_FORCE_BK       (0 = auto)
  int v5_force_bd_tile; // MFA_V5_FORCE_BD_TILE   (0 = auto)
  int v5_force_bq;      // MFA_V5_FORCE_BQ       (0 = auto)
  int v5_force_wm;      // MFA_V5_FORCE_WM       (0 = auto)

  /// Thread-safe singleton — initialized on first call (Meyers' singleton).
  static const MFAEnvConfig& get() {
    static MFAEnvConfig instance = load();
    return instance;
  }

  /// Re-read all env vars. Call after setenv() in tests/benchmarks.
  /// NOT thread-safe — only call from single-threaded test/bench code.
  static void invalidate() {
    // Bypass const to reload — safe because callers guarantee single-thread.
    const_cast<MFAEnvConfig&>(get()) = load();
  }

private:
  static bool env_bool(const char* name) {
    return std::getenv(name) != nullptr;
  }

  /// Tristate: -1 (unset), 0 ("0"), 1 ("1" or any other value).
  static int env_tristate(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return -1;
    if (v[0] == '0' && v[1] == '\0') return 0;
    return 1;
  }

  static int env_int(const char* name, int def = 0) {
    if (const char* v = std::getenv(name)) {
      const int p = std::atoi(v);
      if (p > 0) return p;
    }
    return def;
  }

  static MFAEnvConfig load() {
    MFAEnvConfig c{};
    // Dispatch gates: live-read via static methods (not cached).
    // Architecture override
    c.force_gen    = env_int("MFA_FORCE_GEN");
    // V2 config overrides
    c.v2_force_bk      = env_int("MFA_V2_FORCE_BK");
    c.v2_bq64          = env_bool("MFA_V2_BQ64");
    c.v2_force_bk_d256 = env_int("MFA_V2_FORCE_BK_D256");
    c.v2_force_bk_d512 = env_int("MFA_V2_FORCE_BK_D512");
    c.v2_force_bq_d512 = env_int("MFA_V2_FORCE_BQ_D512");
    c.v2_bd_half_d512  = env_int("MFA_V2_BD_HALF_D512");
    // V3 config overrides
    c.v3_force_bk_d64  = env_int("MFA_V3_FORCE_BK_D64");
    c.v3_force_bk_d128 = env_int("MFA_V3_FORCE_BK_D128");
    // V5 config overrides
    c.v5_force_bk      = env_int("MFA_V5_FORCE_BK");
    c.v5_force_bd_tile = env_int("MFA_V5_FORCE_BD_TILE");
    c.v5_force_bq      = env_int("MFA_V5_FORCE_BQ");
    c.v5_force_wm      = env_int("MFA_V5_FORCE_WM");
    return c;
  }
};

}  // namespace mlx_mfa

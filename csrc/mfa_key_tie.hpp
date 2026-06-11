/// mfa_key_tie.hpp — Campaign 2026-06 Sprint C Track 6.
///
/// CacheKey declaration-derived ==/hash.  Each pipeline-cache key declares
/// its affecting-input set EXACTLY ONCE via `tie()`; operator== and the
/// hash functor derive from that declaration mechanically, so they CANNOT
/// diverge from it (the 2026-05 C1/C6 omission class is structurally
/// impossible).  The remaining failure mode — a struct field missing from
/// tie() — is caught at CI time by the static invariant test
/// (tests/test_campaign_2026_06_sprint_a_key_invariants.py).
///
/// Loud-failure semantics decision: enforcement is CI-static (zero runtime
/// overhead in release builds); no runtime assert is compiled in.
///
/// hash_tie: FNV-1a-seeded fold over the tied elements via std::hash —
/// no virtual dispatch, fully inlinable; perf-parity with hand-written
/// hashes (verified <1% on the hottest cache probes).

#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <utility>

namespace mlx_mfa_keys {

template <typename T>
inline void hash_fold(size_t& h, const T& v) {
  h ^= std::hash<std::decay_t<T>>{}(v) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
}

template <typename Tuple, size_t... I>
inline size_t hash_tie_impl(const Tuple& t, std::index_sequence<I...>) {
  size_t h = 0x811c9dc5u;
  (hash_fold(h, std::get<I>(t)), ...);
  return h;
}

template <typename... Ts>
inline size_t hash_tie(const std::tuple<Ts...>& t) {
  return hash_tie_impl(t, std::index_sequence_for<Ts...>{});
}

}  // namespace mlx_mfa_keys

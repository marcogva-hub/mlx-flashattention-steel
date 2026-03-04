/// mfa_compat.h  –  replaces ccv_nnc_mfa.hpp / ccv_nnc_mfa_error.hpp /
///                   ccv_nnc_mfa_hash.hpp for the standalone mlx-mfa build.
///
/// Keep this header pure C++ (no metal-cpp, no ObjC) so it can be compiled
/// by any translation unit in csrc/mfa/.

#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <simd/simd.h>

// ---------------------------------------------------------------------------
// Precondition / error macros  (ccv_nnc_mfa_error.hpp replacement)
// ---------------------------------------------------------------------------

#define CCV_NNC_MFA_PRECONDITION(x) assert(x)
#define CCV_NNC_MFA_CHECK_ERROR(err) (void)(err)

// ---------------------------------------------------------------------------
// Hash utilities  (ccv_nnc_mfa_hash.hpp replacement)
// ---------------------------------------------------------------------------

namespace {

template<typename T>
T xorshift(const T& n, int i) { return n ^ (n >> i); }

inline uint32_t distribute_32(const uint32_t& n) {
    uint32_t p = 0x55555555ul;
    uint32_t c = 3423571495ul;
    return c * xorshift(p * xorshift(n, 16), 16);
}

inline uint64_t distribute_64(const uint64_t& n) {
    uint64_t p = 0x5555555555555555ull;
    uint64_t c = 17316035218449499591ull;
    return c * xorshift(p * xorshift(n, 32), 32);
}

template<typename T, typename S>
typename std::enable_if<std::is_unsigned<T>::value, T>::type
constexpr rotl(const T n, const S i) {
    const T m = std::numeric_limits<T>::digits - 1;
    const T c = i & m;
    return (n << c) | (n >> ((T(0) - c) & m));
}

} // anonymous namespace

namespace ccv {
namespace nnc {
namespace mfa {
namespace hash {

inline size_t combine_32(std::size_t& seed, const uint32_t& v) {
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute_32(v);
}

inline size_t combine_64(std::size_t& seed, const uint64_t& v) {
    return rotl(seed, std::numeric_limits<size_t>::digits / 3) ^ distribute_64(v);
}

inline uint32_t pack_32(const simd::uchar4& v) {
    return reinterpret_cast<const uint32_t&>(v);
}

inline uint32_t pack_32(const simd::ushort2& v) {
    return reinterpret_cast<const uint32_t&>(v);
}

inline uint64_t pack_64(const simd::ushort4& v) {
    return reinterpret_cast<const uint64_t&>(v);
}

} // namespace hash
} // namespace mfa
} // namespace nnc
} // namespace ccv

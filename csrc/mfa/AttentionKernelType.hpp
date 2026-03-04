/// AttentionKernelType.hpp — Enum for the three ccv-path attention kernel variants.
///
/// loopForward:         standard forward pass (Q × K^T → softmax → P × V)
/// loopBackwardQuery:   backward dQ loop (parallelizes over Q rows)
/// loopBackwardKeyValue: backward dK/dV loop (parallelizes over K/V cols)
///
/// Note: mlx-mfa uses the Python/SDPA backward for gradients (not these loops).
/// The enum is retained for completeness with the ccv code path (f32 forward).

#ifndef AttentionKernelType_hpp
#define AttentionKernelType_hpp

#include <stdint.h>
#include <string>

class AttentionKernelType {
  // Hijack some C++ syntax, making it look like Swift's enumerations with
  // member functions.
  //
  // Source: https://stackoverflow.com/a/53284026
public:
  enum Value: uint16_t {
    forward = 0,
    backwardQuery = 1,
    backwardKeyValue = 2,
  };

  AttentionKernelType() = default;
  constexpr AttentionKernelType(Value aKernelType) : value(aKernelType) { }

  explicit operator bool() const = delete;

  constexpr bool operator==(const AttentionKernelType &rhs) const { return value == rhs.value; }
  constexpr bool operator!=(const AttentionKernelType &rhs) const { return value != rhs.value; }

  std::string name() const noexcept {
    switch (value) {
      case forward:
        return "forward";
      case backwardQuery:
        return "backwardQuery";
      case backwardKeyValue:
        return "backwardKeyValue";
    }
  }

  Value value;
};

#endif

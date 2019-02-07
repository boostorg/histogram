// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_OPTION_HPP
#define BOOST_HISTOGRAM_AXIS_OPTION_HPP

namespace boost {
namespace histogram {
namespace axis {

/**
  Options for builtin axis types.

  Options should be combined with `operator|`.
*/
enum class option {
  none = 0,         ///< all options are off.
  underflow = 0b1,  ///< axis has underflow bin.
  overflow = 0b10,  ///< axis has overflow bin.
  circular = 0b100, ///< axis is circular, mutually exclusive with underflow.
  growth = 0b1000,  ///< axis can grow, mutually exclusive with circular.
  use_default = static_cast<int>(underflow) | static_cast<int>(overflow),
};

/// Invert options.
constexpr inline option operator~(option a) {
  return static_cast<option>(~static_cast<int>(a));
}

/// Logical AND of options.
constexpr inline option operator&(option a, option b) {
  return static_cast<option>(static_cast<int>(a) & static_cast<int>(b));
}

/// Logical OR or options.
constexpr inline option operator|(option a, option b) {
  return static_cast<option>(static_cast<int>(a) | static_cast<int>(b));
}

/// Test whether the bits in b are also set in a.
constexpr inline bool test(option a, option b) {
  return static_cast<int>(a) & static_cast<int>(b);
}

/// Logical OR of options and corrects for mutually exclusive options.
constexpr inline option join(option a, option b) {
  // circular turns off underflow and vice versa
  a = a | b;
  if (test(b, option::underflow)) a = a & ~option::circular;
  if (test(b, option::circular)) a = a & ~option::underflow;
  return a;
}

} // namespace axis
} // namespace histogram
} // namespace boost

#endif

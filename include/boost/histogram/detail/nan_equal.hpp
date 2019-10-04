// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_NAN_EQUAL_HPP
#define BOOST_HISTOGRAM_DETAIL_NAN_EQUAL_HPP

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
bool nan_equal(const T a, const T b) {
  return std::is_floating_point<T>::value ? (std::isnan(a) && std::isnan(b)) || a == b
                                          : a == b;
}

// faster specialization for float, compare bits
inline bool nan_equal(const float a, const float b) {
  static_assert(sizeof(float) == 4, "sizeof(float) expected to be 4 bytes");
  std::int32_t ia, ib;
  std::memcpy(&ia, &a, 4);
  std::memcpy(&ib, &b, 4);
  return ia == ib;
}

// faster specialization for double, compare bits
inline bool nan_equal(const double a, const double b) {
  static_assert(sizeof(double) == 8, "sizeof(double) expected to be 8 bytes");
  std::int64_t ia, ib;
  std::memcpy(&ia, &a, 8);
  std::memcpy(&ib, &b, 8);
  return ia == ib;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif
// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <boost/assert.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace detail {

// the following is highly optimized code that runs in a hot loop;
// please measure the performance impact of changes
inline void lin(std::size_t& out, std::size_t& stride, const int axis_size,
                const int axis_shape, int j) noexcept {
  BOOST_ASSERT_MSG(stride == 0 || (-1 <= j && j <= axis_size),
                   "index must be in bounds for this algorithm");
  j += (j < 0) * (axis_size + 2); // wrap around if j < 0
  out += j * stride;
  stride *=
      (j < axis_shape) * axis_shape; // stride == 0 indicates out-of-range
}

template <typename T>
typename std::enable_if<(is_castable_to_int<T>::value), int>::type
indirect_int_cast(T&& t) noexcept {
  return static_cast<int>(std::forward<T>(t));
}

template <typename T>
typename std::enable_if<!(is_castable_to_int<T>::value), int>::type
indirect_int_cast(T&&) noexcept {
  // Cannot use static_assert here, because this function is created as a
  // side-effect of TMP. It must be valid at compile-time.
  BOOST_ASSERT_MSG(false, "bin argument not convertible to int");
  return 0;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

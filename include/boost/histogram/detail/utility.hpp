// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_
#define _BOOST_HISTOGRAM_DETAIL_UTILITY_HPP_

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/config.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/type_index.hpp>
#include <boost/utility/string_view.hpp>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace boost {
namespace histogram {
namespace detail {

// two_pi can be found in boost/math, but it is defined here to reduce deps
constexpr double two_pi = 6.283185307179586;

inline void escape(std::ostream& os, const string_view s) {
  os << '\'';
  for (auto sit = s.begin(); sit != s.end(); ++sit) {
    if (*sit == '\'' && (sit == s.begin() || *(sit - 1) != '\\')) {
      os << "\\\'";
    } else {
      os << *sit;
    }
  }
  os << '\'';
}

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

template <typename S, typename T>
void fill_storage(S& s, std::size_t idx, weight<T>&& w) {
  s.add(idx, w);
}

template <typename S>
void fill_storage(S& s, std::size_t idx) {
  s.increase(idx);
}

template <typename S>
auto storage_get(const S& s, std::size_t idx, bool error) ->
    typename S::const_reference {
  if (error) throw std::out_of_range("bin index out of range");
  return s[idx];
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

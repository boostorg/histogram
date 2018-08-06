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

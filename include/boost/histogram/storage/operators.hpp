// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_STORAGE_OPERATORS_HPP_

#include <boost/histogram/detail/meta.hpp>

namespace boost {
namespace histogram {
namespace detail {
template <typename S1, typename S2>
bool equal_impl(std::true_type, std::true_type, const S1 &s1, const S2 &s2) {
  for (std::size_t i = 0, n = s1.size(); i < n; ++i) {
    auto x1 = s1[i];
    auto x2 = s2[i];
    if (x1.value() != x2.value() || x1.variance() != x2.variance())
      return false;
  }
  return true;
}

template <typename S1, typename S2>
bool equal_impl(std::true_type, std::false_type, const S1 &s1, const S2 &s2) {
  for (std::size_t i = 0, n = s1.size(); i < n; ++i) {
    auto x1 = s1[i];
    auto x2 = s2[i];
    if (x1.value() != x2 || x1.value() != x1.variance())
      return false;
  }
  return true;
}

template <typename S1, typename S2>
bool equal_impl(std::false_type, std::true_type, const S1 &s1, const S2 &s2) {
  return equal_impl(std::true_type(), std::false_type(), s2, s1);
}

template <typename S1, typename S2>
bool equal_impl(std::false_type, std::false_type, const S1 &s1, const S2 &s2) {
  for (std::size_t i = 0, n = s1.size(); i < n; ++i)
    if (s1[i] != s2[i])
      return false;
  return true;
}
} // namespace detail

template <typename S1, typename S2, typename = detail::requires_storage<S1>,
          typename = detail::requires_storage<S2>>
bool operator==(const S1 &s1, const S2 &s2) noexcept {
  if (s1.size() != s2.size())
    return false;
  return detail::equal_impl(detail::has_variance_support_t<S1>(),
                            detail::has_variance_support_t<S2>(), s1, s2);
}

template <typename S1, typename S2, typename = detail::requires_storage<S1>,
          typename = detail::requires_storage<S2>>
bool operator!=(const S1 &s1, const S2 &s2) noexcept {
  return !operator==(s1, s2);
}

} // namespace histogram
} // namespace boost

#endif

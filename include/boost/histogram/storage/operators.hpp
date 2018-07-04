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

template <typename S1, typename S2, typename = detail::requires_storage<S1>,
          typename = detail::requires_storage<S2>>
bool operator==(const S1 &s1, const S2 &s2) noexcept {
  if (s1.size() != s2.size())
    return false;
  for (std::size_t i = 0, n = s1.size(); i < n; ++i)
    if (!(s1[i] == s2[i]))
      return false;
  return true;
}

template <typename S1, typename S2, typename = detail::requires_storage<S1>,
          typename = detail::requires_storage<S2>>
bool operator!=(const S1 &s1, const S2 &s2) noexcept {
  return !operator==(s1, s2);
}

} // namespace histogram
} // namespace boost

#endif

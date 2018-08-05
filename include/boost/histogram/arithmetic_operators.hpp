// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_ARITHMETIC_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_ARITHMETIC_OPERATORS_HPP_

#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {

template <typename A, typename S>
histogram<A, S>&& operator+(histogram<A, S>&& a, const histogram<A, S>& b) {
  a += b;
  return std::move(a);
}

template <typename A, typename S>
histogram<A, S>&& operator+(histogram<A, S>&& a, histogram<A, S>&& b) {
  a += b;
  return std::move(a);
}

template <typename A, typename S>
histogram<A, S>&& operator+(const histogram<A, S>& a, histogram<A, S>&& b) {
  b += a;
  return std::move(b);
}

template <typename A, typename S>
histogram<A, S> operator+(const histogram<A, S>& a,
                          const histogram<A, S>& b) {
  histogram<A, S> r(a);
  r += b;
  return r;
}

template <typename A, typename S>
histogram<A, S>&& operator*(histogram<A, S>&& a, const double x) {
  a *= x;
  return std::move(a);
}

template <typename A, typename S>
histogram<A, S>&& operator*(const double x, histogram<A, S>&& b) {
  b *= x;
  return std::move(b);
}

template <typename A, typename S>
histogram<A, S> operator*(const histogram<A, S>& a, const double x) {
  auto r = a;
  r *= x;
  return r;
}

template <typename A, typename S>
histogram<A, S> operator*(const double x, const histogram<A, S>& b) {
  auto r = b;
  r *= x;
  return r;
}

template <typename A, typename S>
histogram<A, S>&& operator/(histogram<A, S>&& a, const double x) {
  a /= x;
  return std::move(a);
}

template <typename A, typename S>
histogram<A, S> operator/(const histogram<A, S>& a, const double x) {
  auto r = a;
  r /= x;
  return r;
}

} // namespace histogram
} // namespace boost

#endif

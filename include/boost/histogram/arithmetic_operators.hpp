// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_HISTOGRAM_ARITHMETIC_OPERATORS_HPP_
#define _BOOST_HISTOGRAM_HISTOGRAM_ARITHMETIC_OPERATORS_HPP_

#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator+(H<A, S> &&a,
          const H<A, S> &b) {
  a += b;
  return std::move(a);
}

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator+(H<A, S> &&a,
          H<A, S> &&b) {
  a += b;
  return std::move(a);
}

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator+(const H<A, S> &a,
          H<A, S> &&b) {
  b += a;
  return std::move(b);
}

template <template <class, class> class H, typename A, typename S>
H<A, S>
operator+(const H<A, S> &a,
          const H<A, S> &b) {
  H<A, S> r(a);
  r += b;
  return r;
}

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator*(H<A, S> &&a, const double x) {
  a *= x;
  return std::move(a);
}

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator*(const double x, H<A, S> &&b) {
  b *= x;
  return std::move(b);
}

template <template <class, class> class H, typename A, typename S>
H<A, S>
operator*(const H<A, S> &a, const double x) {
  H<A, S> r(a);
  r *= x;
  return r;
}

template <template <class, class> class H, typename A, typename S>
H<A, S>
operator*(const double x, const H<A, S> &b) {
  H<A, S> r(b);
  r *= x;
  return r;
}

template <template <class, class> class H, typename A, typename S>
H<A, S> &&
operator/(H<A, S> &&a, const double x) {
  a /= x;
  return std::move(a);
}

template <template <class, class> class H, typename A, typename S>
H<A, S>
operator/(const H<A, S> &a, const double x) {
  H<A, S> r(a);
  r /= x;
  return r;
}

} // namespace histogram
} // namespace boost

#endif

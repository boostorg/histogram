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

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator+(histogram<Variant, Axes, Storage> &&a,
          const histogram<Variant, Axes, Storage> &b) {
  a += b;
  return std::move(a);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator+(histogram<Variant, Axes, Storage> &&a,
          histogram<Variant, Axes, Storage> &&b) {
  a += b;
  return std::move(a);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator+(const histogram<Variant, Axes, Storage> &a,
          histogram<Variant, Axes, Storage> &&b) {
  b += a;
  return std::move(b);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage>
operator+(const histogram<Variant, Axes, Storage> &a,
          const histogram<Variant, Axes, Storage> &b) {
  histogram<Variant, Axes, Storage> r(a);
  r += b;
  return r;
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator*(histogram<Variant, Axes, Storage> &&a, const double x) {
  a *= x;
  return std::move(a);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator*(const double x, histogram<Variant, Axes, Storage> &&b) {
  b *= x;
  return std::move(b);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage>
operator*(const histogram<Variant, Axes, Storage> &a, const double x) {
  histogram<Variant, Axes, Storage> r(a);
  r *= x;
  return r;
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage>
operator*(const double x, const histogram<Variant, Axes, Storage> &b) {
  histogram<Variant, Axes, Storage> r(b);
  r *= x;
  return r;
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage> &&
operator/(histogram<Variant, Axes, Storage> &&a, const double x) {
  a /= x;
  return std::move(a);
}

template <typename Variant, typename Axes, typename Storage>
histogram<Variant, Axes, Storage>
operator/(const histogram<Variant, Axes, Storage> &a, const double x) {
  histogram<Variant, Axes, Storage> r(a);
  r /= x;
  return r;
}

} // namespace histogram
} // namespace boost

#endif

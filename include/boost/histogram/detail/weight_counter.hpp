// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_DETAIL_WEIGHT_HPP_
#define _BOOST_HISTOGRAM_DETAIL_WEIGHT_HPP_

namespace boost {
namespace histogram {
namespace detail {

/// Double counter which holds a sum of weights and a sum of squared weights
struct weight_counter {
  double w, w2;
  weight_counter() = default;
  weight_counter(const weight_counter &) = default;
  weight_counter(weight_counter &&) = default;
  weight_counter &operator=(const weight_counter &) = default;
  weight_counter &operator=(weight_counter &&) = default;

  weight_counter(double value, double variance) : w(value), w2(variance) {}

  weight_counter &operator++() {
    ++w;
    ++w2;
    return *this;
  }

  weight_counter &operator+=(const weight_counter &rhs) {
    w += rhs.w;
    w2 += rhs.w2;
    return *this;
  }

  weight_counter &operator*=(const double x) {
    w *= x;
    w2 *= x*x;
    return *this;
  }

  bool operator==(const weight_counter &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }
  bool operator!=(const weight_counter &rhs) const { return !operator==(rhs); }
  template <typename T> bool operator==(const T &rhs) const {
    return w == static_cast<double>(rhs) && w2 == static_cast<double>(rhs);
  }
  template <typename T> bool operator!=(const T &rhs) const {
    return !operator==(rhs);
  }

  weight_counter &operator+=(double t) {
    w += t;
    w2 += t * t;
    return *this;
  }

  template <typename T>
  explicit weight_counter(const T &t)
      : w(static_cast<double>(t)), w2(static_cast<double>(t)) {}

  template <typename T> weight_counter &operator=(const T &t) {
    w = static_cast<double>(t);
    w2 = static_cast<double>(t);
    return *this;
  }
  template <typename T> weight_counter &operator+=(const T &t) {
    w += static_cast<double>(t);
    w2 += static_cast<double>(t);
    return *this;
  }
};

template <typename T> bool operator==(const T &t, const weight_counter &w) {
  return w == t;
}

template <typename T> bool operator!=(const T &t, const weight_counter &w) {
  return !(w == t);
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif

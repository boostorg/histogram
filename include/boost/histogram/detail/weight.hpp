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

/// Used by nstore to hold a sum of weighted counts and a variance estimate
struct weight {
  double w, w2;
  weight() = default;
  weight(const weight &) = default;
  weight(weight &&) = default;
  weight &operator=(const weight &) = default;
  weight &operator=(weight &&) = default;

  weight &operator+=(const weight &rhs) {
    w += rhs.w;
    w2 += rhs.w2;
    return *this;
  }
  weight &operator++() {
    ++w;
    ++w2;
    return *this;
  }

  bool operator==(const weight &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }
  bool operator!=(const weight &rhs) const { return !operator==(rhs); }
  template <typename T> bool operator==(const T &rhs) const {
    return w == static_cast<double>(rhs) && w2 == static_cast<double>(rhs);
  }
  template <typename T> bool operator!=(const T &rhs) const {
    return !operator==(rhs);
  }

  weight &add_weight(double t) {
    w += t;
    w2 += t * t;
    return *this;
  }

  template <typename T>
  explicit weight(const T &t) : w(static_cast<double>(t)), w2(w) {}

  template <typename T> weight &operator=(const T &t) {
    w = static_cast<double>(t);
    w2 = static_cast<double>(t);
    return *this;
  }
  template <typename T> weight &operator+=(const T &t) {
    w += static_cast<double>(t);
    w2 += static_cast<double>(t);
    return *this;
  }
};

template <typename T> bool operator==(const T &t, const weight &w) {
  return w == t;
}

template <typename T> bool operator!=(const T &t, const weight &w) {
  return !(w == t);
}
} // namespace detail
} // namespace histogram
} // namespace boost

#endif

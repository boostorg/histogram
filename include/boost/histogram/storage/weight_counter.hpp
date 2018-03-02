// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_WEIGHT_COUNTER_HPP_
#define _BOOST_HISTOGRAM_STORAGE_WEIGHT_COUNTER_HPP_

#include <boost/histogram/histogram_fwd.hpp>
#include <stdexcept>

namespace boost {

namespace serialization {
class access;
} // namespace serialization

namespace histogram {

/// Double counter which holds a sum of weights and a sum of squared weights
template <typename RealType> class weight_counter {
public:
  /// Beware: For performance reasons counters are not initialized
  weight_counter() = default;
  weight_counter(const weight_counter &) = default;
  weight_counter(weight_counter &&) = default;
  weight_counter &operator=(const weight_counter &) = default;
  weight_counter &operator=(weight_counter &&) = default;

  weight_counter(const RealType& value, const RealType& variance) : w(value), w2(variance) {}

  weight_counter &operator++() {
    ++w;
    ++w2;
    return *this;
  }

  weight_counter &operator+=(const RealType &x) {
    w += x;
    w2 += x;
    return *this;
  }

  template <typename T>
  weight_counter &operator+=(const weight_counter<T> &rhs) {
    w += static_cast<RealType>(rhs.w);
    w2 += static_cast<RealType>(rhs.w2);
    return *this;
  }

  template <typename T>
  weight_counter &operator+=(const detail::weight_t<T> &rhs) {
    const auto x = static_cast<RealType>(rhs.value);
    w += x;
    w2 += x * x;
    return *this;
  }

  weight_counter &operator*=(const RealType &x) {
    w *= x;
    w2 *= x * x;
    return *this;
  }

  bool operator==(const weight_counter &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }

  bool operator!=(const weight_counter &rhs) const { return !operator==(rhs); }

  template <typename T> bool operator==(const weight_counter<T> &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }

  template <typename T> bool operator!=(const weight_counter<T> &rhs) const {
    return !operator==(rhs);
  }

  template <typename T> bool operator==(const T &rhs) const {
    return w == w2 && w == static_cast<RealType>(rhs);
  }

  template <typename T> bool operator!=(const T &rhs) const {
    return !operator==(rhs);
  }

  const RealType& value() const noexcept { return w; }
  const RealType& variance() const noexcept { return w2; }

  bool has_trivial_variance() const noexcept { return w == w2; }

  // conversion
  template <typename T> explicit weight_counter(const T &t) : w(static_cast<T>(t)), w2(w) {}
  explicit operator RealType() const {
    if (!has_trivial_variance())
      throw std::logic_error("cannot convert weight_counter to RealType, value and variance differ");
    return w;
  }
  template <typename T> explicit operator T() const {
    if (!has_trivial_variance())
      throw std::logic_error("cannot convert weight_counter to RealType, value and variance differ");
    return w;
  }
  template <typename T> weight_counter &operator=(const T &x) {
    w = w2 = static_cast<RealType>(x);
    return *this;
  }

private:
  friend class ::boost::serialization::access;

  template <class Archive> void serialize(Archive &ar, unsigned /* version */);

  RealType w, w2;
};

template <typename T, typename U>
bool operator==(const T &t, const weight_counter<U> &w) {
  return w == t;
}

template <typename T, typename U>
bool operator!=(const T &t, const weight_counter<U> &w) {
  return !(w == t);
}

} // namespace histogram
} // namespace boost

#endif

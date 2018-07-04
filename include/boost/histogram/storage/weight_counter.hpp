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

  weight_counter(const RealType &value, const RealType &variance) noexcept
    : w(value), w2(variance) {}

  explicit weight_counter(const RealType &value) noexcept
    : w(value), w2(value) {}

  weight_counter &operator++() {
    ++w;
    ++w2;
    return *this;
  }

  // TODO: explain why this is needed
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
  weight_counter &operator+=(const detail::weight<T> &rhs) {
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

  bool operator==(const weight_counter &rhs) const noexcept {
    return w == rhs.w && w2 == rhs.w2;
  }

  bool operator!=(const weight_counter &rhs) const noexcept {
    return !operator==(rhs);
  }

  template <typename T>
  bool operator==(const weight_counter<T> &rhs) const noexcept {
    return w == rhs.w && w2 == rhs.w2;
  }

  template <typename T>
  bool operator!=(const weight_counter<T> &rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType &value() const noexcept { return w; }
  const RealType &variance() const noexcept { return w2; }

  // conversion
  template <typename T>
  explicit weight_counter(const T &t) { operator=(t); }
  template <typename T> weight_counter &operator=(const T &x) {
    w = w2 = static_cast<RealType>(x);
    return *this;
  }

  // lossy conversion must be explicit
  template <typename T>
  explicit operator T() const { return static_cast<T>(w); }

private:
  friend class ::boost::serialization::access;

  template <class Archive> void serialize(Archive &, unsigned /* version */);

  RealType w, w2;
};

template <typename T, typename U>
bool operator==(const weight_counter<T> &w, const U & u) {
  return w.value() == w.variance() && w.value() == static_cast<T>(u);
}

template <typename T, typename U>
bool operator==(const T & t, const weight_counter<U> &w) {
  return operator==(w, t);
}

template <typename T, typename U>
bool operator!=(const weight_counter<T> &w, const U & u) {
  return !operator==(w, u);
}

template <typename T, typename U>
bool operator!=(const T & t, const weight_counter<U> &w) {
  return operator!=(w, t);
}

template <typename T>
weight_counter<T> operator+(const weight_counter<T>& a, const weight_counter<T>& b) noexcept {
  weight_counter<T> c = a;
  return c += b;
}

template <typename T>
weight_counter<T>&& operator+(weight_counter<T>&& a, const weight_counter<T>& b) noexcept {
  a += b;
  return std::move(a);
}

template <typename T>
weight_counter<T>&& operator+(const weight_counter<T>& a, weight_counter<T>&& b) noexcept {
  return operator+(std::move(b), a);
}

template <typename T>
weight_counter<T> operator+(const weight_counter<T>& a, const T& b) noexcept
{
  auto r = a;
  return r += b;
}

template <typename T>
weight_counter<T> operator+(const T& a, const weight_counter<T>& b) noexcept
{
  auto r = b;
  return r += a;
}

} // namespace histogram
} // namespace boost

#endif

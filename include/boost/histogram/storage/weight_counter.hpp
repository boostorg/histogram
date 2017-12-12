// Copyright 2015-2016 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_STORAGE_WEIGHT_COUNTER_HPP_
#define _BOOST_HISTOGRAM_STORAGE_WEIGHT_COUNTER_HPP_

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

  weight_counter(RealType value, RealType variance) : w(value), w2(variance) {}

  template <typename T>
  explicit weight_counter(const T &t)
      : w(static_cast<RealType>(t)), w2(static_cast<RealType>(t)) {}

  template <typename T> weight_counter &operator=(const T &t) {
    w = static_cast<RealType>(t);
    w2 = static_cast<RealType>(t);
    return *this;
  }

  weight_counter &operator++() {
    ++w;
    ++w2;
    return *this;
  }

  template <typename U>
  weight_counter &operator+=(const weight_counter<U> &rhs) {
    w += rhs.w;
    w2 += rhs.w2;
    return *this;
  }

  weight_counter &operator*=(const RealType x) {
    w *= x;
    w2 *= x * x;
    return *this;
  }

  weight_counter &increase_by_count(const RealType n) {
    w += n;
    w2 += n;
    return *this;
  }

  weight_counter &increase_by_weight(const RealType x) {
    w += x;
    w2 += x * x;
    return *this;
  }

  bool operator==(const weight_counter &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }

  bool operator!=(const weight_counter &rhs) const { return !operator==(rhs); }

  template <typename U> bool operator==(const weight_counter<U> &rhs) const {
    return w == rhs.w && w2 == rhs.w2;
  }

  template <typename U> bool operator!=(const weight_counter<U> &rhs) const {
    return !operator==(rhs);
  }

  template <typename T> bool operator==(const T &rhs) const {
    return w == w2 && w == static_cast<RealType>(rhs);
  }

  template <typename T> bool operator!=(const T &rhs) const {
    return !operator==(rhs);
  }

  RealType value() const noexcept { return w; }
  RealType variance() const noexcept { return w2; }

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

namespace detail {
template <typename T> struct counter_traits; // fwd declaration

// specialization
template <typename T> struct counter_traits<weight_counter<T>> {
  using value_type = T;
  static value_type value(const weight_counter<T> &w) noexcept {
    return w.value();
  }
  static void increase_by_count(weight_counter<T> &lhs, const T &n) noexcept {
    lhs.increase_by_count(n);
  }
  static void increase_by_weight(weight_counter<T> &lhs, const T &w) noexcept {
    lhs.increase_by_weight(w);
  }
};
} // namespace detail

} // namespace histogram
} // namespace boost

#endif

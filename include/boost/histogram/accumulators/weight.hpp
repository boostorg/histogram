// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHT_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHT_HPP

#include <boost/histogram/weight.hpp>
#include <stdexcept>

namespace boost {
namespace histogram {
namespace accumulators {

/// Holds sum of weights and sum of weights squared
template <typename RealType = double>
class weight {
public:
  weight() = default;
  weight(const weight&) = default;
  weight(weight&&) = default;
  weight& operator=(const weight&) = default;
  weight& operator=(weight&&) = default;

  weight(const RealType& value, const RealType& variance) noexcept
      : w_(value), w2_(variance) {}

  explicit weight(const RealType& value) noexcept : w_(value), w2_(value) {}

  void operator()() noexcept {
    ++w_;
    ++w2_;
  }

  template <typename T>
  void operator()(const ::boost::histogram::weight_type<T>& w) noexcept {
    w_ += w.value;
    w2_ += w.value * w.value;
  }

  // used when adding non-weighted histogram to weighted histogram
  weight& operator+=(const RealType& x) noexcept {
    w_ += x;
    w2_ += x;
    return *this;
  }

  template <typename T>
  weight& operator+=(const weight<T>& rhs) {
    w_ += static_cast<RealType>(rhs.w_);
    w2_ += static_cast<RealType>(rhs.w2_);
    return *this;
  }

  weight& operator*=(const RealType& x) noexcept {
    w_ *= x;
    w2_ *= x * x;
    return *this;
  }

  bool operator==(const weight& rhs) const noexcept {
    return w_ == rhs.w_ && w2_ == rhs.w2_;
  }

  bool operator!=(const weight& rhs) const noexcept { return !operator==(rhs); }

  template <typename T>
  bool operator==(const weight<T>& rhs) const noexcept {
    return w_ == rhs.w_ && w2_ == rhs.w2_;
  }

  template <typename T>
  bool operator!=(const weight<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& value() const noexcept { return w_; }
  const RealType& variance() const noexcept { return w2_; }

  // conversion
  template <typename T>
  explicit weight(const T& t) {
    operator=(t);
  }
  template <typename T>
  weight& operator=(const T& x) {
    w_ = w2_ = static_cast<RealType>(x);
    return *this;
  }

  // lossy conversion must be explicit
  template <typename T>
  explicit operator T() const {
    return static_cast<T>(w_);
  }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType w_ = 0, w2_ = 0;
};

template <typename T, typename U>
bool operator==(const weight<T>& w, const U& u) {
  return w.value() == w.variance() && w.value() == static_cast<T>(u);
}

template <typename T, typename U>
bool operator==(const T& t, const weight<U>& w) {
  return operator==(w, t);
}

template <typename T, typename U>
bool operator!=(const weight<T>& w, const U& u) {
  return !operator==(w, u);
}

template <typename T, typename U>
bool operator!=(const T& t, const weight<U>& w) {
  return operator!=(w, t);
}

template <typename T>
weight<T> operator+(const weight<T>& a, const weight<T>& b) noexcept {
  weight<T> c = a;
  return c += b;
}

template <typename T>
weight<T>&& operator+(weight<T>&& a, const weight<T>& b) noexcept {
  a += b;
  return std::move(a);
}

template <typename T>
weight<T>&& operator+(const weight<T>& a, weight<T>&& b) noexcept {
  return operator+(std::move(b), a);
}

template <typename T>
weight<T> operator+(const weight<T>& a, const T& b) noexcept {
  auto r = a;
  return r += b;
}

template <typename T>
weight<T> operator+(const T& a, const weight<T>& b) noexcept {
  auto r = b;
  return r += a;
}

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

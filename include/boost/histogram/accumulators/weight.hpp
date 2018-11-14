// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHT_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHT_HPP

#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {
namespace accumulators {

/// Holds sum of weights and sum of weights squared
template <typename RealType>
class weight {
public:
  weight() = default;
  weight(const RealType& value) noexcept : w_(value), w2_(value) {}
  weight(const RealType& value, const RealType& variance) noexcept
      : w_(value), w2_(variance) {}

  void operator()() noexcept {
    w_ += 1;
    w2_ += 1;
  }

  void operator()(const RealType& w) noexcept {
    w_ += w;
    w2_ += w * w;
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

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_SUM_HPP

#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {
namespace accumulators {

/// Holds sum of weights and its variance estimate
template <typename RealType>
class weighted_sum {
public:
  weighted_sum() = default;
  explicit weighted_sum(const RealType& value) noexcept : sum_(value), sum2_(value) {}
  weighted_sum(const RealType& value, const RealType& variance) noexcept
      : sum_(value), sum2_(variance) {}

  void operator()() {
    sum_ += 1;
    sum2_ += 1;
  }

  template <typename T>
  void operator()(const T& w) {
    sum_ += w;
    sum2_ += w * w;
  }

  // used when adding non-weighted histogram to weighted histogram
  template <typename T>
  weighted_sum& operator+=(const T& x) {
    sum_ += x;
    sum2_ += x;
    return *this;
  }

  template <typename T>
  weighted_sum& operator+=(const weighted_sum<T>& rhs) {
    sum_ += static_cast<RealType>(rhs.sum_);
    sum2_ += static_cast<RealType>(rhs.sum2_);
    return *this;
  }

  weighted_sum& operator*=(const RealType& x) {
    sum_ *= x;
    sum2_ *= x * x;
    return *this;
  }

  bool operator==(const RealType& rhs) const noexcept {
    return sum_ == rhs && sum2_ == rhs;
  }

  template <typename T>
  bool operator==(const weighted_sum<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && sum2_ == rhs.sum2_;
  }

  template <typename T>
  bool operator!=(const T& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& value() const noexcept { return sum_; }
  const RealType& variance() const noexcept { return sum2_; }

  // lossy conversion must be explicit
  template <typename T>
  explicit operator T() const {
    return static_cast<T>(sum_);
  }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType sum_ = RealType(), sum2_ = RealType();
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_MEAN_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_WEIGHTED_MEAN_HPP

#include <boost/histogram/histogram_fwd.hpp>

namespace boost {
namespace histogram {
namespace accumulators {

/**
  Calculates mean and variance of weighted sample.

  Uses West's incremental algorithm to improve numerical stability
  of mean and variance computation.
*/
template <typename RealType>
class weighted_mean {
public:
  weighted_mean() = default;
  weighted_mean(const RealType& wsum, const RealType& wsum2, const RealType& mean,
                const RealType& variance)
      : sum_(wsum), sum2_(wsum2), mean_(mean), dsum2_(variance * (sum_ - sum2_ / sum_)) {}

  void operator()(const RealType& x) { operator()(1, x); }

  void operator()(const RealType& w, const RealType& x) {
    sum_ += w;
    sum2_ += w * w;
    const auto delta = x - mean_;
    mean_ += w * delta / sum_;
    dsum2_ += w * delta * (x - mean_);
  }

  template <typename T>
  weighted_mean& operator+=(const weighted_mean<T>& rhs) {
    const auto tmp = mean_ * sum_ + static_cast<RealType>(rhs.mean_ * rhs.sum_);
    sum_ += static_cast<RealType>(rhs.sum_);
    sum2_ += static_cast<RealType>(rhs.sum2_);
    mean_ = tmp / sum_;
    dsum2_ += static_cast<RealType>(rhs.dsum2_);
    return *this;
  }

  weighted_mean& operator*=(const RealType& s) {
    mean_ *= s;
    dsum2_ *= s * s;
    return *this;
  }

  template <typename T>
  bool operator==(const weighted_mean<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && sum2_ == rhs.sum2_ && mean_ == rhs.mean_ &&
           dsum2_ == rhs.dsum2_;
  }

  template <typename T>
  bool operator!=(const T& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& sum() const noexcept { return sum_; }
  const RealType& value() const noexcept { return mean_; }
  RealType variance() const { return dsum2_ / (sum_ - sum2_ / sum_); }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType sum_ = RealType(), sum2_ = RealType(), mean_ = RealType(), dsum2_ = RealType();
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

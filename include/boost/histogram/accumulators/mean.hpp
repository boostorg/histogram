// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <cstddef>

namespace boost {
namespace histogram {
namespace accumulators {

/**
  Calculates mean and variance of sample.

  Uses Welfords's incremental algorithm to improve the numerical
  stability of mean and variance computation.
*/
template <typename RealType>
class mean {
public:
  mean() = default;
  mean(const std::size_t n, const RealType& mean, const RealType& variance)
      : sum_(n), mean_(mean), dsum2_(variance * (sum_ - 1)) {}

  void operator()(const RealType& x) {
    sum_ += 1;
    const auto delta = x - mean_;
    mean_ += delta / sum_;
    dsum2_ += delta * (x - mean_);
  }

  template <typename T>
  mean& operator+=(const mean<T>& rhs) {
    const auto tmp = mean_ * sum_ + static_cast<RealType>(rhs.mean_ * rhs.sum_);
    sum_ += static_cast<RealType>(rhs.sum_);
    mean_ = tmp / sum_;
    dsum2_ += static_cast<RealType>(rhs.dsum2_);
    return *this;
  }

  mean& operator*=(const RealType& s) {
    mean_ *= s;
    dsum2_ *= s * s;
    return *this;
  }

  template <typename T>
  bool operator==(const mean<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && mean_ == rhs.mean_ && dsum2_ == rhs.dsum2_;
  }

  template <typename T>
  bool operator!=(const mean<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  std::size_t sum() const noexcept { return sum_; }
  const RealType& value() const noexcept { return mean_; }
  RealType variance() const { return dsum2_ / (sum_ - 1); }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  std::size_t sum_ = 0;
  RealType mean_ = 0, dsum2_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

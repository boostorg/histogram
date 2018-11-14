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
      : wsum_(wsum)
      , wsum2_(wsum2)
      , mean_(mean)
      , dsum2_(variance * (wsum_ - wsum2_ / wsum_)) {}

  void operator()(const RealType& x) noexcept { operator()(1, x); }

  void operator()(const RealType& w, const RealType& x) noexcept {
    wsum_ += w;
    wsum2_ += w * w;
    const auto delta = x - mean_;
    mean_ += w * delta / wsum_;
    dsum2_ += w * delta * (x - mean_);
  }

  template <typename T>
  weighted_mean& operator+=(const weighted_mean<T>& rhs) {
    const auto tmp = mean_ * wsum_ + static_cast<RealType>(rhs.mean_ * rhs.wsum_);
    wsum_ += static_cast<RealType>(rhs.wsum_);
    wsum2_ += static_cast<RealType>(rhs.wsum2_);
    mean_ = tmp / wsum_;
    dsum2_ += static_cast<RealType>(rhs.dsum2_);
    return *this;
  }

  weighted_mean& operator*=(const RealType& s) noexcept {
    mean_ *= s;
    dsum2_ *= s * s;
    return *this;
  }

  template <typename T>
  bool operator==(const weighted_mean<T>& rhs) const noexcept {
    return wsum_ == rhs.wsum_ && wsum2_ == rhs.wsum2_ && mean_ == rhs.mean_ &&
           dsum2_ == rhs.dsum2_;
  }

  template <typename T>
  bool operator!=(const weighted_mean<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& sum() const noexcept { return wsum_; }
  const RealType& sum2() const noexcept { return wsum2_; }
  const RealType& value() const noexcept { return mean_; }
  RealType variance() const noexcept { return dsum2_ / (wsum_ - wsum2_ / wsum_); }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType wsum_ = 0, wsum2_ = 0, mean_ = 0, dsum2_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

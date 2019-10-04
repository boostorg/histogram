// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_MEAN_HPP

#include <boost/assert.hpp>
#include <boost/histogram/fwd.hpp>
#include <type_traits>

namespace boost {
namespace histogram {
namespace accumulators {

/** Calculates mean and variance of sample.

  Uses Welfords's incremental algorithm to improve the numerical
  stability of mean and variance computation.
*/
template <class RealType>
class mean {
public:
  mean() = default;
  mean(const RealType& n, const RealType& mean, const RealType& variance)
      : sum_(n), mean_(mean), sum_of_deltas_squared_(variance * (n - 1)) {}

  void operator()(const RealType& x) {
    sum_ += static_cast<RealType>(1);
    const auto delta = x - mean_;
    BOOST_ASSERT(sum_ != 0);
    mean_ += delta / sum_;
    sum_of_deltas_squared_ += delta * (x - mean_);
  }

  void operator()(const RealType& w, const RealType& x) {
    sum_ += w;
    const auto delta = x - mean_;
    BOOST_ASSERT(sum_ != 0);
    mean_ += w * delta / sum_;
    sum_of_deltas_squared_ += w * delta * (x - mean_);
  }

  template <class T>
  mean& operator+=(const mean<T>& rhs) {
    const auto tmp = mean_ * sum_ + static_cast<RealType>(rhs.mean_ * rhs.sum_);
    sum_ += rhs.sum_;
    BOOST_ASSERT(sum_ != 0);
    mean_ = tmp / sum_;
    sum_of_deltas_squared_ += static_cast<RealType>(rhs.sum_of_deltas_squared_);
    return *this;
  }

  mean& operator*=(const RealType& s) {
    mean_ *= s;
    sum_of_deltas_squared_ *= s * s;
    return *this;
  }

  template <class T>
  bool operator==(const mean<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && mean_ == rhs.mean_ &&
           sum_of_deltas_squared_ == rhs.sum_of_deltas_squared_;
  }

  template <class T>
  bool operator!=(const mean<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& count() const noexcept { return sum_; }
  const RealType& value() const noexcept { return mean_; }
  RealType variance() const { return sum_of_deltas_squared_ / (sum_ - 1); }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType sum_ = 0, mean_ = 0, sum_of_deltas_squared_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U>
/// Specialization for boost::histogram::accumulators::mean.
struct common_type<boost::histogram::accumulators::mean<T>,
                   boost::histogram::accumulators::mean<U>> {
  using type = boost::histogram::accumulators::mean<common_type_t<T, U>>;
};
} // namespace std
#endif

#endif

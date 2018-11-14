// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_NEUMAIER_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_NEUMAIER_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <cmath>

namespace boost {
namespace histogram {
namespace accumulators {

/**
  Uses Neumaier algorithm to compute accurate sums.

  The algorithm is about four times slower compared to using
  a simple floating point number to accumulate a sum, but the
  relative error of the sum for non-negative numbers is
  constant and at the level of the machine precision.

  A. Neumaier, Zeitschrift fuer Angewandte Mathematik
  und Mechanik 54 (1974) 39–51.
*/
template <typename RealType>
class neumaier {
public:
  neumaier() = default;
  neumaier(const RealType& value) noexcept : sum_(value), cor_(0) {}
  neumaier& operator=(const RealType& value) noexcept {
    sum_ = value;
    cor_ = 0;
    return *this;
  }

  void operator()() noexcept { operator+=(1); }

  void operator()(const RealType& x) noexcept { operator+=(x); }

  neumaier& operator+=(const RealType& x) noexcept {
    volatile auto temp = sum_ + x; // prevent optimization
    if (std::abs(sum_) >= std::abs(x))
      cor_ += (sum_ - temp) + x;
    else
      cor_ += (x - temp) + sum_;
    sum_ = temp;
    return *this;
  }

  neumaier& operator*=(const RealType& x) noexcept {
    sum_ *= x;
    cor_ *= x;
    return *this;
  }

  template <typename T>
  bool operator==(const neumaier<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && cor_ == rhs.cor_;
  }

  template <typename T>
  bool operator!=(const neumaier<T>& rhs) const noexcept {
    return !operator==(rhs);
  }

  RealType value() const noexcept { return sum_ + cor_; }
  const RealType& large() const noexcept { return sum_; }
  const RealType& small() const noexcept { return cor_; }

  operator RealType() const noexcept { return value(); }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType sum_ = 0, cor_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#endif

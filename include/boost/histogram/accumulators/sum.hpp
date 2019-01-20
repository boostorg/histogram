// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ACCUMULATORS_SUM_HPP
#define BOOST_HISTOGRAM_ACCUMULATORS_SUM_HPP

#include <boost/histogram/fwd.hpp>
#include <cmath>
#include <type_traits>

namespace boost {
namespace histogram {
namespace accumulators {

/**
  Uses Neumaier algorithm to compute accurate sums.

  The algorithm uses memory for two floats and is three to
  five times slower compared to a simple floating point
  number used to accumulate a sum, but the relative error
  of the sum is at the level of the machine precision,
  independent of the number of samples.

  A. Neumaier, Zeitschrift fuer Angewandte Mathematik
  und Mechanik 54 (1974) 39–51.
*/
template <typename RealType>
class sum {
public:
  sum() = default;
  explicit sum(const RealType& value) noexcept : sum_(value), cor_(0) {}
  sum& operator=(const RealType& value) noexcept {
    sum_ = value;
    cor_ = 0;
    return *this;
  }

  void operator()() { operator+=(1); }

  void operator()(const RealType& x) { operator+=(x); }

  sum& operator+=(const RealType& x) {
    auto temp = sum_ + x; // prevent optimization
    if (std::abs(sum_) >= std::abs(x))
      cor_ += (sum_ - temp) + x;
    else
      cor_ += (x - temp) + sum_;
    sum_ = temp;
    return *this;
  }

  sum& operator*=(const RealType& x) {
    sum_ *= x;
    cor_ *= x;
    return *this;
  }

  template <typename T>
  bool operator==(const sum<T>& rhs) const noexcept {
    return sum_ == rhs.sum_ && cor_ == rhs.cor_;
  }

  template <typename T>
  bool operator!=(const T& rhs) const noexcept {
    return !operator==(rhs);
  }

  const RealType& large() const noexcept { return sum_; }
  const RealType& small() const noexcept { return cor_; }

  // allow implicit conversion to RealType
  operator RealType() const { return sum_ + cor_; }

  template <class Archive>
  void serialize(Archive&, unsigned /* version */);

private:
  RealType sum_ = 0, cor_ = 0;
};

} // namespace accumulators
} // namespace histogram
} // namespace boost

#ifndef BOOST_HISTOGRAM_DOXYGEN_INVOKED
namespace std {
template <class T, class U>
struct common_type<boost::histogram::accumulators::sum<T>,
                   boost::histogram::accumulators::sum<U>> {
  using type = boost::histogram::accumulators::sum<common_type_t<T, U>>;
};
} // namespace std
#endif

#endif

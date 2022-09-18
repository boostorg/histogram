// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_WALD_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_WALD_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

/**
  Wald interval or normal approximation interval.

  The Wald interval is a symmetric interval. It is simple to compute,
  but has poor statistical properties and should always be replaced
  by another iternal. The Wilson interval is also easy to compute
  and has far superior statistical properties.

  The Wald interval is commonly used by practitioners, since it can
  be derived easily using the plug-in estimate of the variance for
  the binomial distribution. Without further insight into statistical
  theory, it is not clear that this derivation is flawed and that
  better alternatives exist.

  The Wald interval is an approximation based on the central limit
  theorem, which is unsuitable when the sample size is small or when
  the fraction is close to 0 or 1. It undercovers on average. Its
  limits are not naturally bounded by 0 or 1. It produces empty
  intervals if the number of successes or failures is zero.

  For a critique of the Wald interval, see
  R. D. Cousins, K. E. Hymes, J. Tucker,
  Nucl. Instrum. Meth. A 612 (2010) 388-398.
*/
template <class ValueType>
class wald_interval : public binomial_proportion_interval<ValueType> {
  using base_t = binomial_proportion_interval<ValueType>;

public:
  using value_type = typename base_t::value_type;
  using interval_type = typename base_t::interval_type;

  /** Construct Wald interval computer.

    @param d Number of standard deviations for the interval. The default value 1
    corresponds to a confidence level of 68 %. Both `deviation` and `confidence_level`
    objects can be used to initialize the interval.
  */
  explicit wald_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<value_type>(d)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    // See https://en.wikipedia.org/wiki/
    //   Binomial_proportion_confidence_interval
    //   #Normal_approximation_interval_or_Wald_interval
    const value_type s = successes, f = failures;
    const value_type n = s + f;
    const value_type ninv = 1 / n;
    const value_type a = s * ninv;
    const value_type b = (z_ * ninv) * std::sqrt(s * f * ninv);
    return std::make_pair(a - b, a + b);
  }

private:
  value_type z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
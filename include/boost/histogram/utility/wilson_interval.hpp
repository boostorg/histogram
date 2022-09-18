// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_WILSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_WILSON_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

/**
  Wilson interval.

  The Wilson score interval is simple to compute, has good coverage. Intervals are
  automatically bounded between 0 and 1 and never empty. The interval is asymmetric.

  Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical
  inference". Journal of the American Statistical Association. 22 (158): 209-212.
  doi:10.1080/01621459.1927.10502953. JSTOR 2276774.

  The coverage probability is close to the nominal value. Unlike the Clopper-Pearson
  interval, it is not conservative. For some values of the fractions, the interval
  undercovers and overcovers for neighboring values. On "average", the intervals have
  close to nomimal coverage. The Wilson score intervals and the Clopper-Pearson are
  recommended for general use (the second if the interval has to be conservative)
  in a review of the existing literature by R. D. Cousins, K. E. Hymes, J. Tucker,
  Nucl. Instrum. Meth. A 612 (2010) 388-398.
*/
template <class ValueType>
class wilson_interval : public binomial_proportion_interval<ValueType> {
  using base_t = binomial_proportion_interval<ValueType>;

public:
  using value_type = typename base_t::value_type;
  using interval_type = typename base_t::interval_type;

  explicit wilson_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<value_type>(d)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    // See https://en.wikipedia.org/wiki/
    //   Binomial_proportion_confidence_interval
    //   #Wilson_score_interval

    // We make sure calculation is done in single precision if value_type is float
    // by converting all literal constants to value_type. If the literals remain
    // in the equation, they turn the calculation of intermediate falues to double,
    // since mixed arithemetic between float and double results in double.
    const value_type half{0.5}, quarter{0.25}, zsq{z_ * z_};
    const value_type s = successes, f = failures;
    const value_type n = s + f;
    const value_type a = (s + half * zsq) / (n + zsq);
    const value_type b = z_ / (n + zsq) * std::sqrt(s * f / n + quarter * zsq);
    return std::make_pair(a - b, a + b);
  }

private:
  value_type z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/distributions/beta.hpp>
#include <cmath>

namespace boost {
namespace histogram {
namespace utility {

template <class ValueType>
class clopper_pearson_interval : public binomial_proportion_interval<ValueType> {
public:
  using value_type = typename clopper_pearson_interval::value_type;
  using interval_type = typename clopper_pearson_interval::interval_type;

  explicit clopper_pearson_interval(confidence_level cl = deviation{1}) noexcept
      : alpha_half_{static_cast<value_type>(0.5 - 0.5 * static_cast<double>(cl))} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    // Source:
    // https://en.wikipedia.org/wiki/
    //   Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type zero{0}, one{1};
    const value_type total = successes + failures;

    if (successes == 0) return {zero, one - std::pow(alpha_half_, one / total)};
    if (failures == 0) return {std::pow(alpha_half_, one / total), one};

    math::beta_distribution<value_type> beta_a(successes, failures + one);
    const value_type a = math::quantile(beta_a, alpha_half_);
    math::beta_distribution<value_type> beta_b(successes + one, failures);
    const value_type b = math::quantile(beta_b, one - alpha_half_);
    return {a, b};
  }

private:
  value_type alpha_half_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
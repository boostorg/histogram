// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP

#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

template <class ValueType>
class clopper_pearson_interval : public binomial_proportion_interval<ValueType> {
  using base_t = binomial_proportion_interval<ValueType>;

public:
  using value_type = typename base_t::value_type;
  using interval_type = typename base_t::interval_type;

  explicit clopper_pearson_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<double>(d)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const value_type half{0.5}, one{1.0};
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    const value_type alpha1 = (one - d) * half;
    const value_type alpha2 = one - ((one - d) * half);
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type m1 = ns;
    const value_type n1 = n - ns + one;
    const value_type m2 = ns + 1;
    const value_type n2 = n - ns;
    // Source: https://en.wikipedia.org/wiki/Beta_distribution
    const value_type beta1 = (tgamma(m1)*tgamma(n1)) / tgamma(m1+n1);
    const value_type beta2 = (tgamma(m2)*tgamma(n2)) / tgamma(m2+n2);
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type a = (one / beta1) * std::pow(alpha1, m1 - one) * std::pow(1 - alpha1, n1 - one);
    const value_type b = (one / beta2) * std::pow(alpha2, m2 - one) * std::pow(1 - alpha2, n2 - one);
    return std::make_pair(a, b);
  }

private:
  double z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
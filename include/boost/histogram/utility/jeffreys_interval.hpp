// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_JEFFREYS_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_JEFFREYS_INTERVAL_HPP

#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <cmath>
#include <utility>

namespace boost {
namespace histogram {
namespace utility {

template <class ValueType>
class jeffreys_interval : public binomial_proportion_interval<ValueType> {
  using base_t = binomial_proportion_interval<ValueType>;

public:
  using value_type = typename base_t::value_type;
  using interval_type = typename base_t::interval_type;

  explicit jeffreys_interval(deviation d = deviation{1.0}) noexcept
      : z_{static_cast<double>(d)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const value_type half{0.5}, one{1.0}, two{2.0};
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    const value_type alpha1 = (one - d) * half;
    const value_type alpha2 = one - ((one - d) * half);
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type m = ns + half;
    const value_type n = n - ns + half;
    // Source: https://en.wikipedia.org/wiki/Beta_distribution
    const value_type beta = (tgamma(m)*tgamma(n)) / tgamma(m+n);
    // Source: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1440&context=statistics_papers
    const value_type a = (one / beta) * std::pow(alpha1, m - one) * std::pow(1 - alpha1, n - one);
    const value_type b = (one / beta) * std::pow(alpha2, m - one) * std::pow(1 - alpha2, n - one);
    return std::make_pair(a, b);
  }

private:
  double z_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
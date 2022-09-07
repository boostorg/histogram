// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions.hpp>
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

  explicit clopper_pearson_interval(confidence_level cl = deviation{1}) noexcept
      : cl_{static_cast<value_type>(cl)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const value_type half{0.5}, one{1};
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    const value_type alpha = cl_ * half;
    // Source:
    // https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type m1 = ns;
    const value_type n1 = n - ns + one;
    const value_type m2 = ns + one;
    const value_type n2 = n - ns;
    // if ((m1 == 0) || (m2 == 0) || (n1 == 0) || (n2 == 0)){
    //   throw std::invalid_argument("Beta distribution based arguments' value cannot be zero.");
    //   return std::make_pair(zero, zero);
    // }
    // Source: https://en.wikipedia.org/wiki/Beta_distribution
    // Source:
    // https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/dist_ref/dists/beta_dist.html
    const value_type a = boost::math::quantile(boost::math::beta_distribution<>(m1, n1), alpha * half);
    const value_type b = boost::math::quantile(boost::math::beta_distribution<>(m2, n2), one - (alpha * half));
    return std::make_pair(a, b);
  }

private:
  value_type cl_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
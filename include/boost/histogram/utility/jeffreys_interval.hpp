// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_JEFFREYS_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_JEFFREYS_INTERVAL_HPP

#include <boost/histogram/fwd.hpp>
#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <boost/math/special_functions/beta.hpp>
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

  explicit jeffreys_interval(confidence_level cl = deviation{1}) noexcept
      : cl_{static_cast<value_type>(cl)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const value_type half{0.5}, one{1};
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    const value_type alpha = cl_ * half;
    // Source:
    // https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Jeffreys_interval
    const value_type m1 = ns + half;
    const value_type m2 = n - ns + half;
    // Source: https://en.wikipedia.org/wiki/Beta_distribution
    // Source:
    // https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/dist_ref/dists/beta_dist.html
    const value_type a = boost::math::quantile(boost::math::beta_distribution<>(m1, m2), alpha * half);
    const value_type b = boost::math::quantile(boost::math::beta_distribution<>(m1, m2), one - (alpha * half));
    return std::make_pair(a, b);
  }

private:
  value_type cl_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_CLOPPER_PEARSON_INTERVAL_HPP

#include <boost/histogram/utility/binomial_proportion_interval.hpp>
#include <boost/math/special_functions/beta.hpp>
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

  explicit clopper_pearson_interval(confidence_level cl = confidence_level{0.683}) noexcept
      : cl_{static_cast<double>(cl)} {}

  interval_type operator()(value_type successes, value_type failures) const noexcept {
    const double half{0.5}, one{1.0};
    const value_type ns = successes, nf = failures;
    const value_type n = ns + nf;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    const double alpha = cl_ * half;
    // Source: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    const value_type m1 = ns;
    const value_type n1 = n - ns + one;
    const value_type m2 = ns + 1;
    const value_type n2 = n - ns;
    // Source: https://en.wikipedia.org/wiki/Beta_distribution
    // Source: https://www.boost.org/doc/libs/1_79_0/libs/math/doc/html/math_toolkit/dist_ref/dists/beta_dist.html
    const double a = boost::math::ibeta_inv(m1, n1, alpha * half);
    const double b = boost::math::ibeta_inv(m2, n2, one - (alpha * half));
    return std::make_pair(a, b);
  }

private:
  double cl_;
};

} // namespace utility
} // namespace histogram
} // namespace boost

#endif
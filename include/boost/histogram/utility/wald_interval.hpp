
// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_UTILITY_WALD_INTERVAL_HPP
#define BOOST_HISTOGRAM_UTILITY_WALD_INTERVAL_HPP

#include <cmath>

namespace boost {
namespace histogram {
namespace utility {

// TODO for Jay
template <class T>
T wald_interval(T n_failure, T n_success, T z) {
  return (z * (std::pow((n_failure * n_success), T(0.5)))) /
         std::pow(n_failure + n_success, T(1.5));
  // Source: Wald Interval from Confidence Interval, Wikipedia |
  // https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval_or_Wald_interval
}

} // namespace utility
} // namespace histogram
} // namespace boost

#endif

// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_WILSON_INTERVAL_HPP
#define BOOST_HISTOGRAM_DETAIL_WILSON_INTERVAL_HPP

#include <cmath>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
auto wilson_interval(T n_failure, T n_success, T z) {
  return (z/(n_failure+n_success+(std::pow(z, T(2)))))*(std::pow(((n_failure*n_success)/(n_failure + n_success))+((std::pow(z,T(2)))*T(0.25)), T(0.5)));
  // make pair here
  // Source: Wilson Interval from Confidence Interval, Wikipedia | https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

// Copyright 2022 Jay Gohil, Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_WALD_INTERVAL_HPP
#define BOOST_HISTOGRAM_DETAIL_WALD_INTERVAL_HPP

#include <cmath>

namespace boost {
namespace histogram {
namespace detail {

template <class T>
T waldInterval(T n_failure_, T n_success_, T z) {
  return (z*(std::pow((n_failure_*n_success_), 0.5))) / std::pow(n_failure_+n_success_, 1.5); // Source: Wald Interval from Confidence Interval | Wikipedia
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

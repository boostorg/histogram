// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_SUM_HPP
#define BOOST_HISTOGRAM_ALGORITHM_SUM_HPP

#include <boost/histogram/histogram_fwd.hpp>
#include <numeric>
#include <type_traits>

namespace boost {
namespace histogram {
namespace algorithm {
template <typename A, typename S>
typename histogram<A, S>::value_type sum(const histogram<A, S>& h) {
  using T = typename histogram<A, S>::value_type;
  return std::accumulate(h.begin(), h.end(),
                         std::conditional_t<std::is_integral<T>::value, double, T>());
}
} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

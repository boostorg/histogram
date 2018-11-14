// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_SUM_HPP
#define BOOST_HISTOGRAM_ALGORITHM_SUM_HPP

#include <boost/histogram/accumulators/sum.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <numeric>
#include <type_traits>

namespace boost {
namespace histogram {
namespace algorithm {
template <
    typename A, typename S, typename T = typename histogram<A, S>::value_type,
    typename ReturnType = std::conditional_t<std::is_arithmetic<T>::value, double, T>,
    typename InternalSum =
        std::conditional_t<std::is_arithmetic<T>::value, accumulators::sum<double>, T>>
ReturnType sum(const histogram<A, S>& h) {
  InternalSum sum;
  for (auto x : h) sum += x;
  return sum;
}
} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_MAKE_PROFILE_HPP
#define BOOST_HISTOGRAM_MAKE_PROFILE_HPP

#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/make_histogram.hpp>

namespace boost {
namespace histogram {

/// profile factory from compile-time axis configuration
template <typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_profile(T&& axis, Ts&&... axes) {
  return make_histogram_with(profile_storage(), std::forward<T>(axis),
                             std::forward<Ts>(axes)...);
}

/// profile factory from compile-time axis configuration with weighted mean
template <typename T, typename... Ts, typename = detail::requires_axis<T>>
auto make_weighted_profile(T&& axis, Ts&&... axes) {
  return make_histogram_with(weighted_profile_storage(), std::forward<T>(axis),
                             std::forward<Ts>(axes)...);
}

/// profile factory from vector-like
template <typename Iterable, typename = detail::requires_axis_vector<Iterable>>
auto make_profile(Iterable&& c) {
  return make_histogram_with(profile_storage(), std::forward<Iterable>(c));
}

/// profile factory from vector-like with weighted mean
template <typename Iterable, typename = detail::requires_axis_vector<Iterable>>
auto make_weighted_profile(Iterable&& c) {
  return make_histogram_with(weighted_profile_storage(), std::forward<Iterable>(c));
}

/// profile factory from iterator range
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_profile(Iterator begin, Iterator end) {
  return make_histogram_with(profile_storage(), begin, end);
}

/// profile factory from iterator range with weighted mean
template <typename Iterator, typename = detail::requires_iterator<Iterator>>
auto make_weighted_profile(Iterator begin, Iterator end) {
  return make_histogram_with(weighted_profile_storage(), begin, end);
}

} // namespace histogram
} // namespace boost

#endif

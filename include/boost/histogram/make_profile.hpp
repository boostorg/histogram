// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_MAKE_PROFILE_HPP
#define BOOST_HISTOGRAM_MAKE_PROFILE_HPP

#include <boost/histogram/accumulators/mean.hpp>
#include <boost/histogram/accumulators/weighted_mean.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/histogram/make_histogram.hpp>

/**
  \file boost/histogram/make_profile.hpp
  Collection of factory functions to conveniently create profiles.

  Profiles are histograms which accept an additional sample and compute the mean of the
  sample in each cell.
*/

namespace boost {
namespace histogram {

/// profile factory from compile-time axis configuration
template <typename Axis, typename... Axes, typename = detail::requires_axis<Axis>>
auto make_profile(Axis&& axis, Axes&&... axes) {
  return make_histogram_with(profile_storage(), std::forward<Axis>(axis),
                             std::forward<Axes>(axes)...);
}

/// profile factory from compile-time axis configuration with weighted mean
template <typename Axis, typename... Axes, typename = detail::requires_axis<Axis>>
auto make_weighted_profile(Axis&& axis, Axes&&... axes) {
  return make_histogram_with(weighted_profile_storage(), std::forward<Axis>(axis),
                             std::forward<Axes>(axes)...);
}

/// profile factory from vector-like
template <typename Iterable, typename = detail::requires_sequence_of_any_axis<Iterable>>
auto make_profile(Iterable&& c) {
  return make_histogram_with(profile_storage(), std::forward<Iterable>(c));
}

/// profile factory from vector-like with weighted mean
template <typename Iterable, typename = detail::requires_sequence_of_any_axis<Iterable>>
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

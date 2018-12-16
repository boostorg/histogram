// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP
#define BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP

#include <algorithm>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/meta.hpp>
#include <boost/histogram/histogram.hpp>
#include <boost/histogram/indexed.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <type_traits>

namespace boost {
namespace histogram {
namespace algorithm {

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  Arguments are the source histogram and compile-time numbers, the remaining indices of
  the axes. Returns a new histogram which only contains the subset of axes. The source
  histogram is summed over the removed axes.
*/
template <typename A, typename S, unsigned N, typename... Ns>
auto project(const histogram<A, S>& hist, std::integral_constant<unsigned, N> n,
             Ns... ns) {
  using LN = mp11::mp_list<decltype(n), Ns...>;
  static_assert(mp11::mp_is_set<LN>::value, "indices must be unique");

  auto axes = detail::make_sub_axes(unsafe_access::axes(hist), n, ns...);
  auto result = histogram<decltype(axes), S>(
      std::move(axes), detail::make_default(unsafe_access::storage(hist)));

  detail::axes_buffer<typename decltype(result)::axes_type, int> idx(result.rank());
  for (auto x : indexed(hist, true)) {
    auto i = idx.begin();
    mp11::mp_for_each<LN>([&i, &x](auto I) { *i++ = x[I]; });
    unsafe_access::add_value(result, idx, *x);
  }
  return result;
}

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  This version accepts a source histogram and an iterable range containing the remaining
  indices.
*/
template <typename A, typename S, typename Iterable,
          typename = detail::requires_iterable<Iterable>>
auto project(const histogram<A, S>& hist, const Iterable& c) {
  static_assert(detail::is_axis_vector<A>::value,
                "dynamic version of project requires a histogram with dynamic axis");

  const auto& hist_axes = unsafe_access::axes(hist);
  auto axes = detail::make_default(hist_axes);
  axes.reserve(c.size());
  detail::axes_buffer<A, bool> seen(hist_axes.size(), false);
  for (auto d : c) {
    if (seen[d]) BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    seen[d] = true;
    axes.emplace_back(hist_axes[d]);
  }

  auto result = histogram<A, S>(std::move(axes),
                                detail::make_default(unsafe_access::storage(hist)));

  detail::axes_buffer<A, int> idx(result.rank());
  for (auto x : indexed(hist, true)) {
    auto i = idx.begin();
    for (auto d : c) *i++ = x[d];
    unsafe_access::add_value(result, idx, *x);
  }
  return result;
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

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
  Returns a lower-dimensional grid, summing over removed axes.

  Arguments are the source histogram and compile-time numbers, the remaining indices of
  the axes. Returns a new histogram which only contains the subset of axes. The source
  histogram is summed over the removed axes.
*/
template <template <class, class> class Grid, class A, class S, unsigned N,
          typename... Ns>
auto project(const Grid<A, S>& grid, std::integral_constant<unsigned, N> n, Ns... ns) {
  using LN = mp11::mp_list<decltype(n), Ns...>;
  static_assert(mp11::mp_is_set<LN>::value, "indices must be unique");

  auto axes = detail::make_sub_axes(unsafe_access::axes(grid), n, ns...);
  auto result = Grid<decltype(axes), S>(
      std::move(axes), detail::make_default(unsafe_access::storage(grid)));

  detail::axes_buffer<decltype(axes), int> idx(result.rank());
  for (auto x : indexed(grid, true)) {
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
template <template <class, class> class Grid, class A, class S, class Iterable,
          class = detail::requires_iterable<Iterable>>
auto project(const Grid<A, S>& grid, const Iterable& c) {
  static_assert(detail::is_sequence_of_any_axis<A>::value,
                "dynamic version of project requires a grid with dynamic axis");

  const auto& old_axes = unsafe_access::axes(grid);
  auto axes = detail::make_default(old_axes);
  axes.reserve(c.size());
  detail::axes_buffer<A, bool> seen(old_axes.size(), false);
  for (auto d : c) {
    if (seen[d]) BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    seen[d] = true;
    axes.emplace_back(old_axes[d]);
  }

  auto result =
      Grid<A, S>(std::move(axes), detail::make_default(unsafe_access::storage(grid)));

  detail::axes_buffer<A, int> idx(result.rank());
  for (auto x : indexed(grid, true)) {
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

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
template <class A, class S, unsigned N, typename... Ns>
auto project(const histogram<A, S>& h, std::integral_constant<unsigned, N> n, Ns... ns) {
  using LN = mp11::mp_list<decltype(n), Ns...>;
  static_assert(mp11::mp_is_set<LN>::value, "indices must be unique");

  auto axes = detail::make_sub_axes(unsafe_access::axes(h), n, ns...);
  auto result = histogram<decltype(axes), S>(
      std::move(axes), detail::make_default(unsafe_access::storage(h)));

  detail::axes_buffer<decltype(axes), int> idx(result.rank());
  for (auto x : indexed(h, true)) {
    auto i = idx.begin();
    mp11::mp_for_each<LN>([&i, &x](auto I) { *i++ = x[I]; });
    result.at(idx) += *x;
  }
  return result;
}

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  This version accepts a source histogram and an iterable range containing the remaining
  indices.
*/
template <class A, class S, class Iterable, class = detail::requires_iterable<Iterable>>
auto project(const histogram<A, S>& h, const Iterable& c) {
  static_assert(detail::is_sequence_of_any_axis<A>::value,
                "this version of project requires histogram with non-static axes");

  const auto& old_axes = unsafe_access::axes(h);
  auto axes = detail::make_default(old_axes);
  axes.reserve(c.size());
  detail::axes_buffer<A, bool> seen(old_axes.size(), false);
  for (auto d : c) {
    if (seen[d]) BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    seen[d] = true;
    axes.emplace_back(old_axes[d]);
  }

  auto result =
      histogram<A, S>(std::move(axes), detail::make_default(unsafe_access::storage(h)));

  detail::axes_buffer<A, int> idx(result.rank());
  for (auto x : indexed(h, true)) {
    auto i = idx.begin();
    for (auto d : c) *i++ = x[d];
    result.at(idx) += *x;
  }
  return result;
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

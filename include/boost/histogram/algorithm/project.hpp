// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP
#define BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP

#include <algorithm>
#include <array>
#include <boost/assert.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/detail/is_set.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>
#include <stdexcept>
#include <tuple>

namespace boost {
namespace histogram {
namespace algorithm {

// TODO: make generic reduce, which can sum over axes, shrink, rebin

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  Arguments are the source histogram and compile-time numbers, representing the indices of
  axes that are kept. Returns a new histogram which only contains the subset of axes.
  The source histogram is summed over the removed axes.
*/
template <typename A, typename S, std::size_t I, typename... Ns>
auto project(const histogram<A, S>& h, mp11::mp_size_t<I> n, Ns... ns) {
  using LN = mp11::mp_list<mp11::mp_size_t<I>, Ns...>;
  static_assert(mp11::mp_is_set<LN>::value, "indices must be unique");

  const auto& axes = unsafe_access::axes(h);
  auto r_axes = detail::make_sub_axes(axes, n, ns...);
  auto r_h = histogram<decltype(r_axes), S>(
      std::move(r_axes),
      detail::static_if<detail::has_allocator<S>>(
          [&h](auto) { return S(unsafe_access::storage(h).get_allocator()); },
          [](auto) { return S(); }, 0));

  detail::index_mapper im(h.rank());
  auto iter = im.begin();
  std::size_t s = 1;
  h.for_each_axis([&](const auto& a) {
    const auto n = axis::traits::extend(a);
    im.ntotal *= n;
    iter->first = s;
    s *= n;
    iter->second = 0;
    ++iter;
  });

  s = 1;
  mp11::mp_for_each<LN>([&](auto J) {
    im[J].second = s;
    s *= axis::traits::extend(detail::axis_get<J>(axes));
  });

  do {
    const auto x = unsafe_access::storage(h)[im.first];
    unsafe_access::storage(r_h).add(im.second, x);
  } while (im.next());
  return r_h;
}

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  This version accepts an iterator range that represents the indices which are kept.
*/
template <typename A, typename S, typename Iterator,
          typename = detail::requires_axis_vector<A>,
          typename = detail::requires_iterator<Iterator>>
auto project(const histogram<A, S>& h, Iterator begin, Iterator end) {
  BOOST_ASSERT_MSG(detail::is_set(begin, end), "indices must be unique");
  using H = histogram<A, S>;

  const auto& axes = unsafe_access::axes(h);
  auto r_axes = typename H::axes_type(axes.get_allocator());
  r_axes.reserve(std::distance(begin, end));

  detail::index_mapper im(h.rank());
  auto iter = im.begin();
  std::size_t s = 1;
  h.for_each_axis([&](const auto& a) {
    const auto n = axis::traits::extend(a);
    im.ntotal *= n;
    iter->first = s;
    s *= n;
    iter->second = 0;
    ++iter;
  });

  s = 1;
  for (auto it = begin; it != end; ++it) {
    r_axes.emplace_back(axes[*it]);
    im[*it].second = s;
    s *= axis::traits::extend(axes[*it]);
  }

  auto r_h = H(std::move(r_axes),
               detail::static_if<detail::has_allocator<S>>(
                   [&h](auto) { return S(unsafe_access::storage(h).get_allocator()); },
                   [](auto) { return S(); }, 0));

  do {
    const auto x = unsafe_access::storage(h)[im.first];
    unsafe_access::storage(r_h).add(im.second, x);
  } while (im.next());
  return r_h;
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

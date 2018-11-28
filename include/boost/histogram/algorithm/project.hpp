// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP
#define BOOST_HISTOGRAM_ALGORITHM_PROJECT_HPP

#include <algorithm>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/index_mapper.hpp>
#include <boost/histogram/histogram_fwd.hpp>
#include <boost/histogram/unsafe_access.hpp>
#include <boost/mp11.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace boost {
namespace histogram {
namespace algorithm {

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

  detail::index_mapper<A> im(h.rank());
  auto iter = im.begin();
  std::size_t s = 1;
  h.for_each_axis([&](const auto& a) {
    const auto n = axis::traits::extend(a);
    im.total *= n;
    iter->stride[0] = s;
    s *= n;
    iter->stride[1] = 0;
    ++iter;
  });

  s = 1;
  mp11::mp_for_each<LN>([&](auto J) {
    im[J].stride[1] = s;
    s *= axis::traits::extend(detail::axis_get<J>(axes));
  });

  im(unsafe_access::storage(r_h), unsafe_access::storage(h));
  return r_h;
}

/**
  Returns a lower-dimensional histogram, summing over removed axes.

  This version accepts an iterable range that represents the indices which are kept.
*/
template <typename A, typename S, typename C, typename = detail::requires_axis_vector<A>,
          typename = detail::requires_iterable<C>>
auto project(const histogram<A, S>& h, C c) {
  using H = histogram<A, S>;

  using std::begin;
  using std::end;

  const auto& axes = unsafe_access::axes(h);
  auto r_axes = detail::static_if<detail::has_allocator<A>>(
      [](const auto& axes) {
        using T = detail::unqual<decltype(axes)>;
        return T(axes.get_allocator());
      },
      [](const auto& axes) {
        using T = detail::unqual<decltype(axes)>;
        return T();
      },
      axes);
  r_axes.reserve(std::distance(begin(c), end(c)));

  detail::index_mapper<A> im(h.rank());
  auto iter = im.begin();
  std::size_t stride = 1;
  h.for_each_axis([&](const auto& a) {
    const auto n = axis::traits::extend(a);
    im.total *= n;
    iter->stride[0] = stride;
    stride *= n;
    iter->stride[1] = 0;
    ++iter;
  });

  stride = 1;
  for (auto idx : c) {
    r_axes.emplace_back(axes[idx]);
    auto& stride_ref = im[idx].stride[1];
    if (stride_ref)
      BOOST_THROW_EXCEPTION(std::invalid_argument("indices must be unique"));
    else
      stride_ref = stride;
    stride *= axis::traits::extend(axes[idx]);
  }

  auto r_h = H(std::move(r_axes),
               detail::static_if<detail::has_allocator<S>>(
                   [&h](auto) { return S(unsafe_access::storage(h).get_allocator()); },
                   [](auto) { return S(); }, 0));

  im(unsafe_access::storage(r_h), unsafe_access::storage(h));
  return r_h;
}

} // namespace algorithm
} // namespace histogram
} // namespace boost

#endif

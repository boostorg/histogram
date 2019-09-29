// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_AT_HPP
#define BOOST_HISTOGRAM_DETAIL_AT_HPP

#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11.hpp>
#include <tuple>

namespace boost {
namespace histogram {
namespace detail {

template <class A>
std::size_t linearize_index(optional_index& out, const std::size_t stride, const A& ax,
                            const axis::index_type i) {
  // cannot use static_options here, since A may be variant
  const auto opt = axis::traits::options(ax);
  const axis::index_type begin = opt & axis::option::underflow ? -1 : 0;
  const axis::index_type end = opt & axis::option::overflow ? ax.size() + 1 : ax.size();
  const axis::index_type extent = end - begin;
  // i may be arbitrarily out of range
  if (begin <= i && i < end)
    out += i * stride;
  else
    out = invalid_index;
  return extent;
}

template <class A, class... Us>
optional_index at(std::size_t offset, const A& axes,
                  const std::tuple<Us...>& args) noexcept {
  optional_index idx{offset};
  using namespace boost::mp11;
  mp_for_each<mp_iota_c<sizeof...(Us)>>([&, stride = 1ull](auto i) mutable {
    stride *= linearize_index(idx, stride, axis_get<i>(axes),
                              static_cast<axis::index_type>(std::get<i>(args)));
  });
  return idx;
}

template <class A, class U>
optional_index at(std::size_t offset, const A& axes, const U& args) noexcept {
  optional_index idx{offset};
  using std::begin;
  auto it = begin(args);
  for_each_axis(axes, [&, stride = 1ull](const auto& a) mutable {
    stride *= linearize_index(idx, stride, a, static_cast<axis::index_type>(*it++));
  });
  return idx;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

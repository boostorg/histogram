// Copyright 2015-2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_AT_HPP
#define BOOST_HISTOGRAM_DETAIL_AT_HPP

#include <boost/assert.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/detail/axes.hpp>
#include <boost/histogram/detail/linearize.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/fwd.hpp>
#include <boost/mp11.hpp>
#include <tuple>

namespace boost {
namespace histogram {
namespace detail {

template <class A>
std::size_t linearize_index(optional_index& out, const std::size_t stride, const A& axis,
                            const axis::index_type i) {
  // A may be axis or variant, cannot use static option detection here
  const auto opt = axis::traits::options(axis);
  const auto shift = opt & axis::option::underflow ? 1 : 0;
  const auto extent = axis.size() + (opt & axis::option::overflow ? 1 : 0) + shift;
  // i may be arbitrarily out of range
  using namespace boost::mp11;
  linearize(mp_false{}, mp_false{}, out, stride, extent, i + shift);
  return extent;
}

template <class A, class... Us>
optional_index at(const A& axes, const std::tuple<Us...>& args) noexcept {
  optional_index idx{0};
  std::size_t stride = 1;
  using namespace boost::mp11;
  mp_for_each<mp_iota_c<sizeof...(Us)>>([&](auto i) {
    stride *= linearize_index(idx, stride, axis_get<i>(axes),
                              static_cast<axis::index_type>(std::get<i>(args)));
  });
  return idx;
}

template <class A, class U>
optional_index at(const A& axes, const U& args) noexcept {
  optional_index idx{0};
  std::size_t stride = 1;
  using std::begin;
  auto it = begin(args);
  for_each_axis(axes, [&](const auto& a) {
    stride *= linearize_index(idx, stride, a, static_cast<axis::index_type>(*it++));
  });
  return idx;
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif

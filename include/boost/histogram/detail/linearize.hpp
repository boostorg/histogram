// Copyright 2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP
#define BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP

#include <boost/assert.hpp>
#include <boost/histogram/axis/option.hpp>
#include <boost/histogram/axis/traits.hpp>
#include <boost/histogram/axis/variant.hpp>
#include <boost/histogram/detail/optional_index.hpp>
#include <boost/histogram/fwd.hpp>

namespace boost {
namespace histogram {
namespace detail {

template <class Index, class Opts>
std::size_t linearize(Opts, Index& out, const std::size_t stride,
                      const axis::index_type size, const axis::index_type idx) {
  constexpr bool u = Opts::test(axis::option::underflow);
  constexpr bool o = Opts::test(axis::option::overflow);
  if (std::is_same<Index, std::size_t>::value) {
    BOOST_ASSERT(idx >= (u ? -1 : 0));
    BOOST_ASSERT(idx < (o ? size + 1 : size));
    out += idx * stride;
  } else {
    BOOST_ASSERT(idx >= -1);
    BOOST_ASSERT(idx < size + 1);
    if ((u || idx >= 0) && (o || idx < size))
      out += idx * stride;
    else
      out = invalid_index;
  }
  return size + u + o;
}

template <class Index, class Axis, class Value>
std::size_t linearize(Index& out, const std::size_t stride, const Axis& ax,
                      const Value& v) {
  // mask options to reduce no. of template instantiations
  constexpr auto opts = axis::traits::static_options<Axis>{} &
                        (axis::option::underflow | axis::option::overflow);
  return linearize(opts, out, stride, ax.size(), axis::traits::index(ax, v));
}

template <class Index, class Axis, class Value>
std::size_t linearize_growth(Index& out, axis::index_type& shift,
                             const std::size_t stride, Axis& a, const Value& v) {
  axis::index_type idx;
  std::tie(idx, shift) = axis::traits::update(a, v);
  constexpr bool u = axis::traits::static_options<Axis>::test(axis::option::underflow);
  // initial offset of out must be zero
  if (u) ++idx;
  if (std::is_same<Index, std::size_t>::value) {
    BOOST_ASSERT(idx < axis::traits::extent(a));
    out += idx * stride;
  } else {
    if (0 <= idx && idx < axis::traits::extent(a))
      out += idx * stride;
    else
      out = invalid_index;
  }
  return axis::traits::extent(a);
}

template <class Index, class... Ts, class Value>
std::size_t linearize(Index& o, const std::size_t s, const axis::variant<Ts...>& a,
                      const Value& v) {
  return axis::visit([&o, &s, &v](const auto& a) { return linearize(o, s, a, v); }, a);
}

template <class Index, class... Ts, class Value>
std::size_t linearize_growth(Index& o, axis::index_type& sh, const std::size_t st,
                             axis::variant<Ts...>& a, const Value& v) {
  return axis::visit([&](auto& a) { return linearize_growth(o, sh, st, a, v); }, a);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP

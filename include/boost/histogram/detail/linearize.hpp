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

template <class Opts>
std::size_t linearize(Opts, optional_index& out, const std::size_t stride,
                      const axis::index_type size, const axis::index_type idx) {
  constexpr bool u = Opts::test(axis::option::underflow);
  constexpr bool o = Opts::test(axis::option::overflow);
  BOOST_ASSERT(idx >= -1);
  BOOST_ASSERT(idx <= size);
  if ((u || idx >= 0) && (o || idx < size))
    out += idx * stride;
  else
    out = invalid_index;
  return size + u + o;
}

template <class Opts>
std::size_t linearize(Opts, std::size_t& out, const std::size_t stride,
                      const axis::index_type size, const axis::index_type idx) {
  constexpr bool u = Opts::test(axis::option::underflow);
  constexpr bool o = Opts::test(axis::option::overflow);
  BOOST_ASSERT(idx >= (u ? -1 : 0));
  BOOST_ASSERT(idx < (o ? size + 1 : size));
  out += idx * stride;
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

template <class Axis, class Value>
std::size_t linearize_growth(optional_index& o, axis::index_type& shift,
                             const std::size_t stride, Axis& a, const Value& v) {
  // mask options to reduce no. of template instantiations
  constexpr auto opts = axis::traits::static_options<Axis>{} &
                        (axis::option::underflow | axis::option::overflow);
  axis::index_type idx;
  std::tie(idx, shift) = axis::traits::update(a, v);
  linearize(opts, o, stride, a.size(), idx);
  return axis::traits::extent(a);
}

template <class Index, class... Ts, class Value>
std::size_t linearize(Index& o, const std::size_t s, const axis::variant<Ts...>& a,
                      const Value& v) {
  return axis::visit([&o, &s, &v](const auto& a) { return linearize(o, s, a, v); }, a);
}

template <class... Ts, class Value>
std::size_t linearize_growth(optional_index& o, axis::index_type& sh,
                             const std::size_t st, axis::variant<Ts...>& a,
                             const Value& v) {
  return axis::visit([&](auto& a) { return linearize_growth(o, sh, st, a, v); }, a);
}

} // namespace detail
} // namespace histogram
} // namespace boost

#endif // BOOST_HISTOGRAM_DETAIL_LINEARIZE_HPP

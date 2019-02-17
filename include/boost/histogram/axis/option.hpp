// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_OPTION_HPP
#define BOOST_HISTOGRAM_AXIS_OPTION_HPP

#include <type_traits>

/**
  \file Options for builtin axis types.

  Options circular and growth are mutually exclusive.
  Options circular and underflow are mutually exclusive.
*/

namespace boost {
namespace histogram {
namespace axis {

/// Holder of axis options.
template <unsigned Bits>
struct option_set : std::integral_constant<unsigned, Bits> {};

namespace option {
template <unsigned N>
struct bit : option_set<(1 << N)> {};

/// All bits set to zero.
using none = option_set<0>;
/// Axis has underflow bin. Mutually exclusive with circular.
using underflow = bit<0>;
/// Axis has overflow bin.
using overflow = bit<1>;
/// Axis is circular. Mutually exclusive with growth and underflow.
using circular = bit<2>;
/// Axis can grow. Mutually exclusive with circular.
using growth = bit<3>;
} // namespace option

} // namespace axis

namespace detail {

constexpr inline unsigned join_impl(unsigned a) { return a; }

template <unsigned N, class... Ts>
constexpr unsigned join_impl(unsigned a, axis::option::bit<N>, Ts... ts) {
  using namespace axis::option;
  const auto o = bit<N>::value;
  const auto c = a | o;
  if (o == underflow::value) return join_impl(c & ~circular::value, ts...);
  if (o == circular::value)
    return join_impl(c & ~(underflow::value | growth::value), ts...);
  if (o == growth::value) return join_impl(c & ~circular::value, ts...);
  return join_impl(c, ts...);
}

} // namespace detail

namespace axis {
/// Combines options and corrects for mutually exclusive options.
template <class T, class... Ts>
using join = axis::option_set<detail::join_impl(T::value, Ts{}...)>;

/// Test whether the bits in b are also set in a.
template <class T, class U>
using test = std::integral_constant<bool, (T::value & U::value ? 1 : 0)>;
} // namespace axis
} // namespace histogram
} // namespace boost

#endif

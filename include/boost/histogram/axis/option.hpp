// Copyright 2015-2019 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_AXIS_OPTION_HPP
#define BOOST_HISTOGRAM_AXIS_OPTION_HPP

#include <boost/mp11.hpp>
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

template <class T, class U>
struct join_impl : axis::option_set<(T::value | U::value)> {};
template <class T>
struct join_impl<T, axis::option::underflow>
    : axis::option_set<((T::value & ~axis::option::circular::value) |
                        axis::option::underflow::value)> {};
template <class T>
struct join_impl<T, axis::option::circular>
    : axis::option_set<(
          (T::value & ~(axis::option::growth::value | axis::option::underflow::value)) |
          axis::option::circular::value)> {};
template <class T>
struct join_impl<T, axis::option::growth>
    : axis::option_set<((T::value & ~axis::option::circular::value) |
                        axis::option::growth::value)> {};

} // namespace detail

namespace axis {
/// Combines options and corrects for mutually exclusive options.
template <class... Ts>
using join = mp11::mp_fold<mp11::mp_list<Ts...>, option_set<0>, detail::join_impl>;

/// Test whether the bits in b are also set in a.
template <class T, class U>
using test = std::integral_constant<bool, (T::value & U::value ? 1 : 0)>;
} // namespace axis
} // namespace histogram
} // namespace boost

#endif

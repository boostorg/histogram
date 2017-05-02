// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_HISTOGRAM_LITERALS_HPP_
#define _BOOST_HISTOGRAM_LITERALS_HPP_

#include <type_traits>

namespace boost {
namespace histogram {
namespace literals {
namespace detail {
template <char C> struct char2int;
template <> struct char2int<'0'> { static constexpr unsigned value = 0; };
template <> struct char2int<'1'> { static constexpr unsigned value = 1; };
template <> struct char2int<'2'> { static constexpr unsigned value = 2; };
template <> struct char2int<'3'> { static constexpr unsigned value = 3; };
template <> struct char2int<'4'> { static constexpr unsigned value = 4; };
template <> struct char2int<'5'> { static constexpr unsigned value = 5; };
template <> struct char2int<'6'> { static constexpr unsigned value = 6; };
template <> struct char2int<'7'> { static constexpr unsigned value = 7; };
template <> struct char2int<'8'> { static constexpr unsigned value = 8; };
template <> struct char2int<'9'> { static constexpr unsigned value = 9; };

template <unsigned N> constexpr unsigned parse() { return N; }

template <unsigned N, char First, char... Rest> constexpr unsigned parse() {
  return parse<N * 10 + char2int<First>::value, Rest...>();
}
} // namespace detail

template <char... Digits>
auto operator"" _c() -> decltype(
    std::integral_constant<unsigned, detail::parse<0, Digits...>()>()) {
  return std::integral_constant<unsigned, detail::parse<0, Digits...>()>();
}

} // namespace literals
} // namespace histogram
} // namespace boost

#endif

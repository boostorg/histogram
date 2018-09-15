// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_LITERALS_HPP
#define BOOST_HISTOGRAM_LITERALS_HPP

#include <boost/mp11.hpp>
#include <cstddef>

namespace boost {
namespace histogram {
namespace literals {
namespace detail {
template <char C>
struct char2int;
template <>
struct char2int<'0'> {
  static constexpr std::size_t value = 0;
};
template <>
struct char2int<'1'> {
  static constexpr std::size_t value = 1;
};
template <>
struct char2int<'2'> {
  static constexpr std::size_t value = 2;
};
template <>
struct char2int<'3'> {
  static constexpr std::size_t value = 3;
};
template <>
struct char2int<'4'> {
  static constexpr std::size_t value = 4;
};
template <>
struct char2int<'5'> {
  static constexpr std::size_t value = 5;
};
template <>
struct char2int<'6'> {
  static constexpr std::size_t value = 6;
};
template <>
struct char2int<'7'> {
  static constexpr std::size_t value = 7;
};
template <>
struct char2int<'8'> {
  static constexpr std::size_t value = 8;
};
template <>
struct char2int<'9'> {
  static constexpr std::size_t value = 9;
};

template <std::size_t N>
constexpr std::size_t parse() {
  return N;
}

template <std::size_t N, char First, char... Rest>
constexpr std::size_t parse() {
  return parse<N * 10 + char2int<First>::value, Rest...>();
}
} // namespace detail

template <char... Digits>
auto operator"" _c() -> ::boost::mp11::mp_size_t<detail::parse<0, Digits...>()> {
  return ::boost::mp11::mp_size_t<detail::parse<0, Digits...>()>();
}

} // namespace literals
} // namespace histogram
} // namespace boost

#endif

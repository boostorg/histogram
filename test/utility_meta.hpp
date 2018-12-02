// Copyright 2018 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_META_HPP
#define BOOST_HISTOGRAM_TEST_UTILITY_META_HPP

#include <array>
#include <boost/mp11/tuple.hpp>
#include <ostream>
#include <tuple>
#include <vector>

namespace std {
// never add to std, we only do it here to get ADL working :(
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}

template <typename... Ts>
ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
  os << "[ ";
  ::boost::mp11::tuple_for_each(t, [&os](const auto& x) { os << x << " "; });
  os << "]";
  return os;
}

template <typename T, std::size_t N>
ostream& operator<<(ostream& os, const array<T, N>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}
} // namespace std

#endif

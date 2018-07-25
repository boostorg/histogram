// Copyright 2015-2017 Hans Dembinski
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_HISTOGRAM_TEST_UTILITY_HPP_
#define BOOST_HISTOGRAM_TEST_UTILITY_HPP_

#include <boost/mp11/integral.hpp>
#include <boost/mp11/tuple.hpp>
#include <ostream>
#include <tuple>
#include <vector>

using i0 = boost::mp11::mp_int<0>;
using i1 = boost::mp11::mp_int<1>;
using i2 = boost::mp11::mp_int<2>;
using i3 = boost::mp11::mp_int<3>;

namespace std { // never add to std, we only do it to get ADL working
template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v) {
  os << "[ ";
  for (const auto& x : v) os << x << " ";
  os << "]";
  return os;
}

struct ostreamer {
  ostream& os;
  template <typename T>
  void operator()(const T& t) const {
    os << t << " ";
  }
};

template <typename... Ts>
ostream& operator<<(ostream& os, const tuple<Ts...>& t) {
  os << "[ ";
  ::boost::mp11::tuple_for_each(t, ostreamer{os});
  os << "]";
  return os;
}
} // namespace std

#endif
